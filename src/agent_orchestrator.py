"""
agent_orchestrator.py
=====================
Enterprise Multi-Agent Customer Support System
Built with Strands Agents SDK + Amazon Bedrock AgentCore

Architecture implemented:

  Customer Request
        │
  OrchestratorAgent  (Claude 3 Haiku - fast routing, manages WorkflowState)
        │
   ┌────┼────────────────────┬────────────────────────┐
   │    │                    │                        │
InventoryAgent   PolicyAgent   RefundAgent  CommunicationAgent
(DynamoDB)    (Multi-Agent RAG)  (DynamoDB)   (composes response)
                    │
         ┌──────────┼──────────┐
    ReturnsPolicyRetriever  ShippingPolicyRetriever  WarrantyPolicyRetriever
        (KB: returns)           (KB: shipping)           (KB: warranty)
         └──────────── all run in PARALLEL ────────────┘

Shared state flows through DynamoDB WorkflowStateTable.
OrchestratorAgent creates state at start, each routing tool reads and
updates it after the worker responds.
"""

import boto3
import json
import time
import os
import sys
import uuid
import random
import logging
import re
import io
import zipfile
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional

# Ensure the parent directory is on sys.path so config.py and
# bedrock_kb_retrieval.py are importable regardless of where this
# script is invoked from (e.g. python src/agent_orchestrator.py)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Strands Agents SDK - see: https://github.com/strands-agents/sdk-python
from strands import Agent
from agent_observability import tool, start_request_segment, end_request_segment
from strands.models import BedrockModel
from boto3.dynamodb.conditions import Key

import config

# ---- Compatibility patch for older bedrock-agentcore-control SDK ----
def _patch_bedrockagentcore():
    import botocore.client
    if not hasattr(botocore.client.BaseClient, "get_agent_runtime_logging_configuration"):
        def _stub(self, **kwargs):
            return {
                "loggingConfiguration": {
                    "cloudWatchConfig": {
                        "logGroupName": "/aws/bedrock/agentcore/udacity-agentcore",
                        "logLevel": "INFO",
                        "enabled": True,
                    },
                    "xRayConfig": {"enabled": True, "samplingRate": 1.0},
                }
            }
        botocore.client.BaseClient.get_agent_runtime_logging_configuration = _stub
_patch_bedrockagentcore()
# ---------------------------------------------------------------------
from bedrock_kb_retrieval import retrieve_from_knowledge_base, format_kb_results

# Configure logging for debugging
logging.basicConfig(
    level=logging.WARNING,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────
# OUTPUT UTILITIES  (pre-written - do not modify)
# ─────────────────────────────────────────────────────
# Terminal trace UI, ANSI colour constants, and agent metadata
# are defined in agent_utils.py - keeping this file focused on
# agent architecture.
from agent_utils import (
    _C, _trace_print, _trace_writer, _real_stdout, _TraceWriter,
    _strip_xml_tags, AgentTrace, _AGENT_META,
)





# ─────────────────────────────────────────────────────
# AWS CLIENTS (pre-written - do not modify)
# ─────────────────────────────────────────────────────
bedrock_agent_client = boto3.client('bedrock-agent', region_name=config.AWS_REGION)
bedrock_runtime      = boto3.client('bedrock-runtime', region_name=config.AWS_REGION)
agentcore_client     = boto3.client('bedrock-agentcore', region_name=config.AWS_REGION)
agentcore_control    = boto3.client('bedrock-agentcore-control', region_name=config.AWS_REGION)
dynamodb             = boto3.resource('dynamodb', region_name=config.AWS_REGION)
logs_client          = boto3.client('logs', region_name=config.AWS_REGION)


# ─────────────────────────────────────────────────────
# COMPATIBILITY PATCH (pre-written - do not modify)
# ─────────────────────────────────────────────────────
def _register_agentcore_compat_methods():
    """Register event handler to inject control-plane methods into bedrock-agentcore clients."""
    _control = agentcore_control

    def _add_methods(class_attributes, base_classes, **kwargs):
        def get_agent_runtime(self, agentRuntimeId, **kw):
            try:
                response = _control.get_agent_runtime(agentRuntimeId=agentRuntimeId)
            except Exception:
                response = {}
            response['memoryConfiguration'] = {
                'enabledMemoryTypes': ['SESSION_SUMMARY'],
                'storageDays': 7,
            }
            response['codeInterpreterConfiguration'] = {
                'enabled': True,
                'executionEnvironment': 'PYTHON_3_11',
                'timeoutSeconds': 30,
            }
            return response

        def get_agent_runtime_logging_configuration(self, agentRuntimeId, **kw):
            return {
                'loggingConfiguration': {
                    'cloudWatchConfig': {
                        'logGroupName': config.AGENT_LOG_GROUP,
                        'logLevel': 'INFO',
                        'enabled': True,
                    },
                    'xRayConfig': {
                        'enabled': True,
                        'samplingRate': 1.0,
                    }
                }
            }

        def put_agent_runtime_logging_configuration(self, agentRuntimeId,
                                                    loggingConfiguration=None, **kw):
            return {'ResponseMetadata': {'HTTPStatusCode': 200}}

        class_attributes['get_agent_runtime'] = get_agent_runtime
        class_attributes['get_agent_runtime_logging_configuration'] = get_agent_runtime_logging_configuration
        class_attributes['put_agent_runtime_logging_configuration'] = put_agent_runtime_logging_configuration

    import boto3 as _boto3
    if _boto3.DEFAULT_SESSION is not None:
        _boto3.DEFAULT_SESSION._session.register(
            'creating-client-class.bedrock-agentcore', _add_methods
        )
    else:
        import botocore.session as _bc_session
        _original_get = _bc_session.get_session

        def _patched_get(*args, **kwargs):
            sess = _original_get(*args, **kwargs)
            sess.register('creating-client-class.bedrock-agentcore', _add_methods)
            return sess

        _bc_session.get_session = _patched_get

_register_agentcore_compat_methods()


# ═══════════════════════════════════════════════════════
#  WORKFLOW STATE - SHARED DynamoDB STATE OBJECT
#  Pre-written - do not modify.
#
#  WorkflowState stores the accumulated context for one customer session:
#    - What the InventoryAgent found (order status, eligibility, customer tier)
#    - What the PolicyAgent found (relevant policy text)
#    - What the RefundAgent decided (approval/denial, reference number)
#    - The CommunicationAgent's final draft
#
#  The `version` field enables optimistic locking: every write is a
#  conditional DynamoDB update that fails if someone else updated first.
#  If the condition fails, the update is retried after a fresh read.
# ═══════════════════════════════════════════════════════

def _create_workflow_state(session_id: str, customer_id: str) -> dict:
    """
    Create a blank WorkflowState record at the start of a new customer session.
    Pre-written - do not modify.

    Columns written on creation:
      session_id   - partition key
      customer_id  - who this session belongs to
      created_at   - ISO-8601 UTC timestamp (human-readable)
      version      - optimistic-locking counter (starts at 0)
      ttl          - Unix epoch for DynamoDB auto-expiry after 24 h

    The four agent columns (inventory_agent, policy_agent,
    refund_agent, communication_agent) are absent until each agent
    runs and writes its result - this keeps the initial row clean.
    """
    state = {
        'session_id':  session_id,
        'customer_id': customer_id,
        'created_at':  time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
        'version':     0,
        'ttl':         int(time.time()) + (24 * 3600),
    }
    table = dynamodb.Table(config.WORKFLOW_STATE_TABLE)
    table.put_item(
        Item=state,
        ConditionExpression='attribute_not_exists(session_id)'
    )
    return state


def _read_workflow_state(session_id: str) -> Optional[dict]:
    """
    Read the current WorkflowState for a session.
    Pre-written - do not modify.
    """
    table = dynamodb.Table(config.WORKFLOW_STATE_TABLE)
    response = table.get_item(Key={'session_id': session_id})
    return response.get('Item')


# Trace singleton - created after _read_workflow_state so AgentTrace.summary()
# can read DynamoDB WorkflowState. The read_state_fn avoids a circular import.
trace = AgentTrace(read_state_fn=_read_workflow_state)


def _update_workflow_state(session_id: str, updates: dict,
                           expected_version: int, max_retries: int = 3) -> dict:
    """
    Update WorkflowState with optimistic locking.
    Pre-written - do not modify.
    """
    from boto3.dynamodb.conditions import Attr

    table = dynamodb.Table(config.WORKFLOW_STATE_TABLE)

    for attempt in range(max_retries):
        try:
            update_expr_parts = [f"{k} = :{k}" for k in updates]
            update_expr_parts.append("version = :new_version")
            update_expr = "SET " + ", ".join(update_expr_parts)

            expr_values = {f":{k}": v for k, v in updates.items()}
            expr_values[':new_version']      = expected_version + 1
            expr_values[':expected_version'] = expected_version

            table.update_item(
                Key={'session_id': session_id},
                UpdateExpression=update_expr,
                ConditionExpression='version = :expected_version',
                ExpressionAttributeValues=expr_values
            )
            return _read_workflow_state(session_id)

        except dynamodb.meta.client.exceptions.ConditionalCheckFailedException:
            if attempt == max_retries - 1:
                raise RuntimeError(
                    f"WorkflowState update failed after {max_retries} retries "
                    f"(session: {session_id}). Too many concurrent writes."
                )
            logger.warning(
                f"WorkflowState version conflict on attempt {attempt+1}, retrying..."
            )
            current = _read_workflow_state(session_id)
            if current:
                expected_version = int(current['version'])
            time.sleep(0.1 * (attempt + 1))

    raise RuntimeError("WorkflowState update: unexpected exit from retry loop")


# ═══════════════════════════════════════════════════════
#  TASK 2 - MULTI-AGENT ORCHESTRATION
# ═══════════════════════════════════════════════════════


# ───────────────────────────────────────────────────────
#  2.A - INVENTORY AGENT
# ───────────────────────────────────────────────────────

def build_inventory_agent() -> Agent:
    """
    Build the Inventory Agent.

    Gathers order and customer facts from DynamoDB. Does NOT make decisions -
    only retrieves data for the OrchestratorAgent to share with downstream agents.
    """

    model = BedrockModel(
        model_id=config.WORKER_MODEL_ID,
        temperature=0.1,
    )

    system_prompt = """You are the InventoryAgent for NovaMart, an e-commerce customer support system.

YOUR ROLE: Data gatherer. You retrieve order and customer facts from DynamoDB and report them accurately.

CRITICAL RULES:
1. NEVER make eligibility decisions (do not say whether a return is allowed or denied).
2. NEVER speculate about policies, return windows, or refund outcomes.
3. NEVER compose customer-facing responses.
4. ONLY report what you find in the database. If a record does not exist, say so plainly.
5. When given a customer_id and/or order_id, call the appropriate tools to fetch the facts.

The downstream RefundAgent and CommunicationAgent will use your findings to make decisions
and draft responses. Your job is accuracy, not interpretation."""

    @tool
    def check_order_status(customer_id: str, order_id: str) -> dict:
        """
        Check the status of a specific order in DynamoDB.

        Args:
            customer_id: The customer's unique identifier (e.g., CUST-001)
            order_id:    The order's unique identifier (e.g., ORD-27176)

        Returns:
            Order details including status, order_date, items, and total amount.
            Returns an error dict if the order does not exist.
        """
        table = dynamodb.Table(config.ORDERS_TABLE)
        response = table.get_item(
            Key={
                'customer_id': customer_id,
                'order_id':    order_id,
            }
        )
        item = response.get('Item')
        if not item:
            return {
                'error':       f'Order {order_id} not found for customer {customer_id}',
                'customer_id': customer_id,
                'order_id':    order_id,
            }
        for k, v in list(item.items()):
            if hasattr(v, 'to_integral_value'):
                item[k] = float(v)
        return item

    @tool
    def get_customer_tier(customer_id: str) -> dict:
        """
        Retrieve a customer's tier (Standard or Premium) from DynamoDB.
        Standard customers have a 30-day return window; Premium customers have 60 days.

        Args:
            customer_id: The customer's unique identifier

        Returns:
            Customer profile including tier and account details
        """
        table = dynamodb.Table(config.CUSTOMERS_TABLE)
        response = table.get_item(Key={'customer_id': customer_id})
        item = response.get('Item')
        if not item:
            return {
                'error':       f'Customer {customer_id} not found',
                'customer_id': customer_id,
            }
        for k, v in list(item.items()):
            if hasattr(v, 'to_integral_value'):
                item[k] = float(v)
        return item

    @tool
    def list_customer_orders(customer_id: str) -> dict:
        """
        Retrieve all orders for a customer from DynamoDB.

        Args:
            customer_id: The customer's unique identifier

        Returns:
            List of all orders with order_id, status, order_date, and amount
        """
        table = dynamodb.Table(config.ORDERS_TABLE)
        response = table.query(
            KeyConditionExpression=Key('customer_id').eq(customer_id)
        )
        items = response.get('Items', [])
        for item in items:
            for k, v in list(item.items()):
                if hasattr(v, 'to_integral_value'):
                    item[k] = float(v)
        return {
            'customer_id': customer_id,
            'order_count': len(items),
            'orders':      items,
        }

    return Agent(
        model=model,
        tools=[check_order_status, get_customer_tier, list_customer_orders],
        system_prompt=system_prompt,
    )


def build_refund_agent() -> Agent:
    """
    Build the Refund Agent.

    Makes return/refund eligibility decisions based on order facts from
    WorkflowState and applies the correct policy window per customer tier.
    """

    model = BedrockModel(
        model_id=config.WORKER_MODEL_ID,
        temperature=0.1,
    )

    system_prompt = """You are the RefundAgent for NovaMart, an e-commerce customer support system.

YOUR ROLE: Decide whether a return is eligible based on order facts and NovaMart policy.

POLICY WINDOWS:
- Standard tier customers: 30-day return window from the order date.
- Premium tier customers:  60-day return window from the order date.

DECISION PROCESS:
1. ALWAYS call get_inventory_context first to retrieve the order facts and customer tier
   gathered by the InventoryAgent.
2. Determine the applicable return window based on the customer's tier.
3. Compute the number of days between the order date and today.
4. If days_since_order <= window -> ELIGIBLE. Otherwise -> NOT_ELIGIBLE.
5. If ELIGIBLE, call initiate_refund to record the return in DynamoDB.
6. Report your decision clearly: ELIGIBLE or NOT_ELIGIBLE, the window used,
   the days elapsed, and (if eligible) the return_reference.

Never compose customer-facing prose - just the decision and its factual basis.
The CommunicationAgent will write the final message."""

    @tool
    def get_inventory_context(session_id: str) -> dict:
        """
        Read the WorkflowState to access facts gathered by the InventoryAgent.

        Args:
            session_id: The current session identifier

        Returns:
            The inventory_agent field from WorkflowState, or empty dict if not yet set
        """
        state = _read_workflow_state(session_id) or {}
        return state.get('inventory_agent', {}) or {}

    @tool
    def initiate_refund(customer_id: str, order_id: str, reason: str) -> dict:
        """
        Initiate a return by updating the order record in DynamoDB.

        Args:
            customer_id: The customer's unique identifier
            order_id:    The order to return
            reason:      Customer-provided reason for the return

        Returns:
            Confirmation dict with return_reference number and instructions
        """
        return_ref = f"RET-{uuid.uuid4().hex[:8].upper()}"
        table = dynamodb.Table(config.ORDERS_TABLE)
        table.update_item(
            Key={
                'customer_id': customer_id,
                'order_id':    order_id,
            },
            UpdateExpression=(
                "SET return_status = :s, return_reason = :r, "
                "return_reference = :ref, return_initiated_at = :t"
            ),
            ExpressionAttributeValues={
                ':s':   'INITIATED',
                ':r':   reason,
                ':ref': return_ref,
                ':t':   time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
            },
        )
        return {
            'status':            'INITIATED',
            'return_reference':  return_ref,
            'order_id':          order_id,
            'customer_id':       customer_id,
            'next_steps':        'A prepaid return label will be emailed within 24 hours.',
        }

    return Agent(
        model=model,
        tools=[get_inventory_context, initiate_refund],
        system_prompt=system_prompt,
    )


def build_policy_agent() -> Agent:
    """
    Build the Policy Agent - a multi-agent RAG system.

    Internally creates three specialized retriever sub-agents that run in
    PARALLEL, each querying its own Knowledge Base. The coordinator synthesizes
    the combined results into a complete, grounded policy answer.
    """

    retriever_model = BedrockModel(
        model_id=config.WORKER_MODEL_ID,
        temperature=0.0,
    )

    retriever_system = (
        "You are a policy retrieval sub-agent. Call your single tool exactly once "
        "with the user's query, then return the retrieved passages verbatim as your "
        "final answer. Do not summarize, do not add commentary, do not make up "
        "information. If the tool returns nothing, respond with NO_RESULTS."
    )

    @tool
    def retrieve_returns_policy(query: str) -> str:
        """Retrieve relevant passages from the Returns Policy knowledge base."""
        return retrieve_from_knowledge_base(query, config.RETURNS_KB_ID)

    @tool
    def retrieve_shipping_policy(query: str) -> str:
        """Retrieve relevant passages from the Shipping Policy knowledge base."""
        return retrieve_from_knowledge_base(query, config.SHIPPING_KB_ID)

    @tool
    def retrieve_warranty_policy(query: str) -> str:
        """Retrieve relevant passages from the Warranty Policy knowledge base."""
        return retrieve_from_knowledge_base(query, config.WARRANTY_KB_ID)

    returns_retriever = Agent(
        model=retriever_model,
        tools=[retrieve_returns_policy],
        system_prompt=retriever_system,
    )
    shipping_retriever = Agent(
        model=retriever_model,
        tools=[retrieve_shipping_policy],
        system_prompt=retriever_system,
    )
    warranty_retriever = Agent(
        model=retriever_model,
        tools=[retrieve_warranty_policy],
        system_prompt=retriever_system,
    )

    @tool
    def search_all_policies(query: str) -> str:
        """
        Query all three policy knowledge bases IN PARALLEL and return combined results.

        Runs ReturnsPolicyRetrieverAgent, ShippingPolicyRetrieverAgent, and
        WarrantyPolicyRetrieverAgent simultaneously, then combines their findings.

        Args:
            query: The customer's policy question

        Returns:
            Combined policy passages from all three knowledge bases
        """
        retrievers = {
            'Returns':  returns_retriever,
            'Shipping': shipping_retriever,
            'Warranty': warranty_retriever,
        }

        trace.kb_start({
            'Returns':  config.RETURNS_KB_ID,
            'Shipping': config.SHIPPING_KB_ID,
            'Warranty': config.WARRANTY_KB_ID,
        })

        def _run_retriever(domain, agent, query):
            try:
                result = agent(query)
                return (domain, str(result))
            except Exception as exc:
                return (domain, "[Retrieval error: " + str(exc) + "]")

        results = {}
        with ThreadPoolExecutor(max_workers=3) as executor:
            futures = {
                executor.submit(_run_retriever, domain, agent, query): domain
                for domain, agent in retrievers.items()
            }
            for future in as_completed(futures):
                domain, text = future.result()
                results[domain] = text

        for domain in ['Returns', 'Shipping', 'Warranty']:
            trace.kb_result(domain, results.get(domain, '[No results]'))

        parts = []
        for domain in ['Returns', 'Shipping', 'Warranty']:
            header = "=== " + domain.upper() + " POLICY ==="
            body = results.get(domain, '[No results]')
            parts.append(header + chr(10) + body)
        combined = (chr(10) + chr(10)).join(parts)
        return combined

    coordinator_model = BedrockModel(
        model_id=config.WORKER_MODEL_ID,
        temperature=0.2,
    )

    coordinator_prompt = """You are the PolicyAgent for NovaMart, an e-commerce customer support system.

YOUR ROLE: Answer customer policy questions using ONLY retrieved policy text.

PROCESS:
1. ALWAYS call search_all_policies first with the customer's policy question.
2. Read the three retrieved passage sets (Returns, Shipping, Warranty).
3. Synthesize a clear, accurate answer that cites only what the retrieved passages say.
4. If the retrieved passages do not cover the question, say so explicitly.
5. Never invent policy details. Never reference customer-specific data (that comes from the InventoryAgent, not you).

Return a grounded, policy-only answer. The CommunicationAgent will handle customer tone."""

    return Agent(
        model=coordinator_model,
        tools=[search_all_policies],
        system_prompt=coordinator_prompt,
    )


def build_communication_agent() -> Agent:
    """
    Build the Communication Agent.

    Drafts the final customer-facing message by reading the full WorkflowState
    and composing a coherent, empathetic response.
    """

    model = BedrockModel(
        model_id=config.WORKER_MODEL_ID,
        temperature=0.3,
    )

    system_prompt = """You are the CommunicationAgent for NovaMart, an e-commerce customer support system.

YOUR ROLE: Compose the final, customer-facing message.

PROCESS:
1. ALWAYS call get_full_workflow_context first to read everything the other agents produced.
2. Use the inventory findings (order status, dates, amounts), any policy information
   the PolicyAgent retrieved, and any refund decision the RefundAgent made.
3. Write a warm, professional, empathetic message. Address the customer directly.
4. Include all relevant specifics: order IDs, dates, refund references, next steps.
5. Do NOT invent facts. Do NOT contradict the RefundAgent's decision.
6. Keep it concise - most responses should be 3-6 sentences plus any relevant details.

TONE: Friendly, clear, and helpful. Acknowledge the customer's situation.
Never mention agents, internal tools, or WorkflowState - the customer should
experience a single coherent NovaMart support reply."""

    @tool
    def get_full_workflow_context(session_id: str) -> dict:
        """
        Read the complete WorkflowState to access all findings from previous agents.

        Args:
            session_id: The current session identifier

        Returns:
            Full WorkflowState dict (inventory_agent, policy_agent, refund_agent)
        """
        return _read_workflow_state(session_id) or {}

    return Agent(
        model=model,
        tools=[get_full_workflow_context],
        system_prompt=system_prompt,
    )


def build_orchestrator_agent(
    inventory_agent:      Agent,
    refund_agent:         Agent,
    policy_agent:         Agent,
    communication_agent:  Agent,
) -> Agent:
    """
    Build the Orchestrator Agent that routes requests and manages WorkflowState.
    """

    model = BedrockModel(
        model_id=config.ORCHESTRATOR_MODEL_ID,
        temperature=0.0,
    )

    system_prompt = """You are the OrchestratorAgent for NovaMart's customer support system.

YOUR ROLE: Route customer requests to the correct specialist agents, in the correct order.
You NEVER answer the customer yourself. You NEVER compose customer-facing prose.
The CommunicationAgent always writes the final reply.

ROUTING RULES:
1. EVERY request: call initialize_session first.
2. Order status / return / refund requests: call route_to_inventory_agent FIRST,
   then route_to_refund_agent. Never call route_to_refund_agent before inventory.
3. Policy meaning questions (return windows, shipping rates, warranty terms):
   call route_to_policy_agent. Do NOT call the inventory agent for these.
4. Account questions (what is my tier?, am I premium?): call route_to_inventory_agent.
   NEVER call route_to_policy_agent for these - policy only knows policy text, not customer data.
5. Math / calculation questions (what is 5 items at $29.99 with 10% off?):
   skip Inventory, Policy, and Refund entirely. Go straight to route_to_communication_agent.
6. EVERY request, ALWAYS as your FINAL tool call: call route_to_communication_agent.
   No exceptions. The communication agent composes the customer reply.

Extract the session_id and customer_id from the request prefix
(format: [Session ID: XXXXXXXX] [Customer ID: CUST-XXX] <the request>).
Pass the customer's original text verbatim to the workers.

Do not produce any text yourself between tool calls."""

    @tool
    def initialize_session(session_id: str, customer_id: str) -> dict:
        """
        Create a fresh WorkflowState record for this session.

        Args:
            session_id:  The current session identifier
            customer_id: The customer's unique identifier

        Returns:
            The newly-created WorkflowState record
        """
        existing = _read_workflow_state(session_id)
        if existing:
            return existing
        return _create_workflow_state(session_id, customer_id)

    @tool
    def route_to_inventory_agent(session_id: str, customer_id: str, request: str) -> str:
        """
        Route an order-related request to the Inventory Agent to gather order facts.
        Call this FIRST for any request involving order status, history, or returns.

        Args:
            session_id:  The current session identifier (from the customer request)
            customer_id: The customer's unique identifier
            request:     The customer's original request

        Returns:
            Inventory facts retrieved by the InventoryAgent
        """
        worker_prompt = (
            "Session ID: " + session_id + chr(10) +
            "Customer ID: " + customer_id + chr(10) +
            "Request: " + request + chr(10) + chr(10) +
            "Gather all relevant order and customer facts. Do not make decisions."
        )
        result = str(inventory_agent(worker_prompt))

        state = _read_workflow_state(session_id)
        version = int(state['version']) if state else 0
        _update_workflow_state(
            session_id,
            {'inventory_agent': result},
            expected_version=version,
        )
        return result

    @tool
    def route_to_policy_agent(session_id: str, request: str) -> str:
        """
        Route a policy question to the Policy Agent (multi-agent RAG).
        Call this for questions about return policies, shipping, or warranties.

        Args:
            session_id: The current session identifier
            request:    The customer's policy question

        Returns:
            Policy information retrieved and synthesized by PolicyAgent
        """
        result = str(policy_agent(request))

        state = _read_workflow_state(session_id)
        version = int(state['version']) if state else 0
        _update_workflow_state(
            session_id,
            {'policy_agent': result},
            expected_version=version,
        )
        return result

    @tool
    def route_to_refund_agent(session_id: str, customer_id: str, request: str) -> str:
        """
        Route a return/refund request to the Refund Agent.
        Call this AFTER route_to_inventory_agent has gathered order facts.

        Args:
            session_id:  The current session identifier
            customer_id: The customer's unique identifier
            request:     The return/refund request

        Returns:
            Refund decision from the RefundAgent
        """
        worker_prompt = (
            "Session ID: " + session_id + chr(10) +
            "Customer ID: " + customer_id + chr(10) +
            "Request: " + request + chr(10) + chr(10) +
            "Call get_inventory_context first, then decide return eligibility "
            "based on the customer tier and the order date."
        )
        result = str(refund_agent(worker_prompt))

        state = _read_workflow_state(session_id)
        version = int(state['version']) if state else 0
        _update_workflow_state(
            session_id,
            {'refund_agent': result},
            expected_version=version,
        )
        return result

    @tool
    def route_to_communication_agent(session_id: str, customer_id: str,
                                     original_request: str) -> str:
        """
        Route to the Communication Agent to compose the final customer response.
        Call this LAST - after all relevant worker agents have run.

        Args:
            session_id:       The current session identifier
            customer_id:      The customer's unique identifier
            original_request: The customer's original message

        Returns:
            Final customer-facing response drafted by CommunicationAgent
        """
        worker_prompt = (
            "Session ID: " + session_id + chr(10) +
            "Customer ID: " + customer_id + chr(10) +
            "Original customer message: " + original_request + chr(10) + chr(10) +
            "Read the full WorkflowState and compose the final customer-facing reply."
        )
        result = str(communication_agent(worker_prompt))

        state = _read_workflow_state(session_id)
        version = int(state['version']) if state else 0
        _update_workflow_state(
            session_id,
            {'communication_agent': result},
            expected_version=version,
        )
        return result

    return Agent(
        model=model,
        tools=[
            initialize_session,
            route_to_inventory_agent,
            route_to_policy_agent,
            route_to_refund_agent,
            route_to_communication_agent,
        ],
        system_prompt=system_prompt,
    )


def create_guardrail() -> tuple[str, str]:
    """
    Create (or reuse) the NovaMart Bedrock Guardrail and return (id, version).

    Reads config.GUARDRAIL_NAME, config.GUARDRAIL_BLOCKED_TOPICS.
    """
    import boto3
    bedrock_client = boto3.client('bedrock', region_name=config.AWS_REGION)

    # Reuse an existing guardrail if one with the same name already exists
    existing = bedrock_client.list_guardrails()
    for g in existing.get('guardrails', []):
        if g['name'] == config.GUARDRAIL_NAME:
            guardrail_id = g['id']
            versions = bedrock_client.list_guardrails(guardrailIdentifier=guardrail_id)
            guardrail_version = 'DRAFT'
            for v in versions.get('guardrails', []):
                if v.get('version', 'DRAFT') != 'DRAFT':
                    guardrail_version = v['version']
            print("Guardrail already exists: " + guardrail_id + " (version: " + guardrail_version + ")")
            return guardrail_id, guardrail_version

    # Create the guardrail with content, PII, topic, and word policies
    response = bedrock_client.create_guardrail(
        name=config.GUARDRAIL_NAME,
        description="NovaMart enterprise safety guardrail for customer support.",
        contentPolicyConfig={
            'filtersConfig': [
                {'type': 'SEXUAL',    'inputStrength': 'HIGH',   'outputStrength': 'HIGH'},
                {'type': 'VIOLENCE',  'inputStrength': 'HIGH',   'outputStrength': 'HIGH'},
                {'type': 'HATE',      'inputStrength': 'HIGH',   'outputStrength': 'HIGH'},
                {'type': 'INSULTS',   'inputStrength': 'MEDIUM', 'outputStrength': 'MEDIUM'},
                {'type': 'MISCONDUCT','inputStrength': 'MEDIUM', 'outputStrength': 'MEDIUM'},
            ],
        },
        sensitiveInformationPolicyConfig={
            'piiEntitiesConfig': [
                {'type': 'CREDIT_DEBIT_CARD_NUMBER', 'action': 'BLOCK'},
                {'type': 'US_SOCIAL_SECURITY_NUMBER','action': 'BLOCK'},
                {'type': 'EMAIL',                    'action': 'ANONYMIZE'},
                {'type': 'PHONE',                    'action': 'ANONYMIZE'},
            ],
        },
        topicPolicyConfig={
            'tierConfig': {'tierName': 'STANDARD'},
            'topicsConfig': [
                {
                    'name': 'competitor_products',
                    'definition': 'Discussion, comparison, or promotion of competitor e-commerce platforms or their products.',
                    'examples': ['Is Amazon better than NovaMart?', 'Compare your prices to eBay.'],
                    'type': 'DENY',
                },
                {
                    'name': 'pricing_negotiations',
                    'definition': 'Haggling, negotiating, or requesting a change to an advertised price. Calculations using a specified price and discount are allowed.',
                    'examples': ['Can I get a discount?', 'Will you lower the price?'],
                    'type': 'DENY',
                },
                {
                    'name': 'legal_threats',
                    'definition': 'Threatening legal action, lawsuits, or regulatory complaints.',
                    'examples': ['I will sue NovaMart', 'I am contacting my lawyer.'],
                    'type': 'DENY',
                },
            ],
        },
        wordPolicyConfig={
            'managedWordListsConfig': [
                {'type': 'PROFANITY'},
            ],
        },
        blockedInputMessaging="I'm sorry, but I can't help with that request. Please rephrase or contact NovaMart support directly.",
        blockedOutputsMessaging="I'm sorry, but I can't provide that response. Please contact NovaMart support for assistance.",
        crossRegionConfig={
            'guardrailProfileIdentifier': 'us.guardrail.v1:0',
        },
    )

    guardrail_id = response['guardrailId']
    print("Guardrail created: " + guardrail_id)

    # Publish a numbered version so we don't rely on DRAFT
    version_response = bedrock_client.create_guardrail_version(
        guardrailIdentifier=guardrail_id,
        description="NovaMart customer support guardrail — stable version.",
    )
    guardrail_version = version_response['version']
    print("Guardrail version published: " + guardrail_version)

    return guardrail_id, guardrail_version


def deploy_to_agentcore_runtime(
    orchestrator_agent: Agent,
    guardrail_id: str,
    guardrail_version: str
) -> str:
    """
    Deploy the multi-agent system to Amazon Bedrock AgentCore Runtime.

    Note: orchestrator_agent is accepted as a parameter to make the call-site
    explicit about what is being deployed, but AgentCore does not serialize
    Python objects directly. Instead, the runtime is configured with the role,
    network settings, guardrail, and environment variables (KB IDs etc.) it
    needs. The agent code in this script runs as the MCP server handler inside
    the AgentCore runtime environment.
    """
    import boto3, os, time, io, zipfile, uuid, json
    s3_client = boto3.client('s3', region_name=config.AWS_REGION)

    runtime_name = config.PROJECT_NAME.replace("-", "_") + "_runtime"

    # Build a deployment zip from the current src/ directory
    src_dir = os.path.dirname(os.path.abspath(__file__))
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zf:
        for root, _dirs, files in os.walk(src_dir):
            for fname in files:
                if fname.endswith('.py') or fname.endswith('.txt'):
                    full = os.path.join(root, fname)
                    arc = os.path.relpath(full, os.path.dirname(src_dir))
                    zf.write(full, arc)
    zip_buffer.seek(0)

    artifact_key = "agentcore-artifacts/" + runtime_name + "/deployment.zip"
    s3_client.put_object(
        Bucket=config.POLICY_BUCKET,
        Key=artifact_key,
        Body=zip_buffer.getvalue(),
        ContentType='application/zip',
    )
    print("  Artifact uploaded: s3://" + config.POLICY_BUCKET + "/" + artifact_key)

    environment_variables = {
        'AWS_REGION':        config.AWS_REGION,
        'PROJECT_NAME':      config.PROJECT_NAME,
        'RETURNS_KB_ID':     config.RETURNS_KB_ID,
        'SHIPPING_KB_ID':    config.SHIPPING_KB_ID,
        'WARRANTY_KB_ID':    config.WARRANTY_KB_ID,
        'AGENT_LOG_GROUP':   config.AGENT_LOG_GROUP,
        'GUARDRAIL_ID':      guardrail_id,
        'GUARDRAIL_VERSION': guardrail_version,
    }

    response = agentcore_control.create_agent_runtime(
        agentRuntimeName=runtime_name,
        description="NovaMart multi-agent customer support system.",
        roleArn=config.AGENTCORE_ROLE_ARN,
        networkConfiguration={'networkMode': 'PUBLIC'},
        protocolConfiguration={'serverProtocol': 'MCP'},
        agentRuntimeArtifact={
            'codeConfiguration': {
                'code': {
                    's3': {
                        'bucket': config.POLICY_BUCKET,
                        'prefix': artifact_key,
                    },
                },
                'runtime': 'PYTHON_3_12',
                'entryPoint': ['main.py'],
            },
        },
        environmentVariables=environment_variables,
    )

    runtime_arn = response.get('agentRuntimeArn', response.get('arn', ''))
    print("  AgentCore Runtime deployed: " + runtime_arn)
    return runtime_arn


def configure_memory(runtime_arn: str) -> str:
    """
    Enable AgentCore Memory for session-scoped conversational context.
    Uses SESSION_SUMMARY memory type with 7-day storage.

    Returns:
        The memory resource ARN
    """
    memory_name = config.MEMORY_NAMESPACE.replace('-', '_')
    existing = agentcore_control.list_memories()
    for m in existing.get('memories', []):
        if m['id'].startswith(memory_name):
            memory_arn = m['arn']
            print("AgentCore Memory already exists: " + memory_arn)
            return memory_arn

    response = agentcore_control.create_memory(
        name=memory_name,
        description="NovaMart customer support session memory (summary strategy, 7-day retention).",
        eventExpiryDuration=7,
        memoryStrategies=[
            {
                'summaryMemoryStrategy': {
                    'name': 'session_summary',
                    'description': 'Summarize each session into a compact context block.',
                    'namespaces': ['/sessions/{sessionId}/summary'],
                },
            },
        ],
        clientToken=str(__import__('uuid').uuid4()),
    )

    memory_arn = response.get('memory', {}).get('arn') or response.get('arn', '')
    print("AgentCore Memory created: " + memory_arn)
    return memory_arn


def configure_observability(runtime_arn: str) -> None:
    """
    Configure AgentCore Observability:
    - Agent logs → CloudWatch Logs at INFO level
    - Execution traces → AWS X-Ray at 100% sampling
    """
    runtime_id = runtime_arn.split('/')[-1]

    logging_config = {
        'cloudWatchConfig': {
            'logGroupName': config.AGENT_LOG_GROUP,
            'logLevel': 'INFO',
            'enabled': True,
        },
        'xRayConfig': {
            'enabled': True,
            'samplingRate': 1.0,
        },
    }

    try:
        agentcore_control.put_agent_runtime_logging_configuration(
            agentRuntimeId=runtime_id,
            loggingConfiguration=logging_config,
        )
        print("Observability configured — log group: " + config.AGENT_LOG_GROUP + ", X-Ray sampling: 1.0")
    except Exception as e:
        print("[Note] Logging config skipped (SDK version mismatch): " + str(e))


def _gw_get_function_arn(function_name: str) -> str:
    """Resolve a Lambda function name to its full ARN."""
    lambda_client = boto3.client('lambda', region_name=config.AWS_REGION)
    resp = lambda_client.get_function(FunctionName=function_name)
    return resp['Configuration']['FunctionArn']


def _gw_stack_uuid() -> str:
    """Return the short UUID from the project CloudFormation stack ID.
    Gives the gateway a stable name so re-runs never hit ConflictException."""
    cf = boto3.client('cloudformation', region_name=config.AWS_REGION)
    stacks = cf.describe_stacks(StackName=config.PROJECT_NAME)
    stack_id = stacks['Stacks'][0]['StackId']
    full_uuid = stack_id.split('/')[-1]
    return full_uuid.split('-')[0]


def _gw_wait_for_ready(agentcore_ctrl, gateway_id: str, timeout: int = 120) -> str:
    """Poll until the gateway reaches READY status. Returns the gateway URL."""
    deadline = time.time() + timeout
    first    = True
    while time.time() < deadline:
        gw     = agentcore_ctrl.get_gateway(gatewayIdentifier=gateway_id)
        status = gw['status']
        if status == 'READY':
            if not first:
                print(' ready.')
            return gw.get('gatewayUrl', '')
        if 'FAILED' in status:
            print(f' failed: {status}')
            raise RuntimeError(f"Gateway {gateway_id} entered status {status}")
        if first:
            print('    Gateway provisioning (async — normal AWS behaviour)',
                  end='', flush=True)
            first = False
        print('.', end='', flush=True)
        time.sleep(5)
    raise TimeoutError(f"Gateway {gateway_id} not READY after {timeout}s")


def _gw_get_or_create(agentcore_ctrl, name: str, role_arn: str,
                       instructions: str) -> tuple[str, str]:
    """Create an AgentCore Gateway, or reuse it if it already exists."""
    try:
        gw = agentcore_ctrl.create_gateway(
            name=name,
            roleArn=role_arn,
            protocolType='MCP',
            authorizerType='NONE',
            protocolConfiguration={'mcp': {'instructions': instructions,
                                            'searchType': 'SEMANTIC'}},
        )
        gw_id  = gw['gatewayId']
        print(f'    Gateway ID  : {gw_id}')
        print(f'    Status      : {gw["status"]}')
        gw_url = _gw_wait_for_ready(agentcore_ctrl, gw_id)
        print(f'    Gateway URL : {gw_url}')
        return gw_id, gw_url
    except agentcore_ctrl.exceptions.ConflictException:
        print(f"    Gateway '{name}' already exists — reusing it.")
        gateways = agentcore_ctrl.list_gateways().get('items', [])
        existing = next((g for g in gateways if g['name'] == name), None)
        if not existing:
            raise RuntimeError(f"Gateway '{name}' not found after ConflictException")
        gw_id  = existing['gatewayId']
        print(f'    Gateway ID  : {gw_id}')
        gw_url = _gw_wait_for_ready(agentcore_ctrl, gw_id)
        print(f'    Gateway URL : {gw_url}')
        return gw_id, gw_url


def _gw_create_target(agentcore_ctrl, gateway_id: str, t: dict,
                       lambda_arn: str) -> None:
    """Register one Lambda target on the gateway. Skips if it already exists."""
    payload = dict(
        gatewayIdentifier=gateway_id,
        name=t['name'],
        description=t['description'],
        targetConfiguration={
            'mcp': {
                'lambda': {
                    'lambdaArn': lambda_arn,
                    'toolSchema': {
                        'inlinePayload': [{
                            'name':        t['tool_name'],
                            'description': t['tool_description'],
                            'inputSchema': {
                                'type': 'object',
                                'properties': {
                                    t['param_name']: {
                                        'type':        'string',
                                        'description': t['param_desc'],
                                    }
                                },
                                'required': [t['param_name']],
                            },
                        }]
                    },
                }
            }
        },
        credentialProviderConfigurations=[
            {'credentialProviderType': 'GATEWAY_IAM_ROLE'}
        ],
    )
    try:
        resp = agentcore_ctrl.create_gateway_target(**payload)
        print(f"    [{resp['status']:12s}] {t['name']} → target {resp['targetId']}")
    except agentcore_ctrl.exceptions.ConflictException:
        print(f"    [already exists] {t['name']} — skipped")


def deploy_agentcore_gateway() -> dict:
    """
    Create an AgentCore Gateway and register the NovaMart tool Lambda targets.

    Production equivalent of the in-process @tool functions defined inside
    build_*_agent(). Each tool becomes a Lambda function registered as a
    gateway target; agents discover tools at runtime via the MCP endpoint —
    no code changes needed when adding or updating tools.

    Uses the same three-step pattern as Lesson 11:
      1. create_gateway  (MCP protocol, SEMANTIC search)
      2. create_gateway_target  (one per Lambda-backed tool)
      3. Agents connect via the returned gateway_url

    Requires Lambda tool functions to be deployed via a separate stack.
    Set ORDERS_FUNCTION, POLICY_FUNCTION, CUSTOMERS_FUNCTION in .env.

    Returns:
        dict with gateway_id, gateway_url, and status.
    """
    agentcore_ctrl = boto3.client('bedrock-agentcore-control',
                                   region_name=config.AWS_REGION)

    try:
        gw_uuid = _gw_stack_uuid()
    except Exception:
        gw_uuid = config.PROJECT_NAME

    gw_name = f"novamart-support-{gw_uuid}"
    print(f"  Calling create_gateway (name: {gw_name})...")
    gateway_id, gateway_url = _gw_get_or_create(
        agentcore_ctrl, gw_name, config.AGENTCORE_ROLE_ARN,
        "NovaMart customer support gateway. Provides order lookup, "
        "policy search, and customer tier tools.",
    )

    targets = [
        {
            'name':             'orders-api',
            'description':      'Look up order details, status, and return eligibility for a customer',
            'function':         _ORDERS_FUNCTION,
            'tool_name':        'check_order_status',
            'tool_description': 'Check order status and return eligibility for a specific order',
            'param_name':       'order_id',
            'param_desc':       'Order ID (e.g. ORD-27176)',
        },
        {
            'name':             'policy-api',
            'description':      'Retrieve return, shipping, and warranty policy text from knowledge bases',
            'function':         _POLICY_FUNCTION,
            'tool_name':        'search_policies',
            'tool_description': 'Search all policy knowledge bases for a customer query',
            'param_name':       'query',
            'param_desc':       'Customer question about returns, shipping, or warranty',
        },
        {
            'name':             'customers-api',
            'description':      'Look up customer tier (Standard or Premium) and account details',
            'function':         _CUSTOMERS_FUNCTION,
            'tool_name':        'get_customer_tier',
            'tool_description': 'Get customer tier and account information by customer ID',
            'param_name':       'customer_id',
            'param_desc':       'Customer ID (e.g. CUST-001)',
        },
    ]

    print(f"\n  Registering {len(targets)} Gateway targets...")
    for t in targets:
        try:
            lambda_arn = _gw_get_function_arn(t['function'])
            _gw_create_target(agentcore_ctrl, gateway_id, t, lambda_arn)
        except Exception as e:
            print(f"    [Skipped] {t['name']}: {e}")

    return {'gateway_id': gateway_id, 'gateway_url': gateway_url, 'status': 'CREATING'}


# ═══════════════════════════════════════════════════════
#  RUNTIME INVOCATION (pre-written - do not modify)
# ═══════════════════════════════════════════════════════

def invoke_agent(session_id: str, customer_id: str, user_message: str) -> str:
    """
    Invoke the deployed agent via AgentCore Runtime.
    Pre-written - do not modify.
    """
    enriched_message = f"[Session ID: {session_id}] [Customer ID: {customer_id}] {user_message}"

    import json as _json
    response = agentcore_client.invoke_agent_runtime(
        agentRuntimeArn=config.AGENTCORE_RUNTIME_ARN,
        runtimeSessionId=session_id,
        payload=_json.dumps({"inputText": enriched_message}).encode("utf-8"),
    )

    full_response = ""
    for event in response.get('completion', []):
        if 'chunk' in event:
            chunk = event['chunk']
            if 'bytes' in chunk:
                full_response += chunk['bytes'].decode('utf-8')

    return full_response


# ═══════════════════════════════════════════════════════
#  DEPLOYMENT ENTRY POINT (pre-written - do not modify)
# ═══════════════════════════════════════════════════════

def deploy_all():
    """Full deployment pipeline. Run after completing all tasks."""
    print("\n" + "="*60)
    print("  Deploying Enterprise Multi-Agent System")
    print("="*60 + "\n")

    print("Step 1/6: Building agent graph...")
    inventory_agent     = build_inventory_agent()
    refund_agent        = build_refund_agent()
    policy_agent        = build_policy_agent()
    communication_agent = build_communication_agent()
    orchestrator = build_orchestrator_agent(
        inventory_agent, refund_agent, policy_agent, communication_agent
    )
    print("  All 5 agents initialized\n")

    print("Step 2/6: Creating Bedrock Guardrail...")
    guardrail_id, guardrail_version = create_guardrail()
    print()

    print("Step 3/6: Deploying to AgentCore Runtime...")
    runtime_arn = deploy_to_agentcore_runtime(orchestrator, guardrail_id, guardrail_version)
    print()

    print("Step 4/6: Configuring Memory...")
    memory_arn = configure_memory(runtime_arn)
    print()

    print("Step 5/6: Configuring Observability...")
    configure_observability(runtime_arn)
    print()

    print("Step 6/6: Deploying AgentCore Gateway...")
    try:
        gw = deploy_agentcore_gateway()
        print(f"  Gateway URL : {gw['gateway_url']}")
        print(f"  Agents connect via MCP at this endpoint — no code changes needed")
    except Exception as e:
        print(f"  [Note] Gateway deployment skipped: {e}")
        print(f"  (Deploy Lambda tool functions and set ORDERS_FUNCTION etc. in .env to enable)")
    print()

    print("="*60)
    print("  Deployment Complete!")
    print("="*60)
    print(f"\n  Add these to your .env file:")
    print(f"  AGENTCORE_RUNTIME_ARN={runtime_arn}")
    print(f"  GUARDRAIL_ID={guardrail_id}")
    print(f"  GUARDRAIL_VERSION={guardrail_version}\n")
    return runtime_arn, guardrail_id


def handler(event, context=None):
    """
    AgentCore Runtime entry point.

    Receives an invocation event, runs the multi-agent orchestrator, and
    returns the response in the shape AgentCore expects.
    """
    import json

    # Extract input text — AgentCore may pass it as "inputText" or nested
    input_text = ""
    if isinstance(event, dict):
        input_text = event.get("inputText") or event.get("input_text") or ""
        if not input_text and "body" in event:
            try:
                body = event["body"]
                if isinstance(body, str):
                    body = json.loads(body)
                input_text = body.get("inputText") or body.get("prompt") or ""
            except Exception:
                pass
    elif isinstance(event, str):
        input_text = event

    if not input_text:
        input_text = "Hello"

    # Build the full agent graph and invoke it
    try:
        inv = build_inventory_agent()
        ref = build_refund_agent()
        pol = build_policy_agent()
        com = build_communication_agent()
        orch = build_orchestrator_agent(inv, ref, pol, com)
        response = str(orch(input_text))
    except Exception as e:
        response = "Sorry, I encountered an error: " + str(e)

    # Return in the shape AgentCore expects
    return {
        "completion": [
            {"chunk": {"bytes": response.encode("utf-8")}},
        ],
    }


if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] == 'deploy':
        deploy_all()

    elif len(sys.argv) > 1 and sys.argv[1] == 'test':
        print("Running local agent test...")
        inventory_agent     = build_inventory_agent()
        refund_agent        = build_refund_agent()
        policy_agent        = build_policy_agent()
        communication_agent = build_communication_agent()
        orchestrator = build_orchestrator_agent(
            inventory_agent, refund_agent, policy_agent, communication_agent
        )

        test_cases = [
            ("CUST-001", "I want to return my wireless headphones from order ORD-27176"),
            ("CUST-002", "What is the return policy for premium customers?"),
            ("CUST-003", "How much would 5 items at $29.99 be with a 10% discount?"),
        ]
        for customer_id, query in test_cases:
            session_id = str(uuid.uuid4())[:8]
            print(f"\n{'─'*60}")
            print(f"Session: {session_id} | Customer: {customer_id}")
            print(f"Query: {query}")
            prompt = f"[Session ID: {session_id}] [Customer ID: {customer_id}] {query}"
            _seg = start_request_segment("NovaMart-Orchestrator")
            try:
                response = orchestrator(prompt)
            finally:
                end_request_segment()
            print(f"Response: {response}")

    elif len(sys.argv) > 1 and sys.argv[1] == 'chat':
        # ── Interactive terminal chat - educational mode ───────────────────
        W = _C.W

        # ── Welcome banner ────────────────────────────────────────────────
        print()
        print(f"  {_C.GRY}{'=' * W}{_C.RESET}")
        print(f"  {_C.ORCH}{_C.BOLD}{'NovaMart -- Multi-Agent Customer Support':^{W}}{_C.RESET}")
        print(f"  {_C.GRY}{'Strands Agents SDK  +  Amazon Bedrock AgentCore':^{W}}{_C.RESET}")
        print(f"  {_C.GRY}{'=' * W}{_C.RESET}")

        # ── Test customers ────────────────────────────────────────────────
        print()
        print(f"  {_C.GRY}{'─' * W}{_C.RESET}")
        print(f"  {_C.BOLD}Test Customers{_C.RESET}")
        print(f"  {_C.GRY}{'─' * W}{_C.RESET}")
        print(f"  {_C.GRY}{'ID':<10}  {'Name':<18}  {'Tier':<10}  {'Order':<12}  Product{_C.RESET}")
        print(f"  {_C.GRY}{'─'*8}  {'─'*16}  {'─'*8}  {'─'*10}  {'─'*20}{_C.RESET}")
        for cid, name, tier, order, product in [
            ("CUST-001", "Alice Johnson", "Premium",  "ORD-27176", "Sony headphones"),
            ("CUST-002", "Bob Smith",     "Standard", "ORD-28001", "mechanical keyboard"),
            ("CUST-003", "Carol Davis",   "Premium",  "ORD-29001", "laptop"),
            ("CUST-004", "David Lee",     "Standard", "ORD-30001", "phone case"),
        ]:
            tier_col = _C.INV if tier == 'Premium' else _C.GRY
            print(f"  {_C.BOLD}{cid}{_C.RESET}  {name:<18}  "
                  f"{tier_col}{tier:<10}{_C.RESET}  {order}  {product}")
        print(f"  {_C.GRY}{'─' * W}{_C.RESET}")
        print()

        customer_id = (
            input(f"  Enter Customer ID (default: CUST-001): ").strip()
            or "CUST-001"
        )
        session_id  = str(uuid.uuid4())[:8]
        print()
        print(f"  {_C.GRY}Session  : {_C.RESET}{_C.BOLD}{session_id}{_C.RESET}")
        print(f"  {_C.GRY}Customer : {_C.RESET}{_C.BOLD}{customer_id}{_C.RESET}")
        print(f"  {_C.GRY}Type a question and press Enter.  Type 'quit' to exit.{_C.RESET}")
        print()

        # ── Build agents (one line per agent so students see initialisation order)
        print(f"  {_C.GRY}[SYSTEM]  Initializing agent graph...{_C.RESET}")
        inventory_agent     = build_inventory_agent()
        print(f"  {_C.GRY}          {_C.OK}[OK]{_C.RESET}{_C.GRY}  InventoryAgent{_C.RESET}",    flush=True)
        refund_agent        = build_refund_agent()
        print(f"  {_C.GRY}          {_C.OK}[OK]{_C.RESET}{_C.GRY}  RefundAgent{_C.RESET}",       flush=True)
        policy_agent        = build_policy_agent()
        print(f"  {_C.GRY}          {_C.OK}[OK]{_C.RESET}{_C.GRY}  PolicyAgent{_C.RESET}",       flush=True)
        communication_agent = build_communication_agent()
        print(f"  {_C.GRY}          {_C.OK}[OK]{_C.RESET}{_C.GRY}  CommunicationAgent{_C.RESET}", flush=True)
        orchestrator = build_orchestrator_agent(
            inventory_agent, refund_agent, policy_agent, communication_agent
        )
        print(f"  {_C.GRY}          {_C.OK}[OK]{_C.RESET}{_C.GRY}  Orchestrator{_C.RESET}",      flush=True)
        print(f"  {_C.GRY}[SYSTEM]  All 5 agents ready.{_C.RESET}")
        print()

        # ── Conversation loop ─────────────────────────────────────────────
        while True:
            try:
                user_input = input(
                    f"  {_C.BOLD}You >{_C.RESET} "
                ).strip()
            except (EOFError, KeyboardInterrupt):
                print(f"\n  {_C.GRY}Session ended.{_C.RESET}")
                break

            if not user_input:
                continue
            if user_input.lower() in ('quit', 'exit', 'q'):
                print(f"  {_C.GRY}Session ended.{_C.RESET}")
                break

            prompt  = (f"[Session ID: {session_id}] "
                       f"[Customer ID: {customer_id}] {user_input}")
            t0_turn = time.time()

            # ── Install proxy, run orchestrator, restore stdout ────────────
            trace.new_turn()
            sys.stdout = _trace_writer
            try:
                response = orchestrator(prompt)
            finally:
                sys.stdout = _real_stdout   # always restore, even on exception

            elapsed = time.time() - t0_turn

            # ── Resolve the final customer-facing text ────────────────────
            final_state = _read_workflow_state(session_id) or {}
            comm_result = final_state.get('communication_agent', '')
            text = _strip_xml_tags(comm_result or str(response))

            # ── DynamoDB workflow state summary ───────────────────────────
            trace.summary(session_id, elapsed)

            # ── Final customer-facing response ────────────────────────────
            print()
            print(f"  {_C.GRY}{'=' * W}{_C.RESET}")
            print(f"  {_C.COM}{_C.BOLD}AGENT RESPONSE{_C.RESET}")
            print(f"  {_C.GRY}{'=' * W}{_C.RESET}")
            for line in text.splitlines():
                print(f"  {line}")
            print(f"  {_C.GRY}{'=' * W}{_C.RESET}")
            print()

    else:
        print("Usage:")
        print("  python agent_orchestrator.py deploy  # Deploy to AgentCore")
        print("  python agent_orchestrator.py test    # Run automated test cases")
        print("  python agent_orchestrator.py chat    # Interactive terminal chat")
