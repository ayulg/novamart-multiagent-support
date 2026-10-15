"""
agent_observability.py
======================
Client-side X-Ray instrumentation for the NovaMart multi-agent system.

Strategy:
  - Patch botocore so every AWS call (Bedrock, DynamoDB, S3, Bedrock-Agent-Runtime)
    is automatically recorded as an X-Ray subsegment.
  - Expose `tool` as a PASSTHROUGH to strands.tool (no wrapping) — Strands' own
    decorator has an incompatible calling convention, so we let it do its job
    and rely on botocore patching for the trace graph.
  - Expose `start_request_segment` / `end_request_segment` to bracket a request.
"""

import logging

logger = logging.getLogger(__name__)

try:
    from aws_xray_sdk.core import xray_recorder, patch_all
    try:
        patch_all()
        logger.info("X-Ray botocore patching applied")
    except Exception as exc:
        logger.warning("X-Ray patch_all failed: %s", exc)

    try:
        xray_recorder.configure(
            service="NovaMart-Orchestrator",
            context_missing="LOG_ERROR",
            sampling=True,
        )
    except Exception as exc:
        logger.warning("X-Ray configure failed: %s", exc)
    _XRAY_AVAILABLE = True
except Exception as exc:
    logger.warning("aws-xray-sdk not available: %s", exc)
    _XRAY_AVAILABLE = False


# Passthrough to strands.tool — no wrapping, no interference.
from strands import tool  # noqa: E402,F401


def start_request_segment(name="NovaMart-Orchestrator"):
    """Begin a top-level X-Ray segment for one customer request."""
    if not _XRAY_AVAILABLE:
        return None
    try:
        return xray_recorder.begin_segment(name)
    except Exception as exc:
        logger.warning("begin_segment failed: %s", exc)
        return None


def end_request_segment():
    if not _XRAY_AVAILABLE:
        return
    try:
        xray_recorder.end_segment()
    except Exception as exc:
        logger.warning("end_segment failed: %s", exc)
