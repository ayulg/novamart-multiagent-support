"""AgentCore Runtime entry file.

This thin wrapper re-exports `handler` from agent_orchestrator.py so the
AgentCore entryPoint can reference a top-level module name without dots.
"""
from agent_orchestrator import handler  # noqa: F401

__all__ = ["handler"]
