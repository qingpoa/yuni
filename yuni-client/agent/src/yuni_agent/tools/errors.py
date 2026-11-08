from __future__ import annotations


class ToolExecutionError(Exception):
    """A tool request was rejected or failed in a way the model should see."""


class ToolTimeoutError(ToolExecutionError):
    """A tool's preview or execution exceeded its configured timeout."""
