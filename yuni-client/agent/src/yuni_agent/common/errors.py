class YuniAgentError(Exception):
    """Base error for stable, user-safe agent failures."""


class AgentNotConfiguredError(YuniAgentError):
    """Raised when a feature is called before its implementation is configured."""

