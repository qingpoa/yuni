from __future__ import annotations

from dataclasses import dataclass

# Upper bound for one business tool result as seen by Planner and Replyer.
TOOL_RESULT_MAX_TOKENS = 2_000
# Must match planner.MAX_BUSINESS_ACTIONS; kept here to avoid an import cycle.
TOOL_LOOP_MAX_ACTIONS = 3


@dataclass(frozen=True, slots=True)
class ContextBudget:
    total_tokens: int = 32_000
    response_reserve_tokens: int = 4_000
    recent_context_tokens: int = 12_000
    background_compaction_tokens: int = 20_000
    summary_max_tokens: int = 2_000
    memory_max_tokens: int = 1_500
    # Room kept free for this turn's tool results, so they do not push out
    # history mid-turn. Sized for every allowed action returning a full result.
    tool_loop_tokens: int = TOOL_LOOP_MAX_ACTIONS * TOOL_RESULT_MAX_TOKENS

    @property
    def input_tokens(self) -> int:
        return max(0, self.total_tokens - self.response_reserve_tokens)

    @property
    def tool_loop_reserve(self) -> int:
        # Small test/edge budgets must still leave room for history.
        return min(self.tool_loop_tokens, self.input_tokens // 4)


def estimate_tokens(value: str) -> int:
    """Conservative tokenizer-independent estimate for mixed Chinese and English text."""
    if not value:
        return 0
    ascii_count = sum(1 for char in value if ord(char) < 128)
    non_ascii_count = len(value) - ascii_count
    return non_ascii_count + (ascii_count + 3) // 4
