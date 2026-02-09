"""Incremental text-state parser for fork-race rollouts.

Tracks parser state through streaming tokens to determine:
- Whether we are inside <think>...</think>
- Whether a <fork> is valid (only inside think, at most once)
- Whether <answer>...</answer> is complete for verification
"""

from enum import Enum, auto
from dataclasses import dataclass, field
from typing import Optional


class ParserState(Enum):
    BEFORE_THINK = auto()
    IN_THINK = auto()
    AFTER_THINK = auto()


@dataclass
class ForkParserResult:
    """Result of parsing a complete rollout or branch."""
    state: ParserState
    fork_count: int
    invalid_fork: bool
    has_complete_answer: bool
    answer_text: Optional[str]
    # Position (char offset) where fork occurred
    fork_position: Optional[int]


class ForkParser:
    """Incremental parser for fork-race token streams.

    Tracks tag boundaries by accumulating text and scanning for
    <think>, </think>, <fork>, <answer>, </answer> tags.
    """

    def __init__(self):
        self.state: ParserState = ParserState.BEFORE_THINK
        self.text: str = ""
        self.fork_count: int = 0
        self.invalid_fork: bool = False
        self.has_complete_answer: bool = False
        self.answer_text: Optional[str] = None
        self.fork_position: Optional[int] = None
        # Track processed length to avoid re-scanning
        self._last_scanned: int = 0

    def reset(self):
        self.__init__()

    def feed(self, new_text: str) -> "ForkParser":
        """Feed new text and update parser state. Returns self for chaining."""
        self.text += new_text
        self._scan()
        return self

    def feed_token_text(self, token_text: str) -> "ForkParser":
        """Alias for feed - accepts decoded token text."""
        return self.feed(token_text)

    def _scan(self):
        """Scan accumulated text for state transitions."""
        text = self.text

        # Detect <think> opening
        if self.state == ParserState.BEFORE_THINK:
            idx = text.find("<think>")
            if idx != -1:
                self.state = ParserState.IN_THINK

        # Detect </think> closing
        if self.state == ParserState.IN_THINK:
            # Search from after <think>
            think_start = text.find("<think>")
            if think_start != -1:
                close_idx = text.find("</think>", think_start + 7)
                if close_idx != -1:
                    self.state = ParserState.AFTER_THINK

        # Detect complete <answer>...</answer>
        answer_open = text.find("<answer>")
        if answer_open != -1:
            answer_close = text.find("</answer>", answer_open + 8)
            if answer_close != -1:
                self.has_complete_answer = True
                self.answer_text = text[answer_open + 8:answer_close].strip()

    def check_fork_valid(self) -> bool:
        """Check if emitting <fork> right now would be valid.

        Valid iff:
        - Currently in IN_THINK state
        - No fork has been emitted yet (max one fork)
        """
        return self.state == ParserState.IN_THINK and self.fork_count == 0

    def register_fork(self) -> bool:
        """Register that a <fork> token was emitted.

        Returns True if the fork is valid, False if invalid.
        Sets invalid_fork flag if invalid.
        """
        is_valid = self.check_fork_valid()
        self.fork_count += 1
        if not is_valid:
            self.invalid_fork = True
        else:
            self.fork_position = len(self.text)
        return is_valid

    def get_result(self) -> ForkParserResult:
        return ForkParserResult(
            state=self.state,
            fork_count=self.fork_count,
            invalid_fork=self.invalid_fork,
            has_complete_answer=self.has_complete_answer,
            answer_text=self.answer_text,
            fork_position=self.fork_position,
        )

    def clone(self) -> "ForkParser":
        """Create a deep copy for branch splitting."""
        new = ForkParser()
        new.state = self.state
        new.text = self.text
        new.fork_count = self.fork_count
        new.invalid_fork = self.invalid_fork
        new.has_complete_answer = self.has_complete_answer
        new.answer_text = self.answer_text
        new.fork_position = self.fork_position
        new._last_scanned = self._last_scanned
        return new
