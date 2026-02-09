from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class Episode:
    """Store all relevant information of an episode."""

    prefix: str
    text: str
    prefix_token_ids: List[int]
    prefix_tokens: List[str]
    generated_token_ids: List[int]
    is_finished: bool
    reward: float
    reward_info: Dict[str, float]
    # Fork-race fields (optional, backward compatible)
    token_weights: Optional[List[float]] = None
    fork_info: Optional[Dict[str, float]] = None
    branch_id: Optional[int] = None  # None=no fork, 0=branch A, 1=branch B


@dataclass
class MiniBatch:
    """Batch of data for each training step."""

    prefix: List[str]
    prefix_tokens: List[List[str]]
    prefix_token_ids: List[List[int]]
    numbers: List[List[int]]
    target: List[int]
