"""Fork-race reward computation.

Implements bounded reward with hardware-calibrated latency proxy
per EXPERIMENT_PLAN.md section 4.
"""

from dataclasses import dataclass
from typing import Dict, Any, List, Optional


@dataclass
class ForkRewardConfig:
    """Configuration for fork-race reward computation."""
    alpha: float = 0.5       # latency penalty weight
    beta: float = 0.2        # invalid fork penalty
    delta: float = 0.2       # failure base penalty
    budget_ms: float = 2500.0  # wall-clock budget for normalization
    c1_ms: float = 1.0       # ms per step with 1 active branch (calibrated)
    c2_ms: float = 1.5       # ms per step with 2 active branches (calibrated)


def compute_latency_proxy(
    steps_before_fork: int,
    steps_after_fork_branch_a: int,
    steps_after_fork_branch_b: int,
    forked: bool,
    config: ForkRewardConfig,
) -> float:
    """Compute hardware-calibrated latency proxy T_proxy_ms.

    The critical-path latency is:
    - Pre-fork steps at c1_ms each
    - Post-fork steps at c2_ms each, using max of both branches
      (since they run in parallel, wall-clock = max branch length * c2_ms)

    Args:
        steps_before_fork: number of decode steps before fork
        steps_after_fork_branch_a: steps in branch A after fork
        steps_after_fork_branch_b: steps in branch B after fork
        forked: whether a fork occurred
        config: reward config with calibration constants
    """
    if not forked:
        # No fork: all steps at single-branch speed
        total_steps = steps_before_fork  # branch_a/b should be 0
        return total_steps * config.c1_ms

    pre_fork_ms = steps_before_fork * config.c1_ms
    # Critical path: both branches run in parallel, wall-clock = max length
    critical_post_fork_steps = max(
        steps_after_fork_branch_a,
        steps_after_fork_branch_b,
    )
    post_fork_ms = critical_post_fork_steps * config.c2_ms
    return pre_fork_ms + post_fork_ms


def compute_fork_reward(
    correct: bool,
    t_proxy_ms: float,
    invalid_fork: bool,
    config: ForkRewardConfig,
) -> Dict[str, Any]:
    """Compute bounded fork-race reward.

    Reward formula from EXPERIMENT_PLAN.md section 4:
    - success: R = 1 - alpha * L_norm - beta * invalid_fork
    - failure: R = -delta - beta * invalid_fork
    - L_norm = min(T_proxy_ms / budget_ms, 1.0)

    Returns dict with reward and breakdown info.
    """
    l_norm = min(t_proxy_ms / config.budget_ms, 1.0) if config.budget_ms > 0 else 0.0
    invalid_penalty = config.beta if invalid_fork else 0.0

    if correct:
        reward = 1.0 - config.alpha * l_norm - invalid_penalty
    else:
        reward = -config.delta - invalid_penalty

    return {
        "reward": reward,
        "reward_info": {
            "correct": float(correct),
            "t_proxy_ms": t_proxy_ms,
            "l_norm": l_norm,
            "invalid_fork": float(invalid_fork),
            "invalid_penalty": invalid_penalty,
            "latency_penalty": config.alpha * l_norm if correct else 0.0,
        },
    }


def compute_token_weights(
    total_tokens: int,
    fork_position: Optional[int],
    branch_id: Optional[int],
    active_branches: int = 1,
) -> List[float]:
    """Compute per-token weights for GRPO objective.

    Per EXPERIMENT_PLAN.md section 4 (Risk D):
    - pre-fork tokens: weight 1.0
    - post-fork tokens: weight 1/active_branches

    Args:
        total_tokens: total generated tokens in this episode
        fork_position: token index where fork occurred (None if no fork)
        branch_id: which branch this episode is (0=A, 1=B, None=no fork)
        active_branches: number of active branches after fork

    Returns:
        List of per-token weights
    """
    if fork_position is None or branch_id is None:
        # No fork: all tokens weight 1.0
        return [1.0] * total_tokens

    weights = []
    post_fork_weight = 1.0 / active_branches
    for i in range(total_tokens):
        if i < fork_position:
            weights.append(1.0)
        else:
            weights.append(post_fork_weight)
    return weights
