"""Fork-race warmup curriculum helpers.

Provides a decaying bonus for emitting a valid fork early in training,
then decays that bonus away so final behavior is governed by task reward.
"""

from __future__ import annotations

import dataclasses
from typing import List, Tuple

from data_types import Episode


def linear_decay(step: int, start: float, end: float, decay_steps: int) -> float:
    """Linearly decay value from step 1 to `decay_steps` (inclusive)."""
    if decay_steps <= 0:
        return 0.0
    if decay_steps == 1 or step <= 1:
        return float(start)
    if step >= decay_steps:
        return float(end)
    progress = (step - 1) / (decay_steps - 1)
    return float(start + (end - start) * progress)


def fork_presence_bonus_coef(step: int, fork_config: dict) -> float:
    """Return current bonus coefficient for fork presence warmup."""
    start = float(fork_config.get("warmup_fork_presence_bonus_start", 0.0))
    end = float(fork_config.get("warmup_fork_presence_bonus_end", 0.0))
    decay_steps = int(fork_config.get("warmup_fork_presence_bonus_steps", 0))
    if decay_steps <= 0:
        return 0.0
    return linear_decay(step, start=start, end=end, decay_steps=decay_steps)


def apply_fork_presence_bonus(
    episodes: List[Episode],
    bonus_coef: float,
    valid_only: bool = True,
) -> Tuple[List[Episode], float]:
    """Add bonus to forked episodes and return updated list + hit-rate."""
    if not episodes or bonus_coef <= 0:
        return episodes, 0.0

    updated = []
    hits = 0
    for episode in episodes:
        reward_info = dict(episode.reward_info or {})
        forked = float(reward_info.get("forked", 0.0)) > 0.5
        invalid_fork = float(reward_info.get("invalid_fork", 0.0)) > 0.5
        qualifies = forked and (not valid_only or not invalid_fork)

        bonus = float(bonus_coef) if qualifies else 0.0
        if qualifies:
            hits += 1

        reward_info.setdefault("base_reward", float(episode.reward))
        reward_info["fork_presence_bonus"] = bonus

        updated.append(
            dataclasses.replace(
                episode,
                reward=float(episode.reward) + bonus,
                reward_info=reward_info,
            )
        )

    hit_rate = hits / len(episodes)
    return updated, hit_rate
