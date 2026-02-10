"""Helpers for fork-race metric accounting.

Fork rollouts can emit two episodes per base sample (branch A/B). For logging
and evaluation we usually want one metric row per base sample, not per branch.
"""

from typing import List

from data_types import Episode


def primary_episodes(episodes: List[Episode]) -> List[Episode]:
    """Return one episode per base sample.

    Keeps:
    - non-forked rows (branch_id is None)
    - branch A rows (branch_id == 0)

    Drops:
    - branch B rows (branch_id == 1)
    """
    return [episode for episode in episodes if episode.branch_id != 1]


def episode_correct(episode: Episode) -> float:
    """Extract correctness metric from an episode.

    For fork-race rollouts this prefers race-level correctness (`correct`),
    which reflects "any branch solved". Falls back to `answer_reward` for
    vanilla episodes.
    """
    reward_info = episode.reward_info or {}
    if "correct" in reward_info:
        return float(reward_info["correct"])
    return float(reward_info.get("answer_reward", 0.0))
