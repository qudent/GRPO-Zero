import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_types import Episode
from fork_metrics import episode_correct, primary_episodes


def _make_episode(branch_id, reward_info):
    return Episode(
        prefix="q",
        text="q a",
        prefix_token_ids=[1, 2],
        prefix_tokens=["q", "a"],
        generated_token_ids=[3, 4],
        is_finished=True,
        reward=0.0,
        reward_info=reward_info,
        branch_id=branch_id,
    )


def test_primary_episodes_filters_branch_b():
    episodes = [
        _make_episode(None, {"answer_reward": 0.0, "format_reward": 1.0}),
        _make_episode(0, {"answer_reward": 1.0, "format_reward": 1.0, "correct": 1.0}),
        _make_episode(1, {"answer_reward": 0.0, "format_reward": 1.0, "correct": 1.0}),
    ]
    primary = primary_episodes(episodes)
    assert len(primary) == 2
    assert [episode.branch_id for episode in primary] == [None, 0]


def test_episode_correct_prefers_race_correct():
    episode = _make_episode(
        0,
        {
            "answer_reward": 0.0,
            "format_reward": 1.0,
            "correct": 1.0,
        },
    )
    assert episode_correct(episode) == 1.0


def test_episode_correct_falls_back_to_answer_reward():
    episode = _make_episode(
        None,
        {
            "answer_reward": 1.0,
            "format_reward": 1.0,
        },
    )
    assert episode_correct(episode) == 1.0
