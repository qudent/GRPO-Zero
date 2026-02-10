"""Tests for fork warmup curriculum helpers."""

import pytest

from data_types import Episode
from fork_curriculum import (
    apply_fork_presence_bonus,
    fork_presence_bonus_coef,
    linear_decay,
)


def _episode(forked: float, invalid_fork: float = 0.0, reward: float = 0.2) -> Episode:
    return Episode(
        prefix="p",
        text="p<think>x</think>\n<answer>1+2</answer>",
        prefix_token_ids=[1],
        prefix_tokens=["p"],
        generated_token_ids=[2, 3],
        is_finished=True,
        reward=reward,
        reward_info={
            "forked": forked,
            "invalid_fork": invalid_fork,
            "answer_reward": 1.0,
            "format_reward": 1.0,
        },
    )


def test_linear_decay_endpoints():
    assert linear_decay(step=1, start=0.3, end=0.0, decay_steps=100) == 0.3
    assert linear_decay(step=100, start=0.3, end=0.0, decay_steps=100) == 0.0
    assert linear_decay(step=150, start=0.3, end=0.0, decay_steps=100) == 0.0


def test_bonus_coef_reads_config_and_decays():
    cfg = {
        "warmup_fork_presence_bonus_start": 0.3,
        "warmup_fork_presence_bonus_end": 0.0,
        "warmup_fork_presence_bonus_steps": 3,
    }
    assert fork_presence_bonus_coef(1, cfg) == 0.3
    assert fork_presence_bonus_coef(2, cfg) == 0.15
    assert fork_presence_bonus_coef(3, cfg) == 0.0


def test_apply_bonus_valid_only():
    episodes = [
        _episode(forked=1.0, invalid_fork=0.0, reward=0.2),
        _episode(forked=1.0, invalid_fork=1.0, reward=0.2),
        _episode(forked=0.0, invalid_fork=0.0, reward=0.2),
    ]

    updated, hit_rate = apply_fork_presence_bonus(
        episodes, bonus_coef=0.3, valid_only=True
    )

    assert hit_rate == 1 / 3
    assert updated[0].reward == 0.5
    assert updated[1].reward == 0.2
    assert updated[2].reward == 0.2
    assert updated[0].reward_info["base_reward"] == 0.2
    assert updated[0].reward_info["fork_presence_bonus"] == 0.3
    assert updated[1].reward_info["fork_presence_bonus"] == 0.0


def test_apply_bonus_all_forks_when_not_valid_only():
    episodes = [
        _episode(forked=1.0, invalid_fork=1.0, reward=0.2),
        _episode(forked=0.0, invalid_fork=0.0, reward=0.2),
    ]

    updated, hit_rate = apply_fork_presence_bonus(
        episodes, bonus_coef=0.1, valid_only=False
    )

    assert hit_rate == 0.5
    assert updated[0].reward == pytest.approx(0.3)
    assert updated[1].reward == 0.2
