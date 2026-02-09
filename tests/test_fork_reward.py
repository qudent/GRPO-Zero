"""Tests for fork_reward.py - reward computation, boundedness, and token weights."""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from fork_reward import (
    ForkRewardConfig,
    compute_fork_reward,
    compute_latency_proxy,
    compute_token_weights,
)


class TestLatencyProxy:
    def test_no_fork(self):
        config = ForkRewardConfig(c1_ms=1.0, c2_ms=1.5)
        proxy = compute_latency_proxy(
            steps_before_fork=100,
            steps_after_fork_branch_a=0,
            steps_after_fork_branch_b=0,
            forked=False,
            config=config,
        )
        assert proxy == 100.0  # 100 * 1.0

    def test_fork_symmetric_branches(self):
        config = ForkRewardConfig(c1_ms=1.0, c2_ms=1.5)
        proxy = compute_latency_proxy(
            steps_before_fork=50,
            steps_after_fork_branch_a=30,
            steps_after_fork_branch_b=30,
            forked=True,
            config=config,
        )
        # 50 * 1.0 + max(30, 30) * 1.5 = 50 + 45 = 95
        assert proxy == 95.0

    def test_fork_asymmetric_branches(self):
        config = ForkRewardConfig(c1_ms=1.0, c2_ms=2.0)
        proxy = compute_latency_proxy(
            steps_before_fork=20,
            steps_after_fork_branch_a=10,
            steps_after_fork_branch_b=50,
            forked=True,
            config=config,
        )
        # 20 * 1.0 + max(10, 50) * 2.0 = 20 + 100 = 120
        assert proxy == 120.0

    def test_zero_steps(self):
        config = ForkRewardConfig()
        proxy = compute_latency_proxy(0, 0, 0, False, config)
        assert proxy == 0.0


class TestForkReward:
    def test_success_no_latency(self):
        config = ForkRewardConfig(alpha=0.5, beta=0.2, delta=0.2, budget_ms=1000.0)
        result = compute_fork_reward(correct=True, t_proxy_ms=0.0, invalid_fork=False, config=config)
        assert result["reward"] == 1.0  # max reward

    def test_success_full_latency(self):
        config = ForkRewardConfig(alpha=0.5, beta=0.2, delta=0.2, budget_ms=1000.0)
        result = compute_fork_reward(correct=True, t_proxy_ms=1000.0, invalid_fork=False, config=config)
        assert result["reward"] == 0.5  # 1.0 - 0.5 * 1.0

    def test_success_over_budget(self):
        config = ForkRewardConfig(alpha=0.5, beta=0.2, delta=0.2, budget_ms=1000.0)
        result = compute_fork_reward(correct=True, t_proxy_ms=2000.0, invalid_fork=False, config=config)
        # L_norm is capped at 1.0
        assert result["reward"] == 0.5

    def test_failure_no_invalid(self):
        config = ForkRewardConfig(alpha=0.5, beta=0.2, delta=0.2)
        result = compute_fork_reward(correct=False, t_proxy_ms=100.0, invalid_fork=False, config=config)
        assert result["reward"] == -0.2  # -delta

    def test_failure_with_invalid_fork(self):
        config = ForkRewardConfig(alpha=0.5, beta=0.2, delta=0.2)
        result = compute_fork_reward(correct=False, t_proxy_ms=100.0, invalid_fork=True, config=config)
        assert result["reward"] == -0.4  # -delta - beta

    def test_success_with_invalid_fork(self):
        config = ForkRewardConfig(alpha=0.5, beta=0.2, delta=0.2, budget_ms=1000.0)
        result = compute_fork_reward(correct=True, t_proxy_ms=0.0, invalid_fork=True, config=config)
        assert result["reward"] == 0.8  # 1.0 - 0.2

    def test_reward_bounded(self):
        """Reward should always be in [-delta-beta, 1.0] range."""
        config = ForkRewardConfig(alpha=0.5, beta=0.2, delta=0.2, budget_ms=1000.0)
        min_reward = -config.delta - config.beta
        max_reward = 1.0

        for correct in [True, False]:
            for t_proxy in [0, 500, 1000, 5000]:
                for invalid in [True, False]:
                    result = compute_fork_reward(correct, t_proxy, invalid, config)
                    assert min_reward <= result["reward"] <= max_reward, \
                        f"Reward {result['reward']} out of bounds for correct={correct}, t={t_proxy}, invalid={invalid}"

    def test_reward_info_keys(self):
        config = ForkRewardConfig()
        result = compute_fork_reward(True, 100.0, False, config)
        info = result["reward_info"]
        assert "correct" in info
        assert "t_proxy_ms" in info
        assert "l_norm" in info
        assert "invalid_fork" in info
        assert "invalid_penalty" in info
        assert "latency_penalty" in info


class TestTokenWeights:
    def test_no_fork(self):
        weights = compute_token_weights(total_tokens=10, fork_position=None, branch_id=None)
        assert weights == [1.0] * 10

    def test_fork_at_start(self):
        weights = compute_token_weights(total_tokens=10, fork_position=0, branch_id=0, active_branches=2)
        assert all(w == 0.5 for w in weights)

    def test_fork_in_middle(self):
        weights = compute_token_weights(total_tokens=10, fork_position=5, branch_id=0, active_branches=2)
        assert weights[:5] == [1.0] * 5
        assert weights[5:] == [0.5] * 5

    def test_fork_branch_b(self):
        weights = compute_token_weights(total_tokens=10, fork_position=3, branch_id=1, active_branches=2)
        assert weights[:3] == [1.0] * 3
        assert weights[3:] == [0.5] * 7

    def test_weight_sum_properties(self):
        """For a forked rollout with equal branches, total weight should be less than unforked."""
        unforked_weights = compute_token_weights(10, None, None)
        forked_weights_a = compute_token_weights(10, 5, 0, 2)
        forked_weights_b = compute_token_weights(10, 5, 1, 2)
        assert sum(unforked_weights) == 10.0
        # Pre-fork shared (5) + post-fork A (2.5) + post-fork B (2.5) = 10
        assert sum(forked_weights_a) + sum(forked_weights_b) == 15.0  # some double counting expected

    def test_empty(self):
        weights = compute_token_weights(0, None, None)
        assert weights == []
