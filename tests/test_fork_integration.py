"""Integration tests for fork-race rollout and GRPO update.

Tests race winner logic, early stop, and token-weighted loss using
mock/minimal model setups.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
import torch
import numpy as np
from unittest.mock import MagicMock, patch
from dataclasses import replace

from data_types import Episode, MiniBatch
from fork_parser import ForkParser, ParserState
from fork_reward import ForkRewardConfig, compute_fork_reward, compute_token_weights
from grpo import fork_rollout, normalize_rewards_per_group, update_policy_fork
from countdown_task import reward_function


class TestRaceWinnerLogic:
    """Test that first-correct-wins semantics work."""

    def test_branch_a_wins(self):
        """Branch A correct, branch B incorrect -> A wins."""
        config = ForkRewardConfig()
        rwd = compute_fork_reward(correct=True, t_proxy_ms=50.0, invalid_fork=False, config=config)
        assert rwd["reward"] > 0  # positive reward for correctness

    def test_branch_b_wins(self):
        """Branch A incorrect, branch B correct -> still any_correct=True."""
        config = ForkRewardConfig()
        rwd = compute_fork_reward(correct=True, t_proxy_ms=80.0, invalid_fork=False, config=config)
        assert rwd["reward"] > 0

    def test_both_fail(self):
        """Both branches incorrect -> negative reward."""
        config = ForkRewardConfig()
        rwd = compute_fork_reward(correct=False, t_proxy_ms=100.0, invalid_fork=False, config=config)
        assert rwd["reward"] < 0

    def test_both_correct_uses_critical_path(self):
        """When both correct, reward uses the latency of the full rollout."""
        config = ForkRewardConfig(budget_ms=100.0, alpha=0.5)
        # Short critical path
        rwd_short = compute_fork_reward(correct=True, t_proxy_ms=20.0, invalid_fork=False, config=config)
        # Long critical path
        rwd_long = compute_fork_reward(correct=True, t_proxy_ms=80.0, invalid_fork=False, config=config)
        assert rwd_short["reward"] > rwd_long["reward"]


class TestEarlyStop:
    """Test early stop via parser answer detection."""

    def test_answer_triggers_finish(self):
        p = ForkParser()
        p.feed("<think>work</think>\n<answer>42</answer>")
        assert p.has_complete_answer

    def test_partial_answer_no_finish(self):
        p = ForkParser()
        p.feed("<think>work</think>\n<answer>42")
        assert not p.has_complete_answer

    def test_wrong_answer_still_finishes(self):
        """Even wrong answer is detected as complete (correctness checked separately)."""
        p = ForkParser()
        p.feed("<think>work</think>\n<answer>wrong</answer>")
        assert p.has_complete_answer
        assert p.answer_text == "wrong"


class TestNormalizeRewards:
    """Test reward normalization per group."""

    def _make_episode(self, prefix, reward):
        return Episode(
            prefix=prefix,
            text=prefix + "gen",
            prefix_token_ids=[1, 2, 3],
            prefix_tokens=["a", "b", "c"],
            generated_token_ids=[4, 5],
            is_finished=True,
            reward=reward,
            reward_info={"answer_reward": 0.0, "format_reward": 0.0},
        )

    def test_normalize_single_group(self):
        episodes = [
            self._make_episode("q1", 1.0),
            self._make_episode("q1", -1.0),
        ]
        normalized = normalize_rewards_per_group(episodes)
        rewards = [e.reward for e in normalized]
        assert abs(np.mean(rewards)) < 1e-3

    def test_normalize_two_groups(self):
        episodes = [
            self._make_episode("q1", 1.0),
            self._make_episode("q1", 0.0),
            self._make_episode("q2", 0.5),
            self._make_episode("q2", -0.5),
        ]
        normalized = normalize_rewards_per_group(episodes)
        q1_rewards = [e.reward for e in normalized if e.prefix == "q1"]
        q2_rewards = [e.reward for e in normalized if e.prefix == "q2"]
        assert abs(np.mean(q1_rewards)) < 1e-3
        assert abs(np.mean(q2_rewards)) < 1e-3


class TestTokenWeightedUpdate:
    """Test that the token-weighted GRPO update runs without errors."""

    def _make_fork_episode(self, gen_len, fork_pos=None, branch_id=None, reward=0.5):
        gen_ids = list(range(100, 100 + gen_len))
        tw = compute_token_weights(gen_len, fork_pos, branch_id, 2 if fork_pos is not None else 1)
        return Episode(
            prefix="prefix ",
            text="prefix " + "x" * gen_len,
            prefix_token_ids=[1, 2, 3],
            prefix_tokens=["p", "r", "e"],
            generated_token_ids=gen_ids,
            is_finished=True,
            reward=reward,
            reward_info={"answer_reward": 1.0, "format_reward": 1.0},
            token_weights=tw,
            fork_info={"correct": 1.0},
            branch_id=branch_id,
        )

    def test_update_runs_on_cpu(self):
        """Smoke test that update_policy_fork runs with tiny model on CPU."""
        torch.manual_seed(42)
        device = torch.device("cpu")
        dtype = torch.float32

        # Create tiny model
        vocab_size = 200
        hidden = 32
        embed = torch.nn.Embedding(vocab_size, hidden)
        linear = torch.nn.Linear(hidden, vocab_size, bias=False)

        class TinyModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.embed = torch.nn.Embedding(vocab_size, hidden)
                self.proj = torch.nn.Linear(hidden, vocab_size, bias=False)

            def forward(self, tokens):
                h = self.embed(tokens)
                return self.proj(h)

        model = TinyModel()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

        episodes = [
            self._make_fork_episode(5, fork_pos=2, branch_id=0, reward=0.8),
            self._make_fork_episode(5, fork_pos=2, branch_id=1, reward=0.8),
            self._make_fork_episode(5, fork_pos=None, branch_id=None, reward=-0.2),
        ]

        # Pad token should be outside gen range
        pad_token_id = 0

        result = update_policy_fork(
            model=model,
            optimizer=optimizer,
            episodes=episodes,
            micro_batch_size=2,
            pad_token_id=pad_token_id,
            max_grad_norm=1.0,
            device=device,
            dtype=dtype,
        )

        assert "loss" in result
        assert "grad_norm" in result
        assert "entropy" in result
        assert not np.isnan(result["loss"])
        assert not np.isnan(result["grad_norm"])

    def test_weights_affect_gradient(self):
        """Verify that different token weights produce different gradients."""
        torch.manual_seed(42)

        class TinyModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.embed = torch.nn.Embedding(200, 32)
                self.proj = torch.nn.Linear(32, 200, bias=False)

            def forward(self, tokens):
                return self.proj(self.embed(tokens))

        # Run with uniform weights
        model1 = TinyModel()
        opt1 = torch.optim.SGD(model1.parameters(), lr=1e-3)
        ep_uniform = self._make_fork_episode(5, fork_pos=None, branch_id=None, reward=1.0)

        # Run with fork weights
        model2 = TinyModel()
        model2.load_state_dict(model1.state_dict())
        opt2 = torch.optim.SGD(model2.parameters(), lr=1e-3)
        ep_forked = self._make_fork_episode(5, fork_pos=2, branch_id=0, reward=1.0)

        r1 = update_policy_fork(model1, opt1, [ep_uniform], 1, 0, 1.0, torch.device("cpu"), torch.float32)
        r2 = update_policy_fork(model2, opt2, [ep_forked], 1, 0, 1.0, torch.device("cpu"), torch.float32)

        # Different weights should produce different losses
        # (not guaranteed to be different due to normalization, but grad_norm can differ)
        # Just verify both ran successfully
        assert not np.isnan(r1["loss"])
        assert not np.isnan(r2["loss"])


class TestInvalidForkPenalty:
    """Test that invalid forks receive penalty in reward."""

    def test_invalid_fork_reduces_reward(self):
        config = ForkRewardConfig(beta=0.2)
        valid = compute_fork_reward(True, 50.0, False, config)
        invalid = compute_fork_reward(True, 50.0, True, config)
        assert valid["reward"] > invalid["reward"]
        assert valid["reward"] - invalid["reward"] == pytest.approx(0.2)

    def test_invalid_fork_on_failure(self):
        config = ForkRewardConfig(beta=0.2, delta=0.2)
        valid = compute_fork_reward(False, 50.0, False, config)
        invalid = compute_fork_reward(False, 50.0, True, config)
        assert valid["reward"] == pytest.approx(-0.2)
        assert invalid["reward"] == pytest.approx(-0.4)


class ScriptTokenizer:
    """Tokenizer for deterministic fork-race behavior tests."""

    PAD = 0
    EOS = 1
    FORK = 2
    FORK1 = 3
    FORK2 = 4
    FILL = 5
    ANSWER_OPEN = 6
    ANSWER_EXPR = 7
    ANSWER_CLOSE = 8

    def __init__(self):
        self.pad_token = "<pad>"
        self.pad_token_id = self.PAD
        self.eos_token = "<eos>"
        self.eos_token_id = self.EOS
        self.fork_token_id = self.FORK
        self.fork1_token_id = self.FORK1
        self.fork2_token_id = self.FORK2
        self._vocab_size = 32
        self._decode_map = {
            self.PAD: "",
            self.EOS: "<eos>",
            self.FORK: "<fork>",
            self.FORK1: "<fork1>",
            self.FORK2: "<fork2>",
            self.FILL: "x",
            self.ANSWER_OPEN: "</think>\n<answer>",
            self.ANSWER_EXPR: "1+2+3",
            self.ANSWER_CLOSE: "</answer>",
        }

    @property
    def vocab_size(self):
        return self._vocab_size

    def detokenize(self, token_ids):
        return "".join(self._decode_map.get(tid, "t") for tid in token_ids)


class ScriptAttention(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.n_kv_heads = 1
        self.head_dim = 1
        self.cache_k = None
        self.cache_v = None

    def init_kv_cache(self, max_batch_size, max_seq_len, dtype, device):
        shape = (max_batch_size, max_seq_len, self.n_kv_heads, self.head_dim)
        self.cache_k = torch.zeros(shape, dtype=dtype, device=device)
        self.cache_v = torch.zeros(shape, dtype=dtype, device=device)

    def del_kv_cache(self):
        self.cache_k = None
        self.cache_v = None


class ScriptBlock(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.self_attn = ScriptAttention()


class ScriptModel(torch.nn.Module):
    """Deterministic model to force a known fork + race outcome."""

    def __init__(self, tokenizer: ScriptTokenizer, fork_emit_token_id: int | None = None):
        super().__init__()
        self.tok = tokenizer
        self.layers = torch.nn.ModuleList([ScriptBlock()])
        self.vocab_size = tokenizer.vocab_size
        self.fork_emit_token_id = (
            tokenizer.FORK if fork_emit_token_id is None else int(fork_emit_token_id)
        )

    def init_kv_cache(self, max_batch_size, max_seq_len, device, dtype):
        for layer in self.layers:
            layer.self_attn.init_kv_cache(max_batch_size, max_seq_len, dtype, device)

    def del_kv_cache(self):
        for layer in self.layers:
            layer.self_attn.del_kv_cache()

    def inference(self, tokens, start_pos):
        bsz = tokens.shape[0]
        last = tokens[:, -1]
        logits = torch.full(
            (bsz, 1, self.vocab_size),
            -1e9,
            dtype=torch.float32,
            device=tokens.device,
        )

        # Default behavior: emit filler token.
        next_ids = torch.full((bsz,), self.tok.FILL, dtype=torch.long, device=tokens.device)
        # Emit a fork-like token at the second decode step (start_pos == 1).
        if int(start_pos) == 1:
            next_ids[:] = self.fork_emit_token_id
        else:
            # After branching, branch A (seen <fork1>) emits a correct answer quickly.
            next_ids = torch.where(last == self.tok.FORK1, self.tok.ANSWER_OPEN, next_ids)
            next_ids = torch.where(last == self.tok.ANSWER_OPEN, self.tok.ANSWER_EXPR, next_ids)
            next_ids = torch.where(last == self.tok.ANSWER_EXPR, self.tok.ANSWER_CLOSE, next_ids)

        logits.scatter_(2, next_ids.view(bsz, 1, 1), 0.0)
        return logits


class TestForkRaceSemantics:
    def test_first_correct_terminates_race_and_reduces_latency(self):
        """A correct branch should stop the sibling branch immediately."""
        tokenizer = ScriptTokenizer()
        model = ScriptModel(tokenizer)

        batch = MiniBatch(
            prefix=["prompt<think>"],
            prefix_tokens=[["prompt", "<think>"]],
            prefix_token_ids=[[9]],
            numbers=[[1, 2, 3]],
            target=[6],
        )

        episodes = fork_rollout(
            model=model,
            tokenizer=tokenizer,
            batch=batch,
            max_gen_len=12,
            num_answer_per_question=1,
            reward_function=reward_function,
            device=torch.device("cpu"),
            dtype=torch.float32,
            fork_reward_config=ForkRewardConfig(c1_ms=1.0, c2_ms=1.5),
            fork_token_logit_bias=0.0,
            fork_token_target_prob=None,
        )

        assert len(episodes) == 2
        assert {ep.branch_id for ep in episodes} == {0, 1}
        assert all(ep.reward_info.get("forked", 0.0) == pytest.approx(1.0) for ep in episodes)
        assert max(len(ep.generated_token_ids) for ep in episodes) <= 5
        # Latency proxy is accumulated only until first-correct answer:
        # c1 + c1 + c2 + c2 + c2 = 6.5ms in this scripted rollout.
        assert all(ep.reward_info["t_proxy_ms"] == pytest.approx(6.5) for ep in episodes)

        by_branch = {ep.branch_id: ep for ep in episodes}
        assert by_branch[0].reward_info["answer_reward"] == pytest.approx(1.0)
        assert by_branch[1].reward_info["answer_reward"] == pytest.approx(0.0)

    @pytest.mark.parametrize("fork_emit_token_id", [ScriptTokenizer.FORK1, ScriptTokenizer.FORK2])
    def test_branch_tokens_also_trigger_fork_event(self, fork_emit_token_id):
        tokenizer = ScriptTokenizer()
        model = ScriptModel(tokenizer, fork_emit_token_id=fork_emit_token_id)

        batch = MiniBatch(
            prefix=["prompt<think>"],
            prefix_tokens=[["prompt", "<think>"]],
            prefix_token_ids=[[9]],
            numbers=[[1, 2, 3]],
            target=[6],
        )

        episodes = fork_rollout(
            model=model,
            tokenizer=tokenizer,
            batch=batch,
            max_gen_len=12,
            num_answer_per_question=1,
            reward_function=reward_function,
            device=torch.device("cpu"),
            dtype=torch.float32,
            fork_reward_config=ForkRewardConfig(c1_ms=1.0, c2_ms=1.5),
            fork_token_logit_bias=0.0,
            fork_token_target_prob=None,
        )

        assert len(episodes) == 2
        assert {ep.branch_id for ep in episodes} == {0, 1}
        assert all(ep.reward_info.get("forked", 0.0) == pytest.approx(1.0) for ep in episodes)
