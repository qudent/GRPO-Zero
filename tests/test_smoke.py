"""Smoke test: end-to-end fork rollout + update with mocked tiny model.

Validates that the full pipeline runs without crashes or NaN values,
including fork token handling, KV cache cloning, parser state tracking,
reward computation, and token-weighted GRPO update.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
import torch
import numpy as np
from unittest.mock import MagicMock, patch
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Union, Tuple

from data_types import Episode, MiniBatch
from fork_parser import ForkParser, ParserState
from fork_reward import ForkRewardConfig, compute_fork_reward, compute_token_weights
from grpo import normalize_rewards_per_group, update_policy_fork, update_policy


@dataclass
class TinyConfig:
    vocab_size: int = 200
    hidden_size: int = 32
    num_attention_heads: int = 2
    num_key_value_heads: int = 2
    num_hidden_layers: int = 1
    intermediate_size: int = 64
    rms_norm_eps: float = 1e-6
    rope_theta: float = 10000.0
    max_position_embeddings: int = 512
    max_window_layers: int = 1
    tie_word_embeddings: bool = True
    attention_dropout: float = 0.0
    hidden_act: str = "silu"
    sliding_window: int = 512
    use_sliding_window: bool = False
    use_cache: bool = True
    torch_dtype: str = "float32"
    bos_token_id: int = 0
    eos_token_id: int = 1
    model_type: str = "qwen2"
    initializer_range: float = 0.02


class TinyAttention(torch.nn.Module):
    def __init__(self, hidden, n_heads):
        super().__init__()
        self.n_kv_heads = n_heads
        self.head_dim = hidden // n_heads
        self.cache_k = None
        self.cache_v = None

    def init_kv_cache(self, max_batch_size, max_seq_len, dtype, device):
        shape = (max_batch_size, max_seq_len, self.n_kv_heads, self.head_dim)
        self.cache_k = torch.zeros(shape, dtype=dtype, device=device)
        self.cache_v = torch.zeros(shape, dtype=dtype, device=device)

    def del_kv_cache(self):
        self.cache_k = None
        self.cache_v = None


class TinyBlock(torch.nn.Module):
    def __init__(self, hidden, n_heads):
        super().__init__()
        self.self_attn = TinyAttention(hidden, n_heads)


class TinyTransformer(torch.nn.Module):
    """Minimal transformer-like model for smoke testing."""
    def __init__(self, vocab_size=200, hidden=32, n_heads=2, n_layers=1):
        super().__init__()
        self.vocab_size = vocab_size
        self.params = TinyConfig(vocab_size=vocab_size, hidden_size=hidden,
                                  num_attention_heads=n_heads, num_hidden_layers=n_layers)
        self.embed_tokens = torch.nn.Embedding(vocab_size, hidden)
        self.proj = torch.nn.Linear(hidden, vocab_size, bias=False)
        self.layers = torch.nn.ModuleList([TinyBlock(hidden, n_heads) for _ in range(n_layers)])

    def init_kv_cache(self, max_batch_size, max_seq_len, device, dtype):
        for layer in self.layers:
            layer.self_attn.init_kv_cache(max_batch_size, max_seq_len, dtype, device)

    def del_kv_cache(self):
        for layer in self.layers:
            layer.self_attn.del_kv_cache()

    def inference(self, tokens, start_pos):
        h = self.embed_tokens(tokens)
        logits = self.proj(h)
        return logits[:, -1:, :]

    def forward(self, tokens):
        h = self.embed_tokens(tokens)
        return self.proj(h)

    def resize_embeddings(self, new_vocab_size):
        old_size = self.vocab_size
        if new_vocab_size == old_size:
            return
        old_embed = self.embed_tokens
        new_embed = torch.nn.Embedding(new_vocab_size, old_embed.weight.shape[1])
        with torch.no_grad():
            new_embed.weight[:old_size] = old_embed.weight
            mean_embed = old_embed.weight.mean(dim=0)
            for i in range(old_size, new_vocab_size):
                new_embed.weight[i] = mean_embed
        self.embed_tokens = new_embed
        old_proj = self.proj
        new_proj = torch.nn.Linear(old_proj.in_features, new_vocab_size, bias=False)
        with torch.no_grad():
            new_proj.weight[:old_size] = old_proj.weight
        self.proj = new_proj
        self.vocab_size = new_vocab_size


class MockTokenizer:
    """Mock tokenizer for smoke tests."""
    def __init__(self, vocab_size=200):
        self.eos_token = "<eos>"
        self.eos_token_id = 1
        self.pad_token = "<pad>"
        self.pad_token_id = 0
        self.fork_token_id = vocab_size
        self.fork1_token_id = vocab_size + 1
        self.fork2_token_id = vocab_size + 2
        self._vocab_size = vocab_size + 3

    @property
    def vocab_size(self):
        return self._vocab_size

    def add_fork_tokens(self):
        return self._vocab_size

    def detokenize(self, token_ids):
        # Produce recognizable text for parser
        parts = []
        for tid in token_ids:
            if tid == self.fork_token_id:
                parts.append("<fork>")
            elif tid == self.fork1_token_id:
                parts.append("<fork1>")
            elif tid == self.fork2_token_id:
                parts.append("<fork2>")
            elif tid == self.eos_token_id:
                parts.append("<eos>")
            elif tid == self.pad_token_id:
                parts.append("")
            else:
                parts.append(f"t{tid}")
        return "".join(parts)


class TestSmokeEndToEnd:
    """Full pipeline smoke test on CPU with mocked model."""

    def test_vanilla_rollout_backward_compat(self):
        """Verify original rollout + update_policy still works unmodified."""
        from grpo import rollout
        from countdown_task import reward_function

        model = TinyTransformer(vocab_size=200, hidden=32, n_heads=2)
        tokenizer = MockTokenizer(vocab_size=200)
        # Remove fork tokens for vanilla test
        tokenizer.fork_token_id = None
        tokenizer.fork1_token_id = None
        tokenizer.fork2_token_id = None

        batch = MiniBatch(
            prefix=["prefix1"],
            prefix_tokens=[["p", "r", "e"]],
            prefix_token_ids=[[10, 20, 30]],
            numbers=[[1, 2, 3]],
            target=[6],
        )

        episodes = rollout(
            model=model,
            tokenizer=tokenizer,
            batch=batch,
            max_gen_len=20,
            num_answer_per_question=2,
            reward_function=reward_function,
            device=torch.device("cpu"),
            dtype=torch.float32,
        )

        assert len(episodes) == 2
        for ep in episodes:
            assert isinstance(ep.reward, float)
            assert not np.isnan(ep.reward)

        # Update policy should work
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
        result = update_policy(
            model=model,
            optimizer=optimizer,
            episodes=episodes,
            micro_batch_size=2,
            pad_token_id=0,
            max_grad_norm=1.0,
            device=torch.device("cpu"),
            dtype=torch.float32,
        )
        assert not np.isnan(result["loss"])
        assert not np.isnan(result["grad_norm"])

    def test_fork_rollout_no_fork_emitted(self):
        """Fork rollout where model never emits <fork> - should behave like vanilla."""
        from grpo import fork_rollout
        from countdown_task import reward_function

        model = TinyTransformer(vocab_size=203, hidden=32, n_heads=2)
        tokenizer = MockTokenizer(vocab_size=200)
        model.resize_embeddings(tokenizer.vocab_size)

        batch = MiniBatch(
            prefix=["prefix1"],
            prefix_tokens=[["p", "r"]],
            prefix_token_ids=[[10, 20]],
            numbers=[[1, 2, 3]],
            target=[6],
        )

        config = ForkRewardConfig()
        episodes = fork_rollout(
            model=model,
            tokenizer=tokenizer,
            batch=batch,
            max_gen_len=15,
            num_answer_per_question=2,
            reward_function=reward_function,
            device=torch.device("cpu"),
            dtype=torch.float32,
            fork_reward_config=config,
            force_fork_prob=0.0,
        )

        assert len(episodes) >= 2
        for ep in episodes:
            assert not np.isnan(ep.reward)

    def test_fork_rollout_with_forced_fork(self):
        """Fork rollout with force_fork_prob=1.0 to ensure forking happens."""
        from grpo import fork_rollout
        from countdown_task import reward_function

        model = TinyTransformer(vocab_size=203, hidden=32, n_heads=2)
        tokenizer = MockTokenizer(vocab_size=200)
        model.resize_embeddings(tokenizer.vocab_size)

        batch = MiniBatch(
            prefix=["prefix1"],
            prefix_tokens=[["p", "r"]],
            prefix_token_ids=[[10, 20]],
            numbers=[[1, 2, 3]],
            target=[6],
        )

        config = ForkRewardConfig()
        episodes = fork_rollout(
            model=model,
            tokenizer=tokenizer,
            batch=batch,
            max_gen_len=30,
            num_answer_per_question=2,
            reward_function=reward_function,
            device=torch.device("cpu"),
            dtype=torch.float32,
            fork_reward_config=config,
            force_fork_prob=1.0,
        )

        # With forced fork, should have more episodes (branches)
        assert len(episodes) >= 2
        for ep in episodes:
            assert not np.isnan(ep.reward)
            assert ep.token_weights is not None

    def test_fork_update_no_nan(self):
        """Full fork rollout -> token-weighted update produces no NaN."""
        from grpo import fork_rollout
        from countdown_task import reward_function

        model = TinyTransformer(vocab_size=203, hidden=32, n_heads=2)
        tokenizer = MockTokenizer(vocab_size=200)
        model.resize_embeddings(tokenizer.vocab_size)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

        batch = MiniBatch(
            prefix=["p1", "p2"],
            prefix_tokens=[["a", "b"], ["c", "d"]],
            prefix_token_ids=[[10, 20], [30, 40]],
            numbers=[[1, 2, 3], [4, 5, 6]],
            target=[6, 15],
        )

        config = ForkRewardConfig()
        episodes = fork_rollout(
            model=model,
            tokenizer=tokenizer,
            batch=batch,
            max_gen_len=15,
            num_answer_per_question=2,
            reward_function=reward_function,
            device=torch.device("cpu"),
            dtype=torch.float32,
            fork_reward_config=config,
            force_fork_prob=0.0,
        )

        result = update_policy_fork(
            model=model,
            optimizer=optimizer,
            episodes=episodes,
            micro_batch_size=2,
            pad_token_id=0,
            max_grad_norm=1.0,
            device=torch.device("cpu"),
            dtype=torch.float32,
        )

        assert not np.isnan(result["loss"])
        assert not np.isnan(result["grad_norm"])
        assert not np.isnan(result["entropy"])


class TestEmbeddingResize:
    def test_resize_preserves_old_weights(self):
        model = TinyTransformer(vocab_size=100, hidden=32)
        old_weights = model.embed_tokens.weight[:100].clone()
        model.resize_embeddings(103)
        assert model.vocab_size == 103
        assert torch.allclose(model.embed_tokens.weight[:100], old_weights)

    def test_resize_new_tokens_initialized(self):
        model = TinyTransformer(vocab_size=100, hidden=32)
        model.resize_embeddings(103)
        # New tokens should not be zero (initialized from mean + noise)
        for i in range(100, 103):
            assert not torch.all(model.embed_tokens.weight[i] == 0)

    def test_resize_noop_same_size(self):
        model = TinyTransformer(vocab_size=100, hidden=32)
        old_id = id(model.embed_tokens)
        model.resize_embeddings(100)
        assert id(model.embed_tokens) == old_id


class TestRewardBoundedness:
    """Exhaustive boundedness check across parameter space."""

    def test_all_combinations_bounded(self):
        configs = [
            ForkRewardConfig(alpha=0.0, beta=0.0, delta=0.0, budget_ms=1000),
            ForkRewardConfig(alpha=1.0, beta=0.5, delta=0.5, budget_ms=1000),
            ForkRewardConfig(alpha=0.5, beta=0.2, delta=0.2, budget_ms=1000),
        ]
        for config in configs:
            min_r = -config.delta - config.beta
            max_r = 1.0
            for correct in [True, False]:
                for t in [0, 100, 1000, 5000]:
                    for invalid in [True, False]:
                        r = compute_fork_reward(correct, t, invalid, config)
                        assert min_r - 1e-6 <= r["reward"] <= max_r + 1e-6, \
                            f"Bound violation: {r['reward']} not in [{min_r}, {max_r}] " \
                            f"for config={config}, correct={correct}, t={t}, invalid={invalid}"
