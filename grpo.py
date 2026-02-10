import dataclasses
import gc
import math
from collections import defaultdict
from typing import Callable, Dict, List, Optional

import numpy as np
import torch

from data_types import Episode, MiniBatch
from fork_parser import ForkParser
from fork_reward import (
    ForkRewardConfig,
    compute_fork_reward,
    compute_token_weights,
)
from qwen2_model import Transformer
from tokenizer import Tokenizer


@torch.inference_mode()
def rollout(
    model: Transformer,
    batch: MiniBatch,
    tokenizer: Tokenizer,
    max_gen_len: int,
    num_answer_per_question: int,
    reward_function: Callable,
    device: torch.device,
    dtype: torch.dtype,
) -> List[Episode]:
    end_token = tokenizer.eos_token
    end_token_id = tokenizer.eos_token_id
    pad_token_id = tokenizer.pad_token_id
    prefix_token_ids = batch.prefix_token_ids
    bsz = len(batch.prefix) * num_answer_per_question
    min_prompt_len = min(len(t) for t in prefix_token_ids)
    max_prompt_len = max(len(t) for t in prefix_token_ids)
    total_len = max_gen_len + max_prompt_len
    model.init_kv_cache(
        max_batch_size=bsz,
        max_seq_len=total_len,
        device=device,
        dtype=dtype,
    )
    tokens = torch.full((bsz, total_len), pad_token_id, dtype=torch.long, device=device)
    for k, t in enumerate(prefix_token_ids):
        offset = k * num_answer_per_question
        for i in range(num_answer_per_question):
            tokens[offset + i, : len(t)] = torch.tensor(
                t, dtype=torch.long, device=device
            )

    prev_pos = 0
    input_text_mask = tokens != pad_token_id
    assert min_prompt_len < total_len
    is_finished = torch.zeros((bsz,), dtype=torch.bool, device=device)

    for cur_pos in range(min_prompt_len, total_len):
        print(
            f"\r* Generating trajectories: {cur_pos-min_prompt_len:>4d}/{total_len-min_prompt_len:>4d}",
            flush=True,
            end="",
        )
        with torch.autocast(device_type=device.type, dtype=dtype):
            logits = model.inference(tokens[:, prev_pos:cur_pos], prev_pos)
        probs = torch.softmax(logits[:, -1], dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        next_token = next_token.reshape(-1)
        next_token = torch.where(
            input_text_mask[:, cur_pos], tokens[:, cur_pos], next_token
        )
        # if an rollout is finished, we fill the rest of the tokens with pad_token_id
        next_token = torch.where(is_finished, pad_token_id, next_token)
        tokens[:, cur_pos] = next_token
        if end_token_id is not None:
            is_end_token = next_token == end_token_id
            is_generated_token = ~input_text_mask[:, cur_pos]
            is_finished = is_finished | (is_end_token & is_generated_token)
        prev_pos = cur_pos
        if is_finished.all():
            break
    model.del_kv_cache()
    gc.collect()
    torch.cuda.empty_cache()
    is_finished_list = is_finished.tolist()
    tokens_list = tokens.tolist()

    # prepare the output episodes
    episodes = []
    for i in range(bsz // num_answer_per_question):
        for j in range(num_answer_per_question):
            idx = i * num_answer_per_question + j
            generated_token_ids = tokens_list[idx][len(batch.prefix_token_ids[i]) :]
            # remove padding tokens
            if pad_token_id in generated_token_ids:
                generated_token_ids = generated_token_ids[
                    : generated_token_ids.index(pad_token_id)
                ]
            generated_text = tokenizer.detokenize(generated_token_ids)
            rewards = reward_function(
                response=generated_text,
                numbers=batch.numbers[i],
                target=batch.target[i],
                end_token=end_token,
            )
            episode = Episode(
                prefix=batch.prefix[i],
                text=batch.prefix[i] + generated_text,
                prefix_token_ids=batch.prefix_token_ids[i],
                prefix_tokens=batch.prefix_tokens[i],
                generated_token_ids=generated_token_ids,
                is_finished=is_finished_list[idx],
                reward=rewards["reward"],
                reward_info=rewards["reward_info"],
            )
            episodes.append(episode)
    # clear the output line
    print("\r", end=" " * 100, flush=True)
    return episodes


@torch.inference_mode()
def fork_rollout(
    model: Transformer,
    batch: MiniBatch,
    tokenizer: Tokenizer,
    max_gen_len: int,
    num_answer_per_question: int,
    reward_function: Callable,
    device: torch.device,
    dtype: torch.dtype,
    fork_reward_config: Optional[ForkRewardConfig] = None,
    fork_token_logit_bias: float = 0.0,
    fork_token_target_prob: Optional[float] = None,
    warmup_fork_fraction: float = 1.0,
) -> List[Episode]:
    """Fork-aware rollout with branch racing and early stop.

    Each row can fork at most once, creating a branch B copy.
    Pre-allocates 2*bsz KV rows. Branch B rows live at index bsz+i.

    warmup_fork_fraction: fraction of rows (per question group) that get
    the fork probability injection during warmup. Set < 1.0 to keep some
    non-forked baselines so GRPO can learn the value of forking.
    """
    if fork_reward_config is None:
        fork_reward_config = ForkRewardConfig()

    end_token = tokenizer.eos_token
    end_token_id = tokenizer.eos_token_id
    pad_token_id = tokenizer.pad_token_id
    fork_token_id = tokenizer.fork_token_id
    fork1_token_id = tokenizer.fork1_token_id
    fork2_token_id = tokenizer.fork2_token_id
    fork_event_token_ids = {fork_token_id, fork1_token_id, fork2_token_id}

    prefix_token_ids = batch.prefix_token_ids
    bsz = len(batch.prefix) * num_answer_per_question
    min_prompt_len = min(len(t) for t in prefix_token_ids)
    max_prompt_len = max(len(t) for t in prefix_token_ids)
    total_len = max_gen_len + max_prompt_len

    # Pre-allocate for worst case: each row can fork into 2
    max_rows = 2 * bsz
    model.init_kv_cache(
        max_batch_size=max_rows,
        max_seq_len=total_len,
        device=device,
        dtype=dtype,
    )

    tokens = torch.full((max_rows, total_len), pad_token_id, dtype=torch.long, device=device)
    for k, t in enumerate(prefix_token_ids):
        offset = k * num_answer_per_question
        for i in range(num_answer_per_question):
            tokens[offset + i, : len(t)] = torch.tensor(
                t, dtype=torch.long, device=device
            )

    prev_pos = 0
    input_text_mask = tokens != pad_token_id
    assert min_prompt_len < total_len

    # Track active rows: first bsz are branch A, bsz..2*bsz are branch B
    is_active = torch.zeros((max_rows,), dtype=torch.bool, device=device)
    is_active[:bsz] = True
    is_finished = torch.zeros((max_rows,), dtype=torch.bool, device=device)
    has_forked = torch.zeros((bsz,), dtype=torch.bool, device=device)

    # Track which question each row belongs to
    question_idx = [i // num_answer_per_question for i in range(bsz)] + \
                   [i // num_answer_per_question for i in range(bsz)]

    def _init_parser_for_prompt(prompt_text: str) -> ForkParser:
        """Initialize parser state from prompt suffix only.

        Prompt text may include example tags in instructions. We only care
        whether the model starts generation inside an unclosed <think> block.
        """
        parser = ForkParser()
        think_open = prompt_text.rfind("<think>")
        think_close = prompt_text.rfind("</think>")
        if think_open != -1 and think_open > think_close:
            parser.feed("<think>")
        return parser

    # Per-row parser state (tracked on CPU). Branch A rows are initialized
    # from prompt state so valid-fork detection is prompt-aware.
    parsers = [ForkParser() for _ in range(max_rows)]
    for row in range(bsz):
        parsers[row] = _init_parser_for_prompt(batch.prefix[question_idx[row]])

    # Per-row fork metadata
    fork_step = [None] * max_rows  # step at which fork occurred
    invalid_fork_flags = [False] * max_rows
    # Per-base-row race status (branch A row i and optional branch B row bsz+i)
    race_solved = [False] * bsz
    # Latency proxy accumulated online until first-correct answer.
    t_proxy_ms = [0.0] * bsz

    # Select which rows get warmup fork injection (per-question balanced)
    warmup_eligible = [False] * bsz
    if fork_token_target_prob is not None and warmup_fork_fraction < 1.0:
        import random as _rand
        for q in range(len(batch.prefix)):
            q_rows = [r for r in range(bsz) if question_idx[r] == q]
            n_inject = max(1, round(len(q_rows) * warmup_fork_fraction))
            chosen = _rand.sample(q_rows, n_inject)
            for r in chosen:
                warmup_eligible[r] = True
    else:
        warmup_eligible = [True] * bsz

    for cur_pos in range(min_prompt_len, total_len):
        active_count = is_active.sum().item()
        if active_count == 0:
            break
        print(
            f"\r* Fork rollout: {cur_pos-min_prompt_len:>4d}/{total_len-min_prompt_len:>4d} "
            f"active={active_count}",
            flush=True,
            end="",
        )

        # Build indices of active, non-finished rows
        active_mask = is_active & ~is_finished
        if not active_mask.any():
            break

        # Accumulate wall-clock latency proxy per base row up to first-correct.
        for base_row in range(bsz):
            if race_solved[base_row]:
                continue
            active_branches = int(active_mask[base_row].item())
            active_branches += int(active_mask[base_row + bsz].item())
            if active_branches == 1:
                t_proxy_ms[base_row] += fork_reward_config.c1_ms
            elif active_branches >= 2:
                t_proxy_ms[base_row] += fork_reward_config.c2_ms

        with torch.autocast(device_type=device.type, dtype=dtype):
            logits = model.inference(tokens[:max_rows, prev_pos:cur_pos], prev_pos)

        # Optional warmup signal: increase <fork> probability where a fork is valid.
        # Uses direct probability injection to avoid bfloat16 precision issues
        # with the logit adjustment approach (fork token logits are extremely
        # negative for unused embeddings, and eps clamping makes delta insufficient).
        probs = torch.softmax(logits[:, -1], dim=-1)

        if fork_token_logit_bias != 0.0 and fork_token_target_prob is None:
            # Legacy logit bias path (not used during warmup)
            for row in range(bsz):
                if not is_active[row].item() or is_finished[row].item():
                    continue
                if has_forked[row].item():
                    continue
                if parsers[row].check_fork_valid():
                    logits[row, -1, fork_token_id] += fork_token_logit_bias
            probs = torch.softmax(logits[:, -1], dim=-1)

        if fork_token_target_prob is not None:
            target_p = float(min(max(fork_token_target_prob, 1e-6), 1.0 - 1e-6))
            for row in range(bsz):
                if not warmup_eligible[row]:
                    continue
                if not is_active[row].item() or is_finished[row].item():
                    continue
                if has_forked[row].item():
                    continue
                if input_text_mask[row, cur_pos].item():
                    continue  # still in prompt, skip
                if not parsers[row].check_fork_valid():
                    continue
                # Zero out fork prob, rescale others to (1-target_p), set fork to target_p
                row_probs = probs[row]
                fork_p = row_probs[fork_token_id].item()
                other_sum = 1.0 - fork_p
                if other_sum > 0:
                    scale = (1.0 - target_p) / other_sum
                    probs[row] = row_probs * scale
                probs[row, fork_token_id] = target_p

        next_token = torch.multinomial(probs, num_samples=1).reshape(-1)

        # For prompt positions, use the original token
        for row in range(max_rows):
            if not is_active[row] or is_finished[row]:
                continue
            if cur_pos < len(prefix_token_ids[question_idx[row]]) if row < bsz else False:
                if input_text_mask[row, cur_pos]:
                    next_token[row] = tokens[row, cur_pos]

        next_token = torch.where(
            input_text_mask[:max_rows, cur_pos],
            tokens[:max_rows, cur_pos],
            next_token,
        )
        next_token = torch.where(is_finished[:max_rows], pad_token_id, next_token)
        next_token = torch.where(~is_active[:max_rows], pad_token_id, next_token)
        tokens[:max_rows, cur_pos] = next_token

        # Process each active row for fork/answer detection
        gen_step = cur_pos - min_prompt_len
        for row in range(max_rows):
            if not is_active[row].item() or is_finished[row].item():
                continue
            base_row = row if row < bsz else row - bsz
            if race_solved[base_row]:
                is_finished[row] = True
                continue
            if input_text_mask[row, cur_pos].item():
                continue  # still in prompt

            tok_id = next_token[row].item()
            tok_text = tokenizer.detokenize([tok_id])

            # Treat <fork>, <fork1>, and <fork2> as fork events.
            if tok_id in fork_event_token_ids:
                base_row = row if row < bsz else row - bsz
                parser = parsers[row]
                is_valid = parser.register_fork()

                if is_valid and not has_forked[base_row]:
                    # Valid fork: clone to branch B
                    branch_b_row = base_row + bsz
                    has_forked[base_row] = True
                    fork_step[row] = gen_step
                    fork_step[branch_b_row] = gen_step

                    # Replace <fork> with <fork1> in branch A
                    tokens[row, cur_pos] = fork1_token_id
                    # Set <fork2> in branch B
                    tokens[branch_b_row, cur_pos] = fork2_token_id

                    # Clone KV cache
                    for layer in model.layers:
                        attn = layer.self_attn
                        attn.cache_k[branch_b_row, :cur_pos+1] = attn.cache_k[row, :cur_pos+1]
                        attn.cache_v[branch_b_row, :cur_pos+1] = attn.cache_v[row, :cur_pos+1]

                    # Copy tokens up to current position
                    tokens[branch_b_row, :cur_pos] = tokens[row, :cur_pos]
                    input_text_mask[branch_b_row, :cur_pos+1] = input_text_mask[row, :cur_pos+1]

                    # Activate branch B
                    is_active[branch_b_row] = True
                    # Clone parser
                    parsers[branch_b_row] = parsers[row].clone()
                else:
                    # Invalid fork
                    invalid_fork_flags[row] = True
                    parsers[row].invalid_fork = True
            else:
                # Feed token to parser
                parsers[row].feed(tok_text)

            # Check for end token
            if tok_id == end_token_id:
                is_finished[row] = True

            # Check for complete answer (race logic)
            if parsers[row].has_complete_answer:
                verify = reward_function(
                    response=parsers[row].text,
                    numbers=batch.numbers[question_idx[row]],
                    target=batch.target[question_idx[row]],
                    end_token=end_token,
                )
                answer_correct = verify["reward_info"].get("answer_reward", 0.0) > 0.5
                if answer_correct:
                    race_solved[base_row] = True
                    # First correct answer wins; terminate the whole race pair.
                    is_finished[base_row] = True
                    branch_b_row = base_row + bsz
                    if has_forked[base_row]:
                        is_finished[branch_b_row] = True
                else:
                    # Incorrect complete answer only terminates this branch.
                    is_finished[row] = True

        prev_pos = cur_pos

        # Check if all active rows are finished
        if (is_finished[is_active]).all():
            break

    model.del_kv_cache()
    gc.collect()
    torch.cuda.empty_cache()

    tokens_list = tokens.tolist()
    is_finished_list = is_finished.tolist()

    # Build episodes with fork-aware reward
    episodes = []
    for i in range(bsz):
        q_idx = question_idx[i]
        prompt_len = len(batch.prefix_token_ids[q_idx])

        # Branch A (row i)
        branch_a_gen = tokens_list[i][prompt_len:]
        if pad_token_id in branch_a_gen:
            branch_a_gen = branch_a_gen[:branch_a_gen.index(pad_token_id)]
        branch_a_text = tokenizer.detokenize(branch_a_gen)

        # Branch B (row bsz+i) if forked
        branch_b_row = bsz + i
        forked = has_forked[i].item()
        branch_b_gen = []
        branch_b_text = ""
        if forked and is_active[branch_b_row]:
            branch_b_gen = tokens_list[branch_b_row][prompt_len:]
            if pad_token_id in branch_b_gen:
                branch_b_gen = branch_b_gen[:branch_b_gen.index(pad_token_id)]
            branch_b_text = tokenizer.detokenize(branch_b_gen)

        # Compute rewards using the countdown task reward function
        reward_a = reward_function(
            response=branch_a_text,
            numbers=batch.numbers[q_idx],
            target=batch.target[q_idx],
            end_token=end_token,
        )
        correct_a = reward_a["reward_info"]["answer_reward"] > 0.5

        correct_b = False
        reward_b = None
        if forked and branch_b_text:
            reward_b = reward_function(
                response=branch_b_text,
                numbers=batch.numbers[q_idx],
                target=batch.target[q_idx],
                end_token=end_token,
            )
            correct_b = reward_b["reward_info"]["answer_reward"] > 0.5

        # Determine winner and compute fork reward
        any_correct = race_solved[i]
        t_proxy = t_proxy_ms[i]
        fs = fork_step[i]

        fork_rwd = compute_fork_reward(
            correct=any_correct,
            t_proxy_ms=t_proxy,
            invalid_fork=invalid_fork_flags[i] or invalid_fork_flags[branch_b_row],
            config=fork_reward_config,
        )

        # Build episode for branch A
        tw_a = compute_token_weights(
            total_tokens=len(branch_a_gen),
            fork_position=fs if forked else None,
            branch_id=0 if forked else None,
            active_branches=2 if forked else 1,
        )

        episode_a = Episode(
            prefix=batch.prefix[q_idx],
            text=batch.prefix[q_idx] + branch_a_text,
            prefix_token_ids=batch.prefix_token_ids[q_idx],
            prefix_tokens=batch.prefix_tokens[q_idx],
            generated_token_ids=branch_a_gen,
            is_finished=is_finished_list[i],
            reward=fork_rwd["reward"],
            reward_info={
                **fork_rwd["reward_info"],
                "answer_reward": float(correct_a),
                "format_reward": reward_a["reward_info"]["format_reward"],
                "t_proxy_ms": t_proxy,
                "forked": float(forked),
            },
            token_weights=tw_a,
            fork_info=fork_rwd["reward_info"],
            branch_id=0 if forked else None,
        )
        episodes.append(episode_a)

        # Build episode for branch B if forked
        if forked and branch_b_gen:
            tw_b = compute_token_weights(
                total_tokens=len(branch_b_gen),
                fork_position=fs,
                branch_id=1,
                active_branches=2,
            )
            episode_b = Episode(
                prefix=batch.prefix[q_idx],
                text=batch.prefix[q_idx] + branch_b_text,
                prefix_token_ids=batch.prefix_token_ids[q_idx],
                prefix_tokens=batch.prefix_tokens[q_idx],
                generated_token_ids=branch_b_gen,
                is_finished=is_finished_list[branch_b_row],
                reward=fork_rwd["reward"],  # same reward for both branches
                reward_info={
                    **fork_rwd["reward_info"],
                    "answer_reward": float(correct_b),
                    "format_reward": reward_b["reward_info"]["format_reward"] if reward_b else 0.0,
                    "t_proxy_ms": t_proxy,
                    "forked": float(forked),
                },
                token_weights=tw_b,
                fork_info=fork_rwd["reward_info"],
                branch_id=1,
            )
            episodes.append(episode_b)

    print("\r", end=" " * 100, flush=True)
    return episodes


def normalize_rewards_per_group(episodes: List[Episode]) -> List[Episode]:
    """Normalize rewards per group. A group is defined by the prefix."""
    groups = defaultdict(list)
    for episode in episodes:
        groups[tuple(episode.prefix)].append(episode)
    output = []
    for group in groups.values():
        group_rewards = [item.reward for item in group]
        mean_reward = np.mean(group_rewards)
        std_reward = np.std(group_rewards)
        for episode in group:
            normalized_reward = (episode.reward - mean_reward) / (std_reward + 1e-4)
            episode = dataclasses.replace(episode, reward=normalized_reward)
            output.append(episode)
    return output


def compute_entropy(logits: torch.Tensor) -> torch.Tensor:
    probs = torch.nn.functional.softmax(logits, dim=-1)
    entropy = torch.logsumexp(logits, dim=-1) - torch.sum(probs * logits, dim=-1)
    return entropy


def update_policy(
    model,
    optimizer,
    episodes: List[Episode],
    micro_batch_size: int,
    pad_token_id: int,
    max_grad_norm: float,
    device: torch.device,
    dtype: torch.dtype,
):
    """Update the policy using the GRPO algorithm."""
    episodes = normalize_rewards_per_group(episodes)
    # sort episodes by token length for efficient (micro-)batching
    episodes.sort(key=lambda x: len(x.prefix_token_ids) + len(x.generated_token_ids))
    num_micro_batches = math.ceil(len(episodes) / micro_batch_size)
    num_target_tokens = sum(len(episode.generated_token_ids) for episode in episodes)
    entropy = 0.0

    for i in range(0, len(episodes), micro_batch_size):
        print(
            f"\r* Computing policy gradient: {i:>2d}/{len(episodes):>2d}",
            flush=True,
            end="",
        )
        j = min(i + micro_batch_size, len(episodes))
        batch_episodes = episodes[i:j]
        batch_lengths = [
            len(episode.prefix_token_ids) + len(episode.generated_token_ids)
            for episode in batch_episodes
        ]
        batch_max_length = max(batch_lengths)
        batch_token_ids = [
            episode.prefix_token_ids
            + episode.generated_token_ids
            + [pad_token_id] * (batch_max_length - batch_lengths[i])
            for i, episode in enumerate(batch_episodes)
        ]
        batch_masks = [
            [0] * len(episode.prefix_token_ids)
            + [1] * len(episode.generated_token_ids)
            + [0] * (batch_max_length - batch_lengths[i])
            for i, episode in enumerate(batch_episodes)
        ]
        batch_advantages = [episode.reward for episode in batch_episodes]
        batch_token_ids = torch.tensor(batch_token_ids, device=device, dtype=torch.long)
        batch_masks = torch.tensor(batch_masks, device=device, dtype=torch.bool)
        batch_advantages = torch.tensor(
            batch_advantages, device=device, dtype=torch.float32
        )

        with torch.autocast(device_type=device.type, dtype=dtype):
            input_token_ids = batch_token_ids[:, :-1]
            target_token_ids = batch_token_ids[:, 1:]
            target_masks = batch_masks[:, 1:]
            logits = model.forward(input_token_ids).float()

        log_probs = -torch.nn.functional.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            target_token_ids.reshape(-1),
            ignore_index=pad_token_id,
            reduction="none",
        ).reshape(input_token_ids.shape[0], -1)

        with torch.no_grad():
            token_entropy = compute_entropy(logits)
            entropy = entropy + (token_entropy * target_masks).sum() / num_target_tokens

        obj = log_probs * batch_advantages[:, None]
        # per-token objective
        obj = (obj * target_masks).sum() / num_target_tokens
        loss = -obj
        loss.backward()

    # update the policy
    grad_norm = torch.nn.utils.clip_grad_norm_(
        model.parameters(), max_norm=max_grad_norm
    )
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    return {
        "loss": loss.item(),
        "grad_norm": grad_norm.item(),
        "entropy": entropy.item(),
    }


def update_policy_fork(
    model,
    optimizer,
    episodes: List[Episode],
    micro_batch_size: int,
    pad_token_id: int,
    max_grad_norm: float,
    device: torch.device,
    dtype: torch.dtype,
    entropy_coef: float = 0.0,
):
    """Update policy with token-weighted GRPO for fork-race episodes.

    Key difference from update_policy: uses per-token weights from
    episode.token_weights to properly weight pre-fork vs post-fork tokens.
    Pre-fork tokens get weight 1.0, post-fork get weight 1/active_branches.
    Normalizes by sum of token weights instead of raw token count.
    """
    episodes = normalize_rewards_per_group(episodes)
    episodes.sort(key=lambda x: len(x.prefix_token_ids) + len(x.generated_token_ids))
    entropy = 0.0

    # Compute total token weight for normalization
    total_token_weight = 0.0
    for episode in episodes:
        if episode.token_weights is not None:
            total_token_weight += sum(episode.token_weights)
        else:
            total_token_weight += len(episode.generated_token_ids)

    if total_token_weight == 0:
        total_token_weight = 1.0  # avoid div by zero

    for i in range(0, len(episodes), micro_batch_size):
        print(
            f"\r* Computing fork-weighted policy gradient: {i:>2d}/{len(episodes):>2d}",
            flush=True,
            end="",
        )
        j = min(i + micro_batch_size, len(episodes))
        batch_episodes = episodes[i:j]
        batch_lengths = [
            len(episode.prefix_token_ids) + len(episode.generated_token_ids)
            for episode in batch_episodes
        ]
        batch_max_length = max(batch_lengths)

        batch_token_ids = [
            episode.prefix_token_ids
            + episode.generated_token_ids
            + [pad_token_id] * (batch_max_length - batch_lengths[k])
            for k, episode in enumerate(batch_episodes)
        ]

        # Build float token weight masks instead of binary masks
        batch_weight_masks = []
        for k, episode in enumerate(batch_episodes):
            prefix_weights = [0.0] * len(episode.prefix_token_ids)
            if episode.token_weights is not None:
                gen_weights = list(episode.token_weights)
            else:
                gen_weights = [1.0] * len(episode.generated_token_ids)
            pad_weights = [0.0] * (batch_max_length - batch_lengths[k])
            batch_weight_masks.append(prefix_weights + gen_weights + pad_weights)

        batch_advantages = [episode.reward for episode in batch_episodes]
        batch_token_ids = torch.tensor(batch_token_ids, device=device, dtype=torch.long)
        batch_weight_masks = torch.tensor(
            batch_weight_masks, device=device, dtype=torch.float32
        )
        batch_advantages = torch.tensor(
            batch_advantages, device=device, dtype=torch.float32
        )

        with torch.autocast(device_type=device.type, dtype=dtype):
            input_token_ids = batch_token_ids[:, :-1]
            target_token_ids = batch_token_ids[:, 1:]
            target_weights = batch_weight_masks[:, 1:]
            logits = model.forward(input_token_ids).float()

        log_probs = -torch.nn.functional.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            target_token_ids.reshape(-1),
            ignore_index=pad_token_id,
            reduction="none",
        ).reshape(input_token_ids.shape[0], -1)

        with torch.no_grad():
            token_entropy = compute_entropy(logits)
            target_binary_mask = (target_weights > 0).float()
            entropy = entropy + (token_entropy * target_binary_mask).sum() / total_token_weight

        # Weighted objective: log_prob * advantage * token_weight
        obj = log_probs * batch_advantages[:, None] * target_weights
        obj = obj.sum() / total_token_weight
        # Entropy bonus to prevent mode collapse
        if entropy_coef > 0:
            entropy_bonus = (token_entropy * target_binary_mask).sum() / total_token_weight
            obj = obj + entropy_coef * entropy_bonus
        loss = -obj
        loss.backward()

    grad_norm = torch.nn.utils.clip_grad_norm_(
        model.parameters(), max_norm=max_grad_norm
    )
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    return {
        "loss": loss.item(),
        "grad_norm": grad_norm.item(),
        "entropy": entropy.item(),
    }
