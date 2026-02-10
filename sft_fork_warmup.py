"""SFT Fork Warmup: Teach the model to emit <fork> tokens via supervised learning.

This script:
1. Loads a vanilla checkpoint (trained on countdown task without fork)
2. Generates rollouts WITH forced forks (probability injection)
3. Uses the generated sequences as supervised training data
4. Fine-tunes with cross-entropy loss so the model learns P(fork | context) > 0

After this, the model can be fine-tuned with GRPO to learn WHEN to fork optimally.
"""

import gc
import random
import time
from argparse import ArgumentParser
from datetime import datetime
from pathlib import Path
from typing import List

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader
from torch.utils.tensorboard.writer import SummaryWriter

from countdown_task import CountdownTasksDataset, reward_function
from data_types import Episode
from fork_metrics import episode_correct, primary_episodes
from fork_reward import ForkRewardConfig
from grpo import fork_rollout
from optimizer import MemoryEfficientAdamW
from qwen2_model import Transformer
from tokenizer import Tokenizer


def generate_fork_data(
    model: Transformer,
    tokenizer: Tokenizer,
    dataset: CountdownTasksDataset,
    device: torch.device,
    dtype: torch.dtype,
    num_batches: int = 10,
    batch_size: int = 8,
    num_answers: int = 4,
    max_gen_len: int = 512,
    fork_target_prob: float = 0.25,
) -> List[Episode]:
    """Generate rollouts with forced forks to create SFT training data."""
    generator = torch.Generator(device=device)
    dataloader = DataLoader(
        dataset,
        shuffle=True,
        collate_fn=CountdownTasksDataset.collate_fn,
        generator=generator,
        batch_size=batch_size,
    )

    frc = ForkRewardConfig()
    all_episodes = []

    for batch_idx, batch in enumerate(dataloader, start=1):
        if batch_idx > num_batches:
            break
        episodes = fork_rollout(
            model=model,
            tokenizer=tokenizer,
            batch=batch,
            max_gen_len=max_gen_len,
            num_answer_per_question=num_answers,
            reward_function=reward_function,
            device=device,
            dtype=dtype,
            fork_reward_config=frc,
            fork_token_logit_bias=0.0,
            fork_token_target_prob=fork_target_prob,
            warmup_fork_fraction=1.0,  # force ALL rows to get fork injection
        )
        # Keep only forked episodes (branch A) - these contain the fork token
        forked = [e for e in episodes if e.branch_id == 0 and
                  e.reward_info.get("forked", 0) > 0.5]
        all_episodes.extend(forked)
        print(f"\rBatch {batch_idx}/{num_batches}: {len(forked)} forked episodes "
              f"(total: {len(all_episodes)})", flush=True)

    print(f"\nGenerated {len(all_episodes)} forked episodes for SFT")
    return all_episodes


def sft_step(
    model: Transformer,
    optimizer: torch.optim.Optimizer,
    episodes: List[Episode],
    micro_batch_size: int,
    pad_token_id: int,
    max_grad_norm: float,
    device: torch.device,
    dtype: torch.dtype,
    max_seq_len: int = 512,
    episodes_per_step: int = 64,
    fork_token_ids: set = None,
    fork_weight: float = 100.0,
) -> dict:
    """One step of supervised fine-tuning on fork episodes.

    Cross-entropy loss on the FULL generated sequence, with fork token
    positions weighted fork_weight times higher to focus learning on
    the rare fork token emission.
    """
    # Sample a subset and shuffle
    if len(episodes) > episodes_per_step:
        step_episodes = random.sample(episodes, episodes_per_step)
    else:
        step_episodes = list(episodes)
    random.shuffle(step_episodes)
    step_episodes.sort(key=lambda x: len(x.prefix_token_ids) + len(x.generated_token_ids))

    if not step_episodes:
        return {"loss": 0.0, "grad_norm": 0.0}

    total_loss = 0.0
    total_weight = 0.0
    for i in range(0, len(step_episodes), micro_batch_size):
        j = min(i + micro_batch_size, len(step_episodes))
        batch_eps = step_episodes[i:j]

        # Truncate sequences to max_seq_len
        truncated = []
        for e in batch_eps:
            prefix = e.prefix_token_ids
            gen = e.generated_token_ids
            avail = max_seq_len - len(prefix)
            if avail <= 0:
                continue
            truncated.append((prefix, gen[:avail]))
        if not truncated:
            continue

        lengths = [len(p) + len(g) for p, g in truncated]
        max_len = max(lengths)

        token_ids = [
            list(p) + list(g) + [pad_token_id] * (max_len - lengths[k])
            for k, (p, g) in enumerate(truncated)
        ]
        # Weight mask: 1.0 for normal generated tokens, fork_weight for fork targets
        weight_masks = []
        for k, (p, g) in enumerate(truncated):
            prefix_w = [0.0] * len(p)
            gen_w = []
            for t in g:
                gen_w.append(fork_weight if (fork_token_ids and t in fork_token_ids) else 1.0)
            pad_w = [0.0] * (max_len - lengths[k])
            weight_masks.append(prefix_w + gen_w + pad_w)

        token_ids_t = torch.tensor(token_ids, device=device, dtype=torch.long)
        weight_masks_t = torch.tensor(weight_masks, device=device, dtype=torch.float32)

        with torch.autocast(device_type=device.type, dtype=dtype):
            input_ids = token_ids_t[:, :-1]
            target_ids = token_ids_t[:, 1:]
            target_weights = weight_masks_t[:, 1:]
            logits = model.forward(input_ids).float()

        loss_per_token = torch.nn.functional.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            target_ids.reshape(-1),
            ignore_index=pad_token_id,
            reduction="none",
        ).reshape(input_ids.shape[0], -1)

        batch_weight = target_weights.sum()
        total_weight += batch_weight.item()
        masked_loss = (loss_per_token * target_weights).sum()
        masked_loss.backward()
        total_loss += masked_loss.item()

    grad_norm = torch.nn.utils.clip_grad_norm_(
        model.parameters(), max_norm=max_grad_norm
    )
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    norm_loss = total_loss / max(total_weight, 1.0)
    return {"loss": norm_loss, "grad_norm": grad_norm.item()}


def check_fork_probability(
    model: Transformer,
    tokenizer: Tokenizer,
    dataset: CountdownTasksDataset,
    device: torch.device,
    dtype: torch.dtype,
    num_samples: int = 16,
) -> float:
    """Check the model's natural probability of emitting <fork> at valid positions."""
    generator = torch.Generator(device=device)
    dataloader = DataLoader(
        dataset,
        shuffle=True,
        collate_fn=CountdownTasksDataset.collate_fn,
        generator=generator,
        batch_size=min(4, num_samples),
    )
    batch = next(iter(dataloader))
    fork_id = tokenizer.fork_token_id

    bsz = len(batch.prefix) * 2  # 2 answers per question for speed
    prefix_token_ids = batch.prefix_token_ids
    min_prompt_len = min(len(t) for t in prefix_token_ids)
    max_prompt_len = max(len(t) for t in prefix_token_ids)
    total_len = 20 + max_prompt_len  # just a few generation steps

    model.init_kv_cache(
        max_batch_size=bsz, max_seq_len=total_len, device=device, dtype=dtype
    )
    tokens = torch.full(
        (bsz, total_len), tokenizer.pad_token_id, dtype=torch.long, device=device
    )
    for k, t in enumerate(prefix_token_ids):
        offset = k * 2
        for i in range(2):
            tokens[offset + i, : len(t)] = torch.tensor(t, dtype=torch.long, device=device)

    prev_pos = 0
    fork_probs = []

    for cur_pos in range(min_prompt_len, min(min_prompt_len + 10, total_len)):
        with torch.autocast(device_type=device.type, dtype=dtype):
            logits = model.inference(tokens[:bsz, prev_pos:cur_pos], prev_pos)
        probs = torch.softmax(logits[:, -1], dim=-1)

        for row in range(bsz):
            p_fork = probs[row, fork_id].item()
            fork_probs.append(p_fork)

        # Sample next token normally
        next_token = torch.multinomial(probs, num_samples=1).reshape(-1)
        tokens[:bsz, cur_pos] = next_token
        prev_pos = cur_pos

    model.del_kv_cache()
    gc.collect()
    torch.cuda.empty_cache()

    mean_p = np.mean(fork_probs)
    max_p = np.max(fork_probs)
    return mean_p, max_p


def main(config_path: str):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    device = torch.device(config["model"]["device"])
    dtype_map = {
        "bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32
    }
    dtype = dtype_map.get(config["model"]["dtype"], torch.bfloat16)
    torch.set_default_device(device)
    torch.random.manual_seed(config["training"]["random_seed"])
    random.seed(config["training"]["random_seed"])

    pretrained_model_path = Path(config["model"]["pretrained_model_path"])
    tokenizer = Tokenizer(str(pretrained_model_path / "tokenizer.json"))
    tokenizer.add_fork_tokens()

    sft_config = config.get("sft_warmup", {})
    vanilla_ckpt = sft_config.get("vanilla_checkpoint", None)
    num_data_batches = sft_config.get("num_data_batches", 20)
    sft_steps = sft_config.get("sft_steps", 50)
    sft_lr = float(sft_config.get("learning_rate", 5e-6))
    fork_target_prob = float(sft_config.get("fork_target_prob", 0.25))
    data_answers_per_q = sft_config.get("data_answers_per_question", 4)
    data_batch_size = sft_config.get("data_batch_size", 8)
    micro_batch_size = config["training"].get("micro_batch_size", 2)
    max_gen_len = config["training"].get("max_gen_len", 512)
    max_grad_norm = config["training"].get("max_grad_norm", 1.0)
    output_ckpt = sft_config.get("output_checkpoint", "ckpt_sft_fork.pt")
    log_dir = sft_config.get("log_dir", "logs_sft_fork")

    print(f"=== SFT Fork Warmup ===")
    print(f"Vanilla checkpoint: {vanilla_ckpt}")
    print(f"SFT steps: {sft_steps}, LR: {sft_lr}")
    print(f"Data: {num_data_batches} batches x {data_batch_size} questions x {data_answers_per_q} answers")

    # Load model
    model = Transformer.from_pretrained(pretrained_model_path, device=device).train()
    if tokenizer.vocab_size > model.vocab_size:
        model.resize_embeddings(tokenizer.vocab_size)
        print(f"Model embeddings resized to {model.vocab_size}")
    else:
        print(f"Skipping resize: tokenizer vocab ({tokenizer.vocab_size}) <= model vocab ({model.vocab_size})")

    # Load vanilla checkpoint
    if vanilla_ckpt:
        ckpt_path = Path(vanilla_ckpt)
        if ckpt_path.exists():
            state_dict = torch.load(ckpt_path, map_location=device, weights_only=True)
            # Handle potential vocab size mismatch (vanilla didn't have fork tokens)
            model_sd = model.state_dict()
            for key in state_dict:
                if key in model_sd and state_dict[key].shape == model_sd[key].shape:
                    model_sd[key] = state_dict[key]
                elif key in model_sd:
                    # Size mismatch - likely embedding/output layer. Copy what fits.
                    src = state_dict[key]
                    dst = model_sd[key]
                    min_size = [min(s, d) for s, d in zip(src.shape, dst.shape)]
                    slices = tuple(slice(0, m) for m in min_size)
                    dst[slices] = src[slices]
                    model_sd[key] = dst
                    print(f"  Partial load for {key}: {src.shape} -> {dst.shape}")
            model.load_state_dict(model_sd)
            print(f"Loaded vanilla checkpoint from {ckpt_path}")
        else:
            print(f"WARNING: Checkpoint {ckpt_path} not found, using base model")

    dataset = CountdownTasksDataset(
        data_path=config["data"]["path"],
        tokenizer=tokenizer,
        split="train",
        test_size=config["data"]["test_size"],
    )

    # Check initial fork probability
    model.eval()
    mean_p, max_p = check_fork_probability(model, tokenizer, dataset, device, dtype)
    print(f"\nInitial P(fork): mean={mean_p:.2e}, max={max_p:.2e}")

    # Generate SFT data with forced forks
    print(f"\n=== Generating fork training data ===")
    model.eval()
    fork_episodes = generate_fork_data(
        model=model,
        tokenizer=tokenizer,
        dataset=dataset,
        device=device,
        dtype=dtype,
        num_batches=num_data_batches,
        batch_size=data_batch_size,
        num_answers=data_answers_per_q,
        max_gen_len=max_gen_len,
        fork_target_prob=fork_target_prob,
    )

    if not fork_episodes:
        print("ERROR: No fork episodes generated! Check fork_rollout.")
        return

    # Check fork token presence in episodes
    fork_id = tokenizer.fork_token_id
    fork1_id = tokenizer.fork1_token_id
    fork2_id = tokenizer.fork2_token_id
    fork_token_ids_set = {fork_id, fork1_id, fork2_id}
    has_fork = sum(
        1
        for e in fork_episodes
        if any(token_id in fork_token_ids_set for token_id in e.generated_token_ids)
    )
    print(f"Episodes with fork token: {has_fork}/{len(fork_episodes)}")

    # SFT training with fork-weighted loss
    fork_weight = float(sft_config.get("fork_weight", 100.0))
    print(f"\n=== SFT Training ({sft_steps} steps, fork_weight={fork_weight}) ===")
    model.train()
    optimizer = MemoryEfficientAdamW(
        model.parameters(),
        lr=sft_lr,
        weight_decay=0.0,
        betas=(0.9, 0.999),
        enabled=True,
    )

    current_time = datetime.now().strftime(r"%Y%m%d-%H%M%S")
    tb_writer = SummaryWriter(log_dir=f"{log_dir}/{current_time}")

    for step in range(1, sft_steps + 1):
        results = sft_step(
            model=model,
            optimizer=optimizer,
            episodes=fork_episodes,
            micro_batch_size=micro_batch_size,
            pad_token_id=tokenizer.pad_token_id,
            max_grad_norm=max_grad_norm,
            device=device,
            dtype=dtype,
            fork_token_ids=fork_token_ids_set,
            fork_weight=fork_weight,
        )

        tb_writer.add_scalar("sft_loss", results["loss"], step)
        tb_writer.add_scalar("sft_grad_norm", results["grad_norm"], step)

        if step % 10 == 0 or step == 1:
            # Check fork probability
            model.eval()
            mean_p, max_p = check_fork_probability(
                model, tokenizer, dataset, device, dtype
            )
            model.train()
            tb_writer.add_scalar("fork_prob_mean", mean_p, step)
            tb_writer.add_scalar("fork_prob_max", max_p, step)
            print(f"Step {step}: loss={results['loss']:.4f}, grad_norm={results['grad_norm']:.4f}, "
                  f"P(fork) mean={mean_p:.2e}, max={max_p:.2e}")

        # Regenerate data every 20 steps to avoid overfitting
        if step % 20 == 0 and step < sft_steps:
            print(f"\nRegenerating fork data at step {step}...")
            model.eval()
            fork_episodes = generate_fork_data(
                model=model,
                tokenizer=tokenizer,
                dataset=dataset,
                device=device,
                dtype=dtype,
                num_batches=num_data_batches,
                batch_size=data_batch_size,
                num_answers=data_answers_per_q,
                max_gen_len=max_gen_len,
                fork_target_prob=fork_target_prob,
            )
            model.train()

    # Final check
    model.eval()
    mean_p, max_p = check_fork_probability(model, tokenizer, dataset, device, dtype)
    print(f"\nFinal P(fork): mean={mean_p:.2e}, max={max_p:.2e}")

    # Also check if the model still solves the task
    from grpo import fork_rollout as fr
    test_dataset = CountdownTasksDataset(
        data_path=config["data"]["path"],
        tokenizer=tokenizer,
        split="test",
        test_size=config["data"]["test_size"],
    )
    test_dl = DataLoader(
        test_dataset, shuffle=False,
        collate_fn=CountdownTasksDataset.collate_fn,
        batch_size=min(8, config["data"]["test_size"]),
    )
    test_batch = next(iter(test_dl))
    frc = ForkRewardConfig()
    test_episodes = fr(
        model=model, tokenizer=tokenizer, batch=test_batch,
        max_gen_len=max_gen_len, num_answer_per_question=1,
        reward_function=reward_function, device=device, dtype=dtype,
        fork_reward_config=frc,
        fork_token_logit_bias=0.0, fork_token_target_prob=None,
    )
    metric_episodes = primary_episodes(test_episodes)
    success = (
        np.mean([episode_correct(e) for e in metric_episodes])
        if metric_episodes
        else 0.0
    )
    forks = (
        np.mean([e.reward_info.get("forked", 0) for e in metric_episodes])
        if metric_episodes
        else 0.0
    )
    print(f"Post-SFT eval: success_rate={success:.4f}, natural_fork_rate={forks:.4f}")

    # Save checkpoint
    output_path = Path(output_ckpt)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), output_path)
    print(f"\nSaved SFT fork checkpoint to {output_path}")

    tb_writer.flush()
    tb_writer.close()


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--config", type=str, default="config_sft_fork.yaml")
    args = parser.parse_args()
    main(args.config)
