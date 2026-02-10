import html
import random
import time
from argparse import ArgumentParser
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader
from torch.utils.tensorboard.writer import SummaryWriter

from countdown_task import CountdownTasksDataset, reward_function
from fork_curriculum import apply_fork_presence_bonus, fork_presence_bonus_coef
from fork_metrics import episode_correct, primary_episodes
from fork_reward import ForkRewardConfig
from grpo import fork_rollout, rollout, update_policy, update_policy_fork
from optimizer import MemoryEfficientAdamW
from qwen2_model import Transformer
from tokenizer import Tokenizer


def evaluate(
    model,
    tokenizer,
    device,
    dtype,
    config,
    fork_enabled: bool = False,
    fork_reward_config: ForkRewardConfig | None = None,
):
    """Run held-out evaluation and return scalar metrics."""
    test_dataset = CountdownTasksDataset(
        data_path=config["data"]["path"],
        tokenizer=tokenizer,
        split="test",
        test_size=config["data"]["test_size"],
    )
    eval_cfg = config.get("evaluation", {})
    use_fork_rollout = eval_cfg.get("use_fork_rollout", fork_enabled)
    max_eval_batches = eval_cfg.get("max_eval_batches", None)

    generator = torch.Generator(device=device)
    # We reduce the batch size by half as we want to
    # generate twice as long trajectories.
    eval_batch_size = max(1, config["training"]["batch_size"] // 2)
    dataloader = DataLoader(
        test_dataset,
        shuffle=False,
        collate_fn=CountdownTasksDataset.collate_fn,
        generator=generator,
        batch_size=eval_batch_size,
        drop_last=False,
    )

    success = []
    ttfc_ms = []
    fork_rate = []
    t_proxy_ms = []
    for batch_idx, batch in enumerate(dataloader, start=1):
        t0 = time.perf_counter()
        if use_fork_rollout:
            episodes = fork_rollout(
                model=model,
                tokenizer=tokenizer,
                batch=batch,
                max_gen_len=config["training"]["max_gen_len"] * 2,
                num_answer_per_question=1,
                reward_function=reward_function,
                device=device,
                dtype=dtype,
                fork_reward_config=fork_reward_config,
                fork_token_logit_bias=0.0,
                fork_token_target_prob=None,
            )
        else:
            episodes = rollout(
                model=model,
                tokenizer=tokenizer,
                batch=batch,
                max_gen_len=config["training"]["max_gen_len"] * 2,
                num_answer_per_question=1,
                reward_function=reward_function,
                device=device,
                dtype=dtype,
            )
        batch_ms = (time.perf_counter() - t0) * 1000.0
        metric_episodes = primary_episodes(episodes) if use_fork_rollout else episodes

        if metric_episodes:
            # This is a rollout-level timing proxy for evaluation throughput.
            per_episode_ms = batch_ms / len(metric_episodes)
            ttfc_ms.extend([per_episode_ms] * len(metric_episodes))
            success.extend([episode_correct(episode) for episode in metric_episodes])

        if use_fork_rollout:
            fork_rate.extend(
                [episode.reward_info.get("forked", 0.0) for episode in metric_episodes]
            )
            t_proxy_ms.extend(
                [episode.reward_info.get("t_proxy_ms", 0.0) for episode in metric_episodes]
            )

        if max_eval_batches is not None and batch_idx >= max_eval_batches:
            break

    metrics = {
        "success_rate": float(np.mean(success)) if success else 0.0,
        "ttfc_ms_p50": float(np.percentile(ttfc_ms, 50)) if ttfc_ms else 0.0,
        "ttfc_ms_p90": float(np.percentile(ttfc_ms, 90)) if ttfc_ms else 0.0,
        "num_eval_episodes": len(success),
    }
    if use_fork_rollout:
        metrics["fork_rate"] = float(np.mean(fork_rate)) if fork_rate else 0.0
        metrics["t_proxy_ms"] = float(np.mean(t_proxy_ms)) if t_proxy_ms else 0.0
    return metrics


def main(config_path: str, max_steps_override: int | None = None):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    pretrained_model_path = Path(config["model"]["pretrained_model_path"])
    data_path = Path(config["data"]["path"])
    if not pretrained_model_path.exists():
        raise FileNotFoundError(
            f"Pretrained model path not found: {pretrained_model_path}. "
            "Clone/download the model first."
        )
    if not data_path.exists():
        raise FileNotFoundError(
            f"Dataset path not found: {data_path}. "
            "Clone/download the dataset first."
        )
    if not (data_path / "data").exists():
        raise FileNotFoundError(
            f"Dataset parquet missing: {(data_path / 'data')}. "
            "Expected Countdown parquet at <data_path>/data."
        )

    device = torch.device(config["model"]["device"])
    dtype_map = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }
    dtype = dtype_map.get(config["model"]["dtype"], torch.bfloat16)
    torch.set_default_device(device)
    torch.random.manual_seed(config["training"]["random_seed"])
    random.seed(config["training"]["random_seed"])

    BATCH_SIZE = config["training"]["batch_size"]
    NUM_QUESTIONS_PER_BATCH = config["training"]["num_questions_per_batch"]
    NUM_ANSWERS_PER_QUESTION = BATCH_SIZE // NUM_QUESTIONS_PER_BATCH
    if BATCH_SIZE % NUM_QUESTIONS_PER_BATCH != 0:
        raise ValueError("training.batch_size must be divisible by training.num_questions_per_batch")
    max_steps = max_steps_override
    if max_steps is None:
        max_steps = config["training"].get("max_steps", None)
    if max_steps is not None and max_steps <= 0:
        raise ValueError("max_steps must be > 0 if provided")
    text_log_episodes_per_step = int(
        config["training"].get("text_log_episodes_per_step", 4)
    )
    if text_log_episodes_per_step < 0:
        raise ValueError("training.text_log_episodes_per_step must be >= 0")

    current_time = datetime.now().strftime(r"%Y%m%d-%H%M%S")
    tb_writer = SummaryWriter(log_dir=f"{config['training']['log_dir']}/{current_time}")
    tokenizer = Tokenizer(str(pretrained_model_path / "tokenizer.json"))

    # Fork-race configuration
    fork_config = config.get("fork_race", {})
    fork_enabled = fork_config.get("enabled", False)

    if fork_enabled:
        new_vocab_size = tokenizer.add_fork_tokens()
        print(f"Fork-race enabled. Vocab size: {new_vocab_size}")

    train_dataset = CountdownTasksDataset(
        data_path=config["data"]["path"],
        tokenizer=tokenizer,
        split="train",
        test_size=config["data"]["test_size"],
    )
    generator = torch.Generator(device=device)
    train_dataloader = DataLoader(
        train_dataset,
        shuffle=True,
        collate_fn=CountdownTasksDataset.collate_fn,
        generator=generator,
        batch_size=NUM_QUESTIONS_PER_BATCH,
    )

    model = Transformer.from_pretrained(pretrained_model_path, device=device).train()

    if fork_enabled:
        if tokenizer.vocab_size > model.vocab_size:
            model.resize_embeddings(tokenizer.vocab_size)
            print(f"Model embeddings resized to {model.vocab_size}")
        else:
            print(
                f"Skipping resize: tokenizer vocab ({tokenizer.vocab_size}) "
                f"<= model vocab ({model.vocab_size})"
            )

    optimizer_name = str(config["training"].get("optimizer", "adamw")).lower()
    if optimizer_name == "sgd":
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=config["training"]["learning_rate"],
            momentum=float(config["training"].get("sgd_momentum", 0.0)),
            weight_decay=config["training"]["weight_decay"],
        )
    else:
        optimizer = MemoryEfficientAdamW(
            model.parameters(),
            lr=config["training"]["learning_rate"],
            weight_decay=config["training"]["weight_decay"],
            betas=config["training"]["betas"],
            enabled=config["training"]["memory_efficient_adamw"],
        )

    # Fork reward config
    fork_reward_config = None
    if fork_enabled:
        fork_reward_config = ForkRewardConfig(
            alpha=fork_config.get("alpha", 0.5),
            beta=fork_config.get("beta", 0.2),
            delta=fork_config.get("delta", 0.2),
            budget_ms=fork_config.get("budget_ms", 2500.0),
            c1_ms=fork_config.get("c1_ms", 1.0),
            c2_ms=fork_config.get("c2_ms", 1.5),
        )
        n_warmup = fork_config.get("n_warmup", 50)
        warmup_fork_target_prob = float(fork_config.get("warmup_fork_target_prob", 0.25))
        warmup_fork_presence_bonus_valid_only = bool(
            fork_config.get("warmup_fork_presence_bonus_valid_only", True)
        )

    start_time = time.time()
    ckpt_dir = Path(config["training"]["ckpt_dir"])
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    print(
        f"Training started. fork_enabled={fork_enabled}, "
        f"max_steps={max_steps if max_steps is not None else 'full_dataloader'}"
    )

    for step, batch in enumerate(train_dataloader, start=1):
        if max_steps is not None and step > max_steps:
            break

        if fork_enabled:
            # Warmup: set <fork> token probability target on valid fork states.
            current_fork_target_prob = warmup_fork_target_prob if step <= n_warmup else None
            warmup_fork_fraction = float(fork_config.get("warmup_fork_fraction", 0.5))
            episodes = fork_rollout(
                model=model,
                tokenizer=tokenizer,
                batch=batch,
                max_gen_len=config["training"]["max_gen_len"],
                num_answer_per_question=NUM_ANSWERS_PER_QUESTION,
                reward_function=reward_function,
                device=device,
                dtype=dtype,
                fork_reward_config=fork_reward_config,
                fork_token_logit_bias=0.0,
                fork_token_target_prob=current_fork_target_prob,
                warmup_fork_fraction=warmup_fork_fraction,
            )

            # Warmup curriculum: reward valid fork usage early, then decay away.
            current_fork_presence_bonus = fork_presence_bonus_coef(step, fork_config)
            episodes, fork_presence_bonus_rate = apply_fork_presence_bonus(
                episodes=episodes,
                bonus_coef=current_fork_presence_bonus,
                valid_only=warmup_fork_presence_bonus_valid_only,
            )
        else:
            episodes = rollout(
                model=model,
                tokenizer=tokenizer,
                batch=batch,
                max_gen_len=config["training"]["max_gen_len"],
                num_answer_per_question=NUM_ANSWERS_PER_QUESTION,
                reward_function=reward_function,
                device=device,
                dtype=dtype,
            )
            current_fork_presence_bonus = 0.0
            fork_presence_bonus_rate = 0.0

        if config["training"]["skip_unfinished_episodes"]:
            episodes = [episode for episode in episodes if episode.is_finished]
        if not episodes:
            print(f"\rStep {step}: no episodes after filtering, skipping update.")
            continue

        metric_episodes = primary_episodes(episodes) if fork_enabled else episodes
        if not metric_episodes:
            metric_episodes = episodes

        if fork_enabled:
            entropy_coef = fork_config.get("entropy_coef", 0.0)
            results = update_policy_fork(
                model=model,
                optimizer=optimizer,
                episodes=episodes,
                micro_batch_size=config["training"]["micro_batch_size"],
                pad_token_id=tokenizer.pad_token_id,
                max_grad_norm=config["training"]["max_grad_norm"],
                device=device,
                dtype=dtype,
                entropy_coef=entropy_coef,
            )
        else:
            results = update_policy(
                model=model,
                optimizer=optimizer,
                episodes=episodes,
                micro_batch_size=config["training"]["micro_batch_size"],
                pad_token_id=tokenizer.pad_token_id,
                max_grad_norm=config["training"]["max_grad_norm"],
                device=device,
                dtype=dtype,
            )
        if device.type == "cuda":
            torch.cuda.synchronize()
        end_time = time.time()
        duration = end_time - start_time
        start_time = end_time

        # compute and log important metrics
        reward = [episode.reward for episode in metric_episodes]
        formatted_reward = [
            episode.reward_info["format_reward"] for episode in metric_episodes
        ]
        answer_reward = [episode_correct(episode) for episode in metric_episodes]
        num_finished_episodes = sum(episode.is_finished for episode in metric_episodes)
        mean_reward = np.mean(reward)
        std_reward = np.std(reward)
        success_rate = np.mean(answer_reward)
        format_reward = np.mean(formatted_reward)
        grad_norm = results["grad_norm"]
        entropy = results["entropy"]
        lr = optimizer.param_groups[0]["lr"]
        loss = results["loss"]
        mean_response_len = np.mean(
            [len(episode.generated_token_ids) for episode in metric_episodes]
        )

        # Fork-specific metrics
        fork_rate = 0.0
        mean_t_proxy = 0.0
        if fork_enabled:
            fork_rate = np.mean([
                episode.reward_info.get("forked", 0.0) for episode in metric_episodes
            ])
            mean_t_proxy = np.mean([
                episode.reward_info.get("t_proxy_ms", 0.0) for episode in metric_episodes
            ])

        print(
            f"\rStep {step}, mean_reward: {mean_reward:.2f}, "
            f"train success_rate: {success_rate:.2f}, "
            f"grad_norm: {grad_norm:.2f}, duration: {duration:.2f}, "
            f"num_finished_episodes: {num_finished_episodes}, "
            f"mean_response_len: {mean_response_len:.2f}, "
            f"entropy: {entropy:.2f}"
            + (
                f", fork_rate: {fork_rate:.2f}, T_proxy: {mean_t_proxy:.1f}, "
                f"fork_bonus_coef: {current_fork_presence_bonus:.3f}, "
                f"fork_bonus_rate: {fork_presence_bonus_rate:.2f}"
                if fork_enabled
                else ""
            )
        )
        if step % config["training"]["eval_interval"] == 0:
            eval_metrics = evaluate(
                model=model,
                tokenizer=tokenizer,
                device=device,
                dtype=dtype,
                config=config,
                fork_enabled=fork_enabled,
                fork_reward_config=fork_reward_config,
            )
            print(
                f"\rEval success_rate: {eval_metrics['success_rate']:.2f}, "
                f"TTFC p50/p90: {eval_metrics['ttfc_ms_p50']:.1f}/{eval_metrics['ttfc_ms_p90']:.1f} ms"
                + (
                    f", eval_fork_rate: {eval_metrics.get('fork_rate', 0.0):.2f}, "
                    f"eval_T_proxy: {eval_metrics.get('t_proxy_ms', 0.0):.1f}"
                    if fork_enabled
                    else ""
                )
                + " " * 20
            )
            tb_writer.add_scalar("success_rate/eval", eval_metrics["success_rate"], step)
            tb_writer.add_scalar("ttfc_ms_p50/eval", eval_metrics["ttfc_ms_p50"], step)
            tb_writer.add_scalar("ttfc_ms_p90/eval", eval_metrics["ttfc_ms_p90"], step)
            if fork_enabled:
                tb_writer.add_scalar("fork_rate/eval", eval_metrics.get("fork_rate", 0.0), step)
                tb_writer.add_scalar("T_proxy_ms/eval", eval_metrics.get("t_proxy_ms", 0.0), step)

        tb_writer.add_scalar("loss", loss, step)
        tb_writer.add_scalar("mean_reward", mean_reward, step)
        tb_writer.add_scalar("std_reward", std_reward, step)
        tb_writer.add_scalar("success_rate/train", success_rate, step)
        tb_writer.add_scalar("format_reward", format_reward, step)
        tb_writer.add_scalar("grad_norm", grad_norm, step)
        tb_writer.add_scalar("duration", duration, step)
        tb_writer.add_scalar("num_finished_episodes", num_finished_episodes, step)
        tb_writer.add_scalar("learning_rate", lr, step)
        tb_writer.add_scalar("mean_response_len", mean_response_len, step)
        tb_writer.add_scalar("entropy", entropy, step)

        if fork_enabled:
            tb_writer.add_scalar("fork_rate", fork_rate, step)
            tb_writer.add_scalar("T_proxy_ms", mean_t_proxy, step)
            tb_writer.add_scalar(
                "fork_presence_bonus_coef", current_fork_presence_bonus, step
            )
            tb_writer.add_scalar(
                "fork_presence_bonus_rate", fork_presence_bonus_rate, step
            )

        for i, episode in enumerate(metric_episodes[:text_log_episodes_per_step]):
            # TensorBoard treats text as markdown.
            text = html.escape(episode.text)
            tb_writer.add_text(f"text_{i}", f"<pre>{text}</pre>", step)

        # save checkpoint
        if step % config["training"]["ckpt_save_interval"] == 0:
            output_file = ckpt_dir / f"ckpt_{step:06d}.pt"
            torch.save(model.state_dict(), output_file)
            print(f"Saved checkpoint to {output_file}")

    tb_writer.flush()
    tb_writer.close()


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument("--max_steps", type=int, default=None)
    args = parser.parse_args()
    main(args.config, max_steps_override=args.max_steps)
