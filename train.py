import html
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
from fork_reward import ForkRewardConfig
from grpo import fork_rollout, rollout, update_policy, update_policy_fork
from optimizer import MemoryEfficientAdamW
from qwen2_model import Transformer
from tokenizer import Tokenizer


def evaluate(model, tokenizer, device, dtype, config):
    test_dataset = CountdownTasksDataset(
        data_path=config["data"]["path"],
        tokenizer=tokenizer,
        split="test",
        test_size=config["data"]["test_size"],
    )
    generator = torch.Generator(device=device)
    # We reduce the batch size by half as we want to
    # generate twice as long trajectories.
    dataloader = DataLoader(
        test_dataset,
        shuffle=False,
        collate_fn=CountdownTasksDataset.collate_fn,
        generator=generator,
        batch_size=config["training"]["batch_size"] // 2,
        drop_last=False,
    )
    success = []
    for batch in dataloader:
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
        success.extend([episode.reward_info["answer_reward"] for episode in episodes])
    return np.mean(success)


def main(config_path: str):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    pretrained_model_path = Path(config["model"]["pretrained_model_path"])
    device = torch.device(config["model"]["device"])
    dtype_map = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }
    dtype = dtype_map.get(config["model"]["dtype"], torch.bfloat16)
    torch.set_default_device(device)
    torch.random.manual_seed(config["training"]["random_seed"])
    BATCH_SIZE = config["training"]["batch_size"]
    NUM_QUESTIONS_PER_BATCH = config["training"]["num_questions_per_batch"]
    NUM_ANSWERS_PER_QUESTION = BATCH_SIZE // NUM_QUESTIONS_PER_BATCH

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
        model.resize_embeddings(tokenizer.vocab_size)
        print(f"Model embeddings resized to {model.vocab_size}")

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
        force_fork_prob = fork_config.get("force_fork_prob", 0.3)

    start_time = time.time()
    ckpt_dir = Path(config["training"]["ckpt_dir"])
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    for step, batch in enumerate(train_dataloader, start=1):
        if fork_enabled:
            # Warmup: force forks with probability p_force
            current_force_prob = force_fork_prob if step <= n_warmup else 0.0
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
                force_fork_prob=current_force_prob,
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

        if config["training"]["skip_unfinished_episodes"]:
            episodes = [episode for episode in episodes if episode.is_finished]

        if fork_enabled:
            results = update_policy_fork(
                model=model,
                optimizer=optimizer,
                episodes=episodes,
                micro_batch_size=config["training"]["micro_batch_size"],
                pad_token_id=tokenizer.pad_token_id,
                max_grad_norm=config["training"]["max_grad_norm"],
                device=device,
                dtype=dtype,
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
        torch.cuda.synchronize()
        end_time = time.time()
        duration = end_time - start_time
        start_time = end_time

        # compute and log important metrics
        reward = [episode.reward for episode in episodes]
        formatted_reward = [
            episode.reward_info["format_reward"] for episode in episodes
        ]
        answer_reward = [episode.reward_info["answer_reward"] for episode in episodes]
        num_finished_episodes = sum(episode.is_finished for episode in episodes)
        mean_reward = np.mean(reward)
        std_reward = np.std(reward)
        success_rate = np.mean(answer_reward)
        format_reward = np.mean(formatted_reward)
        grad_norm = results["grad_norm"]
        entropy = results["entropy"]
        lr = optimizer.param_groups[0]["lr"]
        loss = results["loss"]
        mean_response_len = np.mean(
            [len(episode.generated_token_ids) for episode in episodes]
        )

        # Fork-specific metrics
        fork_rate = 0.0
        mean_t_proxy = 0.0
        if fork_enabled:
            fork_rate = np.mean([
                episode.reward_info.get("forked", 0.0) for episode in episodes
            ])
            mean_t_proxy = np.mean([
                episode.reward_info.get("t_proxy_ms", 0.0) for episode in episodes
            ])

        print(
            f"\rStep {step}, mean_reward: {mean_reward:.2f}, "
            f"train success_rate: {success_rate:.2f}, "
            f"grad_norm: {grad_norm:.2f}, duration: {duration:.2f}, "
            f"num_finished_episodes: {num_finished_episodes}, "
            f"mean_response_len: {mean_response_len:.2f}, "
            f"entropy: {entropy:.2f}"
            + (f", fork_rate: {fork_rate:.2f}, T_proxy: {mean_t_proxy:.1f}" if fork_enabled else "")
        )
        if step % config["training"]["eval_interval"] == 0:
            eval_success_rate = evaluate(model, tokenizer, device, dtype, config)
            print(f"\rEval success rate: {eval_success_rate:.2f}" + " " * 100)
            tb_writer.add_scalar("success_rate/eval", eval_success_rate, step)

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

        for i, episode in enumerate(episodes):
            # TensorBoard treats text as markdown.
            text = html.escape(episode.text)
            tb_writer.add_text(f"text_{i}", f"<pre>{text}</pre>", step)

        # save checkpoint
        if step % config["training"]["ckpt_save_interval"] == 0:
            output_file = ckpt_dir / f"ckpt_{step:06d}.pt"
            torch.save(model.state_dict(), output_file)
            print(f"Saved checkpoint to {output_file}")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--config", type=str, default="config.yaml")
    args = parser.parse_args()
    main(args.config)
