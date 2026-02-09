# Vast.ai Training Run Report (GRPO-Zero)

Generated: 2026-02-09 21:08:35 UTC  
Repo: `grpo_zero`  
Instance ID: `31135532`  
Host: `ssh8.vast.ai:15532`  
GPU: `NVIDIA A40 (46,068 MiB)`  
Price: `$0.4318518519 / hour` (including disk/network components shown by Vast)

## Executive Summary

- Training is healthy and actively using the GPU (`100%` util at snapshot).
- Run reached **Step 27** and was in **Computing policy gradient (102/256, 39.84%)** when captured.
- Effective throughput so far:
  - **Iterations/sec (steps/sec):** `0.0100` (all steps), `0.00948` (last 10)
  - **Episodes/sec:** `2.56` (all), `2.43` (last 10)
  - **Generated tokens/sec (approx):** `846.81`
- With current code/config, this job is much longer than “a few hours”: expected full pass is **15,320 steps**.
- At observed speed on current A40, remaining full run time is about **17.7–18.7 days** and **$183–$193**.
- Best speed/$ recommendation right now: **switch to RTX 4090 single-GPU** and cap training length (for example 300–1000 steps) instead of running full-epoch.

## Where Logs Are

- Main stdout/stderr log: `/workspace/train.log`
- tmux session output (cleanest for step lines): `tmux` session `train`
- TensorBoard event file:
  - `/workspace/grpo_zero/logs/20260209-202228/events.out.tfevents.1770668548.66e745e36ddf.1069.0`
- Vast container logs (`vastai logs 31135532`) mostly show startup/SSH activity, not detailed training metrics.

## Training Parameters (from `config.yaml`)

- `batch_size: 256`
- `num_questions_per_batch: 32`
- `answers_per_question: 8` (derived from `256/32`)
- `max_gen_len: 1024`
- `micro_batch_size: 2`
- `eval_interval: 10`
- `ckpt_save_interval: 100`
- `skip_unfinished_episodes: false`
- `memory_efficient_adamw: false` (default A40 config)

## Run Snapshot Metrics

### Latest step metrics (Step 27)

- `mean_reward: 0.36`
- `train_success_rate: 0.27`
- `grad_norm: 0.54`
- `duration: 105.89 s`
- `num_finished_episodes: 244`
- `mean_response_len: 436.71 tokens`
- `entropy: 0.57`

### Aggregated throughput

- Parsed completed steps: `27`
- Mean step duration (all): `99.9767 s`
- Mean step duration (last 10): `105.4480 s`
- Iterations/sec (all): `0.010002`
- Iterations/sec (last 10): `0.009483`
- Episodes/sec (all): `2.5606`
- Episodes/sec (last 10): `2.4277`
- Avg tokens/step (approx): `84,661.10`
- Tokens/sec (approx): `846.81`

### Evaluation signal

- Eval success rates seen: `0.23`, `0.39` (latest observed eval metric: `0.39`)

## How Long This Is Likely To Take

Dataset on instance:
- Total rows: `490,364`
- Test rows: `128`
- Train rows: `490,236`
- Steps per full run (`ceil(490236 / 32)`): **15,320**

Current progress:
- Completed steps: `27`
- Remaining steps: `15,293`

Estimated remaining runtime/cost on current A40:
- Using all-step average (`99.98s`): **424.71 hours (17.70 days)**, **$183.41**
- Using last-10 average (`105.45s`): **447.95 hours (18.66 days)**, **$193.45**

## Beefier Machine Projections (Single-GPU)

These estimates use observed A40 step time and the relative speed assumptions from `grpo_zero_gpu_analysis.md` (`3090=1.0x, A40=1.3x, 4090=2.0x, L40S=1.8x, H100=4.0x`).  
Live offer prices used were pulled at report time.

| GPU | Live $/hr | Est. sec/step | Est. iter/sec | Est. episodes/sec | Est. $/step | Remaining full run (h) | Remaining full run ($) |
|---|---:|---:|---:|---:|---:|---:|---:|
| RTX 3090 | 0.201 | 129.97 | 0.00769 | 1.97 | 0.00726 | 552.12 | 110.98 |
| RTX 4090 | 0.2934 | 64.98 | 0.01539 | 3.94 | 0.00530 | 276.06 | 80.99 |
| A40 (current) | 0.4319 | 99.98 | 0.01000 | 2.56 | 0.01199 | 424.71 | 183.41 |
| L40S | 0.5023 | 72.21 | 0.01385 | 3.55 | 0.01007 | 306.73 | 154.08 |
| H100 SXM | 1.6681 | 32.49 | 0.03078 | 7.88 | 0.01506 | 138.03 | 230.24 |

For a more practical target (stop at step 300 from current step 27):

| GPU | Time to step 300 | Cost to step 300 |
|---|---:|---:|
| RTX 3090 | 9.86 h | $1.98 |
| RTX 4090 | 4.93 h | $1.45 |
| A40 | 7.58 h | $3.27 |
| L40S | 5.48 h | $2.75 |
| H100 SXM | 2.46 h | $4.11 |

## Recommendation (Optimal Speed / Cost)

1. **Do not run full-epoch as currently coded** unless you intentionally want ~15k RL updates.  
   Current loop is one full dataloader pass, which is far longer than the README’s “few hours” expectation.
2. **Use a single RTX 4090 offer** for best speed-per-dollar from live options sampled here.
3. **Cap updates explicitly** (for example 300–1000 steps) to keep turnaround hours, not weeks.
4. Keep checkpointing frequent enough for interruption recovery (`ckpt_save_interval` maybe 25–50 while iterating quickly).
5. If wall-clock is the only priority and budget is secondary, use H100; otherwise 4090 is better economics.

## Why Multi-GPU Is Not the Answer Here

- This repository is currently single-GPU only (no DDP/FSDP/DeepSpeed in `train.py`).
- Multi-GPU instances will not speed up one training job without code changes.
- Extra GPUs are only useful for parallel independent runs (sweeps), not faster single-run convergence.
