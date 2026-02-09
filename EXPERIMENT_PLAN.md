# Experiment Plan v0.2: Fork-Race GRPO for Wall-Clock Time-to-First-Correct

## 1) Goal (Primary Metric First)

We optimize for **real wall-clock latency** to a correct answer, not total tokens.

Primary metric on held-out prompts:
- `TTFC_ms` = milliseconds from first generated token to first verifier-correct answer.

Primary report:
- `TTFC_ms` p50 / p90
- success@budget (fraction solved within fixed wall-clock budget)
- cost-per-correct (USD per correct answer using instance hourly price)

Secondary metrics:
- total generated tokens
- fork rate
- branch win rate

## 2) Scope of v0 (kept minimal)

- No KV-join/merge across histories.
- At most one fork per rollout.
- Exactly two branches after fork.
- Fork valid only inside `<think> ... </think>`.
- Race semantics: first **correct** branch wins; incorrect finished branch is terminated.

## 3) Critical Risks and Fixes (From Prior Critique)

### Risk A: No fair baseline for "learned branching" under a latency goal
Fix:
- Use **latency/cost-budgeted controls** (not compute-matched) with the same max parallelism cap:
1. `single-path`: normal decoding (no branching).
2. `external-race-2`: always run two independent samples in parallel from step 0.
3. `internal-fork-race-2`: learned token-level fork timing.
- Compare methods at fixed wall-clock budgets (e.g., 1500 ms, 2500 ms) and report cost-per-correct.

### Risk B: Sparse reward => no fork behavior emerges
Fix:
- Two-phase curriculum:
1. Warmup (`N_warmup` updates): `alpha=0`, force valid fork on fraction `p_force` of rollouts.
2. Latency phase: remove forcing, set `alpha>0` and optimize latency.

### Risk C: Unstable reward scale
Fix:
- Use bounded reward with normalized latency proxy:
  - success: `R = 1 - alpha * L_norm - beta * invalid_fork`
  - failure: `R = -delta - beta * invalid_fork`
  - `L_norm in [0, 1]`

### Risk D: Incorrect gradient attribution (double counting)
Fix:
- Token-weighted objective:
  - pre-fork tokens: weight `1.0`
  - post-fork branch tokens at step `s`: weight `1 / active_branches(s)`
- Normalize by sum of token weights, not raw token count.

### Risk E: Parser ambiguity for “inside think” and answer completion
Fix:
- Implement incremental text-state parser with explicit states:
  - `before_think`, `in_think`, `after_think`
  - answer-open / answer-close detection
- Trigger verification only when `</answer>` is fully observed.

### Risk F: KV memory blow-up after fork
Fix:
- Pre-allocate for worst-case `2 x batch_size` active rows in fork mode.
- Log OOM and automatically fall back to smaller batch preset for fork runs.

## 4) Wall-Clock-Aware Objective Design

Direct per-rollout reward uses a **hardware-calibrated latency proxy** to reduce jitter while matching real timing:

- Calibrate once per machine:
  - `c1_ms` = average ms/step with 1 active branch
  - `c2_ms` = average ms/step with 2 active branches
- During rollout, accumulate:
  - `T_proxy_ms = sum_s c_{b_s}` where `b_s in {1,2}` active branches at step `s`.
- If correct answer appears at step `s*`, latency term uses prefix sum up to `s*`.
- Normalize:
  - `L_norm = min(T_proxy_ms / budget_ms, 1.0)`

Reward:
- success: `R = 1 - alpha * L_norm - beta * invalid_fork`
- failure: `R = -delta - beta * invalid_fork`

Default v0 constants:
- `alpha = 0.5`
- `beta = 0.2`
- `delta = 0.2`

Note: real wall-clock `TTFC_ms` is still the primary evaluation metric; proxy is for lower-variance training signal.

## 5) Exact Behavioral Semantics

Special tokens:
- `<fork>` operator
- `<fork1>` marker for branch A
- `<fork2>` marker for branch B

When `<fork>` is sampled in valid state:
1. Replace sampled `<fork>` with `<fork1>` in branch A and `<fork2>` in branch B.
2. Clone branch state (token buffer + KV row content up to current position).
3. Continue both branches in parallel.

Invalid fork handling:
- If outside `in_think`, do not branch; keep token text but mark `invalid_fork=1` for penalty.

Termination:
- If any branch emits complete `</answer>` and verifier says correct -> stop rollout immediately and select winner.
- If a branch emits complete incorrect answer -> terminate that branch only.
- Stop rollout when all branches terminated or max length reached.

## 6) Wall-Clock-Budget Experimental Protocol (Signal-Focused)

Held-out evaluation set is fixed and shared across all methods.

For each method (`single-path`, `external-race-2`, `internal-fork-race-2`):
- Run with identical model checkpoint.
- Measure across same prompts and same random-seed set.
- Run on the same hardware class with max parallelism capped at 2 branches.
- Evaluate at fixed wall-clock budgets (e.g., 1.5s and 2.5s), then compare cost-per-correct.

Main questions:
1. At equal wall-clock budget, does internal fork-race increase success@budget?
2. At equal success target, does internal fork-race reduce p50/p90 `TTFC_ms`?
3. Versus `external-race-2`, does learned fork timing reduce cost-per-correct at similar success?

## 7) Statistical Plan (Avoid False Signal)

- Use at least 3 training seeds for v0.
- For each metric, report mean and 95% bootstrap CI across prompts.
- Predefine success criterion:
  - internal-fork-race improves success@budget by >= 5 absolute points over single-path
  - and is non-inferior to external-race-2 on success while improving either p50 `TTFC_ms` or cost-per-correct by >= 10%.

If criterion fails, treat v0 as negative/neutral signal and do not escalate complexity.

## 8) Implementation Plan (Minimal, But Complete)

1. Tokenizer/model plumbing:
- add fork tokens
- resize embeddings + output projection safely

2. Fork-aware rollout engine:
- branch state machine
- parser state tracking
- race + early stop + verifier hook

3. Reward and logging:
- compute `T_proxy_ms`, `TTFC_ms`, success, fork diagnostics
- produce rollout-level scalar reward for GRPO

4. GRPO update weighting:
- token weights by active branch count
- shared-prefix counted once

5. Evaluation harness:
- method selector (`single-path`, `external-race-2`, `internal-fork-race-2`)
- budgeted wall-clock comparisons

## 9) Test Plan (Required Before Remote Runs)

Unit tests:
- parser state transitions (`<think>`, `</think>`, `<answer>`, `</answer>`)
- fork validity checks
- early-stop winner selection
- reward bound checks (`R` range and penalties)
- token-weighted loss math (pre-fork vs post-fork weighting)

Integration tests:
- deterministic toy rollout with mocked logits to force:
  - no fork path
  - valid fork + branch A win
  - valid fork + branch B win
  - invalid fork penalty case
- smoke GRPO update with tiny model shapes for no-NaN gradient.

Sanity run before Vast:
- short local run (few steps) to verify:
  - non-zero fork rate during warmup
  - stable loss/grad norm
  - logs include `TTFC_ms`, `T_proxy_ms`, `fork_rate`.

## 10) Vast.ai Execution Strategy (Bang-for-Buck for v0)

Given prior measurements, default to **single RTX 4090** for best speed/$ in this workload family.

Run plan:
1. warmup stage checkpoint
2. latency stage checkpoint
3. fixed-budget evaluation across three methods

Artifacts to keep:
- checkpoints
- tensorboard logs
- evaluation CSV with per-prompt wall-clock metrics
- run metadata (GPU, $/hr, commit SHA, seed)

---

This v0 plan is intentionally strict: if internal fork-race cannot beat latency/cost-budgeted baselines on wall-clock metrics, we stop and reassess instead of adding complexity.
