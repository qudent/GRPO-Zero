# Fork-Race Experiment: Task DAG

## Current State (2026-02-10)

| Resource | Status | Cost |
|----------|--------|------|
| Instance 1 (ssh3:20614, 4090) | **Idle** - fork training stopped | $0.32/hr |
| Instance 2 (ssh4:21106, 4090) | **Running** vanilla GRPO, step 200 | $0.28/hr |
| Vanilla ckpt_000200.pt | 57% eval success, ready to use | — |
| Vanilla ckpt_000100.pt | ~40% eval success, backup | — |

## Task DAG

```
P0 ──→ P1a ──→ P2 ──→ P3
  ╲      ╱            ╱
   P1b ────────────→ ╱
    ╲               ╱
     P1c ─────────→
```

### P0: Vanilla Baseline [RUNNING - Instance 2]
- **Status**: Step 200, 57% eval. Continuing to step 300.
- **Time remaining**: ~3h at 125s/step
- **Cost**: ~$0.84 remaining
- **Output**: ckpt_000200.pt (ready), ckpt_000300.pt (pending)
- **No action needed** - let it run.

### P1a: SFT Fork Warmup [NEXT - Instance 1]
- **Depends on**: P0 checkpoint (ckpt_000200.pt ready now)
- **Goal**: Teach the model to emit `<fork>` token via supervised learning
- **Method**:
  1. Load vanilla checkpoint
  2. Generate rollouts WITH forced forks (use probability injection)
  3. Build SFT dataset: (input_sequence, target_sequence_with_fork) pairs
  4. Fine-tune with cross-entropy loss for ~50-100 steps
  5. Verify: model emits fork tokens with P > 0.01 naturally
- **Time estimate**: ~1-2h (generate data: 30min, SFT: 30min, verify: 15min)
- **Cost**: ~$0.50
- **Output**: ckpt_sft_fork.pt with natural fork ability

### P1b: External-Race-2 Baseline [PARALLEL - Instance 1 or 2]
- **Depends on**: P0 checkpoint
- **Goal**: Evaluate "run 2 independent samples, take first correct" baseline
- **Method**: Use vanilla ckpt, generate 2x samples per prompt, measure TTFC
- **Time estimate**: ~1h
- **Cost**: ~$0.30
- **Can run on instance 2 after P0 finishes, or instance 1 after P1a**
- **Output**: eval_external_race2.csv

### P1c: Vanilla Eval [PARALLEL - during P0]
- **Depends on**: P0 checkpoint
- **Goal**: Measure single-path TTFC baseline with wall-clock timing
- **Method**: Standard eval on vanilla ckpt
- **Time estimate**: ~30min
- **Cost**: ~$0.15
- **Output**: eval_vanilla.csv

### P2: GRPO Fork-Race Training [Instance 1]
- **Depends on**: P1a (SFT checkpoint)
- **Goal**: RL fine-tune to learn WHEN to fork optimally
- **Method**:
  1. Load SFT fork checkpoint
  2. GRPO with fork-race reward (latency-aware)
  3. Model learns timing: when forking helps vs wastes compute
  4. 200-400 steps
- **Time estimate**: ~4-6h at ~50s/step
- **Cost**: ~$1.50-2.00
- **Output**: ckpt_fork_grpo.pt

### P3: Head-to-Head Evaluation [Either instance]
- **Depends on**: P1b, P1c, P2
- **Goal**: Compare all three methods at fixed wall-clock budgets
- **Metrics**:
  - success@budget (at 1.5s and 2.5s)
  - TTFC p50/p90
  - cost-per-correct
- **Time estimate**: ~1-2h
- **Cost**: ~$0.50
- **Output**: eval_comparison.csv, final report

## Timeline (Optimistic)

```
Hour 0-3:  [Instance 2: P0 continues] [Instance 1: Implement + run P1a]
Hour 2-3:  [Instance 1: P1a completes, start P2]
Hour 3-4:  [Instance 2: P0 done → run P1b + P1c]
Hour 3-9:  [Instance 1: P2 GRPO fork-race training]
Hour 9-10: [Either: P3 head-to-head eval]
```

**Total wall-clock: ~10 hours from now**
**Total cost: ~$6-8 across both instances**

## Speed-Up Options

### Option A: Use ckpt_200 instead of waiting for ckpt_300
- **Saves**: ~3h wait time
- **Risk**: 57% success might be enough; 65%+ would be better
- **Recommendation**: Use ckpt_200 now. Can always re-run from ckpt_300 later.
- **Status**: CHOSEN - start P1a immediately

### Option B: Parallelize P1b/P1c on instance 2 while P0 still runs
- **Saves**: ~1h (eval happens while vanilla trains)
- **Method**: Copy ckpt_200 and eval on same instance during training gaps
- **Risk**: Eval uses GPU, might slow ongoing training
- **Recommendation**: Wait for P0 to finish, then run P1b/P1c

### Option C: Rent a 3rd instance for eval parallelism
- **Saves**: ~2h (all evals run in parallel)
- **Cost**: +$0.30/hr * 2h = $0.60
- **Recommendation**: Not worth it unless time is critical

### Option D: Shorter GRPO phase (100 steps instead of 200-400)
- **Saves**: 2-4h
- **Risk**: Might not learn optimal fork timing
- **Recommendation**: Start with 200, checkpoint at 100, evaluate both

### Option E: Lower batch size for faster iteration during P1a
- **Saves**: ~50% time per step during SFT warmup
- **Risk**: Noisier gradients
- **Recommendation**: Use batch_size=32 for SFT (faster iteration)

## Implementation Checklist

- [ ] Copy ckpt_000200.pt from instance 2 → instance 1
- [ ] Implement SFT fork data generation script
- [ ] Implement SFT training loop (cross-entropy on fork positions)
- [ ] Run SFT warmup, verify P(fork) > 0.01 naturally
- [ ] Update config for GRPO phase 2 (from SFT ckpt, no probability injection)
- [ ] Run GRPO fork-race (200 steps)
- [ ] Implement eval harness (3 methods, wall-clock budgets)
- [ ] Run P1b external-race-2 eval
- [ ] Run P1c vanilla eval
- [ ] Run P3 head-to-head comparison
- [ ] Write final report

## Key Risk: Does Fork-Race Even Help?

The experiment plan (section 7) pre-defines the success criterion:
> internal-fork-race improves success@budget by >= 5 absolute points over single-path
> and is non-inferior to external-race-2 on success while improving cost-per-correct by >= 10%

If this fails, we stop and don't add complexity. External-race-2 (just run 2 independent
samples) might already capture most of the benefit without any model changes.
