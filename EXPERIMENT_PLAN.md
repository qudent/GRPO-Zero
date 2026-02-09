# Report: Minimal “fork-token + race” branching CoT experiment in GRPO-Zero (no KV-join)

## Motivation

You want to test a very specific hypothesis that is *systems-realistic*:

> **Parallel exploration can reduce wall-clock latency to a correct answer**, even if it increases total tokens, because GPUs exploit batching/throughput.

Most existing “multiple CoT” methods do branching *outside* the model (sample N complete solutions, then pick). That doesn’t answer your question: can the model learn **when** to branch as a *token-level decision*, and can it learn a policy that optimizes **time-to-first-correct** under a parallel decoding cost model?

To isolate the effect, we avoid any “KV cache join/merge” across incompatible histories. Instead we implement a clean, well-defined concurrency primitive: **fork + race + early stop**. This keeps the transformer semantics intact and makes the experiment interpretable.

---

## Why not “KV-join” for the first experiment

Merging KV caches from different histories does not correspond to any single token history in a standard transformer. It’s not just an approximation—it changes the underlying semantics in a way that is hard to make stationary and hard to interpret. You *might* be able to train a model to tolerate it, but then you’d be conflating:
- learning to branch
- with learning to cope with a nonstandard state operator

Since you explicitly want to “separate effects,” we postpone KV-join to later iterations.

---

## Core design: treat branching like a Unix process primitive

Use a standard mental model:

- **fork**: clone decoding state into two independent continuations
- **race**: both continuations proceed in parallel (batch dimension)
- **wait**: stop as soon as a success condition is met
- **kill**: terminate losing branch(es)

This gives you a clean, minimal primitive with explicit semantics that matches your throughput argument.

---

## Behavioral spec (what “fork” means)

### Tokens
Introduce three special tokens:

- `<fork>`: branching operator that the model may emit
- `<fork1>`: marker token that replaces `<fork>` in branch A
- `<fork2>`: marker token that replaces `<fork>` in branch B

These markers help with debugging and allow explicit reward shaping later (e.g., “fork only inside think”).

### When `<fork>` is emitted
During decoding (ideally within `<think>...</think>`):

1. Replace the just-generated `<fork>` with:
   - branch A: `<fork1>`
   - branch B: `<fork2>`
2. Split the live decoding state into two branches **from the same prefix**.
3. Continue decoding both branches in parallel (batched inference), independent sampling streams.

### Constraints for v0
To keep this minimal and interpretable:
- Allow **at most one fork per rollout**.
- Fork only allowed inside the `<think>` region; fork outside think is treated as invalid or heavily penalized.
- Exactly **two branches** only (no recursive tree yet).

---

## How the model produces a single final answer (no join required)

### Race semantics: “first-correct wins”
Both branches generate concurrently, and the episode ends as soon as the system obtains a **verifiably correct** answer:

1. Each branch generates until it emits a complete answer block (e.g., closes `</answer>`), or hits max length.
2. When a branch completes an answer, immediately verify it with the existing Countdown checker.
3. If **correct**, terminate the entire rollout immediately; output that branch’s answer as the final answer.
4. If incorrect, that branch can be marked “failed” (stop it), and the other branch continues racing.

This creates a single final output (the winner’s answer) without mixing KV states.

Why this matters: It aligns reward with your latency objective and prevents “finish fast even if wrong,” which would happen if you ended the episode at the first answer regardless of correctness.

---

## Cost model and reward design

Your key point is that wall-clock latency is closer to **critical path length** than total tokens. Race semantics naturally defines a latency proxy:

- Let `T_first_correct` be the *token time index* (or step count) when the first correct answer becomes available.
  - If both branches are decoded in lockstep time steps, this is basically “how many decoding steps until success,” not “how many tokens total were generated across branches.”
- Reward should prioritize correctness first, then minimize `T_first_correct`.

### Minimal reward (clean separation)
Use a two-stage objective in one scalar reward:

- If success: `R = 1.0 - α * T_first_correct`
- If failure: `R = 0.0` (or a small negative penalty)

Optional:
- Add a small format reward only if needed to keep outputs parseable.
- Add a small fork-usage penalty later (`-β` if fork used) once behavior emerges, to discourage gratuitous forks.

### Why this separates effects
- With `α = 0`, you test “does forking help correctness at all?”
- With `α > 0`, you test “does forking reduce time-to-first-correct?”

---

## Training integration with GRPO (minimal algorithmic disruption)

GRPO expects:
- M sampled rollouts per prompt
- a scalar reward per rollout
- advantage normalization within each prompt’s group
- policy gradient update on the sampled tokens

In this design:
- Each rollout still produces **one scalar reward** and **one final answer**.
- Forking only changes how the rollout is generated, not the GRPO math.

### Gradient attribution (important practical detail)
During a forked rollout, you generated tokens for *both* branches until termination. Those tokens influenced the probability of reaching the winning answer quickly.

However, you don’t want gradients to explode just because you had two branches. To keep things stable and faithful to “parallel compute,” use one of these minimal normalization rules:

**Recommended (simple):**
- Compute standard per-token policy-gradient loss for all generated tokens across both branches up to termination.
- Divide the total loss by the number of active branches averaged over time (or simply by 2 if you always fork into 2 and decode both until stop).

This keeps gradient magnitudes comparable to non-fork rollouts and matches your “throughput makes parallel cheaper” spirit.

---

## Implementation plan (smallest set of changes)

### 1) Tokenizer and embeddings
- Add `<fork> <fork1> <fork2>` as special tokens (single token IDs).
- Expand embedding matrix by 3 and initialize new rows.
- Expose token IDs to decoding code.

### 2) Fork-aware decoding wrapper
Introduce a new generation function (or modify existing generation) to support:

- monitoring generated tokens for `<fork>`
- splitting state into 2 branches at fork time
- batched decoding of both branches
- maintaining independent RNG per branch

KV caches:
- Duplicate or clone KV cache at the fork boundary (or recompute prefix once per branch if no KV abstraction exists yet—slower but correct).
- Never merge KV states.

### 3) Early stopping + verification hooks
Implement “first-correct wins”:
- detect when a branch closes an answer
- run verifier
- if correct: stop immediately and return winner
- if incorrect: stop that branch; continue the other

### 4) Reward computation
Record:
- `T_first_correct` (or max steps if failure)
- success boolean
- fork used boolean
Compute reward scalar.

### 5) GRPO loop remains the same
Feed reward scalar into GRPO exactly as before.
Only difference: rollout trajectories now may have two token streams internally, which you flatten into a single loss with normalization.

### 6) Instrumentation (to validate the hypothesis)
Log:
- `fork_rate` (fraction of rollouts that fork)
- success rate
- `T_first_correct` distribution (p50/p90)
- total tokens generated (for showing the divergence)
- how often branch A wins vs branch B

The key figure: success vs `T_first_correct` as training progresses.

---

## Suggested ablation schedule

1) Baseline GRPO-Zero (no fork tokens).
2) Fork enabled, **α = 0** (correctness-only; does model learn to use fork at all?).
3) Fork enabled, **α > 0** (latency pressure; does fork become selective and reduce `T_first_correct`?).
4) (Optional) Add small fork penalty β to prevent gratuitous forks once it works.

---

## End-of-report emphasis: the specific choices we made

- **We rejected KV-cache “join/merge across histories” for v0** to avoid changing transformer semantics and conflating effects.
- **We chose a Unix-like concurrency primitive**: fork → race → early stop (winner) → kill losers.
- **We made fork tokenization explicit with three tokens**: `<fork>` operator plus `<fork1>/<fork2>` branch markers (fork rewrites into branch-id tokens).
- **We constrained v0 heavily**: at most **one fork**, only within `<think>`, exactly **two branches**.
- **We defined episode success as “first-correct wins,” not “first-answer wins”**, to prevent incentivizing fast wrong answers.
- **We used a latency proxy based on critical path**: `T_first_correct` (time-to-first-correct), aligning with GPU batching intuition rather than total token count.
- **We kept GRPO unchanged**: each rollout still yields **one scalar reward**; branching only modifies the rollout generator.
- **We normalized gradients for forked rollouts** (e.g., divide by 2) so branching doesn’t artificially double update magnitude.
- **We planned clean ablations**: α=0 to test correctness impact; α>0 to test latency optimization; optional β to discourage gratuitous forks.