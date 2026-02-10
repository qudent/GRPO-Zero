# grpo_zero - Status

## Current State
GRPO policy gradient training for countdown task with fork-race v0 feature. Training runs on Vast.ai RTX 4090 instances. Fork-race metrics and filtering implemented, pending validation on real training run.

## Active Goals
- [ ] Validate fork-race v0 on a real training run
  - [x] Implement fork_metrics.py
  - [x] Integrate fork metric filtering into train.py and grpo.py
  - [x] Add token text cache for parser performance
  - [x] Reduce TensorBoard overhead -- text_log_episodes_per_step: 4
  - [x] Unit tests for fork metrics
  - [ ] Smoke test locally -- short run, confirm reward > 0
  - [ ] Deploy to Vast.ai and run full training
- [ ] Confirm countdown task converges with current config
  - [ ] Diagnose if "rate 0" / no learning signal is config vs. algorithmic

## Blockers
- Need to smoke-test reward function in isolation before next Vast.ai run -- past runs burned hours with zero reward signal.

## Recent Results
- Feb 10: Fork-race v0 code complete -- metrics, filtering, tests all in place. Multiple optimization patches applied.
- Feb 9: Countdown task still slow. Unclear if feedback loop bug or just slow convergence. Removed 6-hour runtime cap.

## Next Steps
1. Run reward function on known-good and known-bad outputs manually -- confirm it returns different values
2. Short local training run -- 50 steps, assert loss moves
3. If smoke test passes, deploy to Vast.ai with config_vast_real.yaml
