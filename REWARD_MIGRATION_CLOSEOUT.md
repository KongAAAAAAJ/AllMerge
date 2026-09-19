# Reward Migration Closeout

Baseline branch: `migrate/reward`

This document closes the reward-migration diagnostic phase. It does not change
reward mathematics or prescribe new weights.

## Completed validation

- trajectory-mode reward package smoke test;
- real AllMerge scenario adapter validation;
- Polynomial planner real-trajectory reward distribution;
- candidate/reference scorer parity;
- interaction-source attribution;
- background constant-speed prediction fidelity check;
- explicit false-positive and false-negative collision diagnostics.

## Current reward contract

- comparison unit: one target vehicle + one trajectory mode;
- target trajectory: raw desired trajectory `tau_d`;
- teammate context during GRPO: frozen Stage-1 argmax trajectories;
- interaction sources: background vehicles and frozen teammates;
- `formation_component=False`;
- environment legacy `_reward()` remains separate;
- common entry point: `evaluate_candidates(...)`.

## Final diagnostic command

Focused reproduction of the known merge-in stress case:

```bash
python -m highway_env.planner.diffusion.trajectory_mode_reward.background_prediction_diagnostics
```

The enhanced report must show both:
- predicted collision -> actual safe;
- predicted safe -> actual collision;
- collision precision and recall.

## Closeout decision rule

Do not change reward weights, collision geometry, or safe-gap thresholds solely
because of the current diagnostic sample.

Treat the constant-speed background predictor as an approximation. If the
remaining false negatives are concentrated near the 4 s horizon and correlate
with large background position/speed prediction error, record this as a known
predictor limitation and close reward migration.

Escalate to a predictor implementation change only if false negatives are:
1. frequent across multiple seeds/scenarios;
2. early-horizon rather than horizon-edge events; or
3. caused by a systematic lane/actor-state mismatch.

## Remaining integration gate

The only required gate after diagnostic closeout is standalone <-> GRPO reward
parity: both paths must call the same `evaluate_candidates(...)` scorer with
identical inputs and produce identical total rewards/components/unsafe flags.

No second GRPO-specific reward implementation should be introduced.
