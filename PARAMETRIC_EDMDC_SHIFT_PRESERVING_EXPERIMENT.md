# Parametric EDMDc shift-preserving stability correction

## Motivation from the uploaded validation artifacts

The no-shrink validation CSV contains 1,057 episodes. The median FDE is 6.247 m,
while the mean FDE is 49.544 m. Fourteen L8-MU08 episodes (payload 7,000 kg,
road friction 0.8) account for approximately 82% of the total FDE sum. Their
mean FDE is 3,068.576 m. Fifteen L7-MU08 episodes are also degraded.

The no-shrink checkpoint has a maximum condition-grid spectral radius of
1.403773, with the worst case at (payload=7,000 kg, mu=0.8). The dominant
unstable eigenvector is concentrated in the quadratic lifted-feature block.
The original global shrink reduces this mode but also scales deterministic
delay-history shifts.

## Correction

For Parametric EDMDc only:

1. Enforce exact deterministic state-history shifts in A.
2. Enforce exact control-history shifts in A and the newest control input in B.
3. Keep all shift rows independent of payload/friction parameters.
4. Search for the largest common factor that brings the learned lifted-dynamics
   subspace below the existing spectral cap over the full observed condition grid.
5. Apply that factor only to the learned next-state and quadratic-feature rows
   of A, A_payload and A_friction. Do not scale the deterministic shift rows.
   B and its condition terms are not globally scaled; exact control-history
   rows are set to their known shift structure.

Original EDMDc retains its previous legacy stabilization path. No changes are
made to K0/K1-P/K1-F/K2, lifting functions, Ridge regression, data split,
training/evaluation metrics, or model architecture.

## Checkpoint metadata

The new checkpoint records:
- ridge_pre_shift_radius: spectral radius before exact shift enforcement;
- pre_stabilization_radius: radius after exact shift enforcement;
- post_stabilization_radius: radius after learned-row stabilization;
- stabilization_factor;
- stabilization_applied;
- stabilization_policy='shift_preserving_learned_rows';
- shift_structure_enforced=true.

Before overwriting an existing Parametric EDMDc checkpoint, the trainer keeps
a variant-specific backup:
- best_uniform_shrink.pt / last_uniform_shrink.pt
- best_no_shrink.pt / last_no_shrink.pt

## Run

```bash
python highway_env/vehicle/koopman/scripts/run_training.py train --model "Parametric EDMDc"
python highway_env/vehicle/koopman/scripts/run_evaluation.py evaluate --model "Parametric EDMDc" --split val
python highway_env/vehicle/koopman/scripts/run_evaluation.py compare --split test
```

Only Parametric EDMDc needs retraining. Use the existing frozen split and
accepted dataset snapshot. Real TruckSim training/evaluation must be run in
the user's local environment.
