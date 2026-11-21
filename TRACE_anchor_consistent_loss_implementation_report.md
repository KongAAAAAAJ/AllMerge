# TRACE Anchor-assigned Multimodal Loss Implementation Report

## 1. Objective switch

The original `goal_endpoint` objective remains the baseline and is kept on the original code path. The new objective is selected with:

```yaml
losses:
  training_scheme: anchor_consistent
```

Configuration: `configs/train/ablation/diffusion_anchor_consistent.yaml`.

## 2. Target Point conditioning and inference

`target_point` remains an input feature of `StructuredSceneEncoder` in both training and inference. The inference selector is unchanged: among valid generated K2 trajectories, it selects the terminal position closest to `target_point`.

During training, the new objective uses the expert trajectory only to route expert supervision to the best-matching Anchor. This is a training-only assignment because expert future trajectories are unavailable at inference. The Target Point remains part of the scene condition and the goal-endpoint selector remains the inference policy. The implementation logs the disagreement between the Anchor-assigned expert mode and the goal-endpoint-selected mode.

## 3. Anchor execution-space assignment

For each sample and each valid Anchor mode, the corresponding `coarse_controls` are passed through the same frozen Koopman K2 execution model and the same observed context as generated controls:

\[
\tau_A^m = \mathcal P\left(\mathcal F_{K2}(h,o,\ell,\mu,U_A^m)\right).
\]

The Anchor rollout is computed under `torch.no_grad()`. K2 parameters remain frozen. The expert-supervised mode is assigned by full-horizon execution-space ADE:

\[
m^* = \arg\min_{m\in\mathcal V}
\frac{1}{H}\sum_{t=1}^{H}
\left\|p_{A,t}^{m}-p_{GT,t}\right\|_2.
\]

The generated trajectory at the same mode index receives the expert trajectory MSE. Invalid Anchor modes are excluded from assignment.

## 4. Mode-preserving Anchor consistency

Every valid generated mode except the expert-assigned mode is matched to its same-index K2-executed Anchor:

\[
\mathcal L_A = \frac{1}{|\mathcal V\setminus\{m^*\}|}
\sum_{m\in\mathcal V\setminus\{m^*\}}
\frac{1}{H}\sum_{t=1}^{H}
\left\|\hat p_t^m-p_{A,t}^{m}\right\|_2^2.
\]

The expert-assigned mode is excluded from Anchor consistency so its trajectory can be refined toward the expert. The default Anchor consistency weight is `0.1`; it is an initial experimental value, not a tuned optimum.

## 5. Auxiliary losses over all valid modes

The new scheme evaluates the following objectives for every valid generated mode:

- Background-vehicle collision and TTC.
- Inter-truck collision and TTC.
- Road-boundary violation.
- LTR stability penalty when enabled.
- Centerline progress when enabled.
- Relative VT-CPFM fuel objective.

Safety modes are processed one at a time to limit peak memory. Losses and rates are averaged over all valid `(sample, mode)` pairs. Invalid modes do not contribute.

## 6. Relative fuel objective

Each generated mode is compared with its own Anchor K2 execution speed profile, using identical payload and calibrated VT-CPFM parameters:

\[
\mathcal L_{fuel}^{rel} = \frac{1}{|\mathcal V|}
\sum_{m\in\mathcal V}
\left(\frac{F(\hat\tau^m)}{\operatorname{sg}(F(\tau_A^m))+\epsilon}-1\right).
\]

A negative value indicates lower estimated fuel consumption than the paired Anchor; zero indicates parity. Anchor fuel is detached from autograd. The new YAML starts with `relative_weight: 0.1` and `relative_epsilon: 1e-8`. The original absolute fuel term remains unchanged in the baseline YAML.

## 7. Diagnostics

The trainer records:

- `anchor_consistency_loss`
- `anchor_execution_gap_m`: sparse geometric Anchor versus its K2-executed reference
- `fuel_relative_loss`, `fuel_relative_ratio`, `fuel_improvement_pct`
- `expert_anchor_ade`
- `goal_selected_ade`
- `expert_goal_mode_disagreement`
- `safety_mode_count`

`selected_mode` in training outputs remains the goal-endpoint mode for compatibility with the inference selector; `expert_mode` separately identifies the mode receiving expert supervision.

## 8. Verification

The following focused regression command was run against the updated project:

```bash
python -m pytest tests/test_trace_anchor_consistent_loss.py tests/test_a1_online_runtime.py tests/test_fixed_anchor_manager.py tests/test_diffusion_resume.py -q
```

Result: **21 passed, 4 warnings**. The warnings are the existing PyTorch Transformer nested-tensor warning. Python compilation also passed for the modified planner, trainer, and test modules.

The Anchor-assigned tests cover full-horizon K2 Anchor ADE assignment with invalid-mode masking, expert/goal selector disagreement, auxiliary losses over all valid modes, frozen and differentiable K2 execution, relative fuel behavior, and YAML scheme selection. The integration test reconstructs a frozen K2 checkpoint and verifies denoiser gradients without K2 parameter gradients.

A broader collection attempt was blocked by the test environment's missing optional `gymnasium` package and a test-module shim interaction; it is not reported as a passing full-suite run.

These are implementation and smoke-level checks. They do not establish that the new objective improves planning performance; that requires controlled training and held-out evaluation against the original baseline.

## 9. Training commands

```bat
python scripts/training/run_diffusion.py ^
  --config configs/train/diffusion.yaml ^
  --max-epochs 60

python scripts/training/run_diffusion.py ^
  --config configs/train/ablation/diffusion_anchor_consistent.yaml ^
  --max-epochs 60
```

Use the same dataset/splits, seed, A1 configuration, frozen K2 checkpoint, model settings, optimizer, and training duration. Keep the two output directories separate and do not resume one scheme from the other's checkpoint.
