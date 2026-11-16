# TRACE Motion Planning Evaluation (November 2026)

This package **replaces** the W2 model logic behind `evaluation.eval_diffusion_open_loop`.
The official planner is `ControlDiffusionPlanner` with 40 control commands at 10 Hz,
followed by frozen conditional Koopman K2 execution. The checkpoint itself
provides the model, execution and anchor architectures; the training YAML
supplies the data root and VT-CPFM calibration; the evaluation YAML specifies
sample size, inference seeds and scenario settings.

## 0. Required files

- `configs/train/diffusion.yaml`: `output_space: control` and calibrated
  `control.losses.fuel.parameters`.
- `configs/evaluation/trace_planning.yaml`: evaluation-only protocol.
- `outputs/final_compare/control_v0/best.pt`: example checkpoint location **only**;
  substitute the real trained `.pt` path.
- `outputs/expert_dataset/expert_trucksim_k2_anchor_pair_100k`:
  expert shards with paired `coarse_controls [N,10,40,2]`,
  `future_trajectory_dense [N,40,2]` and K2 historical context.

### 1. Open loop, smoke test (Windows CMD)

From `D:\KONG_Files\AllMerge\all_merge`:

```bat
python -m evaluation.eval_diffusion_open_loop ^
  --checkpoint outputs\final_compare\control_v0\best.pt ^
  --num-samples 64 --batch-size 2 --seeds 0 ^
  --output-dir outputs\evaluation\trace_open_loop_smoke
```

After smoke test, run the full validation split with five seeds:

```bat
python -m evaluation.eval_diffusion_open_loop ^
  --checkpoint outputs\final_compare\control_v0\best.pt
```

Outputs: `samples.csv`, `metrics_by_seed.csv`, `summary.json`, `manifest.json`.
Metrics: dense ADE/FDE, sparse ADE/FDE, minADE8/minFDE8, goal error,
VT-CPFM fuel from predicted K2 states, and K2-predicted four-condition stability
violations. The open-loop Stability rate **is not** executed TruckSim stability.
No `Success`, `Collision`, true `Spacing` or TruckSim closed-loop fuel is calculated
here. JTP requires live geometry and is **not enabled** in open-loop mode.

### 2. Closed loop: native TruckSim

The current code includes `straight`, `curved`, `merge_in`, `merge_out` scenarios
only. It does **not** yet contain the paper's fifth scenario, emergency braking,
lane narrowing, Cooperative OCP, or a TransFuser adaptation.

In Windows CMD, check the protocol first:

```bat
python -m evaluation.trace_planning.closed_loop --check-only ^
  --method TRACE --checkpoint outputs\final_compare\control_v0\best.pt
```

Configure real `.sim` and `.dll` paths in `configs/koopman/collect.yaml`, and
verify the loaded physical plant, wheel-force signals and fuel coefficients.
Smoke test one native episode in its own subprocess:

```bat
python -m evaluation.trace_planning.closed_loop ^
  --method TRACE ^
  --checkpoint outputs\final_compare\control_v0\best.pt ^
  --scenarios straight --episodes 1 --seed-start 0 ^
  --trucksim-case L4_MU08 --copy-dlls ^
  --output-dir outputs\evaluation\trace_closed_loop\TRACE
```

Then use the same case, scenes and seed range for a true Expert Planner baseline:

```bat
python -m evaluation.trace_planning.closed_loop ^
  --method "Expert Planner" --copy-dlls ^
  --output-dir outputs\evaluation\trace_closed_loop\Expert
```

`closed_loop.py` writes one `.json` and one `.npz` per episode, `episodes.csv`
and `closed_loop_summary.json` if and only if the entire set was evaluated
`closed_loop.py` writes one `.json` and one `.npz` per episode, `episodes.csv`
and `closed_loop_summary.json` if and only if the entire set was evaluated
successfully. TRACE executes every control sequence returned by JTP after
the initial four 10-Hz K2 history warmup samples. Projector-internal constraint
violations do not trigger a switch to nominal Diffusion or Polynomial control.
A JTP exception aborts the episode; no fallback or hybrid policy is implemented
or accepted in the TRACE comparison.

### 3. Comparison table and figure

```bat
python -m evaluation.trace_planning.comparison ^
  --runs outputs\evaluation\trace_closed_loop\TRACE ^
  --runs outputs\evaluation\trace_closed_loop\Expert
```

Outputs: `planning_comparison_table.tex` (blank missing baselines),
`paper_metrics.csv`, `comparison_summary.json`, and if `.npz` executed traces
exist, `planning_comparison_available_scenarios.pdf/png`.

To generate the final four-method, five-scenario paper table, additional
Cooperative OCP and TransFuser policies **must be executed in TruckSim** and
the five representative scenario classes must be implemented. Their episode
records must use the same physical case, scene and seed IDs and include all
six paper fields. When available, use `--complete-paper` to enforce coverage.

### 4. Metric definitions

| Metric | Source | Definition |
|---|---|---|
| Success (%) | TruckSim terminal state | Maneuver task success and no crash/off-road; episode rate |
| Collision (%) | TruckSim episode | Fraction of episodes with any platoon vehicle collision |
| Stability Viol. (%) | TruckSim states + wheel forces | Fraction of vehicle/time steps exceeding at least one yaw/sideslip/roll/LTR JTP limit |
| Spacing Error (m) | TruckSim XY and headings | Mean absolute preceding-vehicle longitudinal center gap minus `platoon_spacing` (m) |
| Fuel (L/100km) | TruckSim speed and distance | VT-CPFM evaluated on **executed** TruckSim longitudinal speeds; total fuel / total driven distance |
| Planning Time (ms) | runtime instrumented around `env.action_type.act` | Mean wall-clock time per full platoon planning cycle, after four warmup cycles, CUDA synchronized |

**Important:** The gap projection uses the preceding vehicle's heading. For
strongly curved or network-transition cases it is a local approximation, not a
full Frenet s-coordinate; refine this geometric definition if needed in the
final paper. Fuel depends on the supplied VT-CPFM coefficients and is
model-estimated, not a TruckSim fuel-flow sensor measurement. The published
stability threshold YAML must be frozen before comparisons.

### 5. Reproducibility contracts

- Do not silently mix 8-point trajectory planner and 40-step control planner.
- All comparison methods should use the same TruckSim `.sim`, physical case,
  10-Hz scenario, 3-vehicle group action, seed, and completion definitions.
- No random unpaired episode sets or missing baseline values are substituted.
- Validate that every scene's `task_success` matches the intended paper maneuver.
- The included code was statically checked and unit-tested, but the uploaded ZIP
  has no trained weights or TruckSim DLLs. Full native execution still needs
  validation on your Windows/TruckSim machine.
