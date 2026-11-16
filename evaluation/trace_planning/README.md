# TRACE Paper Evaluation V3 (2026-11)

**Two canonical entry points**. These evaluate the frozen K2 control diffusion
checkpoint without changing model weights or training code. The accepted paper
layout is **one two-panel Table** (open and closed), **four open-loop grouped
figures**, and **one native TruckSim closed-loop grouped figure**.

## 1. Open-loop: Control Diffusion, predicted K2 state

```bat
python -m evaluation.eval_diffusion_open_loop ^
  --checkpoint outputs\fuel_ablation\F3\best.pt ^
  --num-samples 64 --batch-size 2 --seeds 0 ^
  --output-dir outputs\evaluation\paper_open_F3_smoke
```

For verified paper comparisons, evaluate **F0 and F3 on the exact same held-out
val dataset, sample IDs and inference seeds**. Omit `--num-samples` for the full
split. In the manuscript, panel (a) rows are dynamic control anchors (propagated
through **the same frozen K2**), F0, and F3. See `--help` for parameters.

The evaluator writes:

- `summary.json`, `manifest.json`, `samples.csv`, `metrics_by_seed.csv`;
- `candidate_metrics.csv`, `control_metrics.csv`, `anchor_metrics.csv`,
  `dynamics_metrics.csv`, `error_by_horizon.csv`, `paper_open_loop_panel.json`;
- `figures/open_loop_trajectory_distribution.pdf/png`;
- `figures/open_loop_error_statistics.pdf/png`;
- `figures/open_loop_control_profiles.pdf/png`;
- `figures/open_loop_chassis_responses.pdf/png`;
- `figures/representative_samples.npz`, `figures/representative_selection.json`;
- `report.md`.

All main displacement errors use 40 steps of 0.1s; `minade_k_m/minfde_k_m`
minimize independently across the **valid** diffusion candidates. Historical
`minade8_m/minfde8_m` are retained only as legacy eight-**time-point** metrics,
not used in the paper table. Selected mode uses the deployment goal-endpoint
selection rule without GT. The anchor row uses its own goal-endpoint selection
without GT. The figure compares the corresponding paired anchor for the
**diffusion-selected** mode. Fuel from VT-CPFM and stability from K2 are
**predictions**, not executed TruckSim measurements. Inference throughput does
not equal end-to-end closed-loop planning time.

For a paper run use separate output folders and the same commands with
`--seeds 0,1,2,3,4`, `--split val` and no `--num-samples`. F0/F3 folder labels
are assigned explicitly in the closed-loop command below, and are paired on
all recorded sample identifiers before being placed in the combined table.

## 2. Closed-loop: native TruckSim execution and combined paper artifacts

```bat
python -m evaluation.eval_diffusion_closed_loop ^
  --checkpoint outputs\fuel_ablation\F3\best.pt ^
  --methods "TRACE,Expert Planner" ^
  --scenarios straight,curved,merge_in,merge_out ^
  --episodes 10 --seed-start 0 ^
  --trucksim-case L4_MU08 ^
  --copy-dlls ^
  --open-loop-dir F0=outputs\evaluation\paper_open_F0 ^
  --open-loop-dir F3=outputs\evaluation\paper_open_F3 ^
  --output-dir outputs\evaluation\TRACE_paper_v3
```

The *same* closed-loop entry supports aggregation of **previously executed**
native episodes without new TruckSim runs:

```bat
python -m evaluation.eval_diffusion_closed_loop ^
  --runs outputs\evaluation\TRACE_paper_v3\episodes\TRACE ^
  --runs outputs\evaluation\TRACE_paper_v3\episodes\Expert_Planner ^
  --open-loop-dir F0=outputs\evaluation\paper_open_F0 ^
  --open-loop-dir F3=outputs\evaluation\paper_open_F3 ^
  --output-dir outputs\evaluation\TRACE_paper_v3
```

The folder `paper_results/` contains:

- `planning_comparison_table.tex`: **one** manuscript table with both panels;
- `paper_metrics.csv`, `comparison_summary.json`, `report.md`;
- `figures/open_loop_*.pdf/png` copied from the verified F3 open-loop run;
- `figures/planning_comparison.pdf/png` showing *executed* trajectories on the
  left and *signed measured spacing + target spacing* on the right.

`--check-only` prints commands and settings; it does **not** prove native DLL
availability. Only `TRACE` and `Expert Planner` adapters are implemented in the
provided source snapshot. Cooperative OCP and TransFuser columns remain `--`
until genuine native TruckSim runs with matching schema and physical scenario
conditions are added. The repository currently has four registered scenarios,
not the previously proposed five.

**Important:** The paper comparison **rejects** TRACE records that fell back
to Polynomial after JTP history warmup, even when a diagnostic run was made
with `--allow-unexecuted post-warmup JTP control`. Existing failing TRACE episode JSON is not acceptable.
No outcome is invented, and incomplete closed-loop runs are not published as valid TRACE aggregates.

## Metrics mapped to manuscript

| Table panel | Methods | Six columns |
|---|---|---|
| (a) open-loop | Dynamic anchors / F0 / F3 | ADE, FDE, minADE_K, minFDE_K, K2 stability horizon violation (%), K2 VT-CPFM Fuel (L/100km) |
| (b) closed-loop | Expert Planner / Cooperative OCP / TransFuser / TRACE | Success (%), Collision (%), executed stability vehicle-step violation (%), signed-center-gap absolute error (m), executed VT-CPFM Fuel (L/100km), full planning-call time (ms) |

The open-loop K2 stability metric is horizon-level; the closed-loop metric
is executed vehicle-time-step-level. The two are explicitly different.
The closed-loop spacing calculation projects the relative position onto the
preceding vehicle heading and is an approximation on strongly curved roads.
Fuel is model-estimated via VT-CPFM from executed velocities, not a direct
fuel-flow sensor reading. Validate this definition before paper submission.

The five PDF names agree exactly with
`TRACE_comparative_evaluation_compact.tex`. The figures must be generated from
real model predictions and real native episode recordings on the user's machine.


## Local JTP execution policy (2026-11-16 source merge)

The merged `closed_loop.py` preserves the latest local strict policy: the first
four policy cycles are reserved for K2 history warm-up; after warm-up, every
executed TRACE control sequence must come from the JTP-returned control. Any
post-warm-up unexecuted JTP step invalidates the episode and aborts paper
aggregation. No `--allow-fallback` option is provided. JTP status/context
counters, first failures, and per-step diagnostics remain in each episode JSON.
