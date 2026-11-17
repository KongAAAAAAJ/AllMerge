# Scripts

## Unified TRACE expert collection

There is one public expert-data collection entry point:

```bash
python scripts/data/collect_expert_dataset.py --check-only
python scripts/data/collect_expert_dataset.py
```

The collection command writes only persistent anchor-independent raw episodes to
`outputs/expert_dataset/expert_trucksim_k2_raw_100k/episodes/`, together with
resume state and reports. Build final training shards separately:

```bash
python scripts/data/build_a1_anchors.py --check-only
python scripts/data/build_a1_anchors.py
python scripts/diagnostics/validate_expert_dataset.py
```

The A1 builder uses `configs/anchors/a1.yaml`, reads the completed raw dataset
without modifying it, and creates `expert_trucksim_k2_anchor_pair_100k` with
paired A1 control/trajectory anchors, K2 context and deterministic episode
splits. Use a new `--output-dir` for each A1 parameter variant. The final
training format is unchanged. Do not run collection and A1 build concurrently.
Legacy standard datasets and separate raw-collection commands are unsupported.

## Training and evaluation

For the current TRACE control diffusion model, evaluate via
`python -m evaluation.eval_diffusion_open_loop --checkpoint PATH_TO_CONTROL_PT`
(10-Hz K2 open-loop accuracy), or
`python -m evaluation.trace_planning.closed_loop --method TRACE --checkpoint PATH`
(native TruckSim closed-loop paper metrics). See
`evaluation/trace_planning/README.md` for complete instructions.

`scripts/run_diffusion_open_loop_eval.sh` calls the updated control-space
entry point. Trajectory-space and control-space Diffusion training both read
the same final A1/K2 expert dataset configured in `configs/train/diffusion.yaml`.
