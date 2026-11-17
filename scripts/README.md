# Scripts

## Unified TRACE expert collection

There is one public expert-data collection entry point:

```bash
python scripts/data/collect_expert_dataset.py --check-only
python scripts/data/collect_expert_dataset.py
```

All collection, process isolation, resume handling, pre-anchor episode staging,
A1 anchor generation, final shard packaging and dataset splitting are managed by
this command. Do not call a separate raw-collection or A1-building script.

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
