# TRACE evaluation

**Current entry point:** `python -m evaluation.eval_diffusion_open_loop` now
loads a TRACE **control-space** Diffusion checkpoint and scores its frozen-K2
40-step trajectories; it no longer loads W2 trajectory checkpoints. See
[`trace_planning/README.md`](trace_planning/README.md) for open-loop and
actual TruckSim closed-loop evaluation procedures.

`evaluation/ablation/` retains the existing Anchor and Fuel experiments.
`legacy_open_loop_cli.py`, `open_loop_eval.py`, `open_loop_metrics.py` and
`open_loop_visualization.py` remain in place strictly for historical
trajectory-space experiment compatibility. They are not used by TRACE and
should not be used for the proposed paper's comparison table.
