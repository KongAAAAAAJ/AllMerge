# TRACE evaluation

**Current entry point:** `python -m evaluation.eval_diffusion_open_loop` now
loads a TRACE **control-space** Diffusion checkpoint and scores its frozen-K2
40-step trajectories; it no longer loads W2 trajectory checkpoints. See
[`trace_planning/README.md`](trace_planning/README.md) for open-loop and
actual TruckSim closed-loop evaluation procedures.

`evaluation/ablation/` retains the existing Anchor and Fuel experiments.
All current evaluators read raw expert episodes and generate A1 anchors online.
The retired W1/W2 trajectory-space evaluation and paired dataset loaders have been removed.
