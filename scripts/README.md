# Scripts

For the current TRACE control diffusion model, evaluate via
`python -m evaluation.eval_diffusion_open_loop --checkpoint PATH_TO_CONTROL_PT`
(10-Hz K2 open-loop accuracy), or
`python -m evaluation.trace_planning.closed_loop --method TRACE --checkpoint PATH`
(native TruckSim closed-loop paper metrics). See
`evaluation/trace_planning/README.md` for complete instructions.

`scripts/run_diffusion_open_loop_eval.sh` calls the updated control-space
entry point. Legacy trajectory-space evaluations remain archived separately.
