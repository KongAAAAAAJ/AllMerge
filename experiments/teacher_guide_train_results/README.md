# Teacher Guide Train Results

Independent read-only paired-t0 comparison of frozen Pretrain, a post-training planner checkpoint, and an accepted PGD Teacher trajectory. See `README_INSTALL.md` at ZIP root for Windows installation, invocation, diagnostics, and safety limitations.

All code and output for this analysis live under `experiments/teacher_guide_train_results/`. The module imports existing project's native denoiser, cached paired replay inputs, spline and V6 PGD task reward directly without modifying them. Checkpoint/bank origin and exact t0 replay consistency are mandatory checks.
