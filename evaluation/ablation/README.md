# TRACE Ablation Evaluation

The Anchor Ablation entry point is `run_anchor.py`. It supports staged execution:
`--stage anchors` (four anchor-quality metrics), `--stage feasibility` (planning
minADE and pre-projection feasibility), and `--stage all` (all six metrics).
Iterative JTP projection is disabled; `Proj. Corr.` remains `--`. See
`anchor_ablation/README.md` for current raw-dataset commands and outputs.
