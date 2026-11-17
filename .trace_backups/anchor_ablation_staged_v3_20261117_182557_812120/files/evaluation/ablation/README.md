在 `all_merge/` 目录下运行。

锚点消融：

```bash
# Anchor quality only; no diffusion checkpoints
python evaluation/ablation/run_anchor.py --stage anchors --methods A0 A1 A2

# Evaluate one anchor type
python evaluation/ablation/run_anchor.py --stage anchors --methods A2 --a2-scorer outputs/trace_adaptive_anchors_v2/best.pt

# After diffusion training: feasibility only, one or several methods
python evaluation/ablation/run_anchor.py --stage feasibility --methods A0 --checkpoint-a0 PATH_A0

# Both stages using trained diffusion checkpoints
python evaluation/ablation/run_anchor.py --stage all --methods A0 A1 A2 --checkpoint-a0 PATH_A0 --checkpoint-a1 PATH_A1 --checkpoint-a2 PATH_A2
```

燃油感知消融：

```bash
python evaluation/ablation/run_fuel_aware.py --config configs/train/diffusion.yaml --checkpoint F0=outputs\fuel_ablation\F0\best.pt --checkpoint F3=outputs\fuel_ablation\F3\best.pt --split test
```
