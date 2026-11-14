在 `all_merge/` 目录下运行。

锚点消融：

```bash
python evaluation/ablation/run_anchor.py --config configs/evaluation/anchor_ablation.yaml --checkpoint-a0 PATH_A0 --checkpoint-a1 PATH_A1 --checkpoint-a2 PATH_A2
```

燃油感知消融：

```bash
python evaluation/ablation/run_fuel_aware.py --config configs/train/diffusion.yaml --checkpoint F0=PATH_F0 --checkpoint F3=PATH_F3 --split test
```
