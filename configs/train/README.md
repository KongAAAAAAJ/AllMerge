# Diffusion training configs

`diffusion.yaml` contains shared settings. Experiment configs inherit it and
override only differences; `extends` is relative to the child YAML file.
Nested mappings merge; lists and scalar values are replaced. Dataset,
checkpoint, and output paths remain project-root relative.

Run from `all_merge/`:

```bash
# Default dynamic anchors
python scripts/training/run_diffusion.py --config configs/train/diffusion.yaml

# A0 fixed anchors (requires the fixed bank)
python scripts/data/build_a0_fixed_anchors.py
python scripts/training/run_diffusion.py --config configs/train/ablation/diffusion_a0_fixed.yaml
```
