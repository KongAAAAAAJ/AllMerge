# A2 safety scorer

Training accepts only completed `trace_expert_raw_k2_v2` datasets with
`reports/manifest.json` and `episodes/episode_*.npz`. Episode splits use the
shared `RawExpertDataset` loader. Legacy shard datasets are unsupported.

A2 generates the full candidate pool using the same candidate generator and
raw road conversion as the current A1 builder. It does not require precomputed
A1 anchors: the scorer learns to rank the full pool before selecting K anchors.
K comes from `control.model.num_modes` in the diffusion configuration;
`top_r_fraction` controls the ranking shortlist. Safety labels use observed
traffic and road features, not expert future trajectory labels.

From the `all_merge` directory:

```powershell
python -m highway_env.planner.anchor_scoring.train --config configs/train/adaptive_anchors.yaml --max-epochs 1 --max-batches 2 --max-samples 64 --output-dir outputs/anchor_ablation/ckpt/A2_smoke --cache-dir outputs/anchor_ablation/cache_A2_smoke
```

For full training, omit the three `--max-*` options and use separate output
and cache directories. Raw cache fingerprints include episode inventory,
split manifest, candidate settings, teacher settings and precompute device.
Old cache versions are rebuilt automatically by the training entry point.
