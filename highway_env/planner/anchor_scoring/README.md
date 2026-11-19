# A2 safety scorer

The public entry is `scripts/data/train_a2_anchors.py`; training logic lives in
`highway_env/planner/anchor_scoring/training.py`. Configuration remains in
`configs/train/a2_scorer.yaml`.

`configs/anchors/a2.yaml` owns candidate generation and Top-R/diversity selection
settings shared by scorer training, diffusion and anchor evaluation. The scorer
YAML only owns its data, optimizer and teacher settings. Diffusion A2 uses
`configs/train/ablation/diffusion_a2_adaptive.yaml` and the resulting scorer checkpoint.

Training accepts only completed `trace_expert_raw_k2_v2` datasets with
`reports/manifest.json` and `episodes/episode_*.npz`. Episode splits use the
shared `RawExpertDataset` loader. Legacy shard datasets are unsupported.

A2 generates the full candidate pool using the same candidate generator and
raw road conversion as the current A1 online builder. It does not require precomputed
A1 anchors: the scorer learns to rank the full pool before selecting K anchors.
K comes from `control.model.num_modes` in the diffusion configuration;
`top_r_fraction` controls the ranking shortlist. Safety labels use observed
traffic and road features, not expert future trajectory labels.

From the `all_merge` directory:

```powershell
python scripts/data/train_a2_anchors.py --config configs/train/a2_scorer.yaml --max-epochs 1 --max-batches 2 --max-samples 64 --output-dir outputs/anchor_ablation/ckpt/A2_smoke --cache-dir outputs/anchor_ablation/cache_A2_smoke
```

For full training, omit the three `--max-*` options and use separate output
and cache directories. Raw cache fingerprints include episode inventory,
split manifest, candidate settings, teacher settings and precompute device.
Old cache versions are rebuilt automatically by the training entry point.

Road checks use the geometric union of raw `road_lane_geometry` left/right edges,
with the raw lane and point masks. Internal lane boundaries do not count as road
boundaries; disconnected areas and holes remain outside. DAC requires every
vehicle footprint circle to fit inside this union. Anchor teacher road labels
retain center-point checks and mean squared outside distance (hard threshold
`1e-4 m`). Shapely 2.x is required. A2 cache V5 invalidates old road labels;
rebuild caches and retrain the scorer before using the revised teacher.

After this road-definition change, run from the `all_merge` directory:

```powershell
python scripts/data/train_a2_anchors.py --config configs/train/a2_scorer.yaml --prepare-cache-only
python scripts/data/train_a2_anchors.py --config configs/train/a2_scorer.yaml --output-dir outputs/anchor_ablation/ckpt/A2_road_union
```

Cache V5 automatically uses a different fingerprint; the original candidate/label
cache is not overwritten. Run fresh training rather than resuming an old scorer.
Then rerun anchor/planning evaluation for A0/A1/A2/Expert with the same samples
and inference seeds, passing `--a2-scorer outputs/anchor_ablation/ckpt/A2_road_union/best.pt`
and a new output directory. This scorer override also applies to `all` and
`feasibility` stages; diffusion weights are not retrained by the override.
Old tables must be regenerated under the new road definition. Raw expert data,
A0 anchors, A1 generation and K2 weights do not need to be rebuilt for this fix.
