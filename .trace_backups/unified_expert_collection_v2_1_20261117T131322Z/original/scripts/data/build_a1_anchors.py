"""Build A1 anchors using the shared pool and control.model.num_modes."""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))
from highway_env.planner.a1_anchor_builder import A1AnchorBuilder, mode_names


def _configured_num_modes(diffusion_config: Path) -> int:
    import yaml
    raw = yaml.safe_load(diffusion_config.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("Diffusion config must be a YAML mapping")
    value = (((raw.get("control") or {}).get("model") or {}).get("num_modes", 10))
    value = int(value)
    if value <= 0:
        raise ValueError(f"control.model.num_modes must be positive, got {value}")
    return value


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input-dir', required=True)
    p.add_argument('--output-dir', required=True)
    p.add_argument('--config', default='configs/anchors/a1.yaml')
    p.add_argument('--diffusion-config', default='configs/train/diffusion.yaml')
    p.add_argument('--num-modes', type=int, default=None,
                   help='Optional override; defaults to control.model.num_modes in diffusion config.')
    args = p.parse_args()
    import yaml
    def project_path(x):
        q = Path(x).expanduser()
        return q.resolve() if q.is_absolute() else (ROOT/q).resolve()
    cfg_path, input_dir, output_dir, diffusion_path = map(
        project_path, (args.config, args.input_dir, args.output_dir, args.diffusion_config)
    )
    cfg = yaml.safe_load(cfg_path.read_text(encoding='utf-8'))['a1']
    num_modes = int(args.num_modes) if args.num_modes is not None else _configured_num_modes(diffusion_path)
    if num_modes <= 0:
        raise ValueError('--num-modes must be positive')
    cfg['num_selected'] = num_modes
    builder = A1AnchorBuilder(cfg)
    source_files = sorted(input_dir.glob('episode_*.npz'))
    if not source_files: raise FileNotFoundError(f'No episode_*.npz in {input_dir}')
    output_dir.mkdir(parents=True, exist_ok=True)
    total, valid_total = 0, np.zeros(builder.num_selected, np.int64)
    for source in source_files:
        with np.load(source, allow_pickle=False) as data:
            arrays = {k: data[k] for k in data.files}
            required = ('road_lane_geometry','road_lane_point_mask','road_lane_mask',
                        'road_current_lane_slot','road_target_lane_slot','road_lane_connections',
                        'road_lane_meta','ego_state')
            missing = [k for k in required if k not in arrays]
            if missing: raise KeyError(f'{source.name} missing A1 fields: {missing}')
            outputs = [builder.build({k: v[i] for k,v in arrays.items()}) for i in range(len(arrays['ego_state']))]
            for key in outputs[0]: arrays[key] = np.stack([x[key] for x in outputs])
            valid_total += arrays['a1_valid_mask'].sum(axis=0)
            total += len(outputs)
            dest = output_dir / source.name
            np.savez_compressed(dest, **arrays)
            print(f'[A1] {source.name}: {len(outputs)} samples, valid/mode={arrays["a1_valid_mask"].sum(axis=0).tolist()}')
    manifest = {'builder':'A1_SHARED_CANDIDATE_POOL_DIVERSITY_V4','source':str(input_dir),
                'samples':total,'num_modes':builder.num_selected,
                'mode_names':mode_names(builder.num_selected),
                'valid_count_by_mode':valid_total.tolist(),
                'vehicle_dimensions_m':[builder.length,builder.width],
                'candidate_pool_size':builder.candidate_builder.num_candidates,
                'candidate_generation':builder.candidate_builder.resolved_config,
                'selection':'joint control/trajectory diversity; no safety scorer',
                'trajectory_source':'shared road-conditioned kinematic candidate pool',
                'control_source':'sampled control sequences with forward kinematic propagation'}
    (output_dir/'a1_manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    print(f'[DONE] {total} samples, K={builder.num_selected} -> {output_dir}')
    return 0
if __name__ == '__main__': raise SystemExit(main())
