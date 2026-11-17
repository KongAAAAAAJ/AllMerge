"""Build A1 lattice anchors offline from anchor-independent raw expert episodes."""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))
from highway_env.planner.a1_anchor_builder import A1AnchorBuilder, MODE_NAMES


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input-dir', required=True)
    p.add_argument('--output-dir', required=True)
    p.add_argument('--config', default='configs/anchors/a1.yaml')
    args = p.parse_args()
    import yaml
    def project_path(x):
        q = Path(x).expanduser()
        return q.resolve() if q.is_absolute() else (ROOT/q).resolve()
    cfg_path, input_dir, output_dir = map(project_path, (args.config, args.input_dir, args.output_dir))
    cfg = yaml.safe_load(cfg_path.read_text(encoding='utf-8'))['a1']
    builder = A1AnchorBuilder(cfg)
    source_files = sorted(input_dir.glob('episode_*.npz'))
    if not source_files: raise FileNotFoundError(f'No episode_*.npz in {input_dir}')
    output_dir.mkdir(parents=True, exist_ok=True)
    total, valid_total = 0, np.zeros(10, np.int64)
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
    manifest = {'builder':'A1_LATTICE_ROAD_BOUNDED_V2','source':str(input_dir),
                'samples':total,'mode_names':MODE_NAMES,'valid_count_by_mode':valid_total.tolist(),
                'vehicle_dimensions_m':[builder.length,builder.width],
                'trajectory_source':'explicit road lane center/edge geometry',
                'control_source':'inverse kinematics from bounded lattice trajectories'}
    (output_dir/'a1_manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    print(f'[DONE] {total} samples -> {output_dir}')
    return 0
if __name__ == '__main__': raise SystemExit(main())
