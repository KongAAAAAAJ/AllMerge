"""Validate anchor-independent TRACE raw expert shards for A1 readiness."""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np

REQUIRED = {
    "ego_state", "agent_states", "agent_valid_mask", "map_polylines", "map_valid_mask",
    "target_point", "target_lane_polyline", "expert_trajectory_xy", "future_trajectory_dense",
    "expert_trajectory_10hz_xy", "expert_trajectory_10hz_time_s",
    "road_lane_geometry", "road_lane_point_mask", "road_lane_mask", "road_lane_index",
    "road_lane_meta", "road_lane_line_types", "road_lane_connections",
    "road_current_lane_slot", "road_target_lane_slot", "road_current_lane_index",
    "road_target_lane_index", "road_forward_preview_m", "ego_pose_world", "ego_dimensions",
}
FORBIDDEN = {"coarse_controls", "coarse_trajectories", "mode_valid_mask", "target_mode", "target_semantic"}


ROOT = Path(__file__).resolve().parents[2]


def main():
    p=argparse.ArgumentParser(); p.add_argument("dataset_dir", type=Path); a=p.parse_args()
    dataset_dir = a.dataset_dir.expanduser()
    if not dataset_dir.is_absolute():
        dataset_dir = ROOT / dataset_dir
    dataset_dir = dataset_dir.resolve()
    paths=sorted(dataset_dir.glob("episode_*.npz"))
    if not paths: raise FileNotFoundError(f"No episode_*.npz under {a.dataset_dir}")
    rows=[]; errors=[]
    for path in paths:
        with np.load(path, allow_pickle=False) as z:
            keys=set(z.files); missing=sorted(REQUIRED-keys); forbidden=sorted(FORBIDDEN&keys)
            if missing: errors.append(f"{path.name}: missing {missing}")
            if forbidden: errors.append(f"{path.name}: anchor-dependent fields unexpectedly present {forbidden}")
            if not missing:
                n=len(z["expert_trajectory_xy"])
                if z["road_lane_geometry"].shape[0] != n: errors.append(f"{path.name}: geometry sample count mismatch")
                if z["expert_trajectory_xy"].shape[1:] != (8,2): errors.append(f"{path.name}: sparse expert shape invalid")
                if z["future_trajectory_dense"].shape[1:] != (40,2): errors.append(f"{path.name}: 40-step dense expert shape invalid")
                if z["expert_trajectory_10hz_xy"].shape[1:] != (41,2): errors.append(f"{path.name}: 10-Hz visualization path shape invalid")
                if not np.isfinite(z["road_lane_geometry"]).all(): errors.append(f"{path.name}: nonfinite road geometry")
                if not np.all(z["road_lane_mask"].any(axis=1)): errors.append(f"{path.name}: no road lanes in one or more samples")
                if not np.all(z["road_current_lane_slot"] >= 0): errors.append(f"{path.name}: missing current lane slot")
                if not np.all(z["road_target_lane_slot"] >= 0): errors.append(f"{path.name}: missing target lane slot")
                rows.append({"shard":path.name,"samples":n,"lanes_mean":float(z["road_lane_mask"].sum(1).mean()),
                             "preview_min_m":float(z["road_forward_preview_m"].min()),
                             "preview_max_m":float(z["road_forward_preview_m"].max())})
    report={"schema":"trace_raw_expert_road_geometry_v1","valid":not errors,"episodes":rows,"errors":errors}
    (dataset_dir/"raw_dataset_validation.json").write_text(json.dumps(report,indent=2),encoding="utf-8")
    print(json.dumps(report,indent=2))
    return 0 if not errors else 2

if __name__=="__main__": raise SystemExit(main())
