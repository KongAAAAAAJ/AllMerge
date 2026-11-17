"""Lossless, anchor-independent road geometry export for offline A1 Lattice.

Coordinates are stored in the planning ego frame.  Center and both lane-edge
curves are sampled directly from the live lane geometry object; no boundary is
reconstructed later from the compact 32-point model feature.
"""
from __future__ import annotations

import numpy as np


class RoadGeometryExporter:
    DEFAULTS = {
        "max_lanes": 16,
        "max_points_per_lane": 192,
        "point_spacing_m": 1.5,
        "backward_range_m": 30.0,
        "forward_min_m": 120.0,
        "forward_max_m": 300.0,
        "prediction_horizon_s": 4.0,
        "reaction_margin_s": 0.8,
        "distance_margin_m": 35.0,
        "lateral_range_m": 32.0,
        "ego_length_m": 8.0,
        "ego_width_m": 2.6,
    }

    # geometry channels: center_x, center_y, left_x, left_y, right_x,
    # right_y, heading, width, lane_s
    GEOMETRY_DIM = 9

    def __init__(self, env, config=None):
        self.env = env
        self.cfg = dict(self.DEFAULTS)
        if config:
            self.cfg.update(config)
        for key in ("max_lanes", "max_points_per_lane"):
            self.cfg[key] = int(self.cfg[key])
            if self.cfg[key] <= 0:
                raise ValueError(f"road_geometry.{key} must be positive")
        for key in ("point_spacing_m", "forward_min_m", "forward_max_m"):
            self.cfg[key] = float(self.cfg[key])
            if self.cfg[key] <= 0:
                raise ValueError(f"road_geometry.{key} must be positive")

    def build_batch(self, target_lane_indices=None):
        vehicles = list(self.env.controlled_vehicles)
        if target_lane_indices is None:
            target_lane_indices = [None] * len(vehicles)
        if len(target_lane_indices) != len(vehicles):
            raise ValueError("target lane count does not match controlled vehicles")
        samples = [self.build_single(v, t) for v, t in zip(vehicles, target_lane_indices)]
        return {k: np.stack([s[k] for s in samples], axis=0) for k in samples[0]}

    def build_single(self, ego, target_lane_index=None):
        network = self.env.road.network
        current = tuple(ego.lane_index)
        target = tuple(target_lane_index or getattr(ego, "target_lane_index", None) or current)
        lanes_dict = network.lanes_dict()
        if current not in lanes_dict or target not in lanes_dict:
            raise RuntimeError(f"current/target lane missing from road graph: {current}, {target}")

        speed = max(float(getattr(ego, "speed", 0.0)), 0.0)
        forward = speed * (float(self.cfg["prediction_horizon_s"]) + float(self.cfg["reaction_margin_s"])) + float(self.cfg["distance_margin_m"])
        forward = float(np.clip(forward, self.cfg["forward_min_m"], self.cfg["forward_max_m"]))
        backward = float(self.cfg["backward_range_m"])
        lateral = float(self.cfg["lateral_range_m"])

        scored = []
        for raw_idx, lane in lanes_dict.items():
            idx = tuple(raw_idx)
            s0, _ = lane.local_coordinates(ego.position)
            s0 = float(np.clip(s0, 0.0, float(lane.length)))
            p = lane.position(s0, 0.0)
            dx, dy = np.asarray(p, dtype=float) - np.asarray(ego.position, dtype=float)
            c, s = np.cos(float(ego.heading)), np.sin(float(ego.heading))
            x_local, y_local = c * dx + s * dy, -s * dx + c * dy
            relevant = (-backward <= x_local <= forward and abs(y_local) <= lateral + float(lane.width_at(s0)))
            if relevant or idx in (current, target):
                scored.append((0 if idx == current else 1 if idx == target else 2,
                               float(np.hypot(x_local, y_local)), idx, lane, s0))
        scored.sort(key=lambda z: (z[0], z[1], str(z[2])))
        scored = scored[: int(self.cfg["max_lanes"])]
        # Fail closed: never silently truncate away current or target lane.
        selected_indices = [z[2] for z in scored]
        for required in (current, target):
            if required not in selected_indices:
                raise RuntimeError(f"road geometry lane capacity omitted required lane {required}")

        L, P = int(self.cfg["max_lanes"]), int(self.cfg["max_points_per_lane"])
        geom = np.zeros((L, P, self.GEOMETRY_DIM), np.float32)
        point_mask = np.zeros((L, P), bool)
        lane_mask = np.zeros((L,), bool)
        lane_index = np.full((L, 3), "", dtype="U64")
        lane_meta = np.zeros((L, 7), np.float32)  # speed limit, length, s0, s_start, s_end, actual ds, priority
        line_types = np.full((L, 2), -1, np.int16)
        connections = np.zeros((L, L), bool)
        current_slot = target_slot = -1

        for li, (_, _, idx, lane, s0) in enumerate(scored):
            lane_mask[li] = True
            lane_index[li] = [str(v) for v in idx]
            if idx == current:
                current_slot = li
            if idx == target:
                target_slot = li
            s_start = max(0.0, s0 - backward)
            s_end = min(float(lane.length), s0 + forward)
            if s_end <= s_start:
                s_end = min(float(lane.length), s_start + 1e-3)
            requested_n = max(2, int(np.ceil((s_end - s_start) / float(self.cfg["point_spacing_m"]))) + 1)
            n = min(P, requested_n)
            s_values = np.linspace(s_start, s_end, n, dtype=np.float64)
            for pi, ss in enumerate(s_values):
                width = float(lane.width_at(float(ss)))
                center = np.asarray(lane.position(float(ss), 0.0), dtype=float)
                left = np.asarray(lane.position(float(ss), width / 2.0), dtype=float)
                right = np.asarray(lane.position(float(ss), -width / 2.0), dtype=float)
                heading = float(lane.heading_at(float(ss)))
                c, sn = np.cos(float(ego.heading)), np.sin(float(ego.heading))
                def local(q):
                    d = q - np.asarray(ego.position, dtype=float)
                    return np.asarray([c*d[0] + sn*d[1], -sn*d[0] + c*d[1]], dtype=np.float32)
                geom[li, pi] = np.asarray([*local(center), *local(left), *local(right),
                                          heading - float(ego.heading), width, float(ss)], np.float32)
                point_mask[li, pi] = True
            speed_limit = float(lane.speed_limit) if lane.speed_limit is not None else 0.0
            lt = getattr(lane, "line_types", (None, None)) or (None, None)
            line_types[li] = [int(v) if v is not None else -1 for v in lt]
            lane_meta[li] = [speed_limit, float(lane.length), s0, s_start, s_end,
                             float((s_end-s_start)/max(n-1, 1)), float(scored[li][0])]

        # Preserve both lateral adjacency on a common edge and directed road-graph
        # connectivity.  These are separate from left/right semantic labels.
        for i, (_, _, idx_i, _, _) in enumerate(scored):
            for j, (_, _, idx_j, _, _) in enumerate(scored):
                same_edge_adjacent = idx_i[:2] == idx_j[:2] and abs(int(idx_i[2])-int(idx_j[2])) == 1
                forward_connected = str(idx_i[1]) == str(idx_j[0])
                connections[i, j] = bool(same_edge_adjacent or forward_connected)

        if current_slot < 0 or target_slot < 0:
            raise RuntimeError("failed to encode current/target lane slots")
        return {
            "road_lane_geometry": geom,
            "road_lane_point_mask": point_mask,
            "road_lane_mask": lane_mask,
            "road_lane_index": lane_index,
            "road_lane_meta": lane_meta,
            "road_lane_line_types": line_types,
            "road_lane_connections": connections,
            "road_current_lane_slot": np.asarray(current_slot, np.int16),
            "road_target_lane_slot": np.asarray(target_slot, np.int16),
            "road_current_lane_index": np.asarray([str(v) for v in current], dtype="U64"),
            "road_target_lane_index": np.asarray([str(v) for v in target], dtype="U64"),
            "road_forward_preview_m": np.asarray(forward, np.float32),
            "road_backward_preview_m": np.asarray(backward, np.float32),
            "road_geometry_version": np.asarray(1, np.int16),
        }
