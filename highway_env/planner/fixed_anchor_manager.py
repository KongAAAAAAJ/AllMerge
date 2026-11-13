"""A0 scene-independent fixed control anchors from K-means centroids."""
from __future__ import annotations

import hashlib
import json
import numpy as np
import torch

from highway_env.planner.diffusion.common.kinematics import rollout_bicycle
from highway_env.planner.diffusion.config.experiment import resolve_repo_file


class FixedAnchorManager:
    """Load a fixed [K,40,2] control-centroid bank and broadcast it per scene.

    The controls are fixed across scenes. Their paired 8-point coarse XY
    trajectories are generated at runtime by the same kinematic bicycle
    rollout used by the A1 anchor builder, conditioned only on the observed
    initial speed. No safety scoring, traffic filtering, or adaptive ranking
    is applied in A0.
    """

    IMPLEMENTATION_ID = "A0_FIXED_CONTROL_KMEANS_CENTROIDS_V3"

    def __init__(self, config, *, num_modes: int = 10, dt: float = 0.1,
                 wheelbase_m: float = 5.5, max_accel_mps2: float = 2.5,
                 max_steer_rad: float = 0.08):
        self.config = dict(config or {})
        self.num_modes = int(num_modes)
        configured_count = int(self.config.get("num_selected", self.num_modes))
        if configured_count != self.num_modes:
            raise ValueError(
                f"A0 fixed control bank K={configured_count} does not match model num_modes={self.num_modes}"
            )
        self.dt = float(dt)
        self.wheelbase_m = float(wheelbase_m)
        self.max_accel_mps2 = float(max_accel_mps2)
        self.max_steer_rad = float(max_steer_rad)
        if self.dt <= 0 or self.wheelbase_m <= 0:
            raise ValueError("A0 rollout dt and wheelbase must be positive")

        bank_value = self.config.get(
            "bank_path", "outputs/anchor_ablation/a0_fixed_control_anchors.npz"
        )
        self.bank_path = resolve_repo_file(bank_value)
        if not self.bank_path.is_file():
            raise FileNotFoundError(
                f"A0 fixed control bank is missing: {self.bank_path}. "
                "Run scripts/data/build_a0_fixed_anchors.py first."
            )
        with np.load(self.bank_path, allow_pickle=False) as data:
            required = ("anchors_controls", "cluster_centers_controls", "control_scale", "metadata_json")
            missing = [key for key in required if key not in data.files]
            if missing:
                raise KeyError(f"A0 control bank is missing fields: {missing}")
            anchors = np.asarray(data["anchors_controls"], dtype=np.float32)
            centers = np.asarray(data["cluster_centers_controls"], dtype=np.float32)
            scale = np.asarray(data["control_scale"], dtype=np.float32)
            metadata_text = str(data["metadata_json"].item())

        expected_shape = (self.num_modes, 40, 2)
        if anchors.shape != expected_shape:
            raise ValueError(f"A0 anchors_controls must be {expected_shape}, got {anchors.shape}")
        if centers.shape != expected_shape:
            raise ValueError(f"A0 cluster_centers_controls must be {expected_shape}, got {centers.shape}")
        if scale.shape != (2,) or not np.isfinite(scale).all() or (scale <= 0).any():
            raise ValueError(f"A0 control_scale must be finite positive [2], got {scale}")
        if not np.isfinite(anchors).all() or not np.isfinite(centers).all():
            raise ValueError("A0 control bank contains NaN/Inf")
        if not np.allclose(anchors, centers, atol=1e-6, rtol=0.0):
            raise ValueError("A0 anchors_controls must be the K-means centroids directly")
        if (np.abs(anchors[..., 0]) > self.max_accel_mps2 + 1e-4).any():
            raise ValueError("A0 acceleration centroid exceeds control planner limits")
        if (np.abs(anchors[..., 1]) > self.max_steer_rad + 1e-5).any():
            raise ValueError("A0 steering centroid exceeds control planner limits")

        try:
            metadata = json.loads(metadata_text)
        except (TypeError, json.JSONDecodeError) as exc:
            raise ValueError("A0 control bank metadata_json is invalid") from exc
        if metadata.get("implementation") != self.IMPLEMENTATION_ID:
            raise ValueError("A0 bank implementation/version is incompatible")
        if int(metadata.get("num_anchors", -1)) != self.num_modes:
            raise ValueError("A0 metadata num_anchors does not match model num_modes")
        if metadata.get("source_split") != "train":
            raise ValueError("A0 fixed control bank must be clustered from the training split only")
        if metadata.get("cluster_space") != "flattened_40_step_physical_control_sequence":
            raise ValueError("A0 bank was not clustered directly in control space")
        if metadata.get("prototype_type") != "kmeans_centroid":
            raise ValueError("A0 bank must contain K-means centroids, not medoids")
        if metadata.get("medoid_selection_used") or metadata.get("trajectory_to_control_matching_used"):
            raise ValueError("A0 direct control K-means must not use medoid or trajectory matching")
        if metadata.get("mode_valid_mask_used") or metadata.get("safety_scoring_used") or metadata.get("adaptive_selection_used"):
            raise ValueError("A0 fixed anchors must not inherit safety filtering or adaptive selection")

        digest = hashlib.sha256()
        digest.update(np.ascontiguousarray(anchors, dtype=np.float32).tobytes())
        digest.update(np.ascontiguousarray(centers, dtype=np.float32).tobytes())
        digest.update(np.ascontiguousarray(scale, dtype=np.float32).tobytes())
        digest.update(metadata_text.encode("utf-8"))
        self.bank_sha256 = digest.hexdigest()
        expected_hash = self.config.get("bank_sha256")
        if expected_hash and expected_hash != self.bank_sha256:
            raise RuntimeError(
                "A0 fixed control bank hash mismatch: the K-means centroids or metadata "
                "differ from the artifact used to train this checkpoint."
            )

        self.anchors_controls = torch.from_numpy(anchors.copy())
        self.metadata = metadata
        self.control_scale = torch.from_numpy(scale.copy())

    @torch.no_grad()
    def select(self, features):
        """Broadcast fixed controls and generate scene-specific kinematic rollouts."""
        if "ego_state" not in features:
            raise KeyError("A0 FixedAnchorManager requires features['ego_state']")
        ego = features["ego_state"]
        if ego.ndim != 2 or ego.shape[-1] < 1:
            raise ValueError("features['ego_state'] must be [B,D] and contain longitudinal speed")
        batch_size = int(ego.shape[0])
        device = ego.device
        dtype = ego.dtype if ego.is_floating_point() else torch.float32

        bank = self.anchors_controls.to(device=device, dtype=dtype)
        controls = bank.unsqueeze(0).expand(batch_size, -1, -1, -1).contiguous()
        initial_speed = ego[:, 0].to(dtype=dtype).clamp_min(0.0)
        poses, _ = rollout_bicycle(
            controls.reshape(batch_size * self.num_modes, 40, 2),
            initial_speed.repeat_interleave(self.num_modes),
            dt=self.dt, wheelbase_m=self.wheelbase_m,
        )
        trajectories = poses[:, 5::5, :2].reshape(batch_size, self.num_modes, 8, 2)
        if trajectories.shape != (batch_size, self.num_modes, 8, 2):
            raise RuntimeError(f"A0 runtime rollout produced invalid trajectory shape {tuple(trajectories.shape)}")

        selected = dict(features)
        selected["coarse_controls"] = controls
        selected["coarse_trajectories"] = trajectories
        # Fixed Anchors are not traffic-filtered. The model's normal endpoint
        # selector still compares their denoised, executed candidates.
        selected["mode_valid_mask"] = torch.ones(
            (batch_size, self.num_modes), dtype=torch.bool, device=device
        )
        return selected
