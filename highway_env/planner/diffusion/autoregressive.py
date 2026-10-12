"""AR V1: re-use the unmodified StructuredTrajectoryDenoiser as an 8-step Gaussian policy.

* Every step conditions on generated/teacher-forced past points + untouched future
  coarse anchors, never on future expert points.
* A fixed diagonal Gaussian in physical metres supplies the likelihood; NO new
  trainable layers, changed tensor shapes, or modified denoiser weights.
* Mode classification comes ONLY from the untouched initial coarse anchors.
* The inherited dense execution head/spline remain identical to Diffusion.
"""
from __future__ import annotations

import math
from typing import Dict, Optional

import torch
import torch.nn.functional as F

from .structured_model import (
    StructuredDiffusionPlanner,
    _geometry_auxiliary_losses,
    _local_sensitive_loss,
)
from .mode_assignment import assign_expert_mode


class StructuredAutoregressivePlanner(StructuredDiffusionPlanner):
    """Exact same registered neural modules and parameter keys as the Diffusion planner."""

    def __init__(self, config, adapter, *, sigma_x_m: float = 2.0,
                 sigma_y_m: float = 0.5):
        super().__init__(config, adapter)
        self.ar_sigma_x_m = float(sigma_x_m)
        self.ar_sigma_y_m = float(sigma_y_m)
        if min(self.ar_sigma_x_m, self.ar_sigma_y_m) <= 0.0:
            raise ValueError("AR Gaussian std must be strictly positive")
        if config.horizon_steps > config.num_train_timesteps:
            raise ValueError("AR steps cannot exceed sinusoidal embedding range")

    @property
    def ar_sigma_xy_m(self):
        return (self.ar_sigma_x_m, self.ar_sigma_y_m)

    def _log_prob_xy(self, mu_norm, target_norm):
        """Gaussian log density in *metres*, sum x+y, preserving gradients."""
        scale = self.adapter.trajectory_scale.to(device=mu_norm.device, dtype=mu_norm.dtype)
        sigma = mu_norm.new_tensor(self.ar_sigma_xy_m)
        delta_m = (target_norm - mu_norm) * scale
        return (-0.5 * ((delta_m / sigma) ** 2)
                - sigma.log() - 0.5 * math.log(2.0 * math.pi)).sum(-1)

    def _one_step(self, work_norm, step, scene, coarse_goal_norm,
                  *, return_mode_tokens=False):
        t = torch.full((work_norm.shape[0],), int(step) + 1,
                       device=work_norm.device, dtype=torch.long)
        return self.denoiser(
            work_norm, t, scene,
            navigation_point_norm=coarse_goal_norm,
            return_mode_tokens=return_mode_tokens,
        )

    def _autoregressive_rollout(self, features, *, generator=None,
                               base_noise=None, navigation_point_norm=None):
        """Sample all M modes, cache one Gaussian log density per waypoint.

        base_noise: reproducible N(0,1) tensor [B,M,H,2], used point by point.
        """
        scene = self.scene_encoder(features)
        anchors = self.adapter.normalize_trajectory(features["coarse_trajectories"])
        batch, modes, horizon, dims = anchors.shape
        goal = (anchors[..., -1, :] if navigation_point_norm is None
                else navigation_point_norm)
        if base_noise is None:
            base_noise = torch.randn(anchors.shape, generator=generator,
                                     device=anchors.device, dtype=anchors.dtype)
        if tuple(base_noise.shape) != tuple(anchors.shape):
            raise ValueError("AR base_noise must be [B,M,H,2]")
        sigma_norm = anchors.new_tensor(self.ar_sigma_xy_m) / self.adapter.trajectory_scale
        work = anchors.clone()
        steps_lp = []
        first_logits = None
        last_tokens = None
        for step in range(horizon):
            predicted, logits, tokens = self._one_step(
                work, step, scene, goal, return_mode_tokens=True,
            )
            if step == 0:
                # Mode logits see NO generated or GT trajectory points.
                first_logits = logits
            mu = predicted[:, :, step, :]
            point = mu + base_noise[:, :, step, :] * sigma_norm
            steps_lp.append(self._log_prob_xy(mu, point))
            # Never replace future anchors or change already drawn prefix.
            work = work.clone()
            work[:, :, step, :] = point
            last_tokens = tokens
        return work, first_logits, last_tokens, torch.stack(steps_lp, dim=-1), anchors

    def _runtime_base_sample(self, features, *, generator=None, base_noise=None,
                             navigation_point_norm=None):
        final_norm, logits, mode_tokens, step_log_probs, anchors = (
            self._autoregressive_rollout(
                features, generator=generator, base_noise=base_noise,
                navigation_point_norm=navigation_point_norm,
            )
        )
        masked = self._masked_logits(logits, features["mode_valid_mask"])
        mode_index = masked.argmax(dim=-1)
        batch_idx = torch.arange(final_norm.shape[0], device=final_norm.device)
        best_norm = final_norm[batch_idx, mode_index]
        best_tokens = mode_tokens[batch_idx, mode_index]
        candidates = self.adapter.denormalize_trajectory(final_norm)
        return {
            "trajectory": candidates[batch_idx, mode_index],
            "trajectory_candidates": candidates,
            "_all_x0_norm": final_norm,
            "_all_mode_feature": mode_tokens,
            "trajectory_mode_logits": logits,
            "trajectory_mode_logits_masked": masked,
            "trajectory_mode_idx": mode_index,
            "selected_x0_norm": best_norm,
            "selected_mode_feature": best_tokens,
            # Compatibility-only field: initial *anchors*, NOT initial diffusion noise.
            "trajectory_noisy_initial": self.adapter.denormalize_trajectory(anchors),
            "ar_step_log_prob": step_log_probs,
        }

    @torch.no_grad()
    def sample_group(self, features, group_size: int = 48, *, generator=None):
        """Sample grouped candidate tensor [B,G,M,8,2] without altering GRPO code."""
        self.eval()
        if group_size <= 0:
            raise ValueError("group_size must be >= 1")
        batch = features["ego_state"].shape[0]
        expanded = {key: tensor.repeat_interleave(group_size, dim=0)
                    for key, tensor in features.items()}
        base = self._runtime_base_sample(expanded, generator=generator)
        m, h = self.config.num_modes, self.config.horizon_steps
        return {
            "candidates": base["trajectory_candidates"].reshape(batch, group_size, m, h, 2),
            "logits": base["trajectory_mode_logits"].reshape(batch, group_size, m),
            "step_log_prob": base["ar_step_log_prob"].reshape(batch, group_size, m, h),
        }

    def joint_trajectory_log_prob(self, features, all_mode_trajectories_m):
        """Exact joint Gaussian rollout log density of [B,M,8,2].

        Critical: CrossAttentionBlock has **mode_self_attn**, so mode samples
        are interdependent. Replay ALL mode prefixes; scoring one mode while
        freezing the others at anchors is NOT the sampled policy likelihood.
        """
        anchors = self.adapter.normalize_trajectory(features["coarse_trajectories"])
        target = self.adapter.normalize_trajectory(all_mode_trajectories_m)
        if tuple(target.shape) != tuple(anchors.shape):
            raise ValueError("all_mode_trajectories_m must be [B,M,H,2]")
        scene = self.scene_encoder(features)
        goal = anchors[:, :, -1, :]
        work = anchors.clone()
        total = anchors.new_zeros((anchors.shape[0],))
        for step in range(self.config.horizon_steps):
            predicted, _ = self._one_step(work, step, scene, goal)
            mu = predicted[:, :, step, :]
            total = total + self._log_prob_xy(mu, target[:, :, step, :]).sum(dim=-1)
            work = work.clone()
            work[:, :, step, :] = target[:, :, step, :]
        return total

    def forward_train(self, features, target_trajectory, target_semantic=None,
                      *, target_trajectory_dense=None, dense_loss_enabled=False,
                      dense_loss_lambda_p=0.0, dense_loss_type="smooth_l1",
                      dense_loss_terminal_timestep=0,
                      dense_loss_weight_mode="terminal_constant",
                      dense_loss_terminal_weight=1.0,
                      local_sensitive_loss_enabled=False,
                      local_sensitive_loss_lambda=0.5,
                      local_sensitive_loss_topk=2,
                      geometry_aux_loss_enabled=False,
                      heading_loss_lambda=0.2, curvature_loss_lambda=0.1,
                      lane_feasibility_loss_lambda=0.5,
                      lane_softmin_temperature_m=0.5):
        """Identical W1 expert/mode contract; replaces noise regression with AR NLL."""
        assignment = assign_expert_mode(
            features=features, target_trajectory=target_trajectory,
            target_semantic=target_semantic,
        )
        target_mode = assignment.target_mode
        anchors = self.adapter.normalize_trajectory(features["coarse_trajectories"])
        gt = self.adapter.normalize_trajectory(target_trajectory)
        scene = self.scene_encoder(features)
        goal = anchors[:, :, -1, :]
        rows = torch.arange(anchors.shape[0], device=anchors.device)
        work = anchors.clone()
        log_probs = []
        predicted_means = []
        first_logits = None
        for step in range(self.config.horizon_steps):
            output_norm, logits = self._one_step(work, step, scene, goal)
            if step == 0:
                first_logits = logits
            mu = output_norm[rows, target_mode, step, :]
            log_probs.append(self._log_prob_xy(mu, gt[:, step, :]))
            predicted_means.append(self.adapter.denormalize_trajectory(mu))
            work = work.clone()
            work[rows, target_mode, step, :] = gt[:, step, :]
        pred = torch.stack(predicted_means, dim=1)
        nll = -torch.stack(log_probs, dim=1).mean()
        point_loss = F.smooth_l1_loss(pred, target_trajectory)
        cls_loss = F.cross_entropy(first_logits, target_mode)
        zero = nll.new_zeros(())
        local_raw = zero
        local_weighted = zero
        if local_sensitive_loss_enabled:
            local_raw = _local_sensitive_loss(pred, target_trajectory,
                                              topk=int(local_sensitive_loss_topk))
            local_weighted = float(local_sensitive_loss_lambda) * local_raw
        regression_loss = nll + local_weighted
        heading = curvature = lane = outside_mean = outside_max = geometry_weighted = zero
        if geometry_aux_loss_enabled:
            geo = _geometry_auxiliary_losses(
                pred, target_trajectory, features,
                lane_softmin_temperature_m=float(lane_softmin_temperature_m),
            )
            heading = geo["heading_loss"]
            curvature = geo["curvature_loss"]
            lane = geo["lane_feasibility_loss"]
            outside_mean = geo["lane_outside_mean_m"]
            outside_max = geo["lane_outside_max_m"]
            geometry_weighted = (float(heading_loss_lambda) * heading
                                 + float(curvature_loss_lambda) * curvature
                                 + float(lane_feasibility_loss_lambda) * lane)
        base_loss = regression_loss + cls_loss + geometry_weighted
        dense_values = {key: zero for key in (
            "dense_loss_raw", "dense_loss_weighted", "dense_ade_m", "dense_base_ade_m",
            "dense_ade_gain_m", "dense_residual_mean_abs_m", "dense_residual_max_abs_m",
            "dense_residual_mean_l2_m", "dense_residual_mean_abs_x_m",
            "dense_residual_mean_abs_y_m", "dense_residual_max_abs_x_m",
            "dense_residual_max_abs_y_m", "dense_residual_x_saturation_ratio",
            "dense_residual_y_saturation_ratio", "dense_runtime_mode_match",
            "dense_weight_t", "dense_terminal_fraction",
        )}
        if bool(dense_loss_enabled) and float(dense_loss_lambda_p) > 0.0:
            if target_trajectory_dense is None:
                raise ValueError("AR dense supervision requires real 10Hz expert target")
            # Reuse EXACT same differentiable dense loss and separate-gradient
            # Dense Residual Head. The virtual _runtime_base_sample uses AR.
            dense_values = self._dense_runtime_auxiliary(
                features=features,
                base_noise=torch.randn_like(anchors),
                target_mode=target_mode,
                target_trajectory_dense=target_trajectory_dense,
                dense_loss_type=dense_loss_type,
                dense_loss_lambda_p=float(dense_loss_lambda_p),
                dense_loss_weight_mode=dense_loss_weight_mode,
                dense_loss_terminal_weight=float(dense_loss_terminal_weight),
                navigation_point_norm=goal,
            )
        mode_valid = features["mode_valid_mask"].bool()
        # Diagnostic candidates: all anchors except the GT-assigned mode,
        # whose slot contains the autoregressive one-step predicted means.
        train_candidates = features["coarse_trajectories"].clone()
        train_candidates[rows, target_mode] = pred
        results = {
            "loss": base_loss + dense_values["dense_loss_weighted"],
            "base_loss": base_loss,
            "trajectory_regression_loss": regression_loss,
            "trajectory_point_loss": point_loss,
            "ar_nll": nll,
            "local_sensitive_loss_raw": local_raw,
            "local_sensitive_loss_weighted": local_weighted,
            "heading_loss_raw": heading,
            "curvature_loss_raw": curvature,
            "lane_feasibility_loss_raw": lane,
            "geometry_aux_loss_weighted": geometry_weighted,
            "lane_outside_mean_m": outside_mean,
            "lane_outside_max_m": outside_max,
            "trajectory_classification_loss": cls_loss,
            "target_mode": target_mode,
            "target_semantic": assignment.target_semantic,
            "target_mode_assignment_distance": assignment.anchor_distance[rows, target_mode],
            "target_mode_geometry_valid": assignment.geometry_mask[rows, target_mode],
            "target_mode_traffic_valid": mode_valid[rows, target_mode],
            "trajectory_mode_logits_train": first_logits,
            "trajectory_candidates_train": train_candidates,
        }
        results.update(dense_values)
        return results
