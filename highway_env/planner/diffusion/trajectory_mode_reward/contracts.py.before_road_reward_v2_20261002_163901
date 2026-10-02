"""Stable contracts for trajectory-mode GRPO reward evaluation."""

from __future__ import annotations

import hashlib
import json


TRAJECTORY_MODE_REWARD_CONTRACT = {
    "version": "stage2_trajectory_mode_reward_v1",
    "comparison_unit": "trajectory_mode",
    "candidate_shape": "[3,10,N,8,2]",
    "same_mode_pretrain_shape": "[3,10,8,2]",
    "teammate_context": "frozen Stage-1 argmax raw tau_d",
    "counterfactual": (
        "replace only the target trajectory; keep both teammate trajectories fixed"
    ),
    "reward_scope": (
        "target progress, comfort, road and out-of-drivable plus target-to-"
        "background and target-to-teammate gap, TTC, collision and clearance"
    ),
    "formation_component": False,
    "other_vehicle_self_events": False,
    "formula": (
        "+progress_weight*progress_score"
        "-gap_weight*gap_penalty"
        "-ttc_weight*ttc_penalty"
        "-road_weight*road_penalty"
        "-comfort_weight*comfort_penalty"
        "-collision_penalty*collision"
        "-out_of_drivable_penalty*out_of_drivable"
    ),
    "invalid_mode_values": "zero-filled and excluded by valid_mode_mask",
}

TRAJECTORY_MODE_REWARD_CONTRACT_SHA256 = hashlib.sha256(
    json.dumps(
        TRAJECTORY_MODE_REWARD_CONTRACT,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
).hexdigest()


GRPO_OPEN_REWARD_APPLICATION_CONTRACT = {
    "version": "allmerge_stage2_grpo_trajectory_mode_v1",
    "comparison_unit": "trajectory_mode",
    "policy_sample_domain": "all-mode raw tau_d",
    "reward_input_domain": "tau_d",
    "teammate_reward_context": "frozen Stage-1 argmax raw tau_d",
    "formation_component": False,
    "sampled_candidate_execution": False,
    "environment_execution_policy": (
        "cached deterministic frozen Stage-1 argmax baseline"
    ),
    "execution_input_domain": "tau_cmd",
    "validation_reward_family": "trajectory_mode_counterfactual",
}
