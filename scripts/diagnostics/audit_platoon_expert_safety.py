"""Audit synchronized expert trajectories before TRACE safety-aware training."""
from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from data_pipeline.raw_expert_dataset import RawExpertDataset
from highway_env.planner.diffusion.common.safety_losses import (
    background_constant_velocity_poses,
    expert_xy_to_poses,
    pairwise_vehicle_safety,
    road_boundary_safety_loss,
)


def audit(dataset_root, split='train', batch_size=64, num_workers=0, max_batches=0,
          vehicle_length_m=8.0, vehicle_width_m=2.6):
    dataset = RawExpertDataset(
        Path(dataset_root), split=split, include_platoon_reference=True
    )
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dimensions = torch.tensor(
        [vehicle_length_m, vehicle_width_m], dtype=torch.float32, device=device
    )
    totals = {
        'samples': 0,
        'expert_intertruck_collision_rate': 0.0,
        'expert_intertruck_ttc_violation_rate': 0.0,
        'expert_background_collision_rate': 0.0,
        'expert_background_ttc_violation_rate': 0.0,
        'expert_road_violation_rate': 0.0,
        'expert_intertruck_min_gap_m': 0.0,
        'expert_background_min_gap_m': 0.0,
        'expert_max_road_violation_m': 0.0,
    }
    for batch_index, (features, targets, _) in enumerate(loader):
        if max_batches and batch_index >= max_batches:
            break
        features = {key: value.to(device) for key, value in features.items()}
        targets = {key: value.to(device) for key, value in targets.items()}
        batch = targets['trajectory_dense'].shape[0]
        expert_poses = expert_xy_to_poses(targets['trajectory_dense'])
        ego_dimensions = dimensions.expand(batch, 2)
        partner_poses = targets['platoon_reference_poses']
        partner_dimensions = dimensions.view(1, 1, 2).expand(batch, 2, 2)
        partner_mask = torch.ones((batch, 2), dtype=torch.bool, device=device)

        intertruck = pairwise_vehicle_safety(
            expert_poses, ego_dimensions,
            partner_poses, partner_dimensions, partner_mask,
            dt=0.1, collision_margin_m=0.2, ttc_threshold_s=1.0,
            softplus_temperature_m=0.2,
            collision_distance_scale_m=1.0, ttc_distance_scale_m=5.0,
        )
        background_poses, background_dimensions, background_mask = background_constant_velocity_poses(
            features['agent_states'], features['agent_valid_mask'],
            features['ego_state'], 40, 0.1,
        )
        background = pairwise_vehicle_safety(
            expert_poses, ego_dimensions,
            background_poses, background_dimensions, background_mask,
            dt=0.1, collision_margin_m=0.2, ttc_threshold_s=1.0,
            softplus_temperature_m=0.2,
            collision_distance_scale_m=1.0, ttc_distance_scale_m=5.0,
        )
        road = road_boundary_safety_loss(
            expert_poses, ego_dimensions,
            features['road_lane_geometry'], features['road_lane_point_mask'],
            features['road_lane_mask'], features['road_lane_connections'],
            features['road_current_lane_slot'], features['road_target_lane_slot'],
            softplus_temperature_m=0.2,
            distance_scale_m=1.0,
        )
        totals['samples'] += batch
        totals['expert_intertruck_collision_rate'] += float(intertruck['collision_rate']) * batch
        totals['expert_intertruck_ttc_violation_rate'] += float(intertruck['ttc_violation_rate']) * batch
        totals['expert_background_collision_rate'] += float(background['collision_rate']) * batch
        totals['expert_background_ttc_violation_rate'] += float(background['ttc_violation_rate']) * batch
        totals['expert_road_violation_rate'] += float(road['violation_rate']) * batch
        totals['expert_intertruck_min_gap_m'] += float(intertruck['min_gap']) * batch
        totals['expert_background_min_gap_m'] += float(background['min_gap']) * batch
        totals['expert_max_road_violation_m'] = max(
            totals['expert_max_road_violation_m'], float(road['max_violation'])
        )

    samples = totals.pop('samples')
    if samples == 0:
        raise RuntimeError('No expert samples were audited')
    print(f'[EXPERT SAFETY AUDIT] split={split} samples={samples}')
    for key, value in totals.items():
        if key.endswith('_rate'):
            value /= samples
        elif key.endswith('_gap_m'):
            value /= samples
        print(f'{key}: {value:.6f}')
    return totals


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset-root', default='outputs/expert_dataset/expert_trucksim_k2_raw_100k')
    parser.add_argument('--split', choices=('train', 'val', 'test'), default='train')
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--num-workers', type=int, default=0)
    parser.add_argument('--max-batches', type=int, default=0)
    parser.add_argument('--vehicle-length-m', type=float, default=8.0)
    parser.add_argument('--vehicle-width-m', type=float, default=2.6)
    args = parser.parse_args()
    audit(
        args.dataset_root, split=args.split, batch_size=args.batch_size,
        num_workers=args.num_workers, max_batches=args.max_batches,
        vehicle_length_m=args.vehicle_length_m,
        vehicle_width_m=args.vehicle_width_m,
    )


if __name__ == '__main__':
    main()
