"""Time-aligned dual expert labels; no simulator or legacy schema dependency."""
import math
import numpy as np

TRAJECTORY_SOURCES = ('polynomial', 'trucksim_closed_loop')


def world_poses_to_local(world, origin):
    world = np.asarray(world, dtype=np.float64)
    origin = np.asarray(origin, dtype=np.float64)
    c, s = np.cos(origin[2]), np.sin(origin[2])
    xy = (world[..., :2] - origin[:2]) @ np.array([[c, -s], [s, c]])
    yaw = np.arctan2(np.sin(world[..., 2] - origin[2]), np.cos(world[..., 2] - origin[2]))
    return np.concatenate((xy, yaw[..., None]), axis=-1).astype(np.float32)


def finalize_dual_samples(samples, world_poses, num_egos=3):
    """Keep complete groups at k with real states k+1 through k+40 available."""
    world_poses = np.asarray(world_poses)
    frames = {}
    for sample in samples:
        frames.setdefault(int(sample['frame_index']), []).append(sample)
    result = []
    for frame, group in frames.items():
        if frame + 40 >= len(world_poses) or {int(s['ego_index']) for s in group} != set(range(num_egos)):
            continue
        for sample in group:
            ego = int(sample['ego_index'])
            sample['trucksim_future_poses'] = world_poses_to_local(
                world_poses[frame + 1:frame + 41, ego], sample['koopman_pose0_si'])
            result.append(sample)
    return result


def expert_local_poses(future_xy: np.ndarray) -> np.ndarray:
    """Prepend the observed origin and estimate future yaw from path tangents.

    Raw expert labels contain XY rather than body yaw. The initial local yaw
    is zero; stationary future nodes retain the preceding tangent heading.
    """
    xy = np.asarray(future_xy, dtype=np.float64)
    if xy.ndim != 2 or xy.shape[1] != 2 or len(xy) < 1 or not np.isfinite(xy).all():
        raise ValueError('Expert future trajectory must be finite [H,2] XY points')
    poses = np.zeros((len(xy) + 1, 3), dtype=np.float64)
    poses[1:, :2] = xy
    tangent = np.gradient(poses[:, :2], axis=0)
    for index in range(1, len(poses)):
        if np.linalg.norm(tangent[index]) > 1e-8:
            poses[index, 2] = math.atan2(tangent[index, 1], tangent[index, 0])
        else:
            poses[index, 2] = poses[index - 1, 2]
    return poses

