import os

import gymnasium as gym
from stable_baselines3.common.vec_env import DummyVecEnv, VecVideoRecorder

import highway_env


def env_reset(env_name, seed=None):
    """创建所选环境和离屏录像器，返回向量环境、录像器和初始观测。"""
    video_folders = {
        "merge-platoon-v0": "all_merge/infos/videos/merge/",
        "highway-platoon-v0": "all_merge/infos/videos/slow-down/",
    }
    video_folder = video_folders[env_name]
    highway_env.register_highway_envs()
    env = DummyVecEnv([
        lambda: gym.make(
            env_name,
            render_mode="rgb_array",
            config={"offscreen_rendering": True},
        )
    ])
    video_env = None
    try:
        os.makedirs(video_folder, exist_ok=True)
        video_env = VecVideoRecorder(
            env,
            video_folder,
            record_video_trigger=lambda step: step == 0,
            video_length=100,
            name_prefix="ppo-highway",
        )
        # VecEnv applies this seed at the next reset, only once.
        if seed is not None:
            env.seed(seed)
        obs = reset_env(video_env)
    except Exception:
        try:
            if video_env is not None:
                video_env.close()
        finally:
            env.close()
        raise
    return env, video_env, obs


def reset_env(video_env):
    """重置已有录像环境并返回观测。"""
    return video_env.reset()
