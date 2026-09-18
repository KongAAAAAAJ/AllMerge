"""用来测试全链路闭环"""

import torch
from stable_baselines3.common.utils import set_random_seed

from data_save import DataSaver
from decide.decide import decide
from env_reset import env_reset
from mode_selector.mode_selector import set_platoon_mode

env_name = "highway-platoon-v0"  # merge-platoon-v0 / highway-platoon-v0


SEED = 1


def main(episodes=1, seed=SEED):
    set_random_seed(seed, using_cuda=torch.cuda.is_available())
    print(f"seed={seed}")
    env = video_env = None
    try:
        env, video_env, obs = env_reset(env_name, seed=seed)
        data_saver = DataSaver()
        for episode in range(episodes):
            reward_sum = 0.0
            steps = 0
            while True:
                mode = set_platoon_mode()
                action = decide(obs, mode)

                obs, reward, done, infos = video_env.step(action)

                # # ----------------------------------------------------------
                # # Check features for the planner
                # if steps == 5:
                #     from tests.debug_planner_features import debug_planner_features, get_planner_features
                #     debug_planner_features(env, ego_idx=0, show=True)
                #     features = get_planner_features(env)
                #     for key, value in features.items():
                #         print(
                #             f"{key:24s}",
                #             value.shape,
                #             value.dtype,
                #         )
                # # ----------------------------------------------------------

                reward_sum += float(reward[0])
                steps += 1
                data_saver.record_step(infos[0], action[0])
                if bool(done[0]):
                    summary = data_saver.finish_episode(infos[0])
                    print(f"episode={episode + 1}, steps={steps}, "
                          f"time={summary['time']:.3f}s, reason={summary['reason']}, "
                          f"terminated={summary['terminated']}, "
                          f"truncated={summary['truncated']}, reward={reward_sum:.6f}")
                    # DummyVecEnv has already reset. Its returned obs starts
                    # the next episode; resetting the recorder creates a new clip.
                    break
        return data_saver.save()
    finally:
        try:
            if video_env is not None:
                video_env.close()
        finally:
            if env is not None:
                env.close()


if __name__ == "__main__":
    main()
