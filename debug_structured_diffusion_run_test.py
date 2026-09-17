import numpy as np
import torch
from stable_baselines3.common.utils import set_random_seed

from data_save import DataSaver
from decide.decide import decide
from env_reset import env_reset
from mode_selector.mode_selector import set_platoon_mode


env_name = "highway-platoon-v0"
SEED = 1

DEBUG_STEP = 1


def get_attr_from_vec_env(
    env,
    name,
    env_idx=0,
):
    values = env.get_attr(name)

    if not values or env_idx >= len(values):
        raise AttributeError(name)

    return values[env_idx]


def print_network_output(
    features,
    output,
):
    print("\n" + "=" * 88)
    print("STRUCTURED DIFFUSION SHADOW OUTPUT")
    print("=" * 88)

    print(
        "coarse_trajectories :",
        features[
            "coarse_trajectories"
        ].shape,
    )
    print(
        "mode_valid_mask     :",
        features[
            "mode_valid_mask"
        ].shape,
    )

    print(
        "trajectory          :",
        output[
            "trajectory"
        ].shape,
    )
    print(
        "candidates          :",
        output[
            "trajectory_candidates"
        ].shape,
    )
    print(
        "mode_logits         :",
        output[
            "trajectory_mode_logits"
        ].shape,
    )
    print(
        "mode_idx            :",
        output[
            "trajectory_mode_idx"
        ].shape,
    )

    print(
        "latency_ms          :",
        float(
            output[
                "latency_ms"
            ]
        ),
    )

    print(
        "selected modes      :",
        output[
            "trajectory_mode_idx"
        ].tolist(),
    )

    for ego_idx in range(
        output[
            "trajectory"
        ].shape[0]
    ):
        mode = int(
            output[
                "trajectory_mode_idx"
            ][ego_idx]
        )

        valid = bool(
            features[
                "mode_valid_mask"
            ][ego_idx, mode]
        )

        print(
            f"ego {ego_idx}: "
            f"mode={mode}, "
            f"mode_valid={valid}, "
            f"endpoint="
            f"{output['trajectory'][ego_idx, -1].tolist()}"
        )

        if not valid:
            raise RuntimeError(
                "Masked-invalid mode was selected"
            )

    for name in (
        "trajectory",
        "trajectory_candidates",
        "trajectory_mode_logits",
    ):
        value = output[name]

        if not np.all(
            np.isfinite(value)
        ):
            raise RuntimeError(
                f"{name} contains NaN/Inf"
            )

    print("=" * 88)


def main(
    episodes=1,
    seed=SEED,
):
    set_random_seed(
        seed,
        using_cuda=torch.cuda.is_available(),
    )

    env = None
    video_env = None

    try:
        env, video_env, obs = env_reset(
            env_name,
            seed=seed,
        )

        data_saver = DataSaver()

        for episode in range(episodes):
            reward_sum = 0.0
            steps = 0

            while True:
                mode = set_platoon_mode()

                action = decide(
                    obs,
                    mode,
                )

                (
                    obs,
                    reward,
                    done,
                    infos,
                ) = video_env.step(
                    action
                )

                reward_sum += float(
                    reward[0]
                )
                steps += 1

                data_saver.record_step(
                    infos[0],
                    action[0],
                )

                if steps == DEBUG_STEP:
                    features = (
                        get_attr_from_vec_env(
                            env,
                            "latest_planner_features",
                        )
                    )

                    output = (
                        get_attr_from_vec_env(
                            env,
                            "latest_diffusion_output",
                        )
                    )

                    print_network_output(
                        features,
                        output,
                    )

                if bool(done[0]):
                    summary = (
                        data_saver.finish_episode(
                            infos[0]
                        )
                    )

                    print(
                        f"episode={episode + 1}, "
                        f"steps={steps}, "
                        f"time={summary['time']:.3f}s, "
                        f"reason={summary['reason']}, "
                        f"reward={reward_sum:.6f}"
                    )
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
