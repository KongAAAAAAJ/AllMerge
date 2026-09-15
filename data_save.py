import datetime
import os
from pathlib import Path

import numpy as np
import pandas as pd



TRUCKSIM_OUTPUTS = (
    ("Ax", "g"), ("Ay", "g"), ("Vx", "km/h"), ("Vy", "km/h"),
    ("Xo", "m"), ("Yo", "m"), ("Roll", "deg"), ("AVx", "deg/s"),
    ("Pitch", "deg"), ("AVy", "deg/s"), ("Yaw", "deg"), ("AVz", "deg/s"),
    ("AV_Eng", "rpm"), ("steer_l1", "deg"), ("steer_r1", "deg"),
)


class DataSaver:
    """管理三辆受控车辆的测试数据、回合统计和 CSV 保存。"""

    def __init__(self, plot_dir=None, output_root=None):
        self.plot_dir = (Path(plot_dir) if plot_dir is not None else
                         Path(__file__).resolve().parent / "infos" / "result_plots")
        self.output_root = (Path(output_root) if output_root is not None else
                            Path(__file__).resolve().parent / "infos")
        self.episode_artifacts = []
        self.front_wheel_angles = [[], [], []]
        self.all_front_wheel_angles = [[], [], []]
        self.run_id = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S_%f")
        self.plot_samples = []
        self.plot_paths = []
        self.distances = [[], [], []]
        self.ttcs = [[], [], []]
        self.distance_min = [[], [], []]
        self.distance_avg = [[], [], []]
        self.ttc_min = [[], [], []]
        self.ttc_average = [[], [], []]
        self.speeds = [[], [], []]
        self.speed_avg = [[], [], []]
        self.speed_min = [[], [], []]
        self.episode_steps = []
        self.end_times = []
        self.exit_reasons = []
        self.terminated = []
        self.truncated = []
        self.collision = []
        self.actions = []

        self.all_actions = []
        self.is_success_merge_list = []
        self.all_distances = [[], [], []]
        self.all_ttcs = [[], [], []]
        self.all_speeds = [[], [], []]

    def record_step(self, info, action):
        """Record the returned step info, never the auto-reset environment."""
        samples = info.get("simulation_trace", [])
        for sample in samples:
            time = float(sample["time"])
            if self.plot_samples and np.isclose(
                    time, self.plot_samples[-1]["time"], rtol=0, atol=1e-10):
                continue  # Adjacent policy steps share one boundary sample.
            if self.plot_samples and time < self.plot_samples[-1]["time"]:
                raise ValueError("Plot sample times must increase within an episode")
            self.plot_samples.append({
                "time": time,
                "position": np.array(sample["position"], dtype=float, copy=True),
                "speed": np.array(sample["speed"], dtype=float, copy=True),
                "acceleration": np.array(sample["acceleration"], dtype=float, copy=True),
                "front_wheel_angle_deg": np.array(sample["front_wheel_angle_deg"], dtype=float, copy=True),
            })
            if "trucksim_outputs" in sample:
                outputs = np.array(sample["trucksim_outputs"], dtype=float, copy=True)
                if outputs.shape != (3, 15):
                    raise ValueError("Expected three vehicles with 15 TruckSim outputs each")
                self.plot_samples[-1]["trucksim_outputs"] = outputs
                self.plot_samples[-1]["solver_time"] = np.array(sample["solver_time"], dtype=float, copy=True)
        if not samples:
            raise ValueError("Step info must contain simulation samples")
        for k in range(3):
            self.front_wheel_angles[k].append(float(samples[-1]["front_wheel_angle_deg"][k]))
            self.distances[k].append(float(info["follow_distance"][k]))
            self.ttcs[k].append(float(info["follow_ttc"][k]))
            self.speeds[k].append(float(info["speed"][k]))
        self.actions.append(int(action))

    def data_plot(self, episode_index=None, output_dir=None):
        """Save actual trajectory, speed and longitudinal acceleration as one PNG."""
        if not self.plot_samples:
            raise ValueError("Cannot plot an episode without simulation samples")
        from matplotlib.figure import Figure
        from matplotlib.backends.backend_agg import FigureCanvasAgg

        episode = len(self.episode_steps) + 1 if episode_index is None else episode_index
        directory = self.plot_dir if output_dir is None else Path(output_dir)
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / f"episode_{episode:04d}_{self.run_id}.png"
        times = np.array([s["time"] for s in self.plot_samples])
        positions = np.array([s["position"] for s in self.plot_samples])
        speeds = np.array([s["speed"] for s in self.plot_samples])
        accelerations = np.array([s["acceleration"] for s in self.plot_samples])
        angles = np.array([s["front_wheel_angle_deg"] for s in self.plot_samples])
        fig = Figure(figsize=(14, 9), constrained_layout=True)
        FigureCanvasAgg(fig)
        try:
            axes = fig.subplots(2, 2).ravel()
            for k, color in enumerate(("tab:blue", "tab:orange", "tab:green")):
                label = f"Vehicle {k}"
                axes[0].plot(positions[:, k, 0], positions[:, k, 1], color=color, label=label)
                axes[0].scatter(positions[0, k, 0], positions[0, k, 1], color=color, marker="o")
                axes[0].scatter(positions[-1, k, 0], positions[-1, k, 1], color=color, marker="x")
                axes[1].plot(times, speeds[:, k], color=color, label=label)
                axes[2].plot(times, accelerations[:, k], color=color, label=label)
                axes[3].plot(times, angles[:, k], color=color, label=label)
            axes[0].set(title="Actual trajectory (circle: start, cross: end)", xlabel="X (m)", ylabel="Y (m)")
            axes[1].set(title="Speed", xlabel="Simulation time (s)", ylabel="Speed (m/s)")
            axes[2].set(title="Longitudinal acceleration", xlabel="Simulation time (s)", ylabel="Acceleration (m/s squared)")
            angle_title = ("Front wheel angle (chassis feedback)" if "trucksim_outputs" in self.plot_samples[0]
                           else "Front wheel angle (applied)")
            axes[3].set(title=angle_title, xlabel="Simulation time (s)", ylabel="Angle (deg)")
            for ax in axes:
                ax.grid(True, alpha=0.3)
                ax.legend()
            fig.suptitle(f"Episode {episode}")
            fig.savefig(path, dpi=300)
        finally:
            fig.clear()
        return path

    def trucksim_data_plot(self, episode_index=None, output_dir=None):
        """Plot all raw outputs against each vehicle's native solver time."""
        if not self.plot_samples or "trucksim_outputs" not in self.plot_samples[0]:
            raise ValueError("Cannot plot TruckSim without output samples")
        from matplotlib.figure import Figure
        from matplotlib.backends.backend_agg import FigureCanvasAgg
        episode = len(self.episode_steps) + 1 if episode_index is None else episode_index
        directory = self.plot_dir if output_dir is None else Path(output_dir)
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / f"episode_{episode:04d}_{self.run_id}_trucksim.png"
        outputs = np.array([s["trucksim_outputs"] for s in self.plot_samples])
        times = np.array([s["solver_time"] for s in self.plot_samples])
        fig = Figure(figsize=(18, 20), constrained_layout=True)
        FigureCanvasAgg(fig)
        try:
            for j, ax in enumerate(fig.subplots(5, 3).ravel()):
                name, unit = TRUCKSIM_OUTPUTS[j]
                for k, color in enumerate(("tab:blue", "tab:orange", "tab:green")):
                    ax.plot(times[:, k], outputs[:, k, j], color=color, label=f"Vehicle {k}")
                ax.set(title=name, xlabel="Solver time (s)", ylabel=f"{name} ({unit})")
                ax.grid(True, alpha=0.3)
                ax.legend()
            fig.suptitle(f"Episode {episode} - TruckSim raw outputs")
            fig.savefig(path, dpi=300)
        finally:
            fig.clear()
        return path

    def _save_episode_samples(self):
        """Persist substep data before clearing the episode cache."""
        episode = len(self.episode_steps) + 1
        suffix = f"episode_{episode:04d}_{self.run_id}.csv"
        data = {"Simulation time (s)": [s["time"] for s in self.plot_samples]}
        for k in range(3):
            for axis, name in enumerate(("X", "Y")):
                data[f"Vehicle {k} {name} (m)"] = [s["position"][k, axis] for s in self.plot_samples]
            for key, label in (("speed", "Speed (m/s)"), ("acceleration", "Acceleration (m/s^2)"),
                               ("front_wheel_angle_deg", "Front wheel angle (deg)")):
                data[f"Vehicle {k} {label}"] = [s[key][k] for s in self.plot_samples]
        directory = self.output_root / "all_data"
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / f"substeps_{suffix}"
        pd.DataFrame(data).to_csv(path, index=False)
        artifacts = {"substep_csv_path": path}
        if "trucksim_outputs" in self.plot_samples[0]:
            raw = {"Simulation time (s)": data["Simulation time (s)"]}
            for k in range(3):
                raw[f"Vehicle {k} Solver time (s)"] = [s["solver_time"][k] for s in self.plot_samples]
                for j, (name, unit) in enumerate(TRUCKSIM_OUTPUTS):
                    raw[f"Vehicle {k} {name} ({unit})"] = [s["trucksim_outputs"][k, j] for s in self.plot_samples]
            directory = self.output_root / "trucksim_data"
            directory.mkdir(parents=True, exist_ok=True)
            path = directory / f"trucksim_{suffix}"
            pd.DataFrame(raw).to_csv(path, index=False)
            artifacts["trucksim_csv_path"] = path
            artifacts["trucksim_plot_path"] = self.trucksim_data_plot()
        return artifacts

    @staticmethod
    def exit_reason(state):
        if any(state["crashed"]):
            return "collision"
        if (state["terminated"] and state["offroad_terminal"]
                and not all(state["on_road"])):
            return "offroad"
        if state["truncated"]:
            return "time_limit"
        return "other_termination"

    def finish_episode(self, info):
        """Summarize the terminal snapshot and clear only per-episode data."""
        if not self.actions:
            raise ValueError("Cannot finish an episode without recorded steps")
        plot_path = self.data_plot()
        artifacts = self._save_episode_samples()
        state = info["episode_state"]
        for k, (distances, ttcs, speeds) in enumerate(
                zip(self.distances, self.ttcs, self.speeds)):
            # Infinity in distance means no preceding vehicle. Keep every row
            # in the CSV, but omit absent-front rows from distance/TTC metrics.
            has_front = np.isfinite(distances)
            distance = np.asarray(distances)[has_front]
            ttc = np.asarray(ttcs)[has_front]
            self.distance_avg[k].append(float(np.mean(distance)) if distance.size else np.inf)
            self.distance_min[k].append(float(np.min(distance)) if distance.size else np.inf)
            self.ttc_average[k].append(float(np.mean(ttc)) if ttc.size else np.inf)
            self.ttc_min[k].append(float(np.min(ttc)) if ttc.size else np.inf)
            self.speed_avg[k].append(float(np.mean(speeds)))
            self.speed_min[k].append(float(np.min(speeds)))
            self.all_distances[k].extend(distances)
            self.all_ttcs[k].extend(ttcs)
            self.all_speeds[k].extend(speeds)
            self.all_front_wheel_angles[k].extend(self.front_wheel_angles[k])

        self.all_actions.extend(self.actions)
        self.collision.append(any(state["crashed"]))
        lanes = [tuple(lane) for lane in state["lane_index"]]
        self.is_success_merge_list.append(all(lane == lanes[0] for lane in lanes))
        summary = {"steps": len(self.actions), "time": state["time"],
                   "reason": self.exit_reason(state),
                   "terminated": state["terminated"], "truncated": state["truncated"]}
        summary.update(artifacts)
        self.episode_artifacts.append(dict(artifacts))
        self.front_wheel_angles = [[], [], []]
        summary["plot_path"] = plot_path
        self.plot_paths.append(plot_path)
        self.plot_samples = []
        self.episode_steps.append(summary["steps"])
        self.end_times.append(summary["time"])
        self.exit_reasons.append(summary["reason"])
        self.terminated.append(summary["terminated"])
        self.truncated.append(summary["truncated"])
        self.distances = [[], [], []]
        self.ttcs = [[], [], []]
        self.speeds = [[], [], []]
        self.actions = []
        return summary

    def save(self, output_root="all_merge/infos"):
        """保存评价指标和逐步明细，依次返回两份 CSV 的路径。"""
        evaluate_data = {
            'collision': self.collision,
            'Episode steps': self.episode_steps,
            'End time': self.end_times,
            'Exit reason': self.exit_reasons,
            'Terminated': self.terminated,
            'Truncated': self.truncated,
            'Merge success': self.is_success_merge_list,
            'Min ttc 0': self.ttc_min[0],
            'Min ttc 1': self.ttc_min[1],
            'Min ttc 2': self.ttc_min[2],
            'Average ttc 0': self.ttc_average[0],
            'Average ttc 1': self.ttc_average[1],
            'Average ttc 2': self.ttc_average[2],
            'Min Distance 0': self.distance_min[0],
            'Min Distance 1': self.distance_min[1],
            'Min Distance 2': self.distance_min[2],
            'Average Distance 0': self.distance_avg[0],
            'Average Distance 1': self.distance_avg[1],
            'Average Distance 2': self.distance_avg[2],
            'Average Speed 0': self.speed_avg[0],
            'Average Speed 1': self.speed_avg[1],
            'Average Speed 2': self.speed_avg[2],
            'Min Speed 0': self.speed_min[0],
            'Min Speed 1': self.speed_min[1],
            'Min Speed 2': self.speed_min[2],
        }
        all_data = {
            'All TTC 0': self.all_ttcs[0],
            'All TTC 1': self.all_ttcs[1],
            'All TTC 2': self.all_ttcs[2],
            'All Distance 0': self.all_distances[0],
            'All Distance 1': self.all_distances[1],
            'All Distance 2': self.all_distances[2],
            'All Speed 0': self.all_speeds[0],
            'All Speed 1': self.all_speeds[1],
            'All Speed 2': self.all_speeds[2],
            'All Actions': self.all_actions,
        }

        for k in range(3):
            all_data[f"All Front wheel angle {k} (deg)"] = self.all_front_wheel_angles[k]

        time_str = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        filename = f"vehicle_test_info {time_str}"
        evaluate_dir = os.path.join(output_root, 'evaluate_data')
        all_data_dir = os.path.join(output_root, 'all_data')
        os.makedirs(evaluate_dir, exist_ok=True)
        os.makedirs(all_data_dir, exist_ok=True)
        evaluate_path = os.path.join(
            evaluate_dir, f'ppo_add_reward_64000_eva+{filename}.csv'
        )
        all_data_path = os.path.join(
            all_data_dir, f'ppo_add_reward_64000_all+{filename}.csv'
        )
        pd.DataFrame(evaluate_data).to_csv(evaluate_path, index=False)
        pd.DataFrame(all_data).to_csv(all_data_path, index=False)
        return evaluate_path, all_data_path
