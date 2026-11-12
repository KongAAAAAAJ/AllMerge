from stable_baselines3 import PPO
from pathlib import Path

MODEL_PATH = (
    Path(__file__).resolve().parents[1]
    / "outputs"
    / "checkpoints"
    / "slow_down_1_best_model.zip"
)

_model = None


def decide(obs, mode):
    """首次调用时加载 PPO，之后复用模型，返回确定性决策的 action。"""
    global _model
    if mode == 2:
        if _model is None:
            if not MODEL_PATH.is_file():
                raise FileNotFoundError(f"PPO checkpoint not found: {MODEL_PATH}")
            _model = PPO.load(MODEL_PATH)
        action, _ = _model.predict(obs, deterministic=True)
    else:
        raise NotImplementedError(f"暂未实现 mode={mode} 的决策")
    return action
