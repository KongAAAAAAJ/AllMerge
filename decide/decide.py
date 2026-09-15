from stable_baselines3 import PPO
from pathlib import Path

MODEL_PATH = (
    Path(__file__).resolve().parents[1]
    / "ckpts"
    / "ckpt_decision"
    / "case_1_Risk-aware_Attention_PPO"
    / "slow_down_1"
    / "logs"
    / "slow_down_1_best_model.zip"
)

if not MODEL_PATH.is_file():
    raise FileNotFoundError(f"模型文件不存在：{MODEL_PATH}")

_model = None


def decide(obs, mode):
    """首次调用时加载 PPO，之后复用模型，返回确定性决策的 action。"""
    global _model
    if mode == 2:
        if _model is None:
            _model = PPO.load(MODEL_PATH)
        action, _ = _model.predict(obs, deterministic=True)
    else:
        raise NotImplementedError(f"暂未实现 mode={mode} 的决策")
    return action
