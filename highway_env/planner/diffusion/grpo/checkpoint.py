from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Mapping

import torch


def _extract_state_dict(payload: Any) -> Mapping[str, torch.Tensor]:
    if isinstance(payload, Mapping):
        for key in ("state_dict", "model_state_dict", "model"):
            value = payload.get(key)
            if isinstance(value, Mapping):
                payload = value
                break
    if not isinstance(payload, Mapping):
        raise TypeError("checkpoint does not contain a state dict")
    state = dict(payload)
    if state and all(key.startswith("model.") for key in state):
        state = {key[len("model."):]: value for key, value in state.items()}
    return state


def load_pretrained(model, checkpoint: str | Path, *, strict: bool = True) -> Dict[str, Any]:
    payload = torch.load(Path(checkpoint), map_location="cpu")
    state = _extract_state_dict(payload)
    result = model.load_state_dict(state, strict=strict)
    return {
        "missing_keys": list(result.missing_keys),
        "unexpected_keys": list(result.unexpected_keys),
    }


def save_grpo_checkpoint(
    path: str | Path,
    *,
    model,
    optimizer,
    step: int,
    config: Any,
    metrics: Dict[str, float] | None = None,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "step": int(step),
            "grpo_config": vars(config) if hasattr(config, "__dict__") else config,
            "metrics": dict(metrics or {}),
        },
        path,
    )
