from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Mapping

import torch


def _strip_common_prefixes(
    state: Mapping[str, torch.Tensor],
) -> tuple[dict[str, torch.Tensor], str | None]:
    out = dict(state)
    if not out:
        return out, None

    prefixes = (
        "module.planner.",
        "model.planner.",
        "planner.",
        "module.model.",
        "model.",
        "module.",
    )
    keys = tuple(out.keys())
    for prefix in prefixes:
        if all(key.startswith(prefix) for key in keys):
            return (
                {key[len(prefix):]: value for key, value in out.items()},
                prefix,
            )
    return out, None


def _extract_state_dict(
    payload: Any,
) -> tuple[Mapping[str, torch.Tensor], str, str | None]:
    checkpoint_format = "raw_state_dict"

    if isinstance(payload, Mapping):
        for key in (
            "planner_state_dict",
            "state_dict",
            "model_state_dict",
            "model",
        ):
            value = payload.get(key)
            if isinstance(value, Mapping):
                payload = value
                checkpoint_format = key
                break

    if not isinstance(payload, Mapping):
        raise TypeError("checkpoint does not contain a state dict")

    state = dict(payload)
    if not state:
        raise RuntimeError("checkpoint state dict is empty")

    bad_keys = [
        key
        for key, value in state.items()
        if not isinstance(value, torch.Tensor)
    ]
    if bad_keys:
        preview = ", ".join(map(str, bad_keys[:8]))
        raise RuntimeError(
            "selected checkpoint payload is not a pure model state dict; "
            f"non-tensor keys include: {preview}"
        )

    state, stripped_prefix = _strip_common_prefixes(state)
    return state, checkpoint_format, stripped_prefix


def load_pretrained(model, checkpoint: str | Path, *, strict: bool = True) -> Dict[str, Any]:
    checkpoint = Path(checkpoint)
    payload = torch.load(checkpoint, map_location="cpu")
    state, checkpoint_format, stripped_prefix = _extract_state_dict(payload)
    result = model.load_state_dict(state, strict=strict)
    return {
        "checkpoint": str(checkpoint),
        "checkpoint_format": checkpoint_format,
        "stripped_prefix": stripped_prefix,
        "num_tensors": len(state),
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
