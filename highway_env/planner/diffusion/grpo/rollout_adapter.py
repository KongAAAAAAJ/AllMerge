from __future__ import annotations

import copy
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Iterator


@dataclass
class AllMergeRolloutAdapter:
    """Correctness-first branch adapter for AllMerge scenario environments.

    If an environment later exposes get_state/set_state, those native methods
    are used.  Current scenario environments can still branch safely through a
    deepcopy fallback without pushing simulator details into the GRPO trainer.
    """

    env: Any

    @property
    def has_native_snapshot(self) -> bool:
        return callable(getattr(self.env, "get_state", None)) and callable(
            getattr(self.env, "set_state", None)
        )

    def snapshot(self) -> Any:
        if self.has_native_snapshot:
            return self.env.get_state()
        return copy.deepcopy(self.env)

    def restore(self, snapshot: Any) -> Any:
        if self.has_native_snapshot:
            self.env.set_state(snapshot)
            return self.env
        self.env = copy.deepcopy(snapshot)
        return self.env

    def fork(self) -> Any:
        if self.has_native_snapshot:
            clone = copy.deepcopy(self.env)
            clone.set_state(self.env.get_state())
            return clone
        return copy.deepcopy(self.env)

    @contextmanager
    def branch(self) -> Iterator[Any]:
        if self.has_native_snapshot:
            state = self.env.get_state()
            try:
                yield self.env
            finally:
                self.env.set_state(state)
        else:
            yield copy.deepcopy(self.env)
