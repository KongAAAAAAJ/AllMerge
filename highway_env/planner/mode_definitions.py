from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from typing import Optional, Tuple


class BehaviorType(IntEnum):
    KEEP_LANE = 0
    LANE_CHANGE = 1


class SpeedProfile(IntEnum):
    HIGH = 0
    MEDIUM = 1
    LOW = 2


@dataclass(frozen=True)
class ModeSlot:
    index: int
    name: str
    behavior_type: BehaviorType
    speed_profile: SpeedProfile
    lateral_direction: Optional[str] = None
    semantic_group: str = ""


MODE_SLOTS: Tuple[ModeSlot, ...] = (
    ModeSlot(0, "KEEP_HIGH", BehaviorType.KEEP_LANE, SpeedProfile.HIGH, None, "KEEP"),
    ModeSlot(1, "KEEP_MEDIUM", BehaviorType.KEEP_LANE, SpeedProfile.MEDIUM, None, "KEEP"),
    ModeSlot(2, "KEEP_LOW", BehaviorType.KEEP_LANE, SpeedProfile.LOW, None, "KEEP"),
    ModeSlot(3, "LEFT_LC_HIGH", BehaviorType.LANE_CHANGE, SpeedProfile.HIGH, "left", "LEFT_LC"),
    ModeSlot(4, "LEFT_LC_MEDIUM", BehaviorType.LANE_CHANGE, SpeedProfile.MEDIUM, "left", "LEFT_LC"),
    ModeSlot(5, "LEFT_LC_LOW", BehaviorType.LANE_CHANGE, SpeedProfile.LOW, "left", "LEFT_LC"),
    ModeSlot(6, "RIGHT_LC_HIGH", BehaviorType.LANE_CHANGE, SpeedProfile.HIGH, "right", "RIGHT_LC"),
    ModeSlot(7, "RIGHT_LC_MEDIUM", BehaviorType.LANE_CHANGE, SpeedProfile.MEDIUM, "right", "RIGHT_LC"),
    ModeSlot(8, "RIGHT_LC_LOW", BehaviorType.LANE_CHANGE, SpeedProfile.LOW, "right", "RIGHT_LC"),
    ModeSlot(9, "STOP", BehaviorType.KEEP_LANE, SpeedProfile.LOW, None, "STOP"),
)

NUM_MODE_SLOTS = len(MODE_SLOTS)
MODE_NAMES = tuple(slot.name for slot in MODE_SLOTS)
