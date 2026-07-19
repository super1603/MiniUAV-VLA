"""Drones VLA environment adapters and dataset utilities."""

from .drones_adapter import (
    ACTION_NAMES,
    ACTION_TO_DELTA,
    build_drones_prompt,
    build_drones_record,
    get_avail_actions,
    get_drone_positions,
    get_human_positions,
    parse_drones_actions,
    render_drones_vla_image,
    safe_actions,
    save_drones_vla_image,
    unwrap_core_env,
)
try:
    from .drones_dataset import DronesVLMDataset
except ModuleNotFoundError:
    DronesVLMDataset = None

try:
    from .drones_model_wrapper import MiniMindVLMWithDroneActions, MultiDroneActionHead
except ModuleNotFoundError:
    MiniMindVLMWithDroneActions = None
    MultiDroneActionHead = None

__all__ = [
    "ACTION_NAMES",
    "ACTION_TO_DELTA",
    "build_drones_prompt",
    "build_drones_record",
    "get_avail_actions",
    "get_drone_positions",
    "get_human_positions",
    "parse_drones_actions",
    "render_drones_vla_image",
    "safe_actions",
    "save_drones_vla_image",
    "unwrap_core_env",
]

if DronesVLMDataset is not None:
    __all__.append("DronesVLMDataset")

if MiniMindVLMWithDroneActions is not None:
    __all__.extend(["MiniMindVLMWithDroneActions", "MultiDroneActionHead"])
