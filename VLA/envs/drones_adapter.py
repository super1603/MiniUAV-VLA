from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np
from PIL import Image, ImageDraw, ImageFont


ACTION_TO_DELTA = {
    0: (-1, 0),  # up
    1: (1, 0),   # down
    2: (0, -1),  # left
    3: (0, 1),   # right
    4: (0, 0),   # stay
}

ACTION_NAMES = {
    0: "UP",
    1: "DOWN",
    2: "LEFT",
    3: "RIGHT",
    4: "STAY",
}

DRONE_COLORS = [
    (37, 99, 235),
    (8, 145, 178),
    (124, 58, 237),
    (202, 138, 4),
    (220, 38, 38),
    (22, 163, 74),
]

COLOR_UNKNOWN = (142, 148, 160)
COLOR_EMPTY = (248, 250, 252)
COLOR_WALL = (15, 23, 42)
COLOR_TREE = (34, 197, 94)
COLOR_HUMAN = (239, 68, 68)
COLOR_HUMAN_DISCOVERED = (249, 115, 22)
COLOR_GRID = (203, 213, 225)
COLOR_VIEW = (59, 130, 246)
COLOR_ATTACK = (244, 63, 94)


def unwrap_core_env(env_or_wrapper: Any) -> Any:
    """Return the underlying EnvDrones-like object."""
    if hasattr(env_or_wrapper, "land_mark_map") and hasattr(env_or_wrapper, "drone_list"):
        return env_or_wrapper
    if hasattr(env_or_wrapper, "env"):
        core = getattr(env_or_wrapper, "env")
        if hasattr(core, "land_mark_map") and hasattr(core, "drone_list"):
            return core
    raise TypeError("Expected an EnvDrones instance or a wrapper with an .env EnvDrones instance")


def _as_list(values: Any) -> List[Any]:
    if values is None:
        return []
    if isinstance(values, np.ndarray):
        return values.tolist()
    return list(values)


def get_drone_positions(env_or_wrapper: Any) -> List[List[int]]:
    core = unwrap_core_env(env_or_wrapper)
    return [[int(d.pos[0]), int(d.pos[1])] for d in core.drone_list]


def get_human_positions(env_or_wrapper: Any, include_eliminated: bool = False) -> List[List[int]]:
    core = unwrap_core_env(env_or_wrapper)
    positions: List[List[int]] = []
    eliminated = _as_list(getattr(core, "eliminated_humans", []))
    for idx, human in enumerate(core.human_list):
        if not include_eliminated and idx < len(eliminated) and float(eliminated[idx]) > 0.5:
            continue
        positions.append([int(human.pos[0]), int(human.pos[1])])
    return positions


def get_avail_actions(env_or_wrapper: Any) -> List[List[int]]:
    if hasattr(env_or_wrapper, "get_avail_actions"):
        return [[int(v) for v in row] for row in env_or_wrapper.get_avail_actions()]

    core = unwrap_core_env(env_or_wrapper)
    avail_actions: List[List[int]] = []
    for drone in core.drone_list:
        row: List[int] = []
        for action in range(5):
            dx, dy = ACTION_TO_DELTA[action]
            nx = int(drone.pos[0]) + dx
            ny = int(drone.pos[1]) + dy
            if action == 4:
                row.append(1)
            elif hasattr(core, "_is_blocked_for_drone"):
                row.append(0 if core._is_blocked_for_drone(nx, ny) else 1)
            else:
                row.append(1 if 0 <= nx < core.map_size and 0 <= ny < core.map_size else 0)
        avail_actions.append(row)
    return avail_actions


def _visible_mask(core: Any) -> np.ndarray:
    mask = np.zeros((int(core.map_size), int(core.map_size)), dtype=bool)
    for drone in core.drone_list:
        if hasattr(core, "_drone_visible_cells"):
            for x, y in core._drone_visible_cells(drone):
                mask[int(x), int(y)] = True
        else:
            radius = int(getattr(drone, "view_range", getattr(core, "view_range", 1)))
            cx, cy = int(drone.pos[0]), int(drone.pos[1])
            for x in range(max(0, cx - radius), min(core.map_size, cx + radius + 1)):
                for y in range(max(0, cy - radius), min(core.map_size, cy + radius + 1)):
                    dx = x - cx
                    dy = y - cy
                    if dx * dx + dy * dy <= radius * radius:
                        mask[x, y] = True
    return mask


def _cell_box(x: int, y: int, cell: float, margin: int) -> Tuple[int, int, int, int]:
    left = int(round(margin + y * cell))
    top = int(round(margin + x * cell))
    right = int(round(margin + (y + 1) * cell))
    bottom = int(round(margin + (x + 1) * cell))
    return left, top, right, bottom


def _cell_center(x: int, y: int, cell: float, margin: int) -> Tuple[float, float]:
    return margin + (y + 0.5) * cell, margin + (x + 0.5) * cell


def render_drones_vla_image(
    env_or_wrapper: Any,
    image_size: int = 336,
    reveal_full_map: bool = True,
    draw_view_range: bool = True,
    draw_attack_range: bool = False,
    draw_grid: bool = True,
) -> Image.Image:
    """Render an EnvDrones state as a VLA-friendly tactical map."""
    core = unwrap_core_env(env_or_wrapper)
    map_size = int(core.map_size)
    margin = max(8, int(image_size * 0.035))
    board_size = image_size - 2 * margin
    cell = board_size / float(map_size)

    image = Image.new("RGB", (image_size, image_size), COLOR_UNKNOWN)
    draw = ImageDraw.Draw(image, "RGBA")
    font = ImageFont.load_default()
    visible = _visible_mask(core)

    for x in range(map_size):
        for y in range(map_size):
            known = reveal_full_map or bool(visible[x, y])
            if not known:
                color = COLOR_UNKNOWN
            else:
                tile = int(core.land_mark_map[x, y])
                if tile == 1:
                    color = COLOR_WALL
                elif tile == 2:
                    color = COLOR_TREE
                else:
                    color = COLOR_EMPTY
            box = _cell_box(x, y, cell, margin)
            draw.rectangle(box, fill=color + (255,))
            if draw_grid and cell >= 8:
                draw.rectangle(box, outline=COLOR_GRID + (90,), width=1)

    discovered = _as_list(getattr(core, "discovered_humans", []))
    eliminated = _as_list(getattr(core, "eliminated_humans", []))
    for h_idx, human in enumerate(core.human_list):
        if h_idx < len(eliminated) and float(eliminated[h_idx]) > 0.5:
            continue
        x, y = int(human.pos[0]), int(human.pos[1])
        if not reveal_full_map and not bool(visible[x, y]):
            continue
        cx, cy = _cell_center(x, y, cell, margin)
        radius = max(3, cell * 0.34)
        color = COLOR_HUMAN_DISCOVERED if h_idx < len(discovered) and float(discovered[h_idx]) > 0.5 else COLOR_HUMAN
        draw.ellipse((cx - radius, cy - radius, cx + radius, cy + radius), fill=color + (230,))
        label = f"H{h_idx}"
        draw.text((cx + radius * 0.75, cy - radius * 1.15), label, fill=(127, 29, 29, 255), font=font)

    if draw_view_range or draw_attack_range:
        for d_idx, drone in enumerate(core.drone_list):
            cx, cy = _cell_center(int(drone.pos[0]), int(drone.pos[1]), cell, margin)
            if draw_view_range:
                radius = float(getattr(drone, "view_range", getattr(core, "view_range", 1))) * cell
                draw.ellipse(
                    (cx - radius, cy - radius, cx + radius, cy + radius),
                    outline=COLOR_VIEW + (100,),
                    width=max(1, int(cell * 0.10)),
                )
            if draw_attack_range and hasattr(core, "attack_range"):
                radius = float(core.attack_range) * cell
                draw.ellipse(
                    (cx - radius, cy - radius, cx + radius, cy + radius),
                    outline=COLOR_ATTACK + (110,),
                    width=max(1, int(cell * 0.08)),
                )

    for d_idx, drone in enumerate(core.drone_list):
        x, y = int(drone.pos[0]), int(drone.pos[1])
        cx, cy = _cell_center(x, y, cell, margin)
        radius = max(5, cell * 0.45)
        color = DRONE_COLORS[d_idx % len(DRONE_COLORS)]
        draw.ellipse((cx - radius, cy - radius, cx + radius, cy + radius), fill=color + (245,), outline=(255, 255, 255, 255), width=2)
        label = f"D{d_idx}"
        bbox = draw.textbbox((0, 0), label, font=font)
        tw = bbox[2] - bbox[0]
        th = bbox[3] - bbox[1]
        draw.text((cx - tw / 2, cy - th / 2), label, fill=(255, 255, 255, 255), font=font)

    draw.rectangle((margin, margin, margin + board_size, margin + board_size), outline=(15, 23, 42, 255), width=2)
    return image


def save_drones_vla_image(image: Image.Image, path: Union[str, Path]) -> str:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path)
    return str(output_path)


def _format_positions(prefix: str, positions: Sequence[Sequence[int]]) -> str:
    parts = [f"{prefix}{idx}=({int(pos[0])},{int(pos[1])})" for idx, pos in enumerate(positions)]
    return ", ".join(parts)


def _format_compact_positions(positions: Sequence[Sequence[int]]) -> str:
    return "|".join(f"{int(pos[0])},{int(pos[1])}" for pos in positions)


def _format_compact_masks(avail_actions: Sequence[Sequence[int]]) -> str:
    return "|".join("".join(str(int(flag)) for flag in row) for row in avail_actions)


def build_drones_prompt(
    env_or_wrapper: Any,
    include_privileged_state: bool = True,
) -> Tuple[str, str]:
    core = unwrap_core_env(env_or_wrapper)
    drone_positions = get_drone_positions(env_or_wrapper)
    human_positions = get_human_positions(env_or_wrapper, include_eliminated=False)
    avail_actions = get_avail_actions(env_or_wrapper)
    discovered = _as_list(getattr(core, "discovered_humans", []))
    eliminated = _as_list(getattr(core, "eliminated_humans", []))

    targets_found = int(np.sum(np.asarray(discovered, dtype=np.float32))) if discovered else 0
    targets_eliminated = int(np.sum(np.asarray(eliminated, dtype=np.float32))) if eliminated else 0
    targets_total = int(getattr(core, "human_num", len(getattr(core, "human_list", []))))
    mission_mode = str(getattr(core, "mission_mode", "detect"))

    instruction = "Control drones. Output JSON only: {\"actions\":[a0,a1,...]}."
    lines = [
        (
            f"T=detect;mode={mission_mode};vr={int(core.view_range)};"
            f"map={int(core.map_size)};t={int(getattr(core, 'step_count', 0))}/{int(getattr(core, 'episode_limit', 0))};"
            f"A=0U1D2L3R4S;N={len(drone_positions)};"
            f"D={_format_compact_positions(drone_positions)};"
            f"F={targets_found}/{targets_total};E={targets_eliminated}/{targets_total};"
            f"M={_format_compact_masks(avail_actions)}"
        )
    ]

    if include_privileged_state:
        lines[0] += f";H={_format_compact_positions(human_positions)}"

    lines[0] += f";return {len(drone_positions)} actions."
    return instruction, "\n".join(lines)


def build_drones_record(
    image_path: Union[str, Path],
    instruction: str,
    prompt_input: str,
    expert_actions: Sequence[int],
    env_or_wrapper: Any,
    reward: Optional[float] = None,
    terminated: bool = False,
    truncated: bool = False,
    info: Optional[Dict[str, Any]] = None,
    sample_id: Optional[str] = None,
    episode: Optional[int] = None,
    t: Optional[int] = None,
    thinking: str = "Move drones toward uncovered humans and avoid blocked cells.",
) -> Dict[str, Any]:
    core = unwrap_core_env(env_or_wrapper)
    actions = [int(a) for a in expert_actions]
    output_obj = {
        "thinking": thinking,
        "actions": actions,
    }

    record: Dict[str, Any] = {
        "instruction": instruction,
        "input": prompt_input,
        "output": json.dumps(output_obj, ensure_ascii=False, separators=(",", ":")),
        "image_path": str(image_path),
        "meta": {
            "expert_actions": actions,
            "avail_actions": get_avail_actions(env_or_wrapper),
            "drone_positions": get_drone_positions(env_or_wrapper),
            "human_positions": get_human_positions(env_or_wrapper, include_eliminated=False),
            "targets_found": int(np.sum(getattr(core, "discovered_humans", np.zeros(0, dtype=np.float32)))),
            "targets_eliminated": int(np.sum(getattr(core, "eliminated_humans", np.zeros(0, dtype=np.float32)))),
            "targets_total": int(getattr(core, "human_num", len(getattr(core, "human_list", [])))),
            "map_size": int(getattr(core, "map_size", 0)),
            "drone_num": int(getattr(core, "drone_num", len(getattr(core, "drone_list", [])))),
            "view_range": int(getattr(core, "view_range", 0)),
            "attack_range": int(getattr(core, "attack_range", 0)),
            "mission_mode": str(getattr(core, "mission_mode", "")),
            "step_count": int(getattr(core, "step_count", 0)),
            "episode_limit": int(getattr(core, "episode_limit", 0)),
        },
    }

    if sample_id is not None:
        record["id"] = str(sample_id)
    if episode is not None:
        record["episode"] = int(episode)
    if t is not None:
        record["t"] = int(t)
    if reward is not None:
        record["reward"] = float(reward)
    record["terminated"] = bool(terminated)
    record["truncated"] = bool(truncated)
    if info is not None:
        record["info"] = _json_safe(info)
    return record


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value


def parse_drones_actions(text: Union[str, Dict[str, Any]], n_agents: int = 4, default_action: int = 4) -> List[int]:
    obj: Any = text
    if isinstance(text, str):
        raw = text.strip()
        try:
            obj = json.loads(raw)
        except json.JSONDecodeError:
            match = re.search(r"\{.*\}", raw, flags=re.DOTALL)
            if not match:
                return [default_action for _ in range(n_agents)]
            try:
                obj = json.loads(match.group(0))
            except json.JSONDecodeError:
                return [default_action for _ in range(n_agents)]

    actions = None
    if isinstance(obj, dict):
        actions = obj.get("actions", obj.get("action"))
    elif isinstance(obj, list):
        actions = obj

    if not isinstance(actions, Iterable) or isinstance(actions, (str, bytes)):
        return [default_action for _ in range(n_agents)]

    parsed: List[int] = []
    for action in actions:
        try:
            parsed.append(int(action))
        except (TypeError, ValueError):
            parsed.append(default_action)

    if len(parsed) < n_agents:
        parsed.extend([default_action] * (n_agents - len(parsed)))
    return parsed[:n_agents]


def _to_numpy_logits(logits: Any) -> Optional[np.ndarray]:
    if logits is None:
        return None
    if hasattr(logits, "detach"):
        logits = logits.detach()
    if hasattr(logits, "cpu"):
        logits = logits.cpu()
    try:
        return np.asarray(logits, dtype=np.float32)
    except Exception:
        return None


def safe_actions(
    actions: Sequence[int],
    avail_actions: Sequence[Sequence[int]],
    logits: Any = None,
    stay_action: int = 4,
) -> List[int]:
    safe: List[int] = []
    logits_np = _to_numpy_logits(logits)

    for agent_idx, avail_row in enumerate(avail_actions):
        avail = [int(v) for v in avail_row]
        legal = [idx for idx, flag in enumerate(avail) if flag]
        if not legal:
            safe.append(stay_action)
            continue

        action = int(actions[agent_idx]) if agent_idx < len(actions) else stay_action
        if 0 <= action < len(avail) and avail[action]:
            safe.append(action)
            continue

        if logits_np is not None and agent_idx < logits_np.shape[0]:
            row = logits_np[agent_idx]
            best = max(legal, key=lambda idx: float(row[idx]) if idx < row.shape[0] else -np.inf)
            safe.append(int(best))
        elif stay_action in legal:
            safe.append(stay_action)
        else:
            safe.append(int(legal[0]))
    return safe


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
