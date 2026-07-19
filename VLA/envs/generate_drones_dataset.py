from __future__ import annotations

import argparse
import importlib.util
import json
import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from VLA.envs.drones_adapter import (  # noqa: E402
    ACTION_TO_DELTA,
    build_drones_prompt,
    build_drones_record,
    get_avail_actions,
    render_drones_vla_image,
    safe_actions,
    save_drones_vla_image,
)


def load_env_drones_class():
    """Load EnvDrones directly, avoiding package-level optional dependencies."""
    env_path = PROJECT_ROOT / "src" / "envs" / "env_Drones" / "env_Drones.py"
    spec = importlib.util.spec_from_file_location("env_Drones", str(env_path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load EnvDrones from {env_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.EnvDrones


def parse_split_ratios(raw: str) -> Tuple[float, float, float]:
    ratios = tuple(float(part.strip()) for part in raw.split(","))
    if len(ratios) != 3:
        raise ValueError("--split-ratios must contain train,val,test")
    total = sum(ratios)
    if abs(total - 1.0) > 1e-6:
        raise ValueError("--split-ratios must sum to 1")
    return ratios  # type: ignore[return-value]


def split_counts(total: int, ratios: Tuple[float, float, float]) -> Dict[str, int]:
    train = int(total * ratios[0])
    val = int(total * ratios[1])
    test = total - train - val
    return {"train": train, "val": val, "test": test}


def make_env(args: argparse.Namespace, seed: int):
    EnvDrones = load_env_drones_class()
    return EnvDrones(
        map_size=args.map_size,
        drone_num=args.drone_num,
        view_range=args.view_range,
        tree_num=args.tree_num,
        human_num=args.human_num,
        episode_limit=args.episode_limit,
        wall_prob=args.wall_prob,
        seed=seed,
        human_stay_prob=args.human_stay_prob,
        human_random_move_prob=args.human_random_move_prob,
        reward_new_target=args.reward_new_target,
        reward_new_elimination=args.reward_new_elimination,
        reward_view_overlap_penalty=args.reward_view_overlap_penalty,
        reward_collision_penalty=args.reward_collision_penalty,
        reward_step_penalty=args.reward_step_penalty,
        reward_success=args.reward_success,
        reward_timeout=args.reward_timeout,
        reward_approach_coef=args.reward_approach_coef,
        drone_start_mode=args.drone_start_mode,
        allow_drone_through_wall=args.allow_drone_through_wall,
        allow_drone_through_tree=args.allow_drone_through_tree,
        attack_range=args.attack_range,
        mission_mode=args.mission_mode,
    )


def _legal_actions(avail_row: Sequence[int]) -> List[int]:
    return [idx for idx, flag in enumerate(avail_row) if int(flag) > 0]


def choose_random_actions(env: Any, rng: random.Random) -> List[int]:
    actions: List[int] = []
    for avail_row in get_avail_actions(env):
        legal = _legal_actions(avail_row)
        actions.append(int(rng.choice(legal)) if legal else 4)
    return actions


def _unfinished_human_indices(env: Any) -> List[int]:
    if str(getattr(env, "mission_mode", "detect")) == "detect":
        mask = np.asarray(env.discovered_humans) < 0.5
    elif str(getattr(env, "mission_mode", "detect")) == "eliminate":
        mask = np.asarray(env.eliminated_humans) < 0.5
    else:
        mask = (np.asarray(env.discovered_humans) < 0.5) | (np.asarray(env.eliminated_humans) < 0.5)
    return [idx for idx, unfinished in enumerate(mask.tolist()) if bool(unfinished)]


def _manhattan(a: Sequence[int], b: Sequence[int]) -> int:
    return abs(int(a[0]) - int(b[0])) + abs(int(a[1]) - int(b[1]))


def _candidate_pos(pos: Sequence[int], action: int) -> List[int]:
    dx, dy = ACTION_TO_DELTA[int(action)]
    return [int(pos[0]) + dx, int(pos[1]) + dy]


def choose_greedy_detect_actions(env: Any, rng: random.Random) -> List[int]:
    """A simple privileged expert: assign drones to nearest unfinished humans."""
    avail_actions = get_avail_actions(env)
    unfinished = _unfinished_human_indices(env)
    if not unfinished:
        return [4 for _ in env.drone_list]

    assigned_targets = set()
    actions: List[int] = []

    for drone_idx, drone in enumerate(env.drone_list):
        drone_pos = [int(drone.pos[0]), int(drone.pos[1])]
        target_order = sorted(
            unfinished,
            key=lambda h_idx: _manhattan(drone_pos, env.human_list[h_idx].pos),
        )

        target_idx = None
        for candidate in target_order:
            if candidate not in assigned_targets:
                target_idx = candidate
                break
        if target_idx is None:
            target_idx = target_order[0]
        assigned_targets.add(target_idx)

        target_pos = env.human_list[target_idx].pos
        legal = _legal_actions(avail_actions[drone_idx])
        if not legal:
            actions.append(4)
            continue

        scored: List[Tuple[int, int]] = []
        for action in legal:
            next_pos = _candidate_pos(drone_pos, action)
            scored.append((_manhattan(next_pos, target_pos), action))

        best_score = min(score for score, _ in scored)
        best_actions = [action for score, action in scored if score == best_score]
        if 4 in best_actions and len(best_actions) > 1:
            best_actions = [action for action in best_actions if action != 4]
        actions.append(int(rng.choice(best_actions)))

    return safe_actions(actions, avail_actions)


def choose_expert_actions(env: Any, expert: str, rng: random.Random) -> List[int]:
    if expert == "random":
        return choose_random_actions(env, rng)
    if expert == "greedy":
        return choose_greedy_detect_actions(env, rng)
    raise ValueError(f"Unsupported expert: {expert}")


def write_record(writer, record: Dict[str, Any]) -> None:
    writer.write(json.dumps(record, ensure_ascii=False) + "\n")


def collect_split(
    split: str,
    needed: int,
    args: argparse.Namespace,
    writer,
    rng: random.Random,
    start_episode: int,
) -> Tuple[int, int]:
    split_image_dir = Path(args.out_dir) / "images" / split
    split_image_dir.mkdir(parents=True, exist_ok=True)

    written = 0
    episode_idx = start_episode
    while written < needed:
        env_seed = args.seed + episode_idx
        env = make_env(args, env_seed)
        episode_step = 0

        while written < needed:
            actions = choose_expert_actions(env, args.expert, rng)
            image_name = f"{args.prefix}_{split}_ep{episode_idx:06d}_t{episode_step:04d}.png"
            image_path = split_image_dir / image_name
            rel_image_path = image_path.resolve().relative_to(PROJECT_ROOT)

            image = render_drones_vla_image(
                env,
                image_size=args.image_size,
                reveal_full_map=not args.partial_map,
                draw_view_range=not args.no_view_range,
                draw_attack_range=args.draw_attack_range,
                draw_grid=not args.no_grid,
            )
            save_drones_vla_image(image, image_path)

            instruction, prompt_input = build_drones_prompt(
                env,
                include_privileged_state=args.include_privileged_state,
            )
            sample_id = f"{args.prefix}_{split}_ep{episode_idx:06d}_t{episode_step:04d}"
            record = build_drones_record(
                rel_image_path,
                instruction,
                prompt_input,
                actions,
                env,
                sample_id=sample_id,
                episode=episode_idx,
                t=episode_step,
                thinking=args.thinking,
            )

            _, reward, terminated, truncated, info = env.step(
                None,
                actions,
                return_joint_obs=False,
            )
            record["reward"] = float(reward)
            record["terminated"] = bool(terminated)
            record["truncated"] = bool(truncated)
            record["info"] = info
            record["split"] = split
            record["expert"] = args.expert
            write_record(writer, record)

            written += 1
            episode_step += 1
            if terminated or truncated:
                break

        episode_idx += 1
    return written, episode_idx


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate Drones VLA Alpaca-style dataset")
    parser.add_argument("--out-dir", type=str, default="VLA/data/drones")
    parser.add_argument("--prefix", type=str, default="drones")
    parser.add_argument("--num-transitions", type=int, default=100)
    parser.add_argument("--split-ratios", type=str, default="0.8,0.1,0.1")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--expert", choices=["random", "greedy"], default="greedy")
    parser.add_argument("--image-size", type=int, default=336)
    parser.add_argument("--partial-map", action="store_true", help="Render only currently visible cells")
    parser.add_argument("--no-view-range", action="store_true")
    parser.add_argument("--draw-attack-range", action="store_true")
    parser.add_argument("--no-grid", action="store_true")
    parser.add_argument("--include-privileged-state", action="store_true", default=True)
    parser.add_argument("--no-include-privileged-state", dest="include_privileged_state", action="store_false")
    parser.add_argument(
        "--thinking",
        type=str,
        default="Move drones toward uncovered humans and avoid blocked cells.",
    )

    parser.add_argument("--map-size", type=int, default=30)
    parser.add_argument("--drone-num", type=int, default=4)
    parser.add_argument("--view-range", type=int, default=6)
    parser.add_argument("--attack-range", type=int, default=3)
    parser.add_argument("--tree-num", type=int, default=40)
    parser.add_argument("--human-num", type=int, default=6)
    parser.add_argument("--episode-limit", type=int, default=120)
    parser.add_argument("--mission-mode", choices=["detect", "eliminate", "both"], default="detect")
    parser.add_argument("--wall-prob", type=float, default=0.01)
    parser.add_argument("--human-stay-prob", type=float, default=0.2)
    parser.add_argument("--human-random-move-prob", type=float, default=0.2)
    parser.add_argument("--allow-drone-through-wall", action="store_true")
    parser.add_argument("--allow-drone-through-tree", action="store_true")
    parser.add_argument("--drone-start-mode", choices=["random", "corner"], default="random")
    parser.add_argument("--reward-new-target", type=float, default=5.0)
    parser.add_argument("--reward-new-elimination", type=float, default=0.0)
    parser.add_argument("--reward-view-overlap-penalty", type=float, default=-0.02)
    parser.add_argument("--reward-collision-penalty", type=float, default=-0.2)
    parser.add_argument("--reward-step-penalty", type=float, default=-0.01)
    parser.add_argument("--reward-success", type=float, default=50.0)
    parser.add_argument("--reward-timeout", type=float, default=0.0)
    parser.add_argument("--reward-approach-coef", type=float, default=0.05)
    args = parser.parse_args()

    if args.num_transitions <= 0:
        raise ValueError("--num-transitions must be positive")

    ratios = parse_split_ratios(args.split_ratios)
    counts = split_counts(args.num_transitions, ratios)
    out_dir = Path(args.out_dir)
    if not out_dir.is_absolute():
        out_dir = PROJECT_ROOT / out_dir
    args.out_dir = str(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rng = random.Random(args.seed)
    episode_cursor = 0
    summary: Dict[str, int] = {}

    for split in ("train", "val", "test"):
        jsonl_path = out_dir / f"{args.prefix}_{split}_alpaca.jsonl"
        with jsonl_path.open("w", encoding="utf-8") as writer:
            written, episode_cursor = collect_split(
                split,
                counts[split],
                args,
                writer,
                rng,
                episode_cursor,
            )
        summary[split] = written
        print(f"{split}: wrote {written} records -> {jsonl_path}")

    print("Summary:", json.dumps(summary, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
