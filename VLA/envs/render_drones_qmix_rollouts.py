from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
from PIL import Image, ImageDraw, ImageFont


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from VLA.envs.drones_adapter import ACTION_NAMES, render_drones_vla_image  # noqa: E402
from VLA.envs.generate_drones_qmix_dataset import (  # noqa: E402
    QMixDronesExpert,
    load_qmix_config,
    resolve_checkpoint_path,
)


def resolve_project_path(path_value: str) -> Path:
    path = Path(path_value)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value


def fill_env_defaults_from_checkpoint(args: argparse.Namespace) -> None:
    ckpt_dir = resolve_checkpoint_path(args.checkpoint_path, args.load_step)
    cfg = load_qmix_config(args.checkpoint_path, ckpt_dir, args.config_path)
    env_args = cfg.get("env_args", {})

    if args.mission_mode is None:
        args.mission_mode = env_args.get("mission_mode", "detect")

    defaults = {
        "reward_new_target": 5.0,
        "reward_new_elimination": 0.0,
        "reward_view_overlap_penalty": -0.02,
        "reward_collision_penalty": -0.2,
        "reward_step_penalty": -0.01,
        "reward_success": 50.0,
        "reward_timeout": 0.0,
        "reward_approach_coef": 0.05,
    }
    for name, fallback in defaults.items():
        if getattr(args, name) is None:
            setattr(args, name, env_args.get(name, fallback))


def action_text(actions: Optional[Sequence[int]]) -> str:
    if actions is None:
        return "-"
    return ",".join(f"{int(a)}:{ACTION_NAMES.get(int(a), str(a))}" for a in actions)


def render_frame(
    env: Any,
    args: argparse.Namespace,
    episode_idx: int,
    step_idx: int,
    actions: Optional[Sequence[int]] = None,
    reward: Optional[float] = None,
    total_return: float = 0.0,
    info: Optional[Dict[str, Any]] = None,
    final: bool = False,
) -> Image.Image:
    image = render_drones_vla_image(
        env,
        image_size=args.image_size,
        reveal_full_map=not args.partial_map,
        draw_view_range=not args.no_view_range,
        draw_attack_range=args.draw_attack_range,
        draw_grid=not args.no_grid,
    ).convert("RGB")

    panel_height = 86
    frame = Image.new("RGB", (image.width, image.height + panel_height), (248, 250, 252))
    frame.paste(image, (0, 0))
    draw = ImageDraw.Draw(frame)
    font = ImageFont.load_default()
    info = info or {}

    core = env.env if hasattr(env, "env") else env
    discovered = getattr(core, "discovered_humans", [])
    eliminated = getattr(core, "eliminated_humans", [])
    found = int(info.get("targets_found", int(np.sum(discovered))))
    eliminated_count = int(info.get("targets_eliminated", int(np.sum(eliminated))))
    total = int(info.get("targets_total", getattr(core, "human_num", 0)))
    status = "DONE" if final or bool(info.get("mission_success", False)) else "RUN"
    reward_text = "-" if reward is None else f"{float(reward):.2f}"

    y0 = image.height + 8
    lines = [
        f"ep={episode_idx} step={step_idx} status={status} mode={getattr(core, 'mission_mode', '-')}",
        f"found={found}/{total} eliminated={eliminated_count}/{total} reward={reward_text} return={total_return:.2f}",
        f"actions=[{action_text(actions)}]",
        f"invalid={int(info.get('invalid_actions', 0))} collisions={int(info.get('drone_collisions', 0))}",
    ]
    for line_idx, line in enumerate(lines):
        draw.text((10, y0 + line_idx * 18), line, fill=(15, 23, 42), font=font)
    return frame


def save_gif(frames: Sequence[Image.Image], path: Path, fps: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    duration_ms = max(1, int(round(1000.0 / max(1.0, float(fps)))))
    first, rest = frames[0], list(frames[1:])
    first.save(
        path,
        save_all=True,
        append_images=rest,
        duration=duration_ms,
        loop=0,
        optimize=False,
    )


def run_episode(
    expert: QMixDronesExpert,
    args: argparse.Namespace,
    episode_idx: int,
    output_dir: Path,
) -> Dict[str, Any]:
    env_seed = int(args.seed) + episode_idx
    env, batch = expert.start_episode(env_seed)
    max_steps = min(int(env.episode_limit), int(args.max_episode_steps))

    frames: List[Image.Image] = []
    trace: List[Dict[str, Any]] = []
    total_return = 0.0
    total_invalid = 0
    total_collisions = 0
    steps = 0
    terminated = False
    truncated = False
    last_info: Dict[str, Any] = {}

    frames.append(render_frame(env, args, episode_idx, 0, total_return=total_return))

    try:
        for step_idx in range(max_steps):
            actions, actions_tensor = expert.select_actions(env, batch, step_idx)
            _, reward, terminated, truncated, info = env.step(actions)
            expert.commit_actions(batch, step_idx, actions_tensor)

            reward = float(reward)
            total_return += reward
            total_invalid += int(info.get("invalid_actions", 0))
            total_collisions += int(info.get("drone_collisions", 0))
            last_info = dict(info)
            steps = step_idx + 1

            if args.save_traces:
                trace.append(
                    {
                        "t": int(step_idx),
                        "actions": [int(action) for action in actions],
                        "reward": reward,
                        "info": json_safe(info),
                    }
                )

            frames.append(
                render_frame(
                    env,
                    args,
                    episode_idx=episode_idx,
                    step_idx=steps,
                    actions=actions,
                    reward=reward,
                    total_return=total_return,
                    info=info,
                    final=bool(terminated or truncated),
                )
            )

            if terminated or truncated:
                break
    finally:
        if hasattr(env, "close"):
            env.close()

    core = env.env if hasattr(env, "env") else env
    targets_found = int(last_info.get("targets_found", int(np.sum(getattr(core, "discovered_humans", [])))))
    targets_eliminated = int(last_info.get("targets_eliminated", int(np.sum(getattr(core, "eliminated_humans", [])))))
    targets_total = int(last_info.get("targets_total", getattr(core, "human_num", 0)))

    gif_path = output_dir / f"episode_{episode_idx:04d}.gif"
    save_gif(frames, gif_path, args.video_fps)

    result = {
        "episode": int(episode_idx),
        "seed": int(env_seed),
        "video_path": str(gif_path.relative_to(PROJECT_ROOT)),
        "success": bool(last_info.get("mission_success", False)),
        "detect_success": bool(last_info.get("detect_success", targets_found >= targets_total and targets_total > 0)),
        "terminated": bool(terminated),
        "truncated": bool(truncated),
        "steps": int(steps),
        "return": float(total_return),
        "targets_found": int(targets_found),
        "targets_eliminated": int(targets_eliminated),
        "targets_total": int(targets_total),
        "invalid_actions": int(total_invalid),
        "drone_collisions": int(total_collisions),
    }
    if args.save_traces:
        result["trace"] = trace
    return result


def summarize(results: Sequence[Dict[str, Any]]) -> Dict[str, float]:
    if not results:
        return {}
    total_targets = max(1, sum(int(item["targets_total"]) for item in results))
    total_steps = max(1, sum(int(item["steps"]) for item in results))
    return {
        "episodes": float(len(results)),
        "success_rate": float(np.mean([float(item["success"]) for item in results])),
        "detect_success_rate": float(np.mean([float(item["detect_success"]) for item in results])),
        "avg_return": float(np.mean([float(item["return"]) for item in results])),
        "avg_steps": float(np.mean([float(item["steps"]) for item in results])),
        "avg_found": float(np.mean([float(item["targets_found"]) for item in results])),
        "avg_eliminated": float(np.mean([float(item["targets_eliminated"]) for item in results])),
        "found_ratio": float(sum(int(item["targets_found"]) for item in results) / total_targets),
        "eliminated_ratio": float(sum(int(item["targets_eliminated"]) for item in results) / total_targets),
        "invalid_actions_per_step": float(sum(int(item["invalid_actions"]) for item in results) / total_steps),
        "drone_collisions_per_step": float(sum(int(item["drone_collisions"]) for item in results) / total_steps),
    }


def save_results(results: Sequence[Dict[str, Any]], summary: Dict[str, float], output_dir: Path) -> None:
    jsonl_path = output_dir / "rollouts.jsonl"
    with jsonl_path.open("w", encoding="utf-8") as f:
        f.write(json.dumps({"summary": summary}, ensure_ascii=False) + "\n")
        for item in results:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    summary_path = output_dir / "summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2, sort_keys=True)

    print(f"Saved rollout jsonl -> {jsonl_path}")
    print(f"Saved summary -> {summary_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Render saved QMIX Drones rollouts as headless GIF videos.")
    parser.add_argument("--checkpoint-path", type=str, required=True)
    parser.add_argument("--load-step", type=int, default=None, help="Default picks latest checkpoint step")
    parser.add_argument("--config-path", type=str, default="")
    parser.add_argument("--output-dir", type=str, default="VLA/eval_runs/qmix_rollout_videos")
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--max-episode-steps", type=int, default=100000)
    parser.add_argument("--save-traces", action="store_true")

    parser.add_argument("--image-size", type=int, default=336)
    parser.add_argument("--video-fps", type=float, default=6.0)
    parser.add_argument("--partial-map", action="store_true")
    parser.add_argument("--no-view-range", action="store_true")
    parser.add_argument("--draw-attack-range", action="store_true")
    parser.add_argument("--no-grid", action="store_true")

    parser.add_argument("--mission-mode", choices=["detect", "eliminate", "both"], default=None)
    parser.add_argument("--reward-new-target", type=float, default=None)
    parser.add_argument("--reward-new-elimination", type=float, default=None)
    parser.add_argument("--reward-view-overlap-penalty", type=float, default=None)
    parser.add_argument("--reward-collision-penalty", type=float, default=None)
    parser.add_argument("--reward-step-penalty", type=float, default=None)
    parser.add_argument("--reward-success", type=float, default=None)
    parser.add_argument("--reward-timeout", type=float, default=None)
    parser.add_argument("--reward-approach-coef", type=float, default=None)
    args = parser.parse_args()

    if args.episodes <= 0:
        raise ValueError("--episodes must be positive")

    fill_env_defaults_from_checkpoint(args)
    output_dir = resolve_project_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Output dir: {output_dir}")
    print(f"Mission mode: {args.mission_mode}")
    expert = QMixDronesExpert(args)

    results: List[Dict[str, Any]] = []
    for episode_idx in range(args.episodes):
        result = run_episode(expert, args, episode_idx, output_dir)
        results.append(result)
        print(
            f"Episode {episode_idx + 1}/{args.episodes}: "
            f"success={int(result['success'])}, "
            f"found={result['targets_found']}/{result['targets_total']}, "
            f"eliminated={result['targets_eliminated']}/{result['targets_total']}, "
            f"steps={result['steps']}, "
            f"return={result['return']:.2f}, "
            f"video={result['video_path']}"
        )

    summary = summarize(results)
    print("Summary:")
    for key, value in summary.items():
        print(f"  {key}: {value:.6f}")
    save_results(results, summary, output_dir)


if __name__ == "__main__":
    main()
