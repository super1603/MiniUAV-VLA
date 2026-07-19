from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from transformers import AutoTokenizer


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from VLA.envs.drones_adapter import (  # noqa: E402
    build_drones_prompt,
    get_avail_actions,
    render_drones_vla_image,
    safe_actions,
    save_drones_vla_image,
)
from VLA.envs.drones_model_wrapper import MiniMindVLMWithDroneActions, VLMConfig  # noqa: E402


os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


def load_env_drones_class():
    env_path = PROJECT_ROOT / "src" / "envs" / "env_Drones" / "env_Drones.py"
    spec = importlib.util.spec_from_file_location("env_Drones", str(env_path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load EnvDrones from {env_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.EnvDrones


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


def load_checkpoint(model: MiniMindVLMWithDroneActions, checkpoint_path: str) -> Tuple[List[str], List[str]]:
    if not checkpoint_path or checkpoint_path.lower() == "none":
        print("Checkpoint loading disabled.")
        return [], []
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint path not found: {checkpoint_path}")

    state = torch.load(checkpoint_path, map_location="cpu")
    if isinstance(state, dict) and "model" in state and isinstance(state["model"], dict):
        state = state["model"]

    missing, unexpected = model.load_state_dict(state, strict=False)
    expected_missing = [
        key for key in missing
        if key.startswith("vision_encoder.") or key.startswith("vlm.vision_encoder.")
    ]
    other_missing = [key for key in missing if key not in expected_missing]
    print(f"Loaded checkpoint: {checkpoint_path}")
    print(
        "Missing keys: "
        f"{len(missing)} "
        f"(vision encoder expected: {len(expected_missing)}, other: {len(other_missing)}); "
        f"unexpected keys: {len(unexpected)}"
    )
    if other_missing:
        print("First non-vision missing keys:", other_missing[:8])
    if unexpected:
        print("First unexpected keys:", list(unexpected)[:8])
    return list(missing), list(unexpected)


def create_prompt(tokenizer, image_special_token: str, instruction: str, user_input: str,
                  image_token_len: int = 1) -> str:
    content_parts = ["<image>", instruction.strip() if instruction else ""]
    if user_input:
        content_parts.append(user_input.strip())
    content = "\n".join(part for part in content_parts if part)
    content = content.replace("<image>", image_special_token * int(image_token_len))

    messages = [{"role": "user", "content": content}]
    if hasattr(tokenizer, "apply_chat_template"):
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
    return f"<|im_start|>user\n{content}<|im_end|>\n<|im_start|>assistant\n"


def encode_prompt(tokenizer, prompt: str, max_seq_len: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
    input_ids = tokenizer(prompt, add_special_tokens=False).input_ids
    input_ids = input_ids[: max(1, int(max_seq_len) - 1)]
    if not input_ids:
        raise ValueError("Prompt tokenization produced no input ids")

    input_tensor = torch.tensor(input_ids, dtype=torch.long, device=device).unsqueeze(0)
    attention_mask = torch.ones_like(input_tensor, dtype=torch.long, device=device)
    return input_tensor, attention_mask


def image_to_pixel_values(model: MiniMindVLMWithDroneActions, image: Image.Image, device: torch.device, disable_vision: bool):
    if disable_vision:
        return None
    if model.processor is None:
        raise RuntimeError("Vision processor is unavailable. Use --disable_vision for text-only evaluation.")
    pixel_values = model.vlm.image2tensor(image, model.processor)
    if hasattr(pixel_values, "items"):
        return {key: value.to(device) for key, value in pixel_values.items()}
    return pixel_values.to(device)


def count_raw_illegal(raw_actions: Sequence[int], avail_actions: Sequence[Sequence[int]]) -> int:
    illegal = 0
    for agent_idx, action in enumerate(raw_actions):
        if agent_idx >= len(avail_actions):
            illegal += 1
            continue
        row = avail_actions[agent_idx]
        action = int(action)
        if action < 0 or action >= len(row) or int(row[action]) == 0:
            illegal += 1
    return illegal


# action → (row_delta, col_delta)
_ACTION_DELTAS = {0: (-1, 0), 1: (1, 0), 2: (0, -1), 3: (0, 1), 4: (0, 0)}


def predict_actions_random(env: Any, args: argparse.Namespace) -> Tuple[List[int], List[int], int]:
    avail_actions = get_avail_actions(env)
    raw_actions = []
    for agent_avail in avail_actions:
        legal = [i for i, v in enumerate(agent_avail) if v]
        raw_actions.append(int(np.random.choice(legal)) if legal else 4)
    return raw_actions, raw_actions, 0


def predict_actions_greedy(env: Any, args: argparse.Namespace) -> Tuple[List[int], List[int], int]:
    """View-range-limited greedy: navigate to nearest visible human; explore randomly if none visible."""
    avail_actions = get_avail_actions(env)
    view_range = int(getattr(env, "view_range", 4))

    mission_mode = getattr(env, "mission_mode", "detect")
    discovered = np.array(getattr(env, "discovered_humans", []), dtype=float)
    eliminated = np.array(getattr(env, "eliminated_humans", []), dtype=float)
    if mission_mode == "detect":
        active_mask = discovered < 0.5
    else:
        active_mask = eliminated < 0.5

    # (idx, row, col) for active humans
    active_humans = [
        (int(h.pos[0]), int(h.pos[1]))
        for i, h in enumerate(env.human_list)
        if active_mask[i]
    ]

    raw_actions = []
    for drone_idx, drone in enumerate(env.drone_list):
        dr, dc = int(drone.pos[0]), int(drone.pos[1])
        agent_avail = avail_actions[drone_idx] if drone_idx < len(avail_actions) else [0, 0, 0, 0, 1]
        legal = [a for a, v in enumerate(agent_avail) if v]

        # only consider humans within Euclidean view_range
        visible_humans = [
            (hr, hc) for hr, hc in active_humans
            if (hr - dr) ** 2 + (hc - dc) ** 2 <= view_range ** 2
        ]

        if not visible_humans or not legal:
            # no visible target: random exploration
            raw_actions.append(int(np.random.choice(legal)) if legal else 4)
            continue

        # navigate to nearest visible human (Manhattan distance)
        target_r, target_c = min(visible_humans, key=lambda p: abs(p[0] - dr) + abs(p[1] - dc))

        best_action = legal[0]
        best_result_dist = float("inf")
        for a in legal:
            ddr, ddc = _ACTION_DELTAS.get(a, (0, 0))
            new_dist = abs((dr + ddr) - target_r) + abs((dc + ddc) - target_c)
            if new_dist < best_result_dist:
                best_result_dist = new_dist
                best_action = a

        raw_actions.append(best_action)

    return raw_actions, raw_actions, 0


@torch.no_grad()
def predict_actions(
    model: MiniMindVLMWithDroneActions,
    tokenizer,
    env: Any,
    args: argparse.Namespace,
    device: torch.device,
    autocast_ctx,
) -> Tuple[List[int], List[int], int, torch.Tensor]:
    instruction, user_input = build_drones_prompt(
        env,
        include_privileged_state=args.include_privileged_state,
    )
    prompt = create_prompt(
        tokenizer,
        model.config.image_special_token,
        instruction,
        user_input,
        image_token_len=getattr(model.config, "image_token_len", 1),
    )
    input_ids, attention_mask = encode_prompt(tokenizer, prompt, args.max_seq_len, device)

    image = render_drones_vla_image(
        env,
        image_size=args.image_size,
        reveal_full_map=not args.partial_map,
        draw_view_range=not args.no_view_range,
        draw_attack_range=args.draw_attack_range,
        draw_grid=not args.no_grid,
    )
    pixel_values = image_to_pixel_values(model, image, device, disable_vision=args.disable_vision)

    with autocast_ctx:
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
        )

    action_logits = outputs.action_logits[0].float().detach().cpu()
    raw_actions = action_logits.argmax(dim=-1).tolist()
    avail_actions = get_avail_actions(env)
    illegal = count_raw_illegal(raw_actions, avail_actions)
    masked_actions = safe_actions(raw_actions, avail_actions, logits=action_logits)
    return raw_actions, masked_actions, illegal, action_logits


def maybe_save_debug_image(env: Any, args: argparse.Namespace, episode_idx: int, step_idx: int) -> Optional[str]:
    if not args.debug_image_dir or episode_idx >= args.debug_episodes:
        return None

    out_dir = Path(args.debug_image_dir)
    if not out_dir.is_absolute():
        out_dir = PROJECT_ROOT / out_dir
    image_path = out_dir / f"episode_{episode_idx:04d}_step_{step_idx:04d}.png"
    image = render_drones_vla_image(
        env,
        image_size=args.image_size,
        reveal_full_map=not args.partial_map,
        draw_view_range=not args.no_view_range,
        draw_attack_range=args.draw_attack_range,
        draw_grid=not args.no_grid,
    )
    return save_drones_vla_image(image, image_path)


def render_eval_frame(
    env: Any,
    args: argparse.Namespace,
    episode_idx: int,
    step_idx: int,
    raw_actions: Optional[Sequence[int]] = None,
    masked_actions: Optional[Sequence[int]] = None,
    reward: Optional[float] = None,
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

    panel_height = 72
    frame = Image.new("RGB", (image.width, image.height + panel_height), (248, 250, 252))
    frame.paste(image, (0, 0))
    draw = ImageDraw.Draw(frame)
    font = ImageFont.load_default()

    info = info or {}
    targets_found = int(info.get("targets_found", int(np.sum(getattr(env, "discovered_humans", [])))))
    targets_total = int(info.get("targets_total", getattr(env, "human_num", 0)))
    status = "DONE" if final or bool(info.get("mission_success", False)) else "RUN"
    raw_text = "-" if raw_actions is None else ",".join(str(int(a)) for a in raw_actions)
    masked_text = "-" if masked_actions is None else ",".join(str(int(a)) for a in masked_actions)
    reward_text = "-" if reward is None else f"{float(reward):.2f}"

    y0 = image.height + 8
    lines = [
        f"ep={episode_idx} step={step_idx} status={status} found={targets_found}/{targets_total} reward={reward_text}",
        f"raw_actions=[{raw_text}] masked_actions=[{masked_text}]",
        f"invalid={int(info.get('invalid_actions', 0))} collisions={int(info.get('drone_collisions', 0))}",
    ]
    for line_idx, line in enumerate(lines):
        draw.text((10, y0 + line_idx * 18), line, fill=(15, 23, 42), font=font)
    return frame


def resolve_project_path(path_value: str) -> Path:
    path = Path(path_value)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path


def save_episode_gif(frames: Sequence[Image.Image], args: argparse.Namespace, episode_idx: int) -> Optional[str]:
    if not frames or not args.video_dir or episode_idx >= args.video_episodes:
        return None

    video_dir = resolve_project_path(args.video_dir)
    video_dir.mkdir(parents=True, exist_ok=True)
    gif_path = video_dir / f"episode_{episode_idx:04d}.gif"
    duration_ms = max(1, int(round(1000.0 / max(1.0, float(args.video_fps)))))
    first, rest = frames[0], list(frames[1:])
    first.save(
        gif_path,
        save_all=True,
        append_images=rest,
        duration=duration_ms,
        loop=0,
        optimize=False,
    )
    return str(gif_path)


def run_episode(
    model: Optional[MiniMindVLMWithDroneActions],
    tokenizer,
    args: argparse.Namespace,
    episode_idx: int,
    device: torch.device,
    autocast_ctx,
) -> Dict[str, Any]:
    env_seed = args.seed + episode_idx
    env = make_env(args, env_seed)
    env.reset(seed=env_seed, options={"return_joint_obs": False})

    total_reward = 0.0
    raw_illegal_actions = 0
    masked_action_changes = 0
    env_invalid_actions = 0
    drone_collisions = 0
    steps = 0
    last_info: Dict[str, Any] = {}
    trace_steps: List[Dict[str, Any]] = []
    video_frames: List[Image.Image] = []

    max_steps = args.max_steps if args.max_steps > 0 else args.episode_limit
    terminated = False
    truncated = False
    policy = getattr(args, "policy", "vla")

    for step_idx in range(max_steps):
        debug_image = maybe_save_debug_image(env, args, episode_idx, step_idx)

        if policy == "random":
            raw_actions, actions, illegal = predict_actions_random(env, args)
            action_logits = torch.zeros(args.n_agents, args.n_actions)
        elif policy == "greedy":
            raw_actions, actions, illegal = predict_actions_greedy(env, args)
            action_logits = torch.zeros(args.n_agents, args.n_actions)
        else:
            raw_actions, actions, illegal, action_logits = predict_actions(
                model, tokenizer, env, args, device, autocast_ctx,
            )

        raw_illegal_actions += illegal
        masked_action_changes += sum(int(a != b) for a, b in zip(raw_actions, actions))

        _, reward, terminated, truncated, info = env.step(
            None,
            actions,
            return_joint_obs=False,
        )
        total_reward += float(reward)
        env_invalid_actions += int(info.get("invalid_actions", 0))
        drone_collisions += int(info.get("drone_collisions", 0))
        last_info = dict(info)
        steps = step_idx + 1

        if args.video_dir and episode_idx < args.video_episodes:
            video_frames.append(
                render_eval_frame(
                    env,
                    args,
                    episode_idx=episode_idx,
                    step_idx=step_idx,
                    raw_actions=raw_actions,
                    masked_actions=actions,
                    reward=reward,
                    info=info,
                    final=bool(terminated or truncated),
                )
            )

        if args.save_traces:
            trace_steps.append(
                {
                    "t": step_idx,
                    "raw_actions": [int(a) for a in raw_actions],
                    "masked_actions": [int(a) for a in actions],
                    "raw_illegal_actions": int(illegal),
                    "reward": float(reward),
                    "targets_found": int(info.get("targets_found", 0)),
                    "mission_success": bool(info.get("mission_success", False)),
                    "debug_image": debug_image,
                    "action_logits": action_logits.numpy().round(4).tolist() if args.save_logits else None,
                }
            )

        if terminated or truncated:
            break

    video_path = save_episode_gif(video_frames, args, episode_idx)

    targets_found = int(last_info.get("targets_found", int(np.sum(env.discovered_humans))))
    targets_total = int(last_info.get("targets_total", env.human_num))
    result = {
        "episode": episode_idx,
        "seed": env_seed,
        "success": bool(last_info.get("mission_success", terminated)),
        "detect_success": bool(last_info.get("detect_success", targets_found >= targets_total)),
        "terminated": bool(terminated),
        "truncated": bool(truncated),
        "steps": int(steps),
        "return": float(total_reward),
        "targets_found": targets_found,
        "targets_total": targets_total,
        "raw_illegal_actions": int(raw_illegal_actions),
        "masked_action_changes": int(masked_action_changes),
        "env_invalid_actions": int(env_invalid_actions),
        "drone_collisions": int(drone_collisions),
    }
    if video_path is not None:
        result["video_path"] = video_path
    if args.save_traces:
        result["trace"] = trace_steps
    return result


def summarize(results: Sequence[Dict[str, Any]], n_agents: int) -> Dict[str, float]:
    if not results:
        return {}

    total_steps = sum(int(item["steps"]) for item in results)
    total_agent_steps = max(1, total_steps * int(n_agents))
    total_targets = max(1, sum(int(item["targets_total"]) for item in results))

    return {
        "episodes": float(len(results)),
        "success_rate": float(np.mean([float(item["success"]) for item in results])),
        "detect_success_rate": float(np.mean([float(item["detect_success"]) for item in results])),
        "avg_return": float(np.mean([float(item["return"]) for item in results])),
        "avg_steps": float(np.mean([float(item["steps"]) for item in results])),
        "avg_detected_humans": float(np.mean([float(item["targets_found"]) for item in results])),
        "detected_human_ratio": float(sum(int(item["targets_found"]) for item in results) / total_targets),
        "raw_illegal_action_rate": float(sum(int(item["raw_illegal_actions"]) for item in results) / total_agent_steps),
        "masked_action_change_rate": float(sum(int(item["masked_action_changes"]) for item in results) / total_agent_steps),
        "env_invalid_action_rate": float(sum(int(item["env_invalid_actions"]) for item in results) / total_agent_steps),
        "drone_collision_rate": float(sum(int(item["drone_collisions"]) for item in results) / total_agent_steps),
    }


def save_results(results: Sequence[Dict[str, Any]], summary: Dict[str, float], output_path: str) -> None:
    path = Path(output_path)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        f.write(json.dumps({"summary": summary}, ensure_ascii=False) + "\n")
        for item in results:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(f"Saved evaluation results -> {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Closed-loop evaluation for MiniMind-V Drones VLA")
    parser.add_argument("--policy", type=str, choices=["vla", "random", "greedy"], default="vla",
                        help="Action policy: vla=model inference, random=random legal action, greedy=nearest-target heuristic")
    parser.add_argument("--checkpoint_path", type=str, default="VLA/models/sft_vlm_drones_greedy_1k_768.pth")
    parser.add_argument("--tokenizer_path", type=str, default="minimind-v/model")
    parser.add_argument("--vision_model_path", type=str, default="model/siglip2-base-p16-224")
    parser.add_argument("--disable_vision", action="store_true")
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", choices=["float16", "bfloat16"], default="bfloat16")
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--max_steps", type=int, default=0, help="0 means use --episode-limit")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--output_path", type=str, default="VLA/eval_runs/drones_closed_loop_eval.jsonl")
    parser.add_argument("--save_traces", action="store_true")
    parser.add_argument("--save_logits", action="store_true")
    parser.add_argument("--debug_image_dir", type=str, default="")
    parser.add_argument("--debug_episodes", type=int, default=1)
    parser.add_argument("--video_dir", type=str, default="", help="Save per-episode GIFs when set")
    parser.add_argument("--video_episodes", type=int, default=1, help="Number of leading episodes to record")
    parser.add_argument("--video_fps", type=float, default=6.0)

    parser.add_argument("--hidden_size", type=int, default=768)
    parser.add_argument("--num_hidden_layers", type=int, default=8)
    parser.add_argument("--num_attention_heads", type=int, default=8)
    parser.add_argument("--num_key_value_heads", type=int, default=4)
    parser.add_argument("--max_seq_len", type=int, default=512)
    parser.add_argument("--image_token_len", type=int, default=196,
                        help="Number of image marker tokens per image; must match training.")
    parser.add_argument("--use_moe", type=int, choices=[0, 1], default=0)
    parser.add_argument("--n_agents", type=int, default=4)
    parser.add_argument("--n_actions", type=int, default=5)

    parser.add_argument("--image_size", type=int, default=336)
    parser.add_argument("--partial_map", action="store_true")
    parser.add_argument("--no_view_range", action="store_true")
    parser.add_argument("--draw_attack_range", action="store_true")
    parser.add_argument("--no_grid", action="store_true")
    parser.add_argument("--include_privileged_state", action="store_true", default=True)
    parser.add_argument("--no_include_privileged_state", dest="include_privileged_state", action="store_false")

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

    device = torch.device(args.device)
    tokenizer = None
    model = None
    autocast_ctx = nullcontext()

    if args.policy == "vla":
        model_config = VLMConfig(
            hidden_size=args.hidden_size,
            num_hidden_layers=args.num_hidden_layers,
            num_attention_heads=args.num_attention_heads,
            num_key_value_heads=args.num_key_value_heads,
            use_moe=bool(args.use_moe),
            max_seq_len=args.max_seq_len,
            image_token_len=args.image_token_len,
        )
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        vision_model_path = "missing_vision_model" if args.disable_vision else args.vision_model_path
        model = MiniMindVLMWithDroneActions(
            model_config,
            vision_model_path=vision_model_path,
            n_agents=args.n_agents,
            n_actions=args.n_actions,
        )
        load_checkpoint(model, args.checkpoint_path)
        model.to(device)
        model.eval()
        device_type = "cuda" if "cuda" in args.device else "cpu"
        amp_dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16
        autocast_ctx = nullcontext() if device_type == "cpu" else torch.cuda.amp.autocast(dtype=amp_dtype)
    else:
        print(f"Policy: {args.policy} — skipping model loading.")

    results: List[Dict[str, Any]] = []
    for episode_idx in range(args.episodes):
        result = run_episode(model, tokenizer, args, episode_idx, device, autocast_ctx)
        results.append(result)
        print(
            "Episode "
            f"{episode_idx + 1}/{args.episodes}: "
            f"success={int(result['success'])}, "
            f"steps={result['steps']}, "
            f"found={result['targets_found']}/{result['targets_total']}, "
            f"return={result['return']:.2f}, "
            f"raw_illegal={result['raw_illegal_actions']}, "
            f"collisions={result['drone_collisions']}"
        )

    summary = summarize(results, args.n_agents)
    print("Summary:")
    for key, value in summary.items():
        print(f"  {key}: {value:.6f}")

    if args.output_path:
        save_results(results, summary, args.output_path)


if __name__ == "__main__":
    main()
