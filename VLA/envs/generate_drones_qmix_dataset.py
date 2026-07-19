from __future__ import annotations

import argparse
import importlib
import json
import random
import sys
import types
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch as th


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from components.episode_buffer import EpisodeBatch  # noqa: E402
from components.transforms import OneHot  # noqa: E402
from controllers import REGISTRY as mac_REGISTRY  # noqa: E402
from VLA.envs.drones_adapter import (  # noqa: E402
    build_drones_prompt,
    build_drones_record,
    render_drones_vla_image,
    save_drones_vla_image,
)


def load_drones_wrapper_class():
    """Load DronesWrapper without importing src/envs/__init__.py optional deps."""
    if "envs" not in sys.modules:
        pkg = types.ModuleType("envs")
        pkg.__path__ = [str(SRC_ROOT / "envs")]
        sys.modules["envs"] = pkg
    return importlib.import_module("envs.drones_wrapper").DronesWrapper


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


def resolve_checkpoint_path(checkpoint_path: str, load_step: Optional[int]) -> Path:
    root = Path(checkpoint_path)
    if not root.is_absolute():
        root = PROJECT_ROOT / root
    if (root / "agent.th").exists():
        return root

    step_dirs = [p for p in root.iterdir() if p.is_dir() and p.name.isdigit()]
    if not step_dirs:
        raise FileNotFoundError(f"No checkpoint step directories found under: {root}")

    step_dirs = sorted(step_dirs, key=lambda p: int(p.name))
    if load_step is None:
        return step_dirs[-1]
    return min(step_dirs, key=lambda p: abs(int(p.name) - int(load_step)))


def resolve_config_path(checkpoint_path: str, ckpt_dir: Path, config_path: str = "") -> Path:
    candidates: List[Path] = []
    if config_path:
        path = Path(config_path)
        candidates.append(path if path.is_absolute() else PROJECT_ROOT / path)

    root = Path(checkpoint_path)
    if not root.is_absolute():
        root = PROJECT_ROOT / root
    candidates.extend([root / "config.json", ckpt_dir / "config.json", ckpt_dir.parent / "config.json"])

    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Could not find config.json for checkpoint {checkpoint_path}")


def load_qmix_config(checkpoint_path: str, ckpt_dir: Path, config_path: str = "") -> Dict[str, Any]:
    resolved = resolve_config_path(checkpoint_path, ckpt_dir, config_path)
    with resolved.open("r", encoding="utf-8") as f:
        cfg = json.load(f)
    print(f"Loaded QMIX config: {resolved}")
    return cfg


def build_scheme(env_info: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, int], Dict[str, Any]]:
    scheme = {
        "state": {"vshape": env_info["state_shape"]},
        "obs": {"vshape": env_info["obs_shape"], "group": "agents"},
        "actions": {"vshape": (1,), "group": "agents", "dtype": th.long},
        "avail_actions": {
            "vshape": (env_info["n_actions"],),
            "group": "agents",
            "dtype": th.int,
        },
        "terminated": {"vshape": (1,), "dtype": th.uint8},
        "reward": {"vshape": (1,)},
    }
    groups = {"agents": int(env_info["n_agents"])}
    preprocess = {"actions": ("actions_onehot", [OneHot(out_dim=int(env_info["n_actions"]))])}
    return scheme, groups, preprocess


def make_episode_batch(
    scheme: Dict[str, Any],
    groups: Dict[str, int],
    preprocess: Dict[str, Any],
    max_seq_length: int,
    device: str,
) -> EpisodeBatch:
    return EpisodeBatch(
        scheme,
        groups,
        batch_size=1,
        max_seq_length=max_seq_length,
        preprocess=preprocess,
        device=device,
    )


class QMixDronesExpert:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.ckpt_dir = resolve_checkpoint_path(args.checkpoint_path, args.load_step)
        self.cfg = load_qmix_config(args.checkpoint_path, self.ckpt_dir, args.config_path)
        self.cfg = deepcopy(self.cfg)
        self.device = self._resolve_device(args.device)
        self.cfg["device"] = self.device
        self.cfg["use_cuda"] = self.device.startswith("cuda")

        self.DronesWrapper = load_drones_wrapper_class()
        self.env_args = self._build_env_args(seed=args.seed)
        probe_env = self._make_env(seed=args.seed)
        env_info = probe_env.get_env_info()
        probe_env.close()

        self.cfg["n_agents"] = env_info["n_agents"]
        self.cfg["n_actions"] = env_info["n_actions"]
        self.cfg["state_shape"] = env_info["state_shape"]
        self.scheme, self.groups, self.preprocess = build_scheme(env_info)
        self.max_seq_length = int(env_info["episode_limit"]) + 1

        mac_args = argparse.Namespace(**self.cfg)
        seed_batch = make_episode_batch(
            self.scheme,
            self.groups,
            self.preprocess,
            self.max_seq_length,
            self.device,
        )
        self.mac = mac_REGISTRY[self.cfg["mac"]](seed_batch.scheme, self.groups, mac_args)
        if self.device.startswith("cuda"):
            self.mac.agent.to(th.device(self.device))
        self.mac.load_models(str(self.ckpt_dir))
        self.mac.agent.eval()
        print(f"Loaded QMIX checkpoint: {self.ckpt_dir}")

    @staticmethod
    def _resolve_device(device: str) -> str:
        if device.startswith("cuda") and not th.cuda.is_available():
            print(f"CUDA requested ({device}) but unavailable; falling back to CPU.")
            return "cpu"
        return device

    def _build_env_args(self, seed: int) -> Dict[str, Any]:
        env_args = deepcopy(self.cfg.get("env_args", {}))
        env_args["seed"] = int(seed)
        env_args["mission_mode"] = self.args.mission_mode
        env_args["reward_new_target"] = self.args.reward_new_target
        env_args["reward_new_elimination"] = self.args.reward_new_elimination
        env_args["reward_success"] = self.args.reward_success
        env_args["reward_timeout"] = self.args.reward_timeout
        env_args["reward_step_penalty"] = self.args.reward_step_penalty
        env_args["reward_collision_penalty"] = self.args.reward_collision_penalty
        env_args["reward_view_overlap_penalty"] = self.args.reward_view_overlap_penalty
        env_args["reward_approach_coef"] = self.args.reward_approach_coef
        return env_args

    def _make_env(self, seed: int):
        env_args = deepcopy(self.env_args)
        env_args["seed"] = int(seed)
        return self.DronesWrapper(
            **env_args,
            common_reward=self.cfg["common_reward"],
            reward_scalarisation=self.cfg["reward_scalarisation"],
        )

    def start_episode(self, seed: int):
        env = self._make_env(seed)
        env.reset(seed=seed)
        batch = make_episode_batch(
            self.scheme,
            self.groups,
            self.preprocess,
            self.max_seq_length,
            self.device,
        )
        self.mac.init_hidden(batch_size=1)
        return env, batch

    @th.no_grad()
    def select_actions(self, env: Any, batch: EpisodeBatch, t: int) -> Tuple[List[int], th.Tensor]:
        pre_data = {
            "state": [env.get_state()],
            "avail_actions": [env.get_avail_actions()],
            "obs": [env.get_obs()],
        }
        batch.update(pre_data, ts=t)
        actions = self.mac.select_actions(batch, t_ep=t, t_env=0, test_mode=True)
        action_list = actions[0].detach().to("cpu").numpy().astype("int64").reshape(-1).tolist()
        return [int(action) for action in action_list], actions

    def commit_actions(self, batch: EpisodeBatch, t: int, actions_tensor: th.Tensor) -> None:
        batch.update({"actions": actions_tensor}, ts=t, mark_filled=False)


def write_record(writer, record: Dict[str, Any]) -> None:
    writer.write(json.dumps(record, ensure_ascii=False) + "\n")


def _new_split_stats() -> Dict[str, Any]:
    return {
        "episodes": 0,
        "success_episodes": 0,
        "detect_success_episodes": 0,
        "truncated_episodes": 0,
        "total_steps": 0,
        "total_return": 0.0,
        "total_found": 0,
        "total_eliminated": 0,
        "total_targets": 0,
        "total_invalid_actions": 0,
        "total_drone_collisions": 0,
        "kept_records": 0,
        "skipped_episodes": 0,
    }


def _finalise_split_stats(stats: Dict[str, Any]) -> Dict[str, Any]:
    episodes = max(1, int(stats["episodes"]))
    total_targets = max(1, int(stats["total_targets"]))
    total_agent_steps = max(1, int(stats["total_steps"]))
    return {
        "episodes": int(stats["episodes"]),
        "success_episodes": int(stats["success_episodes"]),
        "success_rate": float(stats["success_episodes"] / episodes),
        "detect_success_episodes": int(stats["detect_success_episodes"]),
        "detect_success_rate": float(stats["detect_success_episodes"] / episodes),
        "truncated_episodes": int(stats["truncated_episodes"]),
        "avg_steps": float(stats["total_steps"] / episodes),
        "avg_return": float(stats["total_return"] / episodes),
        "avg_found": float(stats["total_found"] / episodes),
        "found_ratio": float(stats["total_found"] / total_targets),
        "avg_eliminated": float(stats["total_eliminated"] / episodes),
        "invalid_actions_per_step": float(stats["total_invalid_actions"] / total_agent_steps),
        "drone_collisions_per_step": float(stats["total_drone_collisions"] / total_agent_steps),
        "kept_records": int(stats["kept_records"]),
        "skipped_episodes": int(stats["skipped_episodes"]),
    }


def _maybe_log_episode(
    split: str,
    episode_idx: int,
    seed: int,
    stats: Dict[str, Any],
    episode_summary: Dict[str, Any],
    written: int,
    needed: int,
    args: argparse.Namespace,
) -> None:
    interval = int(args.episode_log_interval)
    if interval <= 0 or int(stats["episodes"]) % interval != 0:
        return

    episodes = max(1, int(stats["episodes"]))
    success_rate = float(stats["success_episodes"] / episodes)
    detect_success_rate = float(stats["detect_success_episodes"] / episodes)
    print(
        f"[{split} ep={episode_idx:06d} seed={seed}] "
        f"success={int(episode_summary['success'])} "
        f"detect={int(episode_summary['detect_success'])} "
        f"found={episode_summary['targets_found']}/{episode_summary['targets_total']} "
        f"elim={episode_summary['targets_eliminated']}/{episode_summary['targets_total']} "
        f"steps={episode_summary['steps']} "
        f"return={episode_summary['return']:.2f} "
        f"invalid={episode_summary['invalid_actions']} "
        f"collisions={episode_summary['drone_collisions']} "
        f"kept={episode_summary['kept_records']} "
        f"written={written}/{needed} "
        f"split_success={success_rate:.3f} "
        f"split_detect={detect_success_rate:.3f}"
    )


def collect_split(
    split: str,
    needed: int,
    args: argparse.Namespace,
    expert: QMixDronesExpert,
    writer,
    start_episode: int,
) -> Tuple[int, int, Dict[str, Any]]:
    split_image_dir = Path(args.out_dir) / "images" / split
    split_image_dir.mkdir(parents=True, exist_ok=True)

    written = 0
    stats = _new_split_stats()
    episode_idx = start_episode
    while written < needed:
        env_seed = args.seed + episode_idx
        env, batch = expert.start_episode(env_seed)
        episode_records: List[Dict[str, Any]] = []
        written_before_episode = written
        episode_step = 0
        episode_return = 0.0
        episode_invalid_actions = 0
        episode_drone_collisions = 0
        terminated = False
        truncated = False
        last_info: Dict[str, Any] = {}

        while not (terminated or truncated) and episode_step < env.episode_limit and episode_step < args.max_episode_steps:
            actions, actions_tensor = expert.select_actions(env, batch, episode_step)
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

            _, reward, terminated, truncated, info = env.step(actions)
            expert.commit_actions(batch, episode_step, actions_tensor)
            last_info = dict(info)
            episode_return += float(reward)
            episode_invalid_actions += int(info.get("invalid_actions", 0))
            episode_drone_collisions += int(info.get("drone_collisions", 0))

            record["reward"] = float(reward)
            record["terminated"] = bool(terminated)
            record["truncated"] = bool(truncated)
            record["info"] = info
            record["split"] = split
            record["expert"] = "qmix"
            record["qmix_checkpoint"] = str(expert.ckpt_dir.relative_to(PROJECT_ROOT))

            if args.success_only:
                episode_records.append(record)
            else:
                write_record(writer, record)
                written += 1
            episode_step += 1

            if not args.success_only and written >= needed:
                break

        targets_found = int(last_info.get("targets_found", 0))
        targets_eliminated = int(last_info.get("targets_eliminated", 0))
        targets_total = int(last_info.get("targets_total", getattr(env.env, "human_num", 0)))
        detect_success = bool(last_info.get("detect_success", targets_found >= targets_total and targets_total > 0))
        mission_success = bool(last_info.get("mission_success", False))
        success = bool(mission_success or detect_success)
        if args.success_only and success:
            for record in episode_records:
                if written >= needed:
                    break
                write_record(writer, record)
                written += 1
        elif args.success_only:
            stats["skipped_episodes"] += 1

        kept_records = written - written_before_episode
        stats["episodes"] += 1
        stats["success_episodes"] += int(success)
        stats["detect_success_episodes"] += int(detect_success)
        stats["truncated_episodes"] += int(truncated)
        stats["total_steps"] += int(episode_step)
        stats["total_return"] += float(episode_return)
        stats["total_found"] += int(targets_found)
        stats["total_eliminated"] += int(targets_eliminated)
        stats["total_targets"] += int(targets_total)
        stats["total_invalid_actions"] += int(episode_invalid_actions)
        stats["total_drone_collisions"] += int(episode_drone_collisions)
        stats["kept_records"] += int(kept_records)

        episode_summary = {
            "success": success,
            "detect_success": detect_success,
            "targets_found": targets_found,
            "targets_eliminated": targets_eliminated,
            "targets_total": targets_total,
            "steps": int(episode_step),
            "return": float(episode_return),
            "invalid_actions": int(episode_invalid_actions),
            "drone_collisions": int(episode_drone_collisions),
            "kept_records": int(kept_records),
        }
        _maybe_log_episode(split, episode_idx, env_seed, stats, episode_summary, written, needed, args)

        episode_idx += 1
    return written, episode_idx, _finalise_split_stats(stats)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate Drones VLA dataset with a saved QMIX expert")
    parser.add_argument("--checkpoint-path", type=str, default="results/models/qmix_seed0_drones_2026-04-23_02-22-48")
    parser.add_argument("--load-step", type=int, default=None, help="Default picks the latest checkpoint step")
    parser.add_argument("--config-path", type=str, default="")
    parser.add_argument("--out-dir", type=str, default="VLA/data/drones_qmix")
    parser.add_argument("--prefix", type=str, default="drones_qmix")
    parser.add_argument("--num-transitions", type=int, default=1000)
    parser.add_argument("--split-ratios", type=str, default="0.8,0.1,0.1")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda:0" if th.cuda.is_available() else "cpu")
    parser.add_argument("--success-only", action="store_true")
    parser.add_argument("--max-episode-steps", type=int, default=100000)
    parser.add_argument("--episode-log-interval", type=int, default=1, help="Print rollout quality every N episodes; 0 disables episode logs")

    parser.add_argument("--image-size", type=int, default=336)
    parser.add_argument("--partial-map", action="store_true")
    parser.add_argument("--no-view-range", action="store_true")
    parser.add_argument("--draw-attack-range", action="store_true")
    parser.add_argument("--no-grid", action="store_true")
    parser.add_argument("--include-privileged-state", action="store_true", default=True)
    parser.add_argument("--no-include-privileged-state", dest="include_privileged_state", action="store_false")
    parser.add_argument(
        "--thinking",
        type=str,
        default="Follow the QMIX expert policy to find all humans while avoiding blocked cells.",
    )

    parser.add_argument("--mission-mode", choices=["detect", "eliminate", "both"], default="detect")
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
    random.seed(args.seed)

    expert = QMixDronesExpert(args)
    episode_cursor = 0
    summary: Dict[str, Any] = {}

    for split in ("train", "val", "test"):
        jsonl_path = out_dir / f"{args.prefix}_{split}_alpaca.jsonl"
        with jsonl_path.open("w", encoding="utf-8") as writer:
            written, episode_cursor, split_stats = collect_split(
                split,
                counts[split],
                args,
                expert,
                writer,
                episode_cursor,
            )
        split_stats["records"] = int(written)
        summary[split] = split_stats
        print(
            f"{split}: wrote {written} records -> {jsonl_path}; "
            f"episodes={split_stats['episodes']}, "
            f"success_rate={split_stats['success_rate']:.3f}, "
            f"detect_success_rate={split_stats['detect_success_rate']:.3f}, "
            f"avg_found={split_stats['avg_found']:.2f}, "
            f"avg_steps={split_stats['avg_steps']:.2f}, "
            f"avg_return={split_stats['avg_return']:.2f}, "
            f"skipped_episodes={split_stats['skipped_episodes']}"
        )

    print("Summary:", json.dumps(summary, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
