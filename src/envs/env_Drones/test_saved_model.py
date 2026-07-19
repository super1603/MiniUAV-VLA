import argparse
from copy import deepcopy
import datetime
from pathlib import Path
import sys

import imageio.v2 as imageio
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import numpy as np
import torch as th
import yaml


SRC_ROOT = Path(__file__).resolve().parents[2]
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from components.episode_buffer import EpisodeBatch
from components.transforms import OneHot
from controllers import REGISTRY as mac_REGISTRY
from envs.drones_wrapper import DronesWrapper


ACTION_NAME = {
    0: "U",
    1: "D",
    2: "L",
    3: "R",
    4: "S",
}


def _draw_entities(ax, env_core, drone_trajs, human_trajs, draw_traj):
    if draw_traj:
        for idx, traj in enumerate(drone_trajs):
            if len(traj) >= 2:
                xs = [p[1] for p in traj]
                ys = [p[0] for p in traj]
                ax.plot(xs, ys, linewidth=1.2, alpha=0.8, label=f"D{idx}_traj")
        for idx, traj in enumerate(human_trajs):
            if len(traj) >= 2:
                xs = [p[1] for p in traj]
                ys = [p[0] for p in traj]
                ax.plot(xs, ys, linewidth=1.0, alpha=0.6, linestyle="--", label=f"H{idx}_traj")

    for idx, drone in enumerate(env_core.drone_list):
        x, y = drone.pos
        ax.scatter([y], [x], marker="o", s=35)
        ax.text(y + 0.2, x + 0.2, f"D{idx}", fontsize=8)

    for idx, human in enumerate(env_core.human_list):
        x, y = human.pos
        ax.scatter([y], [x], marker="x", s=30)
        ax.text(y + 0.2, x + 0.2, f"H{idx}", fontsize=8)


def _recursive_update(base, update):
    for k, v in update.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            _recursive_update(base[k], v)
        else:
            base[k] = v
    return base


def _load_yaml(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.load(f, Loader=yaml.FullLoader)


def _resolve_checkpoint_path(checkpoint_path, load_step):
    checkpoint_path = Path(checkpoint_path)
    if (checkpoint_path / "agent.th").exists():
        return checkpoint_path

    step_dirs = [p for p in checkpoint_path.iterdir() if p.is_dir() and p.name.isdigit()]
    if not step_dirs:
        raise FileNotFoundError(
            f"No checkpoint step directories found under: {checkpoint_path}"
        )

    step_dirs = sorted(step_dirs, key=lambda p: int(p.name))
    if load_step is None:
        return step_dirs[-1]
    return min(step_dirs, key=lambda p: abs(int(p.name) - int(load_step)))


def _build_config(alg_name, env_name):
    default_cfg = _load_yaml(SRC_ROOT / "config" / "default.yaml")
    env_cfg = _load_yaml(SRC_ROOT / "config" / "envs" / f"{env_name}.yaml")
    alg_cfg = _load_yaml(SRC_ROOT / "config" / "algs" / f"{alg_name}.yaml")

    cfg = deepcopy(default_cfg)
    cfg = _recursive_update(cfg, env_cfg)
    cfg = _recursive_update(cfg, alg_cfg)
    return cfg


def run_eval(args):
    cfg = _build_config(args.alg_config, args.env_config)
    cfg["use_cuda"] = bool(args.use_cuda) and th.cuda.is_available()
    cfg["env_args"]["seed"] = int(args.seed)

    if args.override_t_max is not None:
        cfg["t_max"] = int(args.override_t_max)

    env = DronesWrapper(
        **cfg["env_args"],
        common_reward=cfg["common_reward"],
        reward_scalarisation=cfg["reward_scalarisation"],
    )
    env_info = env.get_env_info()
    cfg["n_agents"] = env_info["n_agents"]
    cfg["n_actions"] = env_info["n_actions"]
    cfg["state_shape"] = env_info["state_shape"]
    cfg["device"] = "cuda" if cfg["use_cuda"] else "cpu"

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
    }
    scheme["reward"] = {"vshape": (1,)}
    groups = {"agents": cfg["n_agents"]}
    preprocess = {"actions": ("actions_onehot", [OneHot(out_dim=cfg["n_actions"])])}

    runner_batch = EpisodeBatch(
        scheme,
        groups,
        batch_size=1,
        max_seq_length=env.episode_limit + 1,
        preprocess=preprocess,
        device=cfg["device"],
    )

    args_sn = argparse.Namespace(**cfg)
    mac = mac_REGISTRY[cfg["mac"]](runner_batch.scheme, groups, args_sn)
    if cfg["use_cuda"]:
        mac.cuda()

    ckpt_dir = _resolve_checkpoint_path(args.checkpoint_path, args.load_step)
    mac.load_models(str(ckpt_dir))
    print(f"Loaded checkpoint: {ckpt_dir}")

    fig = plt.figure(figsize=(11, 5))
    gs = GridSpec(1, 2, figure=fig)
    ax_full = fig.add_subplot(gs[0:1, 0:1])
    ax_joint = fig.add_subplot(gs[0:1, 1:2])
    plt.ion()
    plt.show(block=False)

    video_writer = None
    if args.save_video:
        if args.video_path is not None:
            video_path = Path(args.video_path)
        else:
            ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            video_path = Path.cwd() / f"drones_policy_rollout_{ts}.mp4"
        video_path.parent.mkdir(parents=True, exist_ok=True)
        video_writer = imageio.get_writer(str(video_path), fps=args.video_fps)
        print(f"Video recording enabled: {video_path}")

    try:
        for ep in range(args.episodes):
            runner_batch = EpisodeBatch(
                scheme,
                groups,
                batch_size=1,
                max_seq_length=env.episode_limit + 1,
                preprocess=preprocess,
                device=cfg["device"],
            )
            env.reset(seed=args.seed + ep)
            mac.init_hidden(batch_size=1)

            drone_trajs = [[] for _ in range(env.n_agents)]
            human_trajs = [[] for _ in range(env.env.human_num)]
            for i, drone in enumerate(env.env.drone_list):
                drone_trajs[i].append((drone.pos[0], drone.pos[1]))
            for i, human in enumerate(env.env.human_list):
                human_trajs[i].append((human.pos[0], human.pos[1]))

            t = 0
            ep_return = 0.0
            terminated = False
            truncated = False

            while not (terminated or truncated):
                pre_data = {
                    "state": [env.get_state()],
                    "avail_actions": [env.get_avail_actions()],
                    "obs": [env.get_obs()],
                }
                runner_batch.update(pre_data, ts=t)

                with th.no_grad():
                    actions = mac.select_actions(
                        runner_batch, t_ep=t, t_env=0, test_mode=True
                    )

                action_list = (
                    actions[0].detach().to("cpu").numpy().astype(np.int64).reshape(-1).tolist()
                )
                _, reward, terminated, truncated, info = env.step(action_list)
                ep_return += float(reward)
                runner_batch.update({"actions": actions}, ts=t, mark_filled=False)

                for i, drone in enumerate(env.env.drone_list):
                    drone_trajs[i].append((drone.pos[0], drone.pos[1]))
                for i, human in enumerate(env.env.human_list):
                    human_trajs[i].append((human.pos[0], human.pos[1]))

                full_obs = env.env.get_full_obs()
                joint_obs = env.env.get_joint_obs()
                ax_full.clear()
                ax_joint.clear()
                ax_full.imshow(full_obs)
                ax_joint.imshow(joint_obs)
                _draw_entities(ax_full, env.env, drone_trajs, human_trajs, args.draw_traj)
                _draw_entities(ax_joint, env.env, drone_trajs, human_trajs, args.draw_traj)
                act_text = ",".join(ACTION_NAME.get(a, str(a)) for a in action_list)
                ax_full.set_title(f"Episode {ep} Step {t} Return {ep_return:.2f}")
                ax_joint.set_title(
                    f"Actions [{act_text}] Found {info['targets_found']}/{info['targets_total']}"
                )
                ax_full.set_xticks([])
                ax_full.set_yticks([])
                ax_joint.set_xticks([])
                ax_joint.set_yticks([])

                fig.canvas.draw()
                if video_writer is not None:
                    frame = np.asarray(fig.canvas.buffer_rgba())[:, :, :3].copy()
                    video_writer.append_data(frame)

                plt.pause(args.pause)
                t += 1

            print(
                f"[Episode {ep}] steps={t}, return={ep_return:.3f}, "
                f"success={info.get('mission_success', False)}, "
                f"found={info.get('targets_found', 0)}/{info.get('targets_total', 0)}"
            )
    finally:
        if video_writer is not None:
            video_writer.close()

    print("Evaluation finished.")
    plt.ioff()
    plt.show()


def parse_args():
    parser = argparse.ArgumentParser(
        description="Load a saved EPyMARL model and visualise Drones decisions."
    )
    parser.add_argument(
        "--checkpoint-path",
        type=str,
        required=True,
        help="Path to checkpoint root or a specific step dir containing agent.th",
    )
    parser.add_argument(
        "--load-step",
        type=int,
        default=None,
        help="If checkpoint-path is a root dir, pick nearest step to this value (default: max step).",
    )
    parser.add_argument("--alg-config", type=str, default="qmix")
    parser.add_argument("--env-config", type=str, default="drones")
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--pause", type=float, default=0.2)
    parser.add_argument("--draw-traj", action="store_true", default=False)
    parser.add_argument("--save-video", action="store_true", default=False)
    parser.add_argument("--video-path", type=str, default=None)
    parser.add_argument("--video-fps", type=int, default=10)
    parser.add_argument("--use-cuda", action="store_true", default=False)
    parser.add_argument("--override-t-max", type=int, default=None)
    return parser.parse_args()


if __name__ == "__main__":
    run_eval(parse_args())



# python src/envs/env_Drones/test_saved_model.py \
#   --checkpoint-path "results/models/<unique_token>" \
#   --episodes 3 \
#   --draw-traj \
#   --save-video \
#   --video-path "results/videos/drones_rollout.mp4" \
#   --video-fps 12 \
#   --pause 0.05 \
#   --use-cuda 