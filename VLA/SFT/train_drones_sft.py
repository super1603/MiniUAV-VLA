from __future__ import annotations

import argparse
import math
import os
import sys
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Dict

import torch
import torch.nn.functional as F
from torch import nn, optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from VLA.envs.drones_dataset import DronesVLMDataset  # noqa: E402
from VLA.envs.drones_model_wrapper import MiniMindVLMWithDroneActions, VLMConfig  # noqa: E402
from VLA.SFT.evaluate_drones import evaluate_drones_model, move_pixel_values  # noqa: E402

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")


def get_lr(current_step: int, total_steps: int, lr: float) -> float:
    if total_steps <= 0:
        return lr
    return lr * (0.1 + 0.45 * (1 + math.cos(math.pi * current_step / total_steps)))


def count_trainable_params(model: torch.nn.Module) -> float:
    return sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6


def unwrap_model(model):
    raw_model = model.module if hasattr(model, "module") else model
    return getattr(raw_model, "_orig_mod", raw_model)


def load_pretrained(model: MiniMindVLMWithDroneActions, path: str, device: torch.device) -> None:
    if not path or path.lower() == "none":
        print("Pretrained loading disabled.")
        return
    if not os.path.exists(path):
        raise FileNotFoundError(f"Pretrained path not found: {path}")

    state = torch.load(path, map_location=device)
    if isinstance(state, dict) and "model" in state and isinstance(state["model"], dict):
        state = state["model"]

    if any(str(k).startswith("vlm.") or str(k).startswith("action_head.") for k in state.keys()):
        missing, unexpected = model.load_state_dict(state, strict=False)
    else:
        missing, unexpected = model.vlm.load_state_dict(state, strict=False)
    print(f"Loaded pretrained: {path}")
    print(f"Missing keys: {len(missing)}; unexpected keys: {len(unexpected)}")


def freeze_by_strategy(model: MiniMindVLMWithDroneActions, freeze_vlm: bool, freeze_llm: int) -> None:
    for param in model.parameters():
        param.requires_grad = True

    if freeze_vlm:
        for param in model.vlm.parameters():
            param.requires_grad = False

    if freeze_llm == 1:
        last_idx = model.config.num_hidden_layers - 1
        for name, param in model.vlm.model.named_parameters():
            param.requires_grad = ("layers.0." in name) or (f"layers.{last_idx}." in name)
    elif freeze_llm == 2:
        for name, param in model.vlm.named_parameters():
            if "vision_proj" not in name:
                param.requires_grad = False

    for param in model.action_head.parameters():
        param.requires_grad = True


def save_checkpoint(model, save_path: str) -> None:
    raw_model = unwrap_model(model)
    state_dict = raw_model.state_dict()
    clean_state_dict = {
        key: value
        for key, value in state_dict.items()
        if not key.startswith("vision_encoder.") and not key.startswith("vlm.vision_encoder.")
    }
    clean_state_dict = {k: v.detach().half().cpu() for k, v in clean_state_dict.items()}
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(clean_state_dict, save_path)


def train_one_epoch(
    epoch: int,
    model,
    loader,
    optimizer,
    scaler,
    autocast_ctx,
    args,
    device: torch.device,
) -> Dict[str, float]:
    model.train()
    loss_fct = nn.CrossEntropyLoss(reduction="none")
    start_time = time.time()
    iter_per_epoch = len(loader)

    running = {
        "loss": 0.0,
        "loss_lm": 0.0,
        "loss_action": 0.0,
        "per_agent_accuracy": 0.0,
        "joint_action_accuracy": 0.0,
    }
    logged_steps = 0

    pbar = tqdm(loader, desc=f"Epoch {epoch + 1}/{args.epochs}")
    for step, batch in enumerate(pbar):
        global_step = epoch * iter_per_epoch + step
        lr = get_lr(global_step, args.epochs * iter_per_epoch, args.learning_rate)
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)
        loss_mask = batch["loss_mask"].to(device)
        actions = batch["actions"].to(device)
        action_positions = None if args.no_prompt_pos else batch["action_positions"].to(device)
        pixel_values = move_pixel_values(batch.get("pixel_values"), device, disable_vision=args.disable_vision)

        with autocast_ctx:
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                pixel_values=pixel_values,
                action_positions=action_positions,
            )

            lm_loss_tok = loss_fct(
                outputs.logits.reshape(-1, outputs.logits.size(-1)),
                labels.reshape(-1),
            ).view(labels.size())
            loss_lm = (lm_loss_tok * loss_mask).sum() / (loss_mask.sum() + 1e-8)

            action_logits = outputs.action_logits
            loss_action = F.cross_entropy(
                action_logits.reshape(-1, action_logits.size(-1)),
                actions.reshape(-1),
            )
            aux_loss = getattr(outputs, "aux_loss", None)
            if aux_loss is None:
                aux_loss = loss_action.new_zeros(())
            loss = loss_lm + aux_loss + args.action_loss_weight * loss_action
            loss = loss / args.accumulation_steps

        scaler.scale(loss).backward()

        if (step + 1) % args.accumulation_steps == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

        with torch.no_grad():
            pred_actions = action_logits.argmax(dim=-1)
            correct = pred_actions == actions
            per_agent_acc = correct.float().mean().item()
            joint_acc = correct.all(dim=1).float().mean().item()

        running["loss"] += float(loss.item() * args.accumulation_steps)
        running["loss_lm"] += float(loss_lm.item())
        running["loss_action"] += float(loss_action.item())
        running["per_agent_accuracy"] += per_agent_acc
        running["joint_action_accuracy"] += joint_acc
        logged_steps += 1

        if step % args.log_interval == 0:
            elapsed = time.time() - start_time
            pbar.set_postfix(
                {
                    "loss": f"{loss.item() * args.accumulation_steps:.3f}",
                    "act": f"{per_agent_acc:.2f}",
                    "joint": f"{joint_acc:.2f}",
                    "lr": f"{lr:.2e}",
                    "t": f"{elapsed:.0f}s",
                }
            )

        if (step + 1) % args.save_interval == 0:
            save_checkpoint(model, args.save_path)

    if logged_steps:
        for key in running:
            running[key] /= logged_steps
    return running


def main() -> None:
    parser = argparse.ArgumentParser(description="MiniMind-V Drones VLA SFT")
    parser.add_argument("--data_path", type=str, default="VLA/data/drones/drones_train_alpaca.jsonl")
    parser.add_argument("--val_data_path", type=str, default="VLA/data/drones/drones_val_alpaca.jsonl")
    parser.add_argument("--out_dir", type=str, default="VLA/models")
    parser.add_argument("--save_name", type=str, default="sft_vlm_drones_768.pth")
    parser.add_argument("--tokenizer_path", type=str, default="minimind-v/model")
    parser.add_argument("--vision_model_path", type=str, default="model/siglip2-base-p16-224")
    parser.add_argument("--pretrained_path", type=str, default="out/pretrain_vlm_768.pth")
    parser.add_argument("--disable_vision", action="store_true")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=5e-6)
    parser.add_argument("--action_loss_weight", type=float, default=0.5)
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", type=str, choices=["float16", "bfloat16"], default="bfloat16")
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--accumulation_steps", type=int, default=1)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--log_interval", type=int, default=1)
    parser.add_argument("--save_interval", type=int, default=100)
    parser.add_argument("--val_interval", type=int, default=1)
    parser.add_argument("--max_val_steps", type=int, default=20)
    parser.add_argument("--hidden_size", type=int, default=768)
    parser.add_argument("--num_hidden_layers", type=int, default=8)
    parser.add_argument("--num_attention_heads", type=int, default=8)
    parser.add_argument("--num_key_value_heads", type=int, default=4)
    parser.add_argument("--max_seq_len", type=int, default=512)
    parser.add_argument("--image_token_len", type=int, default=196,
                        help="Number of <|image_pad|> markers to expand each <image> tag into. "
                             "Must equal the vision encoder's patch count (196 for SigLIP2-base-patch16-224).")
    parser.add_argument("--use_moe", type=int, choices=[0, 1], default=0)
    parser.add_argument("--n_agents", type=int, default=4)
    parser.add_argument("--n_actions", type=int, default=5)
    parser.add_argument("--freeze_vlm", action="store_true")
    parser.add_argument("--freeze_llm", type=int, choices=[0, 1, 2], default=0)
    parser.add_argument("--no_prompt_pos", action="store_true",
                        help="Use last-token pooling (ignore action_positions); default is prompt-end pooling")
    parser.add_argument("--seed", type=int, default=1337)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = torch.device(args.device)
    os.makedirs(args.out_dir, exist_ok=True)
    args.save_path = os.path.join(args.out_dir, args.save_name)

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
    load_pretrained(model, args.pretrained_path, device)
    freeze_by_strategy(model, freeze_vlm=args.freeze_vlm, freeze_llm=args.freeze_llm)
    model.to(device)

    preprocess = None if args.disable_vision else model.processor
    if not args.disable_vision and preprocess is None:
        raise RuntimeError(
            f"Vision processor could not be loaded from {args.vision_model_path}. "
            "Use --disable_vision for text/action smoke tests."
        )

    train_ds = DronesVLMDataset(
        args.data_path,
        tokenizer,
        preprocess=preprocess,
        image_special_token=model_config.image_special_token,
        image_token_len=model_config.image_token_len,
        max_length=args.max_seq_len,
        n_agents=args.n_agents,
    )
    val_ds = DronesVLMDataset(
        args.val_data_path,
        tokenizer,
        preprocess=preprocess,
        image_special_token=model_config.image_special_token,
        image_token_len=model_config.image_token_len,
        max_length=args.max_seq_len,
        n_agents=args.n_agents,
    )
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=("cuda" in args.device),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=("cuda" in args.device),
    )

    optimizer = optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.learning_rate)
    device_type = "cuda" if "cuda" in args.device else "cpu"
    amp_dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16
    autocast_ctx = nullcontext() if device_type == "cpu" else torch.cuda.amp.autocast(dtype=amp_dtype)
    scaler = torch.cuda.amp.GradScaler(enabled=(args.dtype == "float16" and device_type == "cuda"))

    print(f"Train samples: {len(train_ds)}; val samples: {len(val_ds)}")
    print(f"Trainable params: {count_trainable_params(model):.3f}M")
    print(f"Save path: {args.save_path}")

    for epoch in range(args.epochs):
        train_metrics = train_one_epoch(epoch, model, train_loader, optimizer, scaler, autocast_ctx, args, device)
        print(f"Epoch {epoch + 1} train metrics: {train_metrics}")

        if (epoch + 1) % args.val_interval == 0:
            val_metrics = evaluate_drones_model(
                model,
                val_loader,
                device,
                max_eval_steps=args.max_val_steps,
                disable_vision=args.disable_vision,
            )
            print(f"Epoch {epoch + 1} val metrics: {val_metrics}")

        save_checkpoint(model, args.save_path)

    print(f"Saved model -> {args.save_path}")


if __name__ == "__main__":
    main()
