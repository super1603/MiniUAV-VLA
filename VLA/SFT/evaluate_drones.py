from __future__ import annotations

from typing import Any, Dict, Optional

import torch
import torch.nn.functional as F
from tqdm import tqdm


def move_pixel_values(pixel_values: Any, device: torch.device, disable_vision: bool = False):
    if disable_vision or pixel_values is None:
        return None
    if isinstance(pixel_values, dict):
        return {k: v.to(device) for k, v in pixel_values.items()}
    return pixel_values.to(device)


def _aux_loss_value(outputs) -> torch.Tensor:
    aux_loss = getattr(outputs, "aux_loss", None)
    if aux_loss is None:
        logits = getattr(outputs, "logits")
        return logits.new_zeros(())
    return aux_loss


@torch.no_grad()
def evaluate_drones_model(
    model,
    val_loader,
    device: torch.device,
    max_eval_steps: Optional[int] = None,
    disable_vision: bool = False,
) -> Dict[str, float]:
    model.eval()
    loss_fct = torch.nn.CrossEntropyLoss(reduction="none")

    total_samples = 0
    total_agent_actions = 0
    correct_agent_actions = 0
    correct_joint_actions = 0
    total_lm_loss = 0.0
    total_action_loss = 0.0
    num_batches = 0

    iterator = tqdm(val_loader, desc="Evaluating", leave=False)
    for step, batch in enumerate(iterator):
        if max_eval_steps is not None and step >= max_eval_steps:
            break

        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)
        loss_mask = batch["loss_mask"].to(device)
        actions = batch["actions"].to(device)
        action_positions = batch["action_positions"].to(device)
        pixel_values = move_pixel_values(batch.get("pixel_values"), device, disable_vision=disable_vision)

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

        pred_actions = action_logits.argmax(dim=-1)
        correct_by_agent = pred_actions == actions
        correct_agent_actions += int(correct_by_agent.sum().item())
        total_agent_actions += int(actions.numel())
        correct_joint_actions += int(correct_by_agent.all(dim=1).sum().item())
        total_samples += int(actions.size(0))
        total_lm_loss += float(loss_lm.item())
        total_action_loss += float(loss_action.item())
        num_batches += 1

    if num_batches == 0:
        return {
            "avg_lm_loss": 0.0,
            "avg_action_loss": 0.0,
            "per_agent_accuracy": 0.0,
            "joint_action_accuracy": 0.0,
            "total_samples": 0.0,
        }

    return {
        "avg_lm_loss": total_lm_loss / num_batches,
        "avg_action_loss": total_action_loss / num_batches,
        "per_agent_accuracy": correct_agent_actions / max(1, total_agent_actions),
        "joint_action_accuracy": correct_joint_actions / max(1, total_samples),
        "total_samples": float(total_samples),
    }


__all__ = ["evaluate_drones_model", "move_pixel_values"]
