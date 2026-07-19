from __future__ import annotations

import sys
from pathlib import Path
from typing import List, Optional, Tuple, Union

import torch
from torch import nn


PROJECT_ROOT = Path(__file__).resolve().parents[2]
MINIMIND_ROOT = PROJECT_ROOT / "minimind-v"
if str(MINIMIND_ROOT) not in sys.path:
    sys.path.insert(0, str(MINIMIND_ROOT))

from model.model_vlm import MiniMindVLM, VLMConfig  # noqa: E402


class MultiDroneActionHead(nn.Module):
    def __init__(self, hidden_size: int, n_agents: int = 4, n_actions: int = 5):
        super().__init__()
        self.n_agents = int(n_agents)
        self.n_actions = int(n_actions)
        self.fc = nn.Linear(hidden_size, self.n_agents * self.n_actions)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        flat_logits = self.fc(x)
        return flat_logits.view(x.size(0), self.n_agents, self.n_actions)


class MiniMindVLMWithDroneActions(nn.Module):
    """
    MiniMindVLM backbone with a multi-drone discrete action head.

    Forward returns the original MiniMindVLM output with:
      res["action_logits"] = Tensor[B, n_agents, n_actions]
    """

    def __init__(
        self,
        config: VLMConfig,
        vision_model_path: str = "model/siglip2-base-p16-224",
        n_agents: int = 4,
        n_actions: int = 5,
    ):
        super().__init__()
        self.vlm = MiniMindVLM(config, vision_model_path=vision_model_path)
        self.action_head = MultiDroneActionHead(
            hidden_size=config.hidden_size,
            n_agents=n_agents,
            n_actions=n_actions,
        )

        self.n_agents = int(n_agents)
        self.n_actions = int(n_actions)
        self.vision_encoder = self.vlm.vision_encoder
        self.processor = self.vlm.processor
        self.config = self.vlm.config

    @staticmethod
    def _pool_last_hidden(
        last_hidden_state: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        action_positions: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if action_positions is not None:
            positions = action_positions.long().clamp(
                min=0,
                max=last_hidden_state.size(1) - 1,
            )
            batch_idx = torch.arange(last_hidden_state.size(0), device=last_hidden_state.device)
            return last_hidden_state[batch_idx, positions, :]

        if attention_mask is None:
            return last_hidden_state[:, -1, :]

        lengths = attention_mask.long().sum(dim=1).clamp(min=1) - 1
        batch_idx = torch.arange(last_hidden_state.size(0), device=last_hidden_state.device)
        return last_hidden_state[batch_idx, lengths, :]

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
        use_cache: bool = False,
        logits_to_keep: Union[int, torch.Tensor] = 0,
        pixel_values: Optional[torch.FloatTensor] = None,
        action_positions: Optional[torch.Tensor] = None,
        **args,
    ):
        res = self.vlm(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            logits_to_keep=logits_to_keep,
            pixel_values=pixel_values,
            **args,
        )

        hidden_states = getattr(res, "last_hidden_state", None)
        if hidden_states is None:
            hidden_states = getattr(res, "hidden_states")
        pooled_hidden = self._pool_last_hidden(hidden_states, attention_mask, action_positions=action_positions)
        action_logits = self.action_head(pooled_hidden)
        try:
            res.__setitem__("action_logits", action_logits)
        except Exception:
            setattr(res, "action_logits", action_logits)
        return res


__all__ = [
    "MiniMindVLMWithDroneActions",
    "MultiDroneActionHead",
    "VLMConfig",
]
