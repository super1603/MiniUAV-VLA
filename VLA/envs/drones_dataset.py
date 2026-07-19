from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


PROJECT_ROOT = Path(__file__).resolve().parents[2]
MINIMIND_ROOT = PROJECT_ROOT / "minimind-v"
if str(MINIMIND_ROOT) not in sys.path:
    sys.path.insert(0, str(MINIMIND_ROOT))

try:
    from model.model_vlm import MiniMindVLM
except Exception:
    MiniMindVLM = None


class DronesVLMDataset(Dataset):
    """
    Alpaca-style Drones VLA dataset.

    Expected JSONL fields:
    {
      "instruction": str,
      "input": str,
      "output": "{\"thinking\": string, \"actions\": [int, ...]}",
      "image_path": str,
      "meta": {
        "expert_actions": [int, ...],
        "drone_num": int,
        ...
      }
    }

    Returns:
    {
      "input_ids": LongTensor[max_length - 1],
      "labels": LongTensor[max_length - 1],
      "loss_mask": FloatTensor[max_length - 1],
      "attention_mask": LongTensor[max_length - 1],
      "action_positions": LongTensor[],
      "pixel_values": FloatTensor[1, 1, C, H, W] or dict[str, FloatTensor[1, C, H, W]],
      "actions": LongTensor[n_agents],
    }
    """

    def __init__(
        self,
        jsonl_path: str,
        tokenizer,
        preprocess=None,
        max_length: int = 512,
        image_special_token: str = "<|image_pad|>",
        image_token_len: int = 1,
        images_root: Optional[str] = None,
        n_agents: Optional[int] = None,
        fallback_image_size: int = 224,
    ):
        super().__init__()
        self.jsonl_path = os.path.abspath(jsonl_path)
        self.samples = self._load_jsonl(self.jsonl_path)
        self.tokenizer = tokenizer
        self.preprocess = preprocess
        self.max_length = int(max_length)
        self.image_token = image_special_token
        self.image_token_len = int(image_token_len)
        self.images_root = images_root
        self.n_agents = n_agents or self._infer_n_agents()
        self.fallback_image_size = int(fallback_image_size)

        if self.n_agents <= 0:
            raise ValueError("n_agents must be positive or inferable from the dataset")

        if getattr(self.tokenizer, "pad_token_id", None) is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.pad_id = self.tokenizer.pad_token_id

        self.bos_id = tokenizer("<|im_start|>assistant", add_special_tokens=False).input_ids
        self.eos_id = tokenizer("<|im_end|>", add_special_tokens=False).input_ids

    @staticmethod
    def _load_jsonl(path: str) -> List[Dict[str, Any]]:
        items: List[Dict[str, Any]] = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    items.append(json.loads(line))
        return items

    def _infer_n_agents(self) -> int:
        for sample in self.samples:
            meta = sample.get("meta") or {}
            if "drone_num" in meta:
                return int(meta["drone_num"])
            actions = meta.get("expert_actions")
            if isinstance(actions, Sequence) and not isinstance(actions, (str, bytes)):
                return len(actions)
            parsed = self._parse_actions_from_output(sample.get("output", ""))
            if parsed:
                return len(parsed)
        return 0

    def __len__(self) -> int:
        return len(self.samples)

    def _resolve_image_path(self, image_path: str) -> str:
        if os.path.isabs(image_path) and os.path.exists(image_path):
            return image_path
        if os.path.exists(image_path):
            return image_path

        candidates: List[Path] = []
        if self.images_root:
            candidates.append(Path(self.images_root) / image_path)
        candidates.append(PROJECT_ROOT / image_path)
        candidates.append(Path(self.jsonl_path).parent / image_path)

        for candidate in candidates:
            if candidate.exists():
                return str(candidate)
        return image_path

    def _create_prompt(self, instruction: str, user_input: str) -> str:
        content_parts = ["<image>", instruction.strip() if instruction else ""]
        if user_input:
            content_parts.append(user_input.strip())
        content = "\n".join(part for part in content_parts if part)

        messages = [
            {"role": "user", "content": content.replace("<image>", self.image_token * self.image_token_len)}
        ]
        if hasattr(self.tokenizer, "apply_chat_template"):
            return self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        return f"<|im_start|>user\n{messages[0]['content']}<|im_end|>\n<|im_start|>assistant\n"

    def _generate_loss_mask(self, input_ids: List[int]) -> List[int]:
        loss_mask = [0] * len(input_ids)
        i = 0
        while i < len(input_ids):
            if input_ids[i:i + len(self.bos_id)] == self.bos_id:
                start = i + len(self.bos_id)
                end = start
                while end < len(input_ids):
                    if input_ids[end:end + len(self.eos_id)] == self.eos_id:
                        break
                    if self.pad_id is not None and input_ids[end] == self.pad_id:
                        break
                    end += 1
                for j in range(start + 1, min(end + len(self.eos_id), len(input_ids))):
                    if self.pad_id is not None and input_ids[j] == self.pad_id:
                        break
                    loss_mask[j] = 1
                i = end + len(self.eos_id) if end < len(input_ids) else len(input_ids)
            else:
                i += 1
        return loss_mask

    def _tokenize(self, text: str) -> List[int]:
        tokenized = self.tokenizer(text, add_special_tokens=False)
        return list(tokenized.input_ids)

    def _image_to_tensor(self, image: Image.Image):
        if image.mode in ["RGBA", "LA"]:
            image = image.convert("RGB")
        elif image.mode != "RGB":
            image = image.convert("RGB")

        if MiniMindVLM is not None and self.preprocess is not None:
            image_tensor = MiniMindVLM.image2tensor(image, self.preprocess)
            if hasattr(image_tensor, "items"):
                return {k: v.squeeze(0) if v.ndim > 3 and v.shape[0] == 1 else v for k, v in image_tensor.items()}
            return image_tensor

        image = image.resize((self.fallback_image_size, self.fallback_image_size))
        arr = np.asarray(image, dtype=np.float32) / 255.0
        arr = np.transpose(arr, (2, 0, 1))
        return torch.from_numpy(arr).unsqueeze(0)

    @staticmethod
    def _parse_actions_from_output(output: Any) -> List[int]:
        try:
            obj = json.loads(output) if isinstance(output, str) else output
        except Exception:
            return []
        if not isinstance(obj, dict):
            return []
        actions = obj.get("actions", obj.get("action"))
        if not isinstance(actions, Sequence) or isinstance(actions, (str, bytes)):
            return []
        parsed: List[int] = []
        for action in actions:
            try:
                parsed.append(int(action))
            except (TypeError, ValueError):
                return []
        return parsed

    def _extract_actions(self, sample: Dict[str, Any]) -> torch.Tensor:
        meta = sample.get("meta") or {}
        actions = meta.get("expert_actions")
        if not isinstance(actions, Sequence) or isinstance(actions, (str, bytes)):
            actions = self._parse_actions_from_output(sample.get("output", ""))

        if not isinstance(actions, Sequence) or isinstance(actions, (str, bytes)):
            raise ValueError("Sample does not contain valid expert_actions/actions")

        parsed = [int(action) for action in actions]
        if len(parsed) != self.n_agents:
            raise ValueError(
                f"Expected {self.n_agents} actions, got {len(parsed)} in sample {sample.get('id', '<no id>')}"
            )
        if any(action < 0 or action > 4 for action in parsed):
            raise ValueError(f"Actions must be in [0, 4], got {parsed}")
        return torch.tensor(parsed, dtype=torch.long)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        sample = self.samples[index]
        instruction = sample.get("instruction", "")
        user_input = sample.get("input", "")
        target = sample.get("output", "")
        image_path = sample.get("image_path", "")

        prompt = self._create_prompt(instruction, user_input)
        target_text = target if isinstance(target, str) else json.dumps(target, ensure_ascii=False)
        target_with_eos = target_text.strip() + "<|im_end|>"

        prompt_ids = self._tokenize(prompt)
        target_ids = self._tokenize(target_with_eos)
        input_ids = (prompt_ids + target_ids)[: self.max_length]
        pad_len = self.max_length - len(input_ids)
        if pad_len > 0:
            input_ids = input_ids + [self.pad_id] * pad_len

        loss_mask_list = self._generate_loss_mask(input_ids)
        attention_mask_list = [0 if token_id == self.pad_id else 1 for token_id in input_ids]
        X = torch.tensor(input_ids[:-1], dtype=torch.long)
        Y = torch.tensor(input_ids[1:], dtype=torch.long)
        loss_mask = torch.tensor(loss_mask_list[1:], dtype=torch.float)
        attention_mask = torch.tensor(attention_mask_list[:-1], dtype=torch.long)
        action_position = min(max(len(prompt_ids) - 1, 0), X.numel() - 1)

        resolved_path = self._resolve_image_path(image_path)
        if not os.path.exists(resolved_path):
            raise FileNotFoundError(f"Image not found: {resolved_path}")
        image = Image.open(resolved_path)
        image_tensor = self._image_to_tensor(image)
        if hasattr(image_tensor, "items"):
            pixel_values = image_tensor
        else:
            pixel_values = torch.stack([image_tensor], dim=0)

        return {
            "input_ids": X,
            "attention_mask": attention_mask,
            "action_positions": torch.tensor(action_position, dtype=torch.long),
            "labels": Y,
            "loss_mask": loss_mask,
            "pixel_values": pixel_values,
            "actions": self._extract_actions(sample),
        }


__all__ = ["DronesVLMDataset"]
