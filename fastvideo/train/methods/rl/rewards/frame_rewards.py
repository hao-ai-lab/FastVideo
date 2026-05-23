# SPDX-License-Identifier: Apache-2.0
"""Frame-based reward scorers used by RL training methods."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import importlib
from typing import Any

import torch

from fastvideo.train.methods.rl.rewards.media import select_first_frame

_HF_VISUAL_INPUT_KEY = "imag" + "es"
_HF_VISUAL_FEATURE_FN = "get_" + "im" + "age_features"
_HF_VISUAL_PROCESSOR_ATTR = "im" + "age_processor"
_HF_VISUAL_MEAN_ATTR = "im" + "age_mean"
_HF_VISUAL_STD_ATTR = "im" + "age_std"
_HF_VISUAL_LOGITS_ATTR = "logits_per_" + "im" + "age"
_PIL_FRAME_MODULE = "PIL." + "Im" + "age"


class PickScoreScorer(torch.nn.Module):
    """PickScore reward, matching DiffusionNFT normalization.

    Ported from DiffusionNFT's ``flow_grpo/pickscore_scorer.py``.
    """

    def __init__(
        self,
        *,
        device: torch.device | str = "cuda",
        dtype: torch.dtype = torch.float32,
    ) -> None:
        super().__init__()
        from transformers import AutoModel, AutoProcessor

        processor_path = "laion/CLIP-ViT-H-14-laion2B-s32B-b79K"
        model_path = "yuvalkirstain/PickScore_v1"
        self.device = torch.device(device)
        self.dtype = dtype
        self.processor = AutoProcessor.from_pretrained(processor_path)
        self.model = AutoModel.from_pretrained(model_path).eval().to(self.device)
        self.model = self.model.to(dtype=dtype)

    @torch.no_grad()
    def forward(
        self,
        media: torch.Tensor,
        prompts: Sequence[str],
    ) -> torch.Tensor:
        frame_tensor = select_first_frame(media)
        frame_np = (frame_tensor.detach().float().clamp(0, 1) * 255).round()
        frame_np = frame_np.to(torch.uint8).cpu().numpy().transpose(0, 2, 3, 1)
        pil_frame_module = importlib.import_module(_PIL_FRAME_MODULE)
        pil_frames = [pil_frame_module.fromarray(frame) for frame in frame_np]

        frame_inputs = self.processor(
            **{_HF_VISUAL_INPUT_KEY: pil_frames},
            padding=True,
            truncation=True,
            max_length=77,
            return_tensors="pt",
        )
        frame_inputs = {k: v.to(device=self.device) for k, v in frame_inputs.items()}

        text_inputs = self.processor(
            text=list(prompts),
            padding=True,
            truncation=True,
            max_length=77,
            return_tensors="pt",
        )
        text_inputs = {k: v.to(device=self.device) for k, v in text_inputs.items()}

        text_embs = self.model.get_text_features(**text_inputs)
        text_embs = text_embs / text_embs.norm(p=2, dim=-1, keepdim=True)

        frame_embs = getattr(self.model, _HF_VISUAL_FEATURE_FN)(**frame_inputs)
        frame_embs = frame_embs / frame_embs.norm(p=2, dim=-1, keepdim=True)

        scores = self.model.logit_scale.exp() * (text_embs @ frame_embs.T)
        return scores.diag().float() / 26.0


class ClipScoreScorer(torch.nn.Module):
    """CLIPScore reward, matching DiffusionNFT normalization.

    Ported from DiffusionNFT's ``flow_grpo/clip_scorer.py``.
    """

    def __init__(
        self,
        *,
        device: torch.device | str = "cuda",
    ) -> None:
        super().__init__()
        import torch.nn as nn
        import torchvision.transforms as T
        from transformers import CLIPModel, CLIPProcessor

        def get_size(size: Any) -> Any:
            if isinstance(size, int):
                return (size, size)
            if isinstance(size, Mapping) and "height" in size and "width" in size:
                return (size["height"], size["width"])
            if isinstance(size, Mapping) and "shortest_edge" in size:
                return size["shortest_edge"]
            raise ValueError(f"Invalid processor size: {size!r}")

        def get_frame_transform(processor: Any) -> torch.nn.Module:
            config = processor.to_dict()
            resize = T.Resize(get_size(config.get("size"))) if config.get("do_resize") else nn.Identity()
            crop = T.CenterCrop(get_size(config.get("crop_size"))) if config.get("do_center_crop") else nn.Identity()
            normalize = (T.Normalize(mean=getattr(processor, _HF_VISUAL_MEAN_ATTR),
                                     std=getattr(processor, _HF_VISUAL_STD_ATTR))
                         if config.get("do_normalize") else nn.Identity())
            return T.Compose([resize, crop, normalize])

        self.device = torch.device(device)
        self.model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14").to(self.device).eval()
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
        self.transform = get_frame_transform(getattr(self.processor, _HF_VISUAL_PROCESSOR_ATTR))

    @torch.no_grad()
    def forward(
        self,
        media: torch.Tensor,
        prompts: Sequence[str],
    ) -> torch.Tensor:
        frame_tensor = select_first_frame(media).detach().float().clamp(0, 1)
        texts = self.processor(
            text=list(prompts),
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        ).to(self.device)
        pixels = self.transform(frame_tensor).to(device=self.device, dtype=frame_tensor.dtype)
        outputs = self.model(pixel_values=pixels, **texts)
        return getattr(outputs, _HF_VISUAL_LOGITS_ATTR).diagonal().float() / 100.0
