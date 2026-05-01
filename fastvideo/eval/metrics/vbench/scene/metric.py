"""VLM-based scene matching using AVoCaDO (Qwen2.5-Omni).

Replaces VBench's Tag2Text-based scene metric with a modern VLM caption.
The algorithm follows VBench:
  1. Caption the video
  2. Check if all scene keywords appear in the caption
  3. Score = 1.0 if all match, 0.0 otherwise

Unlike VBench (which captions each frame separately with Tag2Text),
AVoCaDO captions the entire video in one pass with rich natural language,
making the keyword check more robust.
"""

from __future__ import annotations

import os
import tempfile

import torch
import torchvision.io

from fastvideo.eval.metrics.base import BaseMetric
from fastvideo.eval.registry import register
from fastvideo.eval.types import MetricResult

_SCENE_PROMPT = (
    "Describe the visual scene in this video, including the location, "
    "environment, objects, and overall setting. Be specific and use "
    "concrete descriptive words."
)


@register("vbench.scene")
class SceneMetric(BaseMetric):

    name = "vbench.scene"
    requires_reference = False
    higher_is_better = True
    needs_gpu = True
    batch_unit = "video"
    dependencies = ["transformers", "qwen_omni_utils"]
    backbone = "avocado"

    def __init__(self, model_path: str = "AVoCaDO-Captioner/AVoCaDO") -> None:
        super().__init__()
        self._model = None
        self._processor = None
        self._model_path = model_path

    def to(self, device):
        super().to(device)
        if self._model is not None:
            self._model = self._model.to(self.device)
        return self

    def setup(self) -> None:
        if self._model is not None:
            return
        from transformers import Qwen2_5OmniForConditionalGeneration, Qwen2_5OmniProcessor

        # AVoCaDO uses Qwen2.5-Omni — large multimodal model
        os.environ.setdefault("VIDEO_MAX_PIXELS", str(20070400))  # 512*28*28*50

        self._model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
            self._model_path,
            torch_dtype=torch.bfloat16,
            device_map=str(self.device) if self.device.type == "cuda" else None,
        )
        self._model.disable_talker()
        self._model.eval()
        self._processor = Qwen2_5OmniProcessor.from_pretrained(self._model_path)

    def _save_temp_video(self, video: torch.Tensor) -> str:
        """Save (T, C, H, W) float [0,1] tensor as a temp mp4 file."""
        # torchvision expects (T, H, W, C) uint8
        frames = (video * 255).clamp(0, 255).to(torch.uint8).permute(0, 2, 3, 1).cpu()
        tmp = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
        tmp.close()
        torchvision.io.write_video(tmp.name, frames, fps=8, video_codec="libx264",
                                    options={"crf": "18"})
        return tmp.name

    def _generate_caption(self, video_path: str) -> str:
        from qwen_omni_utils import process_mm_info

        conversation = [
            {
                "role": "system",
                "content": [{"type": "text", "text": (
                    "You are Qwen, a virtual human developed by the Qwen Team, "
                    "Alibaba Group, capable of perceiving auditory and visual inputs."
                )}],
            },
            {
                "role": "user",
                "content": [
                    {"type": "video", "video": video_path, "max_pixels": 401408},
                    {"type": "text", "text": _SCENE_PROMPT},
                ],
            },
        ]

        text = self._processor.apply_chat_template(
            conversation, add_generation_prompt=True, tokenize=False
        )
        # AVoCaDO is video+audio; for scene matching we don't need audio,
        # but the model expects it so let it process
        audios, images, videos = process_mm_info(conversation, use_audio_in_video=False)
        inputs = self._processor(
            text=text, audio=audios, images=images, videos=videos,
            return_tensors="pt", padding=True, use_audio_in_video=False,
        )
        inputs = inputs.to(self._model.device).to(self._model.dtype)

        with torch.no_grad():
            text_ids = self._model.generate(
                **inputs, use_audio_in_video=False, return_audio=False,
                do_sample=False, thinker_max_new_tokens=512,
            )

        decoded = self._processor.batch_decode(
            text_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]
        return decoded.split("\nassistant\n")[-1].lower()

    @torch.no_grad()
    def compute(self, sample: dict) -> list[MetricResult]:
        video = sample["video"]  # (B, T, C, H, W)
        aux = sample.get("auxiliary_info")
        if aux is None:
            return self._skip(sample, "missing auxiliary_info with 'scene' key")

        B = video.shape[0]
        results = []

        for b in range(B):
            scene_keywords = aux[b]["scene"] if isinstance(aux, list) else aux["scene"]
            keywords = [k.strip().lower() for k in scene_keywords.split() if k.strip()]

            tmp_path = self._save_temp_video(video[b])
            try:
                caption = self._generate_caption(tmp_path)
            finally:
                os.unlink(tmp_path)

            # Check if all keywords appear in the caption
            matched = [kw for kw in keywords if kw in caption]
            score = 1.0 if len(matched) == len(keywords) else 0.0

            results.append(MetricResult(
                name=self.name,
                score=score,
                details={
                    "caption": caption[:500],
                    "keywords": keywords,
                    "matched": matched,
                },
            ))

        return results
