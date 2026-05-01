"""VBench Color — GRiT dense captioning for color accuracy.

Detects the target object via GRiT and checks if the expected color
keyword appears in the object's caption. Score = frames_with_correct_color
/ frames_with_object_detected.
"""

from __future__ import annotations

import torch

from fastvideo.eval.metrics.base import BaseMetric
from fastvideo.eval.registry import register
from fastvideo.eval.types import MetricResult

_COLOR_KEYWORDS = [
    "white", "red", "pink", "blue", "silver", "purple",
    "orange", "green", "gray", "yellow", "black", "grey",
]


@register("vbench.color")
class ColorMetric(BaseMetric):

    name = "vbench.color"
    requires_reference = False
    higher_is_better = True
    needs_gpu = True
    batch_unit = "video"
    dependencies = ["detectron2"]

    def __init__(self) -> None:
        super().__init__()
        self._model = None

    def setup(self) -> None:
        if self._model is not None:
            return
        from fastvideo.eval.metrics.vbench._grit_helper import load_grit_model
        self._model = load_grit_model(self.device)

    @torch.no_grad()
    def compute(self, sample: dict) -> list[MetricResult]:
        from fastvideo.eval.metrics.vbench._grit_helper import prepare_frames

        video = sample["video"]  # (B, T, C, H, W)
        aux = sample.get("auxiliary_info")
        text_prompts = sample.get("text_prompt")
        if aux is None:
            return self._skip(sample, "missing auxiliary_info with 'color' key")

        B = video.shape[0]
        results = []

        for b in range(B):
            color_key = aux[b]["color"] if isinstance(aux, list) else aux["color"]
            prompt = text_prompts[b] if text_prompts else ""
            # Parse object name: remove "a ", "an ", and the color word
            object_key = prompt.replace("a ", "").replace("an ", "").replace(color_key, "").strip()

            frames_np = prepare_frames(video[b])

            # Run GRiT with caption output
            preds = []
            with torch.no_grad():
                for frame in frames_np:
                    ret = self._model.run_caption_tensor(frame)
                    cur_pred = []
                    if len(ret[0]) < 1:
                        cur_pred.append(["", ""])
                    else:
                        for cap_det in ret[0]:
                            cur_pred.append([cap_det[0], cap_det[2][0]])
                    preds.append(cur_pred)

            # Score: matching VBench's check_generate logic
            cur_object = 0
            cur_object_color = 0
            for frame_pred in preds:
                object_flag = False
                color_flag = False
                for pred in frame_pred:
                    if object_key == pred[1]:
                        for cq in _COLOR_KEYWORDS:
                            if cq in pred[0]:
                                object_flag = True
                        if color_key in pred[0]:
                            color_flag = True
                if color_flag:
                    cur_object_color += 1
                if object_flag:
                    cur_object += 1

            score = cur_object_color / cur_object if cur_object > 0 else 0.0
            results.append(MetricResult(
                name=self.name,
                score=float(score),
                details={"object_detected": cur_object, "color_correct": cur_object_color},
            ))

        return results
