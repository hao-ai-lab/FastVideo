"""VideoScore2 — VLM-based video quality scoring.

Uses a Qwen2.5-VL model fine-tuned to score generated videos on three
dimensions: visual quality, text-to-video alignment, and physical
consistency. Scores are extracted from token logits as upstream's
``ll_based_soft_score_normed`` weighting (1-5 scale).

Reference: TIGER-AI-Lab/VideoScore2 (vs2_inference.py).
"""

from __future__ import annotations

import re
from string import Template

import numpy as np
import torch
from PIL import Image

from fastvideo.eval.metrics.base import BaseMetric
from fastvideo.eval.registry import register
from fastvideo.eval.types import MetricResult


# Match upstream verbatim, including leading newline and 4-space indents
# (TIGER-AI-Lab/VideoScore2/vs2_inference.py).
VS2_QUERY_TEMPLATE = Template("""
    You are an expert for evaluating AI-generated videos from three dimensions:
    (1) visual quality – clarity, smoothness, artifacts;
    (2) text-to-video alignment – fidelity to the prompt;
    (3) physical/common-sense consistency – naturalness and physics plausibility.

    Video prompt: $t2v_prompt

    Please output in this format:
    visual quality: <v_score>;
    text-to-video alignment: <t_score>,
    physical/common-sense consistency: <p_score>
    """)


# The released VideoScore2 model emits a <think>...</think> chain-of-
# thought followed by a numbered list of the form:
#
#     (1) visual quality – clarity, smoothness, artifacts: 3
#     (2) text-to-video alignment – fidelity to the prompt: 4
#     (3) physical/common-sense consistency – naturalness and physics …: 3
#
# Upstream's vs2_inference.py regex (``visual quality:\s*(\d+)``) does
# not match this output — it expects the colon directly after the
# header, with no descriptor in between, so upstream's script returns
# ``null`` on its own released model. We anchor on the ``(N)`` prefix
# to avoid matching digits inside the chain-of-thought reasoning.
SCORE_PATTERN = re.compile(
    r"\(1\)\s*visual quality[^\d]*?(\d+).*?"
    r"\(2\)\s*text-to-video alignment[^\d]*?(\d+).*?"
    r"\(3\)\s*physical/common-sense consistency[^\d]*?(\d+)",
    re.DOTALL | re.IGNORECASE,
)


def _find_score_token_index(prompt_text: str, tokenizer, gen_ids: list[int]) -> int:
    """Find the token index where the score digit appears after prompt_text."""
    gen_str = tokenizer.decode(gen_ids, skip_special_tokens=False)
    pattern = r"(?:\(\d+\)\s*|\n\s*)?" + re.escape(prompt_text)
    match = re.search(pattern, gen_str, flags=re.IGNORECASE)
    if not match:
        return -1
    after = gen_str[match.end():]
    num_match = re.search(r"\d", after)
    if not num_match:
        return -1
    target = gen_str[: match.end() + num_match.start() + 1]
    for i in range(len(gen_ids)):
        if tokenizer.decode(gen_ids[: i + 1], skip_special_tokens=False) == target:
            return i
    return -1


def _ll_based_soft_score_normed(hard_val: int | None, token_idx: int, scores, tokenizer, seq_idx: int = 0) -> float | None:
    """Upstream VideoScore2's soft score: argmax_score × (argmax_prob / Σprob).

    Matches ``ll_based_soft_score_normed`` in
    ``TIGER-AI-Lab/VideoScore2/vs2_inference.py``. The ``seq_idx`` arg
    is the only addition (for batched generate). With ``B == 1`` the
    behaviour is identical to upstream.
    """
    if hard_val is None or token_idx < 0:
        return None
    logits = scores[token_idx][seq_idx]
    score_probs = []
    for s in range(1, 6):
        ids = tokenizer.encode(str(s), add_special_tokens=False)
        if len(ids) == 1:
            logp = torch.log_softmax(logits, dim=-1)[ids[0]].item()
            score_probs.append((s, float(np.exp(logp))))
    if not score_probs:
        return None
    scores_list, probs_list = zip(*score_probs)
    total_prob = sum(probs_list)
    max_prob = max(probs_list)
    best_score = scores_list[probs_list.index(max_prob)]
    normalized_prob = max_prob / total_prob if total_prob > 0 else 0
    return round(best_score * normalized_prob, 4)


def _parse_output(output_text: str, scores, tokenizer, gen_ids: list[int], seq_idx: int = 0) -> dict:
    """Parse scores from a single sequence's output."""
    match = SCORE_PATTERN.search(output_text)
    v_hard = int(match.group(1)) if match else None
    t_hard = int(match.group(2)) if match else None
    p_hard = int(match.group(3)) if match else None

    if scores is not None:
        # Anchor on the numbered list to skip the chain-of-thought.
        idx_v = _find_score_token_index("(1) visual quality", tokenizer, gen_ids)
        idx_t = _find_score_token_index("(2) text-to-video alignment", tokenizer, gen_ids)
        idx_p = _find_score_token_index("(3) physical/common-sense consistency", tokenizer, gen_ids)
        v_soft = _ll_based_soft_score_normed(v_hard, idx_v, scores, tokenizer, seq_idx)
        t_soft = _ll_based_soft_score_normed(t_hard, idx_t, scores, tokenizer, seq_idx)
        p_soft = _ll_based_soft_score_normed(p_hard, idx_p, scores, tokenizer, seq_idx)
    else:
        v_soft = float(v_hard) if v_hard is not None else None
        t_soft = float(t_hard) if t_hard is not None else None
        p_soft = float(p_hard) if p_hard is not None else None

    return {
        "visual_quality": v_soft,
        "text_alignment": t_soft,
        "physical_consistency": p_soft,
        "visual_quality_hard": v_hard,
        "text_alignment_hard": t_hard,
        "physical_consistency_hard": p_hard,
        "raw_output": output_text,
    }


@register("videoscore2")
class VideoScore2Metric(BaseMetric):
    """VideoScore2: VLM-based video quality scoring (3 dimensions).

    Requires ``sample["text_prompt"]`` for text-to-video alignment.
    Supports batched generation for GPU efficiency.
    """

    name = "videoscore2"
    requires_reference = False
    higher_is_better = True
    needs_gpu = True
    dependencies = ["transformers", "qwen_vl_utils"]
    batch_unit = "video"

    def __init__(
        self,
        model_name: str = "TIGER-Lab/VideoScore2",
        infer_fps: float = 2.0,
        max_tokens: int = 1024,
        temperature: float = 0.7,
        do_sample: bool = True,
    ) -> None:
        super().__init__()
        self._model_name = model_name
        self.infer_fps = infer_fps
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.do_sample = do_sample
        self._model = None
        self._processor = None
        self._tokenizer = None

    def to(self, device):
        super().to(device)
        if self._model is not None:
            self._model = self._model.to(self.device)
        return self

    def setup(self) -> None:
        if self._model is not None:
            return
        from transformers import AutoProcessor, AutoModelForVision2Seq, AutoTokenizer

        # Match upstream: no dtype kwarg, weights load with whatever the
        # checkpoint was saved as (bf16 for the released VideoScore2).
        self._model = AutoModelForVision2Seq.from_pretrained(
            self._model_name, trust_remote_code=True,
        ).to(self.device)
        self._model.eval()
        self._processor = AutoProcessor.from_pretrained(
            self._model_name, trust_remote_code=True,
        )
        self._tokenizer = getattr(self._processor, "tokenizer", None)
        if self._tokenizer is None:
            self._tokenizer = AutoTokenizer.from_pretrained(
                self._model_name, trust_remote_code=True, use_fast=False,
            )

    def _tensor_to_pil_list(self, video: torch.Tensor) -> list[Image.Image]:
        """Convert (T, C, H, W) float [0,1] tensor to list of PIL images."""
        frames = (video.permute(0, 2, 3, 1).cpu().numpy() * 255).astype(np.uint8)
        return [Image.fromarray(frames[t]) for t in range(frames.shape[0])]

    def _subsample_frames(self, pil_frames: list[Image.Image], max_frames: int = 64, max_resolution: int = 960) -> list[Image.Image]:
        """Subsample to ~infer_fps worth of frames (max 64), resize if too large."""
        n = len(pil_frames)
        target = min(n, max_frames)
        if target < n:
            indices = np.linspace(0, n - 1, target, dtype=int)
            pil_frames = [pil_frames[i] for i in indices]

        w, h = pil_frames[0].size
        if max(w, h) > max_resolution:
            scale = max_resolution / max(w, h)
            new_w, new_h = int(w * scale), int(h * scale)
            pil_frames = [f.resize((new_w, new_h), Image.LANCZOS) for f in pil_frames]

        return pil_frames

    @torch.no_grad()
    def compute(self, sample: dict) -> list[MetricResult]:
        if self._model is None:
            self.setup()

        from qwen_vl_utils import process_vision_info

        video = sample["video"]  # (B, T, C, H, W)
        text = sample.get("text_prompt", "")
        B = video.shape[0]

        if isinstance(text, str):
            text = [text] * B

        # Build batch: each video gets its own message, then batch through processor
        texts = []
        all_videos = []
        for b in range(B):
            pil_frames = self._tensor_to_pil_list(video[b])
            pil_frames = self._subsample_frames(pil_frames)
            user_prompt = VS2_QUERY_TEMPLATE.substitute(t2v_prompt=text[b])

            messages = [{"role": "user", "content": [
                {"type": "video", "video": pil_frames, "fps": self.infer_fps},
                {"type": "text", "text": user_prompt},
            ]}]

            chat_text = self._processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
            )
            texts.append(chat_text)

            _, vid_inputs = process_vision_info(messages)
            if vid_inputs:
                all_videos.extend(vid_inputs)

        # Batched processor call
        inputs = self._processor(
            text=texts,
            videos=all_videos if all_videos else None,
            fps=self.infer_fps,
            padding=True,
            return_tensors="pt",
        ).to(self.device)

        input_len = inputs["input_ids"].shape[1]

        # Batched generation
        gen_kwargs = dict(
            max_new_tokens=self.max_tokens,
            output_scores=True,
            return_dict_in_generate=True,
            do_sample=self.do_sample,
        )
        if self.do_sample:
            gen_kwargs["temperature"] = self.temperature
        gen_out = self._model.generate(**inputs, **gen_kwargs)

        sequences = gen_out.sequences
        scores = gen_out.scores

        # Parse each sequence
        results = []
        for b in range(B):
            gen_ids = sequences[b, input_len:].tolist()
            # Strip padding tokens from gen_ids
            pad_id = self._tokenizer.pad_token_id
            if pad_id is not None:
                gen_ids = [t for t in gen_ids if t != pad_id]

            output_text = self._tokenizer.decode(gen_ids, skip_special_tokens=True)

            parsed = _parse_output(output_text, scores, self._tokenizer, gen_ids, seq_idx=b)

            soft_vals = [v for v in [parsed["visual_quality"],
                                     parsed["text_alignment"],
                                     parsed["physical_consistency"]] if v is not None]
            combined = sum(soft_vals) / len(soft_vals) if soft_vals else 0.0

            results.append(MetricResult(
                name=self.name,
                score=combined,
                details=parsed,
            ))
        return results
