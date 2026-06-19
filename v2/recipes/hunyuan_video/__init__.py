"""HunyuanVideo — rectified flow-match video DiT ported into the v2 substrate.

Self-contained recipe package: the card declares its torch adapters via ``ComponentSpec.adapter``
(``HunyuanVideoDiT`` / ``HunyuanVideoVAE`` / ``HunyuanVideoLlamaEncoder`` / ``HunyuanVideoCLIPEncoder`` in
``v2/platform/backends/torch_hunyuan_video.py``) and reuses the canonical ``WanDenoiseLoop`` flow-match math
via a thin ``HunyuanDenoiseLoop`` (which only threads the CLIP-pooled vector through Wan's ``context=``
channel). The new work vs Wan: dual text-encoder marshalling (LLaMA per-token sequence + CLIP pooled global
vector packed into the DiT's 2-element ``encoder_hidden_states``) and the Hunyuan VAE scalar-``scaling_factor``
normalization. Reuses ``stamp_wan21_checkpoints``. Registered in ``v2/registry.py``.
"""
from __future__ import annotations

from v2.recipes.hunyuan_video.card import build_fast_hunyuan_video_card, build_hunyuan_video_card
from v2.recipes.hunyuan_video.loop import HunyuanDenoiseLoop
from v2.recipes.hunyuan_video.program import build_hunyuan_video_program

__all__ = [
    "build_hunyuan_video_card",
    "build_fast_hunyuan_video_card",
    "build_hunyuan_video_program",
    "HunyuanDenoiseLoop",
]
