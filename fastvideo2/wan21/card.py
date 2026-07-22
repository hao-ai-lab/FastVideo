"""Wan2.1-T2V-1.3B — the card, declared as data.

The constant below is the whole contract: components by upstream class +
checkpoint subfolder, the loop the weights assume (by semantics id), and the
sampling defaults that are part of the trained artifact. Variants derive from
it (``derive(WAN21_T2V_1_3B, model_id=..., ...)``); nothing here is callable.
"""
from __future__ import annotations

from fastvideo2.card import ComponentSpec, LoopSpec, ModelCard, Provenance, SamplingDefaults

# The canonical Wan2.1 negative prompt (shipped with the model release).
WAN_NEG = ("色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，"
           "低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，"
           "毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走")

WAN21_T2V_1_3B = ModelCard(
    model_id="wan2.1-t2v-1.3b",
    family="wan",
    weights="Wan-AI/Wan2.1-T2V-1.3B-Diffusers",
    components={
        "tokenizer": ComponentSpec("tokenizer", kind="tokenizer",
                                   module="transformers:AutoTokenizer",
                                   subfolder="tokenizer", dtype=""),
        "text_encoder": ComponentSpec("text_encoder", kind="text_encoder",
                                      module="transformers:UMT5EncoderModel",
                                      subfolder="text_encoder", dtype="bf16"),
        # The DiT is the OFFICIAL modeling code (vendored verbatim) running the
        # official-layout weights — alignment by construction, not by porting.
        # dtype is STORAGE precision: official's pipeline loads fp32 storage
        # and computes under bf16 autocast (provenance.precision); loading the
        # weights as bf16 rounds them and shows up in the fp32 anchor row.
        "transformer": ComponentSpec("transformer", kind="dit",
                                     module="fastvideo2.wan21.model:WanModel",
                                     subfolder="", dtype="fp32",
                                     source="Wan-AI/Wan2.1-T2V-1.3B"),
        # Wan VAE runs fp32 (bf16 decode visibly degrades output).
        "vae": ComponentSpec("vae", kind="vae",
                             module="diffusers:AutoencoderKLWan",
                             subfolder="vae", dtype="fp32"),
    },
    loops={
        "denoise": LoopSpec("denoise", loop="fastvideo2.wan21.loop:WanDenoiseLoop",
                            params={"latent_channels": 16, "spatial_ratio": 8, "temporal_ratio": 4}),
    },
    capabilities=("text_to_video",),
    provenance=Provenance(method="base", assumes_loop="wan.flow_euler.cfg/v1", precision="bf16"),
    sampling_defaults=SamplingDefaults(num_steps=50, guidance_scale=5.0, height=480, width=832,
                                       num_frames=81, fps=16, shift=3.0, negative_prompt=WAN_NEG),
    determinism="tolerance",
).validate()
