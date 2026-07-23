"""Wan2.1-T2V-1.3B — the card, declared as data.

The constant below is the whole contract: components by upstream class +
checkpoint subfolder, the loop the weights assume (by semantics id), and the
sampling defaults that are part of the trained artifact. Variants derive from
it (``derive(WAN21_T2V_1_3B, model_id=..., ...)``); nothing here is callable.
"""
from __future__ import annotations

from fastvideo2.card import ComponentSpec, LoopSpec, ModelCard, Provenance, SamplingDefaults, derive

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


# --------------------------------------------------------------------------- #
# FastWan-QAD-FP8 — fastvideo-main-trained DMD student served with dynamic FP8.
# Authority: fastvideo-main (see model_fv.py header for the pinned commit);
# alignment target is bit-exactness vs main's own serving path (flash attn,
# fp8 per-tensor, no compile), gated by gates/capture_fastvideo_main goldens.
# --------------------------------------------------------------------------- #
FASTWAN_QAD_FP8_1_3B = derive(
    WAN21_T2V_1_3B,
    model_id="fastwan-qad-fp8-1.3b",
    weights="FastVideo/FastWan-QAD-FP8-1.3B",
    components={
        # main-vendored forward + post-load per-tensor FP8 on the block
        # linears; dtype is the LOAD cast (fp32 checkpoint -> bf16 -> fp8
        # codes; that chain is part of the artifact).
        "transformer": ComponentSpec("transformer", kind="dit",
                                     module="fastvideo2.wan21.model_fv:WanModelFVFP8",
                                     subfolder="transformer", dtype="bf16"),
        # main runs the UMT5 in fp32 (official runs bf16); the fp32 embeds
        # also set the initial-latent dtype in main's DMD stage.
        "text_encoder": ComponentSpec("text_encoder", kind="text_encoder",
                                      module="transformers:UMT5EncoderModel",
                                      subfolder="text_encoder", dtype="fp32"),
    },
    loops={
        "denoise": LoopSpec("denoise", loop="fastvideo2.wan21.loop:WanDMDLoop",
                            params={"timesteps": [1000, 757, 522], "shift": 8.0,
                                    "latent_channels": 16, "spatial_ratio": 8,
                                    "temporal_ratio": 4}),
    },
    provenance={"method": "dmd-qad-fp8", "parents": ("wan2.1-t2v-1.3b",),
                "assumes_loop": "wan.dmd.fvmain/v1", "precision": "bf16",
                "substitution": "quality-changing"},
    # 3 fixed DMD timesteps, CFG-free; negative prompt is not part of this
    # artifact.
    sampling_defaults={"num_steps": 3, "guidance_scale": 1.0, "shift": 8.0,
                       "negative_prompt": ""},
)


# --------------------------------------------------------------------------- #
# FastWan (VSA-distilled) — sparse-attention DMD student, bf16 serving.
# Same authority and gates as QAD; the kernel and 0.8 sparsity are part of
# the released recipe. VSA fails closed when fastvideo_kernel is absent.
# --------------------------------------------------------------------------- #
FASTWAN_T2V_1_3B = derive(
    WAN21_T2V_1_3B,
    model_id="fastwan-t2v-1.3b",
    weights="FastVideo/FastWan2.1-T2V-1.3B-Diffusers",
    components={
        "transformer": ComponentSpec("transformer", kind="dit",
                                     module="fastvideo2.wan21.model_fv:WanModelFVVSA",
                                     subfolder="transformer", dtype="bf16"),
        "text_encoder": ComponentSpec("text_encoder", kind="text_encoder",
                                      module="transformers:UMT5EncoderModel",
                                      subfolder="text_encoder", dtype="fp32"),
    },
    loops={
        "denoise": LoopSpec("denoise", loop="fastvideo2.wan21.loop:WanDMDLoop",
                            params={"timesteps": [1000, 757, 522], "shift": 8.0,
                                    "vsa_sparsity": 0.8,
                                    "latent_channels": 16, "spatial_ratio": 8,
                                    "temporal_ratio": 4}),
    },
    provenance={"method": "dmd-vsa", "parents": ("wan2.1-t2v-1.3b",),
                "assumes_loop": "wan.dmd.fvmain/v1", "precision": "bf16",
                "substitution": "quality-changing"},
    sampling_defaults={"num_steps": 3, "guidance_scale": 1.0, "shift": 8.0,
                       "negative_prompt": ""},
)


# --------------------------------------------------------------------------- #
# SFWan (self-forcing causal) — chunked realtime rollout, 4-step DMD per
# 3-frame block over a KV cache. Authority: fastvideo-main (their released
# checkpoint + CausalDMDDenosingStage); warp semantics and the
# SelfForcingFlowMatchScheduler table are vendored in the loop.
# --------------------------------------------------------------------------- #
SFWAN_T2V_1_3B = derive(
    WAN21_T2V_1_3B,
    model_id="sfwan-t2v-1.3b",
    weights="wlsaidhi/SFWan2.1-T2V-1.3B-Diffusers",
    components={
        "transformer": ComponentSpec("transformer", kind="dit",
                                     module="fastvideo2.wan21.model_fv:WanModelFVCausal",
                                     subfolder="transformer", dtype="bf16"),
        "text_encoder": ComponentSpec("text_encoder", kind="text_encoder",
                                      module="transformers:UMT5EncoderModel",
                                      subfolder="text_encoder", dtype="fp32"),
    },
    loops={
        "denoise": LoopSpec("denoise", loop="fastvideo2.wan21.loop:WanCausalDMDLoop",
                            params={"timesteps": [1000, 750, 500, 250], "shift": 5.0,
                                    "num_frames_per_block": 3, "context_noise": 0,
                                    "latent_channels": 16, "spatial_ratio": 8,
                                    "temporal_ratio": 4}),
    },
    provenance={"method": "self-forcing-dmd", "parents": ("wan2.1-t2v-1.3b",),
                "assumes_loop": "wan.causal_dmd.chunked/v1", "precision": "bf16",
                "substitution": "quality-changing"},
    sampling_defaults={"num_steps": 4, "guidance_scale": 1.0, "shift": 5.0,
                       "negative_prompt": ""},
)
