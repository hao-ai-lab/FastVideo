#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""FastVideo-side AnyFlow 14B T2V demo at NFE=4 and NFE=50.

Loads ``nvidia/AnyFlow-Wan2.1-T2V-14B-Diffusers`` into FastVideo's
``WanTransformer3DModel`` (with the ``param_names_mapping`` regex
handling the ``delta_embedder`` rename), runs the
``FlowMapEulerDiscreteScheduler`` for the requested NFE schedule, and
saves the decoded video as MP4. Matches the prompt / shift / guidance
recipe used by the parallel FastGen demo so the videos are directly
comparable.

Memory tactics (single H200, 141 GB HBM):
- Encode prompts with UMT5, free the encoder.
- Build the FastVideo Wan-14B transformer, load AnyFlow safetensor
  shards via param_names_mapping translation.
- Sample at both NFEs without re-loading the transformer.
- Free the transformer, then load the Wan VAE with tiling for decode.

Run via::

    srun --jobid=304 bash -c '
      source ~/.venvs/anyflow/bin/activate
      export CPATH=/home/guian/.local/share/uv/python/cpython-3.10.20-linux-x86_64-gnu/include/python3.10
      cd ~/projects/anyflow/FastVideo
      PYTHONPATH=$PWD python scripts/demo_anyflow_14b.py
    '
"""

from __future__ import annotations

import gc
import os
import re
import sys
import time
from pathlib import Path

import torch
from safetensors.torch import load_file


ANYFLOW_LOCAL = Path("/home/guian/projects/anyflow/anyflow-14b")
OUT_DIR = Path("/home/guian/projects/anyflow/demo_videos")
OUT_DIR.mkdir(exist_ok=True)

SEED = 0
DEVICE = torch.device("cuda")
DTYPE = torch.bfloat16
NUM_FRAMES = 81
HEIGHT, WIDTH = 480, 832
PROMPT = (
    "CG game concept digital art, a majestic elephant with a vibrant tusk and sleek fur "
    "running swiftly towards a herd of its kind. The elephant has a calm yet determined "
    "expression, with its ears flapping slightly as it moves at high speed. The herd consists "
    "of several other elephants of various ages and sizes, all moving in unison. The landscape "
    "is vast savanna with rolling hills, tall grasses, and scattered acacia trees. The sun "
    "sets behind the horizon, casting a warm golden glow over the scene. Low-angle view, focus "
    "on the elephant as it accelerates towards the herd."
)
NEG_PROMPT = "blurry, low quality, distorted"
# The published nvidia/AnyFlow-* checkpoints are on-policy distilled with
# fuse_guidance_scale=3.0 baked into the weights — inference uses 1.0
# (single conditional forward, no CFG; matches AnyFlow's official demo.py).
GUIDANCE = 1.0


def banner(msg: str) -> None:
    print("\n" + "=" * 80)
    print(msg)
    print("=" * 80)


def free_gpu() -> None:
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    used = torch.cuda.memory_allocated() / 1e9
    print(f"  [mem] allocated {used:.1f} GB after free")


def init_single_rank() -> None:
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29571")
    os.environ.setdefault("WORLD_SIZE", "1")
    os.environ.setdefault("RANK", "0")
    os.environ.setdefault("LOCAL_RANK", "0")
    from fastvideo.distributed import (
        init_distributed_environment,
        initialize_model_parallel,
    )
    init_distributed_environment(world_size=1, rank=0, local_rank=0, backend="nccl")
    initialize_model_parallel(
        tensor_model_parallel_size=1,
        sequence_model_parallel_size=1,
        data_parallel_size=1,
    )


def translate_keys(raw: dict, *, mapping: dict[str, str]) -> dict:
    out: dict = {}
    for k, v in raw.items():
        new_k = k
        for pat, repl in mapping.items():
            new_k = re.sub(pat, repl, new_k)
        out[new_k] = v
    return out


def encode_prompts():
    banner("(1) Encode prompts via UMT5")
    from transformers import AutoTokenizer, UMT5EncoderModel

    tok = AutoTokenizer.from_pretrained(str(ANYFLOW_LOCAL), subfolder="tokenizer", use_fast=False)
    enc = UMT5EncoderModel.from_pretrained(
        str(ANYFLOW_LOCAL), subfolder="text_encoder", torch_dtype=DTYPE,
    ).to(DEVICE).eval()

    @torch.no_grad()
    def encode_one(prompts):
        out = tok(
            prompts, padding="max_length", max_length=512, truncation=True,
            return_attention_mask=True, return_tensors="pt")
        ids = out.input_ids.to(DEVICE)
        mask = out.attention_mask.to(DEVICE)
        seq_lens = mask.gt(0).sum(dim=1).long()
        embeds = enc(ids, mask).last_hidden_state
        padded = []
        for i, l in enumerate(seq_lens):
            e = embeds[i, :l]
            pad = torch.zeros(512 - e.size(0), e.size(1), device=DEVICE, dtype=embeds.dtype)
            padded.append(torch.cat([e, pad], dim=0))
        return torch.stack(padded, dim=0)

    text_e = encode_one([PROMPT])
    neg_e = encode_one([NEG_PROMPT])
    print(f"  prompts encoded: text={tuple(text_e.shape)} neg={tuple(neg_e.shape)}")
    del enc, tok
    free_gpu()
    return text_e, neg_e


def load_transformer():
    banner("(2) Build FastVideo Wan-14B + load AnyFlow weights")
    from fastvideo.configs.models.dits import WanVideoConfig
    from fastvideo.models.dits.wanvideo import WanTransformer3DModel

    cfg = WanVideoConfig()
    arch = cfg.arch_config
    # Wan2.1-T2V-14B arch (per AnyFlow checkpoint config.json).
    arch.num_attention_heads = 40
    arch.attention_head_dim = 128
    arch.num_layers = 40
    arch.ffn_dim = 13824
    arch.r_embedder = True
    arch.r_embedder_fusion = "gated"
    arch.r_embedder_gate_value = 0.25
    arch.r_embedder_deltatime_type = "r"
    arch.__post_init__()

    t0 = time.time()
    model = WanTransformer3DModel(config=cfg, hf_config={}).to(DEVICE, dtype=DTYPE).eval()
    print(f"  transformer built in {time.time() - t0:.1f}s; "
          f"params: {sum(p.numel() for p in model.parameters())/1e9:.2f}B")

    ckpt_dir = ANYFLOW_LOCAL / "transformer"
    sd: dict = {}
    for shard in sorted(ckpt_dir.glob("diffusion_pytorch_model-*.safetensors")):
        sd.update(load_file(str(shard), device="cpu"))
    print(f"  AnyFlow state dict: {len(sd)} tensors loaded")
    sd = translate_keys(sd, mapping=arch.param_names_mapping)
    info = model.load_state_dict(sd, strict=False)
    print(f"  load: missing={len(info.missing_keys)} unexpected={len(info.unexpected_keys)}")
    del sd
    free_gpu()
    return model


@torch.no_grad()
def sample(model, text_e, neg_e, nfe: int) -> torch.Tensor:
    banner(f"(3) Sample 14B NFE={nfe}")
    from fastvideo.forward_context import set_forward_context
    from fastvideo.models.schedulers.scheduling_flow_map_euler_discrete import (
        FlowMapEulerDiscreteScheduler, )

    scheduler = FlowMapEulerDiscreteScheduler(num_train_timesteps=1000, shift=5.0)
    scheduler.set_timesteps(num_inference_steps=nfe, device=DEVICE)
    timesteps = scheduler.timesteps.to(DEVICE, dtype=DTYPE)

    B, C = 1, 16
    F = (NUM_FRAMES - 1) // 4 + 1  # temporal VAE compression = 4 (81 → 21)
    H_l, W_l = HEIGHT // 8, WIDTH // 8
    g = torch.Generator(device=DEVICE).manual_seed(SEED)
    x = torch.randn(B, C, F, H_l, W_l, device=DEVICE, dtype=DTYPE, generator=g)

    t0 = time.time()
    for i, (t_cur, t_next) in enumerate(zip(timesteps[:-1], timesteps[1:])):
        t_in = t_cur.expand(B).to(DTYPE)
        r_in = t_next.expand(B).to(DTYPE)
        with set_forward_context(current_timestep=t_in, attn_metadata=None):
            flow_cond = model(
                hidden_states=x, encoder_hidden_states=text_e,
                timestep=t_in, r_timestep=r_in)
            if GUIDANCE != 1.0:
                flow_uncond = model(
                    hidden_states=x, encoder_hidden_states=neg_e,
                    timestep=t_in, r_timestep=r_in)
                flow = flow_uncond + GUIDANCE * (flow_cond - flow_uncond)
            else:
                flow = flow_cond
        x = scheduler.step(
            flow, sample=x,
            timestep=t_cur.repeat(B), r_timestep=t_next.repeat(B))
    print(f"  NFE={nfe} sample time: {time.time() - t0:.1f}s "
          f"({(time.time() - t0) / nfe:.1f}s/step)")
    xf = x.float()
    print(f"  latents mean={xf.mean().item():+.3f} std={xf.std().item():.3f} "
          f"range=[{xf.min().item():+.2f}, {xf.max().item():+.2f}] "
          f"finite={torch.isfinite(xf).all().item()}")
    return x.detach()


@torch.no_grad()
def decode_to_mp4(latents: torch.Tensor, out_path: Path) -> tuple[Path, tuple]:
    from diffusers.models.autoencoders.autoencoder_kl_wan import AutoencoderKLWan
    import imageio.v3 as iio

    vae = AutoencoderKLWan.from_pretrained(
        str(ANYFLOW_LOCAL), subfolder="vae", torch_dtype=DTYPE,
    ).to(DEVICE).eval()
    try:
        vae.enable_tiling()
        print(f"  VAE tiling enabled")
    except Exception:
        pass

    mean = torch.tensor(vae.config.latents_mean, device=DEVICE, dtype=DTYPE).view(1, -1, 1, 1, 1)
    std = torch.tensor(vae.config.latents_std, device=DEVICE, dtype=DTYPE).view(1, -1, 1, 1, 1)
    latents_unscaled = latents * std + mean
    t0 = time.time()
    frames = vae.decode(latents_unscaled, return_dict=False)[0]
    print(f"  VAE decode time: {time.time() - t0:.1f}s")
    frames = (frames.clamp(-1, 1) + 1) / 2
    frames = frames[0].permute(1, 2, 3, 0).float().cpu().numpy()
    frames = (frames * 255).astype("uint8")
    iio.imwrite(str(out_path), frames, fps=16, codec="libx264", quality=8)
    del vae
    free_gpu()
    return out_path, frames.shape


def main() -> None:
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    init_single_rank()

    text_e, neg_e = encode_prompts()
    model = load_transformer()

    latents_list = []
    for nfe in [4, 50]:
        lat = sample(model, text_e, neg_e, nfe=nfe)
        latents_list.append((nfe, lat))

    del model, text_e, neg_e
    free_gpu()

    for nfe, lat in latents_list:
        banner(f"(4) Decode 14B NFE={nfe}")
        out_path = OUT_DIR / f"fastvideo_anyflow_14b_nfe{nfe}_seed{SEED}.mp4"
        path, shape = decode_to_mp4(lat, out_path)
        print(f"  decoded {shape}")
        print(f"  saved: {path}  ({path.stat().st_size / 1e6:.2f} MB)")

    banner("DONE")


if __name__ == "__main__":
    main()
