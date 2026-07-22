"""Wan2.1 T2V reference — the oracle, in one readable file.

This is the complete generation path in plain eager PyTorch on stock
diffusers/transformers components: encode the prompt with UMT5, run N
classifier-free-guided flow-match Euler steps on the DiT, decode with the Wan
VAE. No fastvideo2 imports, no runtime, no policies — copy this file out of
the repo and it still runs.

It serves three roles:
  * the textbook: what Wan2.1 T2V *is*, in execution order;
  * the T2 oracle: the production pipeline must reproduce this trajectory
    within the card's declared tolerance (``fastvideo2 verify --tier 2``);
  * the porting aid: the starting point an agent copies when building a
    bespoke stack.

Deliberate simplification: the sampler is first-order flow-match Euler (the
same solver the production loop uses), not the multistep UniPC that upstream
Wan defaults to. The reference defines what the production path must match;
it does not chase upstream sampler variants.

Authority ordering: the OFFICIAL implementation (Wan-Video/Wan2.1) is the
numerics ground truth. This file currently runs on diffusers components as a
convenience backend — that fidelity is *certified* against captured official
goldens (``verify --anchor``, see ``capture_official.py``), never assumed;
ports like diffusers are known to drift from official implementations in ways
that surface later as training skew. When a convention conflicts, official wins.

Usage:
    python -m fastvideo2.wan21.reference --prompt "a cat surfing" --out cat.mp4
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from typing import Any

WEIGHTS = "Wan-AI/Wan2.1-T2V-1.3B-Diffusers"
NEGATIVE = ("色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，"
            "低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，"
            "毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走")


@dataclass
class ReferenceResult:
    video: Any                      # uint8 numpy [T, H, W, C]
    latents: Any                    # final fp32 latents [1, 16, T', h, w] (cpu)
    trajectory: list = field(default_factory=list)  # per-step fp32 latents (cpu), when captured


def load_models(root: str | None = None, device: str = "cuda"):
    """Load the four components exactly as declared on the card: DiT and UMT5
    in bf16, VAE in fp32."""
    import torch
    from diffusers import AutoencoderKLWan, WanTransformer3DModel
    from transformers import AutoTokenizer, UMT5EncoderModel
    if root is None:
        from huggingface_hub import snapshot_download
        root = snapshot_download(WEIGHTS)
    tokenizer = AutoTokenizer.from_pretrained(root, subfolder="tokenizer")
    # torch_dtype (not a post-hoc .to()) so the loader's kept-in-fp32 islands
    # (Wan's time_embedder and norms inside the bf16 DiT) survive as intended.
    text_encoder = UMT5EncoderModel.from_pretrained(
        root, subfolder="text_encoder", torch_dtype=torch.bfloat16).to(device).eval()
    dit = WanTransformer3DModel.from_pretrained(
        root, subfolder="transformer", torch_dtype=torch.bfloat16).to(device).eval()
    vae = AutoencoderKLWan.from_pretrained(
        root, subfolder="vae", torch_dtype=torch.float32).to(device).eval()
    return tokenizer, text_encoder, dit, vae


def encode_prompt(tokenizer, text_encoder, text: str):
    """UMT5 encode, padded to 512 with encoder outputs past the true sequence
    length zeroed (the Wan convention)."""
    import torch
    batch = tokenizer([text], padding="max_length", max_length=512, truncation=True,
                      add_special_tokens=True, return_attention_mask=True,
                      return_tensors="pt")
    device = next(text_encoder.parameters()).device
    ids, mask = batch.input_ids.to(device), batch.attention_mask.to(device)
    with torch.no_grad():
        embeds = text_encoder(ids, mask).last_hidden_state  # [1, 512, 4096]
    embeds[:, int(mask[0].sum()):] = 0
    return embeds


def flow_sigmas(num_steps: int, shift: float) -> list[float]:
    """Shifted flow-match schedule: sigma = shift*t / (1 + (shift-1)*t) over
    t = 1 .. 0 in N steps. sigma_0 = 1 (pure noise), sigma_N = 0 (clean)."""
    ts = [(num_steps - i) / num_steps for i in range(num_steps + 1)]
    return [shift * t / (1.0 + (shift - 1.0) * t) for t in ts]


def generate(prompt: str,
             *,
             root: str | None = None,
             device: str = "cuda",
             negative_prompt: str = NEGATIVE,
             seed: int = 0,
             num_steps: int = 50,
             guidance_scale: float = 5.0,
             height: int = 480,
             width: int = 832,
             num_frames: int = 81,
             shift: float = 3.0,
             capture_trajectory: bool = False,
             models: tuple | None = None) -> ReferenceResult:
    """The whole model, in order. Deterministic given (seed, config, env)."""
    import torch
    tokenizer, text_encoder, dit, vae = models or load_models(root, device)

    # 1) Conditioning: one embedding per CFG branch.
    text = encode_prompt(tokenizer, text_encoder, prompt)
    neg = encode_prompt(tokenizer, text_encoder, negative_prompt or "")

    # 2) Initial noise in the VAE latent space: 16 channels, 4x temporal and
    #    8x spatial compression. Latents stay fp32; the DiT runs in bf16.
    shape = (1, 16, (num_frames - 1) // 4 + 1, height // 8, width // 8)
    gen = torch.Generator(device).manual_seed(seed)
    x = torch.randn(shape, generator=gen, device=device, dtype=torch.float32)

    # 3) CFG flow-match Euler: v = v_neg + g*(v_cond - v_neg); x += dsigma * v.
    sigmas = flow_sigmas(num_steps, shift)
    trajectory: list = []
    for i in range(num_steps):
        sigma, sigma_next = sigmas[i], sigmas[i + 1]
        t = torch.tensor([sigma * 1000.0], device=device, dtype=torch.float32)
        x_in = x.to(torch.bfloat16)
        with torch.no_grad():
            v_cond = dit(hidden_states=x_in, timestep=t, encoder_hidden_states=text,
                         return_dict=False)[0].to(torch.float32)
            if guidance_scale == 1.0:
                v = v_cond
            else:
                v_neg = dit(hidden_states=x_in, timestep=t, encoder_hidden_states=neg,
                            return_dict=False)[0].to(torch.float32)
                v = v_neg + guidance_scale * (v_cond - v_neg)
        x = x + (sigma_next - sigma) * v
        if capture_trajectory:
            trajectory.append(x.detach().to("cpu", copy=True))

    # 4) Decode: denormalize by the VAE's per-channel latent stats, then decode.
    mean = torch.tensor(vae.config.latents_mean, device=device,
                        dtype=torch.float32).view(1, -1, 1, 1, 1)
    std = torch.tensor(vae.config.latents_std, device=device,
                       dtype=torch.float32).view(1, -1, 1, 1, 1)
    with torch.no_grad():
        video = vae.decode((x * std + mean).to(torch.float32), return_dict=False)[0]
    video = (video / 2 + 0.5).clamp(0, 1)
    frames = (video[0].permute(1, 2, 3, 0) * 255).round().to(torch.uint8).cpu().numpy()

    return ReferenceResult(video=frames, latents=x.detach().cpu(), trajectory=trajectory)


def save_video(frames, path: str, fps: int = 16) -> str:
    import imageio.v2 as imageio
    imageio.mimsave(path, list(frames), fps=fps, format="mp4")
    return path


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--prompt", required=True)
    p.add_argument("--out", default="reference.mp4")
    p.add_argument("--root", default=None, help="local checkpoint dir (else HF cache)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--steps", type=int, default=50)
    p.add_argument("--guidance", type=float, default=5.0)
    p.add_argument("--height", type=int, default=480)
    p.add_argument("--width", type=int, default=832)
    p.add_argument("--frames", type=int, default=81)
    p.add_argument("--shift", type=float, default=3.0)
    args = p.parse_args()
    res = generate(args.prompt, root=args.root, seed=args.seed, num_steps=args.steps,
                   guidance_scale=args.guidance, height=args.height, width=args.width,
                   num_frames=args.frames, shift=args.shift)
    print(f"video {res.video.shape} -> {save_video(res.video, args.out)}")


if __name__ == "__main__":
    main()
