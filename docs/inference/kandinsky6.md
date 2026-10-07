# Kandinsky 6 Text/Image to Video with Audio (T2IVA)

`Kandinsky6TI2VAPipeline` generates a video and a synchronized audio track from a text prompt, optionally conditioned
on an image: one pipeline serves both, so pass `image_path` to condition on an image and leave it out for text only.
The audio is decoded by the checkpoint's audio VAE (mel decoder plus vocoder in one component) and muxed into the saved
mp4 automatically. To upscale a generated clip, see [Kandinsky 6 Video SR](kandinsky6_sr.md).

## Models

Kandinsky 6 comes in two sizes, Pro (30.1B-parameter DiT) and Lite (3.2B), each with a base and a pi-Flow distilled
checkpoint. All four are official Diffusers repos, loaded directly through their `model_index.json`; Lite shares Pro's
architecture, text encoders, VAEs and schedulers, with a narrower and shallower DiT.

| Size | Variant | Hub repo | Scheduler | Steps | Guidance | Example |
|---|---|---|---|---|---|---|
| Pro | T2IVA | `kandinskylab/Kandinsky-6.0-Pro-5s-Diffusers` | `FlowMatchEulerDiscreteScheduler` (shift 5.0) | 50 | 5.0 | [`basic_kandinsky6_ti2va.py`](https://github.com/hao-ai-lab/FastVideo/blob/main/examples/inference/basic/basic_kandinsky6_ti2va.py) |
| Pro | T2IVA distilled | `kandinskylab/Kandinsky-6.0-Pro-distill-5s-Diffusers` | `PiflowScheduler` (`n_grid` 10, shift 5.0) | 10 | 1.0 | [`basic_kandinsky6_ti2va.py`](https://github.com/hao-ai-lab/FastVideo/blob/main/examples/inference/basic/basic_kandinsky6_ti2va.py) |
| Lite | T2IVA | `kandinskylab/Kandinsky-6.0-Lite-5s-Diffusers` | `FlowMatchEulerDiscreteScheduler` (shift 5.0) | 50 | 5.0 | [`basic_kandinsky6_ti2va.py`](https://github.com/hao-ai-lab/FastVideo/blob/main/examples/inference/basic/basic_kandinsky6_ti2va.py) |
| Lite | T2IVA distilled | `kandinskylab/Kandinsky-6.0-Lite-distill-5s-Diffusers` | `PiflowScheduler` (`n_grid` 10, shift 5.0) | 10 | 1.0 | [`basic_kandinsky6_ti2va.py`](https://github.com/hao-ai-lab/FastVideo/blob/main/examples/inference/basic/basic_kandinsky6_ti2va.py) |

The steps and guidance columns are the defaults of the preset the registry selects for each repo id. Everything else is
shared: 512x768, 121 frames (5 s at 24 fps) and the Diffusers default negative prompt (only used when
`guidance_scale > 1`). The distilled preset is named `kandinsky6_ti2va_distilled`.

## Usage

```bash
python examples/inference/basic/basic_kandinsky6_ti2va.py
KANDINSKY6_MODEL_PATH=kandinskylab/Kandinsky-6.0-Pro-distill-5s-Diffusers \
  python examples/inference/basic/basic_kandinsky6_ti2va.py
```

Set `IMAGE_PATH` in the script to condition on an image.

```python
from fastvideo import VideoGenerator

generator = VideoGenerator.from_pretrained(
    "kandinskylab/Kandinsky-6.0-Pro-5s-Diffusers",
    num_gpus=1,
    dit_cpu_offload=False,
    text_encoder_cpu_offload=True,
)
generator.generate_video(
    "cinematic shot: a giant stone samurai on a stormy cliff above a neon city opens glowing golden eyes and "
    "raises a katana. Blue lightning strikes the blade, creating a massive shockwave through the clouds. The "
    "camera rapidly pulls back from a low angle. Photorealistic, epic scale, dark blue and gold lighting, rain, "
    "sparks, volumetric lightning, blockbuster quality. Audio: heavy rain, deep thunder, metallic sword hum, "
    "rising brass and choir, electrical crackle, perfectly synchronized lightning impact, sub-bass shockwave. "
    "No dialogue, text, or logos.",
    image_path=None,  # or the path of a conditioning image
    output_path="video_samples_kandinsky6_ti2va",
    height=512,
    width=768,
    num_frames=121,
)
```

A local copy of a repo works the same way. A local directory is treated as the distilled variant only when its name is
a Kandinsky-6 name containing `distill` (for example `Kandinsky-6.0-Pro-distill-5s-Diffusers`); any other directory name
selects the base preset (see below).

## Distilled (pi-Flow) checkpoint

- The distilled repo replaces the flow-matching scheduler with `PiflowScheduler` (`n_grid` 10, `eps` 1e-6,
  `final_step_size_scale` 0.5, `num_policy_substeps` 128). Its DiT emits `n_grid` predictions per latent channel
  (`out_visual_dim` 160 = 16 x 10, `out_audio_dim` 400 = 40 x 10).
- pi-Flow runs without classifier-free guidance: `guidance_scale` must be exactly `1.0`. The official Diffusers
  pipeline rejects any other guidance for a `PiflowScheduler` too; FastVideo raises a `ValueError` naming the
  required value. `num_inference_steps` is not constrained by the checkpoint -- `PiflowScheduler.set_timesteps`
  accepts any step count and ignores the scheduler's `nfe`; `10` is only the value the distilled checkpoint was
  trained for and the `kandinsky6_ti2va_distilled` preset's default.
- The `kandinsky6_ti2va_distilled` preset (10 steps, guidance 1.0) is the default for the distilled repo id and for
  local directories named like it. Set `KANDINSKY6_MODEL_PATH` to either one when running the shared
  `basic_kandinsky6_ti2va.py` example. A distilled copy under another directory name selects the base preset, so
  retain `Kandinsky-6.0-Pro-distill-5s-Diffusers` as the final directory name.
- The policy values can be overridden on `Kandinsky6TI2VAConfig` (`piflow_eps`, `piflow_final_step_size_scale`,
  `piflow_num_policy_substeps`); `None` keeps the values from `scheduler_config.json`.

## Differences from the Diffusers pipeline

A few Diffusers pipeline options are not ported, and are surfaced here instead of as a knob that would silently do
nothing:

- `sample_audio=False` (video-only, no audio stream) is not exposed; FastVideo's DiT raises `NotImplementedError` for
  a partial-modality call instead of denoising video alone.
- `expand_prompts` (the Qwen prompt-beautifier pass) is not ported; FastVideo's `PromptEnhancerConfig` is a separate,
  external (Cerebras/Groq streaming) feature, not this pipeline's built-in expansion.
- Of the Diffusers reference's `visual_cond_scheme` values, only `tail_cond_first_frame` (append one clean reference
  frame to the end of the sequence) is implemented; `pretrain` and plain `i2v` are not.
- Precomputed `prompt_embeds`/`negative_prompt_embeds` are not accepted; every call encodes its own prompt text.
- MagCache is not ported: a `magcache` block in a checkpoint's `transformer/config.json` is parsed and dropped
  by `update_model_arch`, not read automatically or exposed as an opt-in cache config.
- RNG differs: FastVideo draws video then audio noise from one per-request CPU generator seeded by `seed` (default
  1024); Diffusers seeds a device generator from a value drawn out of `generator` (audio uses `seed+1`). The same
  numeric seed produces different noise on the two stacks -- pass `latents`/`audio_latents` directly for bit-level
  comparisons.
- Qwen prompt tokens are unpadded and carry no attention mask (Diffusers pads to a fixed length and masks the
  padding in text self/cross-attention). Mathematically equivalent for a single prompt (measured 2e-7 relative
  difference), but FastVideo has no attention-mask plumbing, so a hand-built batch of unequal-length prompts is not
  supported.
- Attention is dense (`LocalAttention`, flash/SDPA); NABLA sparse attention is wired but unverified against a real
  NABLA-flagged checkpoint. This matches the Diffusers pipeline itself, which never enables NABLA either.
- VAE tiling is on by default (`vae_tiling=True`); Diffusers decodes untiled.

## Memory

The Pro DiT has 30.1B parameters, about 60 GB in bf16 (`dit_precision` defaults to `bf16`); the Lite DiT has 3.2B,
about 6.4 GB. Both use the same Qwen2.5-VL text encoder, which adds 16.6 GB. FastVideo enables `dit_cpu_offload` by default; the examples turn it off (`dit_cpu_offload=False`)
to keep the DiT resident on the GPU and offload the text encoder instead (`text_encoder_cpu_offload=True`). See
[Offloading](offloading.md) for the memory knobs.
