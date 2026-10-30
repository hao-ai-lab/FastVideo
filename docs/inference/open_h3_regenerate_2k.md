# Open-H3-Regenerate-2K

`OpenH3Regenerate2KPipeline` regenerates a 768P [MiniMax-H3](https://huggingface.co/MiniMaxAI/MiniMax-H3) draft video
(1344x768, 121 frames, 24 fps) at **2560x1440** with three refine steps. It is video to video: the draft is the input,
and the output is a silent 2K MP4. FastVideo ports NVlabs'
[Open-H3-Regenerate-2K](https://huggingface.co/Efficient-Large-Model/Open-H3-Regenerate-2K)
([project page](https://nvlabs.github.io/Sana/Open-H3-Regenerate-2K/),
[reference code](https://github.com/NVlabs/Sana/tree/sol-engine/models/Open-H3-Regenerate-2K)); the dense BF16
`standard_fwd` path of that repo is the parity reference. The accelerated `sol_engine_fwd` path (Sol-Attn, MXFP8, CUDA
graphs) and the fast 24-layer VAE decoder are not ported.

```text
draft video -> H3 VAE encode -> latent upscaler (2x, 3D conv) -> 3 Euler refine steps -> H3 VAE decode -> 2560x1440
                                                 H3 DiT + merged LoRA + Sharp curve AdaLN branches
```

## What runs

| Step | Details |
|---|---|
| Draft encode | `ffmpeg` resamples to 24 fps, cover-scales (bilinear) and centre-crops to 1344x768 (`draft_size: 768p`, or 1024x576 for `576p`), holds the last frame to 121 frames plus 3 copies. The FP32 H3 VAE posterior is sampled under seed 42. |
| Latent upscale | The author's 3D-conv upscaler (`fastvideo/models/upsamplers/minimax_h3_latent_upscaler.py`), BF16, to the 160x90 latent grid. |
| Refine | Euler on sigmas `[0.909375, 0.857143, 0.666667, 0]`, CFG 1, model time `1 - sigma`, initial state `(1 - 0.909375) * upscaled + 0.909375 * noise`. `num_inference_steps: 2` uses `[0.7, 0.494118, 0]` (softer, about 30% faster). |
| Transformer | The base MiniMax-H3 DiT (the same module as the other H3 pipelines) with the released merged LoRA (FP32, rank 576 on 312 linears, scale 1) and 50 LMS Sharp curve AdaLN branches. |
| Decode | The released H3 VAE; the output is trimmed to 121 frames. |

The merged LoRA loads through the standard H3 community-LoRA path as a constructor-time adapter: the pipeline sets
`lora_path` to `<model>/lora/open_h3_regenerate_2k_lora.safetensors` when you do not pass one. The Sharp branches take an
8-D timestep curve (`lora/adaln_t_table.bin`, 1025x8) instead of the layer input, so they cannot be folded into weights;
`sharp_curve.py` adds them to the AdaLN projection outputs with forward hooks.

**Audio.** The reference feeds an encoded-silence audio stream to the DiT as conditioning and writes the video with no
audio track (`-an`); it neither regenerates nor passes through the draft's audio. FastVideo does the same: the draft's
soundtrack is ignored and the output MP4 has no audio.

## Weights

The release repo keeps the unmodified MiniMax-H3 components (a copy of `MiniMaxAI/MiniMax-H3` at `6818f6c`) under
`base_model/` next to `lora/` and `latent_upsampler/`. Point `model_path` at the Hub id or at a local copy of that layout.
`adaln_t_table.bin` is base-derived and is not redistributed in the release repo: put the 32,800-byte file (SHA-256
`c02a6c11888297688c1e6278185ea1f947023acfc69f9003bbcdcec9a229a8e7`) at `lora/adaln_t_table.bin`. The reference
`download.py` fetches it by byte range from `Comfy-Org/MiniMax-H3` at revision
`e5eb578a89295337b8ff433a035929ce0279e0b6` (tensor `adaln_t_table` of
`diffusion_models/minimax_h3_ref2va_pruned_int8_convrot.safetensors`) and verifies the hash.

A GB200/B200-class GPU is needed (about 150 GB with the DiT, the 2K activations and the VAE).

## Usage

```bash
python examples/inference/basic/basic_open_h3_regenerate_2k.py \
    --model-path Efficient-Large-Model/Open-H3-Regenerate-2K \
    --video draft_768p.mp4 --prompt "A red fox trots through fresh snow" --output outputs/regen2k
```

The prompt conditions the refine steps (use the draft's own prompt). Drafts from other video models are better encoded at
576P (`--draft-size 576p`). `--steps 2` runs the two-step variant. The output is always 2560x1440; the first 121 frames of
the draft are used.

## Serving

```bash
fastvideo serve --config examples/serving/openai_open_h3_regenerate_2k.yaml
```

`examples/serving/openai_open_h3_regenerate_2k.yaml` loads the release on one GPU, sets `media_roots`, and pins the 2K
defaults. The draft arrives as the typed `video_reference` of `POST /v1/videos/sync` (the response body is the MP4) or of
`POST /v1/videos` (an async job; poll `GET /v1/videos/{id}` and download `GET /v1/videos/{id}/content`):

```bash
curl -sS -X POST http://localhost:8000/v1/videos/sync \
  -H 'Content-Type: application/json' \
  -d '{
        "prompt": "A red fox trots through fresh snow",
        "video_reference": {"video_url": "https://example.com/draft_768p.mp4"},
        "seed": 1766580932
      }' \
  -o regen_2k.mp4 -D -
```

The response headers carry `X-Request-Id`, `X-Model`, `X-Inference-Time-S` and `X-Stage-Durations`. Rules:

- Exactly one `video_reference` (or the legacy `video_path` / `video_url`) and no image or audio references.
- `http(s)` URLs must resolve to public addresses unless `server.allow_private_media_urls` is set. A local path is accepted
  only inside a directory listed in `server.media_roots`. `data:` URLs are refused for video references; this policy is
  unchanged.
- `size`, `width` and `height` may be omitted or `2560x1440`; `num_frames` may be omitted or 124 (the frame count of the
  VAE grid; the video has 121 frames); `fps` must be 24. `seconds` is not accepted.
- `num_inference_steps` is 3 (default) or 2; `seed` sets the refine noise.
- Serve with `attention_backend: TORCH_SDPA`: the refine stage selects cuDNN attention, which at about 133k tokens is
  about 3x faster than PyTorch's default flash dispatch. The `FLASH_ATTN` backend bypasses that selection.

## Refining on a FastH3 DiT

`refine_transformer_dir` swaps the refine DiT for another MiniMax-H3 transformer while the rest of the release (VAEs,
encoders, upscaler, merged LoRA, Sharp branches) stays put. Point it at a transformer component directory, or at a
checkpoint whose `transformer/` subdirectory is one, such as a FastH3 export:

```python
PipelineSelection(components=ComponentConfig(),
                  experimental={"refine_transformer_dir": "/models/FastH3-QAD-v2"})
```

The released LoRA and Sharp branches are base-H3-derived adapters, so they keep loading over the chosen DiT. A packed
NVFP4 FastH3 export needs its quantization profile
(`EngineConfig(quantization=QuantizationConfig(transformer_quant="NVFP4", layer_profile="h3_dit_vsa"))`) and refuses a
dense attention backend, because its `to_gate_compress` linears are built only for VSA. Do not set
`FASTVIDEO_H3_ADALN_TABLE` for a different sigma grid than the one a checkpoint's `adaln_tables.pt` covers; the refine
grid is not FastH3's 8-step schedule, so load the real AdaLN projections.

This swap does not reproduce the release's refine. On the fox draft, FastH3 V2 BF16 walking its own DMD rungs from 500, the
best adapter subset reached 26 dB PSNR and 0.84 SSIM against the base-H3 refine (the base reproduces itself at 45 dB), with no
faster forwards at this sequence length. Draft with FastH3 and refine with the base DiT instead.

`draft_latent_path` skips the draft video: a `.pt` holding the normalized native H3 draft latent `[C, T, h, w]`
(float32) goes straight to the upscaler, so a draft generated by a FastH3 t2v run never needs decoding to pixels and
re-encoding. Starting from `vae.encode_pixels(...).latent_dist`, sample it and apply `vae.normalize_latents` before
saving, as the draft encoder does. A batched latent is accepted only with a leading dimension of 1.

```bash
python examples/inference/basic/basic_open_h3_regenerate_2k.py --video draft.mp4 --prompt "..." \
    --refine-transformer /models/FastH3-8-Step-V2 --attention-backend VIDEO_SPARSE_ATTN_H3
```

## Regenerating a FastH3 draft

Only the draft model changes: a FastH3 checkpoint drafts at 768P (1344x768, 124 frames) and the regen stack stays as
released (base MiniMax-H3, merged LoRA, Sharp branches, three Euler steps). All H3-family drafts share the H3 VAE's latent
space (`[24, 37, 48, 84]` for a 124-frame 768P clip), so a FastH3 draft latent goes straight to the upscaler.

```bash
python examples/inference/basic/basic_fasth3_to_regen2k.py \
    --draft-model FastVideo/FastVideo-FastH3-8-Step-V2 --prompt "A red fox trots through fresh snow" \
    --output outputs/fasth3_regen2k            # add --draft-quant h3_dit_vsa for an NVFP4 draft
```

Two handoffs are available:

- **Video:** decode the draft to an MP4 and pass it as `video_path`. The regen decodes it, resamples and VAE-encodes it (about
  11 s).
- **Latent:** pass the draft's normalized native latent `[C, T, h, w]` as `draft_latent_path`, or per request as a `.pt`
  `video_path` (through the OpenAI server it must be a local path under `server.media_roots`; not exercised in this run). The encode drops to 0.6 s and the
  draft is never decoded to pixels.

The two give different 2K videos (24-27 dB PSNR against each other). On smooth content the MP4 route is much sharper (a
portrait: Laplacian variance 180-244 against 84-106 for the latent route). Writing the same draft as a lossless MP4 gives 85,
close to the latent route, so the extra energy comes from the H.264 artifacts of the draft MP4, which the refine turns into
texture. The latent route also leaves PSNR against the draft within 0.8 dB of the MP4 route. Prefer it: it is faster and does
not amplify codec noise.

Measured on one GB200 with nothing else on the node (regen at `TORCH_SDPA`, seed 1766580932, warm requests; fox, portrait
and harbor prompts, 5.2 s clips):

| Draft model | Draft | Regen: MP4 / latent / latent + NVFP4 decoder | Text to 2K (latent handoff) |
|---|---:|---:|---:|
| FastH3 QAD V2.1 NVFP4 | 36 s | 106 s / 96 s / 79 s | 132 s (115 s) |
| FastH3 Trim NVFP4 | 34 s | 106 s / 96 s / 79 s | 130 s (113 s) |
| FastH3 V2 BF16 | 35 s | 106 s / 96 s / 79 s | 131 s (114 s) |
| base H3 50-step | 316 s | 106 s / 96 s / 79 s | 412 s (395 s) |

The regen costs the same for every draft: about 70 s of refine (3 forwards at 133k tokens) and 24 s of decode. Setting
`h3_vae_nvfp4_decoder` to `quality-36L/decoder-nvfp4.safetensors` from `FastVideo/FastH3-NVFP4-VAE-Decoder` cuts the decode
to 7 s (82 s on the first call of a process) with no visible metric change. Against the draft, the 2K output keeps the scene
(PSNR 19-22 dB at 768P: the refine regenerates texture, so pixel agreement stays low for base and FastH3 drafts alike) and
raises Laplacian variance 1.7-3.3x with a 1.3-1.5x flicker ratio for every draft model. FastH3 drafts are sharper than base
drafts before the regen, so their absolute PSNR against the draft is about 2 dB lower.

## Parity with the reference

Measured on one GB200 (BF16, `TORCH_SDPA` with cuDNN attention) against the reference `standard_fwd` with the released
VAE, same draft, seed (1766580932), prompt and VAE posterior seed (42). The reference's noise is a CUDA generator draw
that FastVideo reproduces (the same `torch.Generator(device="cuda")` seed), so no injected noise was needed.

| Stage / quantity | Fox draft (FastVideo H3 t2v, 768P) | Dance draft (576P, project page) |
|---|---|---|
| Draft latent, upscaled latent, DiT input | bit-identical | bit-identical |
| One DiT forward (LoRA + Sharp), relative L2 | 1.1e-2 | 2.6e-2 |
| Refined latent, relative L2 | 1.3e-2 | 3.7e-2 |
| Decoded video PSNR / SSIM (21 frames) | 45.2 dB / 0.993 | 42.1 dB / 0.992 |

The differences come from text-encoder rounding (relative L2 1e-3 on the prompt embeddings) and BF16 kernel choices;
the reference reports the same order (1.4% relative L2 on the refined latent between its own runs).

Warm timings, one GB200 (seconds):

| | Upscale | Refine (3 steps) | VAE decode | Total |
|---|---:|---:|---:|---:|
| Reference `standard_fwd` | 0.87 | 71.6 | 15.0 | 87.5 |
| FastVideo (end to end, incl. draft encode 5-10 s and decode to host) | 5.9-10.7 (encode + upscale + silence) | 69.5 | 25.0 | 107.5 |

The first request builds nothing extra but loads weights (about 70 s once). The decode stage is slower than the
reference because it also copies 121 frames to host memory and the MP4 encode is outside both columns.

## Licences

The release is not covered by one licence; its `README.md` says so, and the runtime code's licence does not replace the
model licences. FastVideo only loads these weights and does not redistribute them.

- The refiner LoRA, the fast H3 VAE decoder and `base_model/` (a copy of `MiniMaxAI/MiniMax-H3` at `6818f6c`: DiT, VAEs,
  text encoder) follow the
  [MiniMax H3 Community License](https://huggingface.co/MiniMaxAI/MiniMax-H3/blob/6818f6c32d12b210915e44ad56a4228c2608f160/LICENSE),
  including its territory, use and redistribution terms (`base_model/LICENSE`, `NOTICE`). The Qwen3-VL text encoder's
  Apache-2.0 text ships as `LICENSE-Qwen.txt`.
- The latent upscaler follows its own licence:
  [LBH-123-AI/Comfyui_Minimax_h3_latent_Upscaler](https://github.com/LBH-123-AI/Comfyui_Minimax_h3_latent_Upscaler)
  (MIT, per the reference code) and its [weights](https://huggingface.co/LBH-123-AI/Minimax_h3_latent_Upscaler)
  (`latent_upsampler/LICENSE`, Apache-2.0). `minimax_h3_latent_upscaler.py` is a port of that node's network.
- The merged LoRA includes [LightX2V FL2VA Turbo](https://huggingface.co/lightx2v/Minimax-h3-Turbo) and
  [LMS Sharp](https://huggingface.co/Alissonerdx/Minimax-H3-ComfyUI), which keep their own licences; the Sharp
  `adaln_t_table` is read from [Comfy-Org/MiniMax-H3](https://huggingface.co/Comfy-Org/MiniMax-H3).
