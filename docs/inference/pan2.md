# PAN 2 Text/Image to Video

PAN 2 is a 29B mixture-of-transformers video DiT with separate video and text streams, a Qwen3.5 text encoder and the
Wan2.2 VAE. One pipeline serves text-to-video and image-to-video: pass `image_path` to condition on a first frame and
leave it out for text only.

## Model

| Hub repo | Pipeline | Workloads | Example |
|---|---|---|---|
| `IFM/PAN2` | `PAN2ModularPipeline` | T2V, I2V | [`basic_pan2.py`](https://github.com/hao-ai-lab/FastVideo/blob/main/examples/inference/basic/basic_pan2.py) |

PAN2 checkpoints use the Modular Diffusers layout (`modular_model_index.json`). Any repo id or local directory whose
name contains `pan2` (or whose manifest declares `PAN2ModularPipeline`) resolves to this pipeline. The defaults come
from the `pan2_t2v` and `pan2_i2v` presets:

| Setting | Default |
|---|---|
| Resolution | 704x1248 (height and width must be multiples of 32) |
| Frames | 313, 13 s at 24 fps (`num_frames - 1` must be a multiple of 4) |
| Steps | 50, uniform sigmas from 1 to 0 with the checkpoint's flow shift (9 for `IFM/PAN2-TEST`, the reference sampler's value) |
| Guidance | 2.5, classifier-free guidance, the reference launcher's default (3 before 2026-10-23). Image-to-video keeps the first latent frame out of guidance (`i2v_skip_first_latent_cfg`, on by default, as in the reference sampler) |
| Negative prompt | None unless the request carries one. The reference sampler always sends the release's structured negative caption (`negative_prompt.json` of the PAN2 repository) with its three `time_range_sec` values set to the clip length, `0 to 20` for 481 frames at 24 fps |

PAN 2 is trained on the structured captions its prompt rewriter produces, serialized as one-line JSON. Short raw
prompts work, with weaker results.

## Usage

```bash
python examples/inference/basic/basic_pan2.py
```

```python
from fastvideo import VideoGenerator

generator = VideoGenerator.from_pretrained(
    "IFM/PAN2",
    num_gpus=1,
    dit_cpu_offload=False,
    dit_layerwise_offload=False,
    text_encoder_cpu_offload=True,
)
generator.generate_video(
    "A red fox trots through fresh snow in a pine forest at dawn.",
    image_path=None,  # or the path of a first frame for image-to-video
    output_path="video_samples_pan2",
)
```

For image-to-video, the first frame is scaled to cover the requested resolution and center cropped, then encoded by the
VAE into the first latent frame of the conditioning input.

## Parallelism

With `num_gpus > 1`, FastVideo runs Ulysses sequence parallelism: the video tokens are sharded across ranks and the
text tokens are replicated. The number of GPUs must divide the 32 attention heads. Sequence-parallel padding of the
video tokens is trimmed before attention, so no attention mask is needed.

Padding is always excluded, without an opt-in setting. At 704x1248 with 313 frames, the 67,782 video tokens need
two padding tokens at SP=4. They are removed before joining the text tokens in every attention block. This provides
the same exclusion as vLLM-Omni's unconditional padding mask in
[revision `e0c4c731`](https://github.com/vllm-project/vllm-omni/commit/e0c4c73158a5e4f44eda8eaa8879f7fd9cedbd74).
The trimmed sequence can use either FlashAttention or Torch SDPA.

```bash
PAN2_NUM_GPUS=4 python examples/inference/basic/basic_pan2.py
```

## Performance switches

PAN2 sampling turns these on by default (`PAN2PipelineConfig`); pass any of them as `False` to
`VideoGenerator.from_pretrained` to turn it off:

| Switch | What it does |
|---|---|
| `vae_decode_kernel` | Decodes with fastvideo-kernel's Wan2.2 decoder on B200 / GB200 / B300 (704x1248, batch 1): bf16 with fp32 accumulation, 7x faster than the fp32 decode on a B300 at PSNR 61-63 dB and SSIM 0.998 against it, at a peak of about 20 GiB for a 5 s video. It builds on first use (minutes) into `WAN22_VAE_DECODE_BUILD_DIR`, against fastvideo-kernel's CUTLASS submodule (`git submodule update --init fastvideo-kernel/include/cutlass`, or set `CUTLASS_DIR`). Other GPUs and sizes keep the fp32 decode. |
| `vae_decode_kernel_parallel` | With 2, 4 or 8 GPUs, splits that decode over them instead of decoding the whole video on each: a 20 s video takes 3.0 s on one B300, 1.55 s on two, 0.79 s on four and 0.47 s on eight. |
| `dit_fused_kernels` | Runs the DiT's fused Triton kernels for Q/K RMSNorm + RoPE and LayerNorm + modulation. Measured with a distilled PAN2 model on eight B300s, they cut the denoising time by 21%. The rounding differs from the unfused path; record by record, fused and unfused videos were equally close to the 50-step reference. |

On GPUs with room for every component (B200, B300), keep the text encoder resident as well:
`text_encoder_cpu_offload=False`.

## Memory and offloading

The example keeps the DiT on the GPUs and offloads the text encoder between requests. Sequence parallelism shards
video tokens; each rank still holds the DiT weights. Increasing `num_gpus` alone does not shard those weights.
For devices that cannot hold the DiT, see [CPU offloading](offloading.md) for `dit_cpu_offload` and
`dit_layerwise_offload`. Hardware recipes and timings reported by vLLM-Omni use its own offload implementation;
they have not been validated for FastVideo.

The full `IFM/PAN2` checkpoint is not public yet. Local parity tests use the public random-weight
`wuqing157/tiny-pan2-modular-pipe` checkpoint; it verifies the integration and produces noise rather than useful video.

## Differences from the vLLM-Omni integration

- The optional prompt and video guardrails (`pan2-guardrail`) are not included.
- The Cache-DiT `quality` policy and CFG-parallel execution are vLLM-Omni serving features and are not ported.
- The text encoder is a FastVideo-native Qwen3.5 port. Its Gated DeltaNet layers use a PyTorch chunked delta rule
  instead of the `fla` kernels.
