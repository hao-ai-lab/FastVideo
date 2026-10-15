# PAN2 Text/Image to Video

PAN2 is a 29B mixture-of-transformers video DiT with separate video and text streams, a Qwen3.5 text encoder and the
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
| Steps | 50, uniform sigmas from 1 to 0 with flow shift 7 |
| Guidance | 3.0, classifier-free guidance with an empty negative prompt |

PAN2 is trained on the structured captions its prompt rewriter produces, serialized as one-line JSON. Short raw
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

## Differences from the vLLM-Omni integration

- The optional prompt and video guardrails (`pan2-guardrail`) are not included.
- The Cache-DiT `quality` policy and CFG-parallel execution are vLLM-Omni serving features and are not ported.
- The text encoder is a FastVideo-native Qwen3.5 port. Its Gated DeltaNet layers use a PyTorch chunked delta rule
  instead of the `fla` kernels.
