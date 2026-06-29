# MixKit training data (QAD 5090 recipe)

The QAD 5090 models are distilled from Wan2.1-T2V-1.3B on a MixKit subset at
**480×832, 77 frames, 16 fps**. FastVideo training consumes **Parquet** shards of
precomputed VAE latents + text embeddings (no text encoder / VAE needed at train
time).

## Option A — download the preprocessed data (recommended)

The encoded dataset is published on the Hugging Face Hub, ready to train:

```bash
# from the repo root
bash examples/training/finetune/wan_t2v_1.3B/mixkit/download_mixkit_data.sh
```

This pulls [`weizhou03/HD-Mixkit-Finetune-Wan`](https://huggingface.co/datasets/weizhou03/HD-Mixkit-Finetune-Wan)
into `data/HD-Mixkit-Finetune-Wan/`:

```
data/HD-Mixkit-Finetune-Wan/
├── combined_parquet_dataset/      # training shards  -> point --data_path here
│   └── worker_0/data_chunk_*.parquet
└── validation_parquet_dataset/    # validation shards
    └── worker_0/data_chunk_0.parquet
```

Each Parquet row holds the VAE latent bytes + text-embedding bytes (plus
shape/dtype metadata), matching FastVideo's standard preprocessing output.

## Option B — build the Parquet from raw videos

To reproduce the encoding from raw videos end-to-end (e.g. to verify the
published preprocessed dataset, or to substitute your own captions), run:

```bash
# from the repo root
bash examples/training/finetune/wan_t2v_1.3B/mixkit/preprocess_mixkit_data.sh \
    [DATA_ROOT] [NUM_GPUS]
```

Defaults to `DATA_ROOT=data/HD-Mixkit-Finetune-Wan` and `NUM_GPUS=2`. The
script pulls [`FastVideo/Mixkit-Src`](https://huggingface.co/datasets/FastVideo/Mixkit-Src)
(~25 GB of full-HD clips + caption JSON), stages the layout the preprocess
workflow expects, and runs FastVideo's standard VAE + text-embed pipeline —
resized to 480×832, resampled to 16 fps, 77 frames per clip — into the same
`combined_parquet_dataset/` / `validation_parquet_dataset/` layout that Option
A's download produces. See
[`docs/training/data_preprocess.md`](https://hao-ai-lab.github.io/FastVideo/training/data_preprocess)
for the full parameter reference.

## Train (QAT finetune)

With the data in place, run the quantization-aware finetune. The 4-bit attention
path is **config-driven** — selected purely by an env var, no monkey-patching:

```bash
bash examples/training/finetune/wan_t2v_1.3B/mixkit/finetune_qat.sh
# or point at your own parquet dir / GPU count:
NUM_GPUS=4 bash .../mixkit/finetune_qat.sh data/HD-Mixkit-Finetune-Wan/combined_parquet_dataset/
```

`FASTVIDEO_ATTENTION_BACKEND=ATTN_QAT_TRAIN` routes attention through the
fake-quantized Triton kernel (straight-through estimator), so the DiT learns to
absorb FP4 attention error. This kernel is Triton, so it runs on both `sm_100`
(B200/GB200) and `sm_120` (RTX 5090).

## Train stage 2 (QAT DMD distillation to 3 steps)

Distill the QAT-finetuned generator down to **3 sampling steps**. Only the
generator is quantized (Attn-QAT); the teacher (`real_score`) and critic
(`fake_score`) stay full precision. This is enforced in the loader
(`component_loader.py`, via the `_loading_teacher_critic_model` flag), so the
same global `ATTN_QAT_TRAIN` env reaches **only** the generator — no per-model
flags or monkey-patching.

```bash
# generator init = the stage-1 finetune checkpoint
bash examples/training/finetune/wan_t2v_1.3B/mixkit/distill_dmd_qat.sh \
    data/HD-Mixkit-Finetune-Wan/combined_parquet_dataset/ \
    checkpoints/wan_t2v_qat_finetune/checkpoint-2000/transformer/diffusion_pytorch_model.safetensors
```

DMD runs a double loop (critic every step, generator every
`generator_update_interval`), and validation samples the distilled student at
3 steps — the final 4-bit-attention model.

## Inference (NVFP4 4-bit linear)

For Wan-2.1, enable the FP4 linear layers with the **`nvfp4_qat`** quantization
config (it matches Wan's `to_q/k/v/out` + `ffn` layers; the plain `NVFP4` config
is LTX2-specific and will not quantize Wan):

```python
from fastvideo import VideoGenerator
from fastvideo.layers.quantization import get_quantization_config

gen = VideoGenerator.from_pretrained(
    "Wan-AI/Wan2.1-T2V-1.3B-Diffusers", num_gpus=1,
    transformer_quant=get_quantization_config("nvfp4_qat")(),  # a config instance, not the string
    use_fsdp_inference=False,
)
gen.generate(request={"prompt": "...", "output": {"save_video": True}})
```

The loader converts the tagged linear weights to FP4 at load time
(`_maybe_convert_model_to_nvfp4`). Combine with
`FASTVIDEO_ATTENTION_BACKEND=ATTN_QAT_INFER` on an RTX 5090 (`sm_120`) for the
full 4-bit path; on other GPUs the attention falls back to Flash while the FP4
linear layers still run. `flashinfer` (and a host C++ compiler for its FP4
kernel JIT) are required.
