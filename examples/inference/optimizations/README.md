# Optimization Examples

## Wan 2.1 QAT Attention 14B Inference

Use these files for Wan 2.1 14B inference with the `ATTN_QAT_INFER` backend:

- `examples/inference/optimizations/download_14B_qat.sh`
- `examples/inference/optimizations/attn_qat_inference_example.py`

### 1. Download the 14B QAT checkpoint

The helper script downloads the QAT safetensors from
`FastVideo/14B_qat_400` into `checkpoints/14B_qat_400` by default.

Prerequisites:

- `huggingface_hub` installed, for example: `uv pip install huggingface_hub`
- access to the model repo if it is private or gated: `huggingface-cli login`

Run:

```bash
bash examples/inference/optimizations/download_14B_qat.sh
```

To download into a custom directory, pass it as the first argument:

```bash
bash examples/inference/optimizations/download_14B_qat.sh /path/to/14B_qat_400
```

### 2. Edit the inference example for Wan 2.1 14B

Open `examples/inference/optimizations/attn_qat_inference_example.py` and
update these two values:

1. Change the base model from `Wan-AI/Wan2.1-T2V-1.3B-Diffusers` to
   `Wan-AI/Wan2.1-T2V-14B-Diffusers`.
2. Replace the placeholder
   `init_weights_from_safetensors="safetensors_path"` with the directory that
   contains the downloaded `.safetensors` files.

Example:

```python
generator = VideoGenerator.from_pretrained(
    "Wan-AI/Wan2.1-T2V-14B-Diffusers",
    num_gpus=1,
    use_fsdp_inference=True,
    dit_cpu_offload=False,
    vae_cpu_offload=False,
    text_encoder_cpu_offload=True,
    pin_cpu_memory=False,
    init_weights_from_safetensors="checkpoints/14B_qat_400",
)
```

The script already sets:

```python
os.environ["FASTVIDEO_ATTENTION_BACKEND"] = "ATTN_QAT_INFER"
```

### 3. Run the example

```bash
python examples/inference/optimizations/attn_qat_inference_example.py
```

The generated videos are written to `video_samples/` by default.

### Notes

- `ATTN_QAT_INFER` requires the in-repo `fastvideo-kernel` build to expose the
  `attn_qat_infer` package.
- If you have not built the kernel yet, run `cd fastvideo-kernel && ./build.sh`
  first.
- If you keep the example on the `1.3B` base model while loading the 14B QAT
  weights, the model/config will not match.
