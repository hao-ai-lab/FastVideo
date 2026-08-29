# LoRA Extraction and Merging

Tools for extracting and merging LoRA adapters for FastVideo models.

## Extract LoRA Adapter

The default remains exact CPU SVD:

```bash
python extract_lora.py \
  --base Wan-AI/Wan2.2-TI2V-5B-Diffusers \
  --ft FastVideo/FastWan2.2-TI2V-5B-FullAttn-Diffusers \
  --out adapter_r32.safetensors \
  --rank 32 \
  --exact-tensor-pattern '^condition_embedder\.' \
  --exact-tensor-pattern '^proj_out\.weight$'
```

The extractor is runtime-agnostic by default: it cannot infer from checkpoint tensors whether a runtime wraps a
particular matrix as a LoRA layer. When extracting a full fine-tune, select matrices unsupported by the target runtime
with `--exact-tensor-pattern`; their changes remain exact `.diff` tensors rather than being discarded. The Wan patterns
above cover its excluded condition embedders and its unwrapped output projection.

For large transformers, stream their indexed safetensors and factorize on a GPU:

```bash
python extract_lora.py \
  --base MiniMaxAI/MiniMax-H3 \
  --ft FastVideo/FastVideo-FastH3-8-step-Preview-v1-VSA-DataFree \
  --out adapter_r64.safetensors \
  --rank 64 \
  --load-mode indexed \
  --device cuda:0 \
  --svd-method randomized \
  --randomized-q 320 \
  --niter 4 \
  --factor-dtype float16 \
  --dense-dtype float32 \
  --replacement-dtype source \
  --exact-tensor-pattern '^audio_proj_(in|out)\\.weight$' \
  --exact-tensor-pattern '^context_embedder\\.weight$' \
  --exact-tensor-pattern '^proj_(in|out)\\.weight$' \
  --exact-tensor-pattern '^time_embedder\\.'
```

`q=320, niter=4` retained 99.9355% of the energy captured by exact rank-64 SVD in a 362-matrix MiniMax-H3 comparison. Exact CPU SVD is still the default; randomized SVD must be requested explicitly.

Important options:

- `--base`, `--ft`: Hugging Face model IDs or local paths.
- `--rank`: requested LoRA rank.
- `--load-mode indexed`: download/read only `transformer/*` and stream one tensor pair at a time.
- `--device`: factorization device, such as `cpu` or `cuda:0`.
- `--svd-method`: `exact` or `randomized`.
- `--randomized-q`, `--niter`, `--seed`: randomized SVD accuracy and reproducibility.
- `--factor-dtype`: storage dtype for `lora_A` and `lora_B`.
- `--dense-dtype`: storage dtype for exact `.diff`, `.diff_b`, and `.diff_param` payloads.
- `--replacement-dtype`: storage dtype for fine-tuned-only `.set_weight` and `.set_param` parameters.
- `--exact-tensor-pattern`: repeatable regex selecting matrices to retain as exact dense deltas.
- `--work-dir`, `--resume`: resume a partially completed streaming extraction.

Fine-tuned parameters that cannot or should not be factorized are retained automatically:

- a changed base weight becomes `.diff`;
- a changed base bias becomes `.diff_b`;
- another changed parameter, such as `scale_shift_table`, becomes `.diff_param`;
- a fine-tuned-only weight, such as a VSA compression gate, becomes `.set_weight`;
- another fine-tuned-only parameter becomes `.set_param`;
- a bit-identical parameter is omitted.

Indexed loading is preferred and downloads only the transformer component. `--load-mode auto` falls back to legacy pipeline loading when indexed safetensors are unavailable.


## Merge Adapter

```bash
python merge_lora.py \
  --base Wan-AI/Wan2.2-TI2V-5B-Diffusers \
  --adapter adapter_r32.safetensors \
  --ft FastVideo/FastWan2.2-TI2V-5B-FullAttn-Diffusers \
  --output merged_model
```

**Options:**
- `--base`: Base model (HuggingFace ID or local path)
- `--adapter`: LoRA adapter file (.safetensors)
- `--ft`: Fine-tuned model (for configuration)
- `--output`: Output directory

## Validate Quality (Optional)

```bash
python lora_inference_comparison.py \
  --base merged_model \
  --ft FastVideo/FastWan2.2-TI2V-5B-FullAttn-Diffusers \
  --adapter NONE \
  --output-dir results \
  --prompt "A cat sitting on a windowsill" \
  --seed 42 \
  --height 480 \
  --width 480 \
  --num-frames 49 \
  --num-inference-steps 32 \
  --compute-ssim \
  --compute-lpips
```

**Options:**
- `--base`: Merged model or base model path
- `--ft`: Fine-tuned model (reference)
- `--adapter`: Path to adapter or NONE
- `--output-dir`: Output directory
- `--prompt`: Text prompt (default: "A cat sitting on a windowsill")
- `--seed`: Random seed (default: 42)
- `--height`: Video height (default: 480)
- `--width`: Video width (default: 832)
- `--num-frames`: Number of frames (default: 49)
- `--num-inference-steps`: Inference steps (default: 32)
- `--compute-ssim`: Compute SSIM metric
- `--compute-lpips`: Compute LPIPS metric
