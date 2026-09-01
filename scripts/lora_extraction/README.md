# LoRA Extraction and Merging

Generic runtime adapter extraction plus the existing legacy merge utilities for FastVideo models.

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

The extractor is runtime-agnostic: it cannot infer from checkpoint tensors whether a runtime wraps a
particular matrix as a LoRA layer. When extracting a full fine-tune, select matrices unsupported by the target runtime
with `--exact-tensor-pattern`; their changes remain exact `.diff` tensors instead of becoming factors the runtime
cannot apply. The Wan patterns above cover its excluded condition embedders and its unwrapped output projection.

For large transformers, stream their indexed safetensors and factorize on a GPU:

```bash
python extract_lora.py \
  --base MiniMaxAI/MiniMax-H3 \
  --base-revision 9bfb6693f2cf6de171db46d1aa586f67d773a1da \
  --ft FastVideo/FastVideo-FastH3-4-step-v1.1 \
  --ft-revision c9e910404950b42f627f07b0c4a09d9a3e087d47 \
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
  --exact-tensor-pattern '^audio_proj_(in|out)\.weight$' \
  --exact-tensor-pattern '^context_embedder\.weight$' \
  --exact-tensor-pattern '^proj_(in|out)\.weight$' \
  --exact-tensor-pattern '^time_embedder\.'
```

`q=320, niter=4` retained 99.9355% of the energy captured by exact rank-64 SVD in a 362-matrix MiniMax-H3 comparison. Exact CPU SVD is still the default; randomized SVD must be requested explicitly.

This FastH3 checkpoint contains VSA compression-gate replacements. Load the extracted adapter with
[`basic_fasth3_lora_preview.py`](../../examples/inference/basic/basic_fasth3_lora_preview.py), which inspects the payload
and selects the required MiniMax-H3 VSA backend.

Important options:

- `--base`, `--ft`: Hugging Face model IDs or local paths.
- `--rank`, `--full-rank`: truncated or full factorization rank.
- `--min-delta`: omit tensors whose maximum absolute FP32 delta is at or below this threshold (default: `1e-8`).
- `--load-mode indexed`: download/read only `transformer/*` and stream one tensor pair at a time.
- `--device`: factorization device, such as `cpu` or `cuda:0`.
- `--svd-method`: `exact` or `randomized`.
- `--randomized-q`, `--niter`, `--seed`: randomized SVD accuracy and reproducibility.
- `--factor-dtype`: storage dtype for `lora_A` and `lora_B`.
- `--dense-dtype`: storage dtype for exact `.diff`, `.diff_b`, and `.diff_param` payloads (default: `float32`).
- `--replacement-dtype`: storage dtype for fine-tuned-only `.set_weight` and `.set_param` parameters.
- `--exact-tensor-pattern`: repeatable regex selecting matrices to retain as exact dense deltas.
- `--base-revision`, `--ft-revision`: pin Hugging Face inputs in indexed mode. Revisions are rejected for local paths and pipeline loading rather than silently ignored.
- `--work-dir`, `--resume`: resume a partially completed streaming extraction. Scratch is written to an
  output-specific directory under `fastvideo-lora-extract/`, and only that namespace is cleaned up. Resume requires indexed
  safetensors and validates both checkpoints' index/shard fingerprints before reusing partial results.

Fine-tuned parameters that cannot or should not be factorized are retained automatically:

- a changed base weight becomes `.diff`;
- a changed base bias becomes `.diff_b`;
- another changed parameter, such as `scale_shift_table`, becomes `.diff_param`;
- a fine-tuned-only weight, such as a VSA compression gate, becomes `.set_weight`;
- another fine-tuned-only parameter becomes `.set_param`;
- a bit-identical parameter is omitted.

Indexed loading is preferred and downloads only the transformer component. `--load-mode auto` falls back to legacy pipeline loading when indexed safetensors are unavailable.

Mixed low-rank/dense adapters produced by this extractor must be supplied when constructing FastVideo through
`ComponentConfig(lora_path=...)`; their dense payload cannot be swapped later with `set_lora_adapter`. The legacy
offline merger below retains its existing scope and is not part of the generic extraction workflow.

## Legacy Merge Adapter

The command below documents the pre-existing merger for adapters it already supports. Do not pass a mixed adapter from
the generic extractor to it: the legacy merger does not apply the adapter's exact dense or replacement payloads.

```bash
python merge_lora.py \
  --base Wan-AI/Wan2.2-TI2V-5B-Diffusers \
  --adapter legacy_factor_only_adapter.safetensors \
  --ft FastVideo/FastWan2.2-TI2V-5B-FullAttn-Diffusers \
  --output merged_model
```

**Options:**
- `--base`: Base model (Hugging Face ID or local path)
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
