# LoRA Extraction and Merging

Tools for extracting and merging LoRA adapters for FastVideo models.

## Extract LoRA Adapter

```bash
python scripts/lora_extraction/extract_lora.py \
  --base Wan-AI/Wan2.2-TI2V-5B-Diffusers \
  --ft FastVideo/FastWan2.2-TI2V-5B-FullAttn-Diffusers \
  --out adapter_r32.safetensors \
  --rank 32 \
  --exact-tensor-pattern '^condition_embedder\.' \
  --exact-tensor-pattern '^proj_out\.weight$'
```

The extractor is runtime-agnostic by default and cannot determine from checkpoint tensors whether the target runtime
wraps a given matrix as a LoRA layer. Use `--exact-tensor-pattern` for changed matrices that the runtime does not wrap;
the extractor preserves them as exact `.diff` tensors. The Wan patterns above cover its excluded condition embedders
and its unwrapped output projection.

Exact CPU SVD remains the default. For a large transformer, stream its indexed safetensors and factorize on a GPU:

```bash
python scripts/lora_extraction/extract_lora.py \
  --base <base-model-or-path> \
  --ft <finetuned-model-or-path> \
  --out adapter_r64.safetensors \
  --rank 64 \
  --load-mode indexed \
  --device cuda:0 \
  --svd-method randomized \
  --randomized-q 320 \
  --niter 4 \
  --factor-dtype float16 \
  --dense-dtype float32 \
  --replacement-dtype source
```

`--load-mode indexed` downloads only `transformer/*` for a Hugging Face model and reads one base/fine-tuned tensor pair at a time. The default `auto` mode tries indexed loading first and falls back to the legacy FastVideo pipeline loader; `pipeline` selects the legacy loader directly.

Important options:

- `--base`, `--ft`: Hugging Face model IDs or local paths.
- `--rank`, `--full-rank`: truncated or full factorization rank.
- `--device`: factorization device, such as `cpu` or `cuda:0`.
- `--svd-method`: `exact` or `randomized`.
- `--randomized-q`, `--niter`, `--seed`: randomized SVD accuracy and reproducibility.
- `--factor-dtype`, `--dense-dtype`, `--replacement-dtype`: adapter storage precision.
- `--exact-tensor-pattern`: repeatable regex for a matrix that should remain an exact dense delta.
- `--work-dir`, `--resume`: resume an interrupted streaming extraction.

The adapter retains changes that do not fit a low-rank product: `.diff` and `.diff_b` hold exact additive weight/bias deltas, `.diff_param` handles standalone parameters such as `scale_shift_table`, and `.set_weight`/`.set_param` hold parameters absent from the base checkpoint. Bit-identical parameters are omitted. The extractor writes an adjacent `*.report.json` with tensor counts, settings, and reconstruction residuals.

For the validated MiniMax-H3 rank-64 command, including its exact-boundary patterns, see [`scripts/lora_extraction/README.md`](https://github.com/hao-ai-lab/FastVideo/blob/main/scripts/lora_extraction/README.md).

## Merge Adapter

```bash
python scripts/lora_extraction/merge_lora.py \
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
python scripts/lora_extraction/lora_inference_comparison.py \
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
