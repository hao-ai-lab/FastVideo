# LTX2 Overfit V2

This is a clean 2-stage dataset pipeline for LTX2 finetuning.

Canonical finetune input is precomputed tensors:

- `<data_root>/.precomputed/latents/*.pt`
- `<data_root>/.precomputed/conditions/*.pt`
- `<data_root>/.precomputed/audio_latents/*.pt` (optional)

## Configure

Edit:

`examples/training/finetune/ltx2/overfit_v2/configs/overfit_v2.yaml`

Important keys:

- `python_env`
- `data_root`
- optional defaults: `prompts_jsonl`, `start_idx`, `end_idx`, `num_gpus_generate`
- preprocess settings: `model_path`, `num_gpus_preprocess`, `with_audio`, resolution/fps
  - keep `preprocess_video_batch_size: 1` for stability in current preprocess path

## Stage A: Build Raw Dataset

From repo root:

```bash
bash examples/training/finetune/ltx2/overfit_v2/prepare_raw_dataset.sh --prompts-jsonl /path/to/data1.jsonl --num-gpus 8
```

One-liner with available args and output override:

```bash
bash examples/training/finetune/ltx2/overfit_v2/prepare_raw_dataset.sh --config examples/training/finetune/ltx2/overfit_v2/configs/overfit_v2.yaml --prompts-jsonl /path/to/data1.jsonl --num-gpus 8 --start-idx 0 --end-idx 200 --output-root /tmp/data1_output
```

For multi-node usage, run this same command once per node with a different `--prompts-jsonl`.

Outputs:

- default output dir is inferred as `<data_root>/<jsonl_stem>_output/`
  - example: `data1.jsonl` -> `<data_root>/data1_output/`
- `<output_root>/videos/*.mp4`
- `<output_root>/videos2caption.json`
- `<output_root>/reports/raw_summary.json`
- `<output_root>/logs/generate_gpu*.log`

Optional flags:

- `--config <path>`
- `--prompts-jsonl <path>`
- `--num-gpus <n>`
- `--start-idx <n>`
- `--end-idx <n>`
- `--output-root <path>`
- `--skip-generation`

## Stage B: Precompute Train Inputs

From repo root:

```bash
bash examples/training/finetune/ltx2/overfit_v2/precompute_dataset.sh
```

One-liner with available args:

```bash
bash examples/training/finetune/ltx2/overfit_v2/precompute_dataset.sh --config examples/training/finetune/ltx2/overfit_v2/configs/overfit_v2.yaml
```

Outputs:

- `<data_root>/.precomputed/...`
- `<data_root>/reports/precompute_summary.json`

Optional flags:

- `--config <path>`

## Validation

Both scripts call:

- `validate_dataset.py --mode raw` before preprocess
- `validate_dataset.py --mode precomputed` after preprocess

The process fails fast on schema/path mismatches and missing files.
