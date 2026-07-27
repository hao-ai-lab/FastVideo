# Kandinsky5 QAD recipe (T2V, 480p)

Quantization-aware distillation for Kandinsky-5.0 Lite T2V at
**512×768, 121 frames, 24 fps** (the checkpoint's native `KANDINSKY5_T2V_LITE_5S`
preset -- 121/24 ~= 5.04s, matching the "5s" in the checkpoint name),
mirroring the Wan2.1 QAD recipe
(`examples/training/finetune/wan_t2v_1.3B/mixkit/`) on the new
`fastvideo/train/` stack. Scope is deliberately narrow: T2V only, 480p only,
dense/local attention only. Kandinsky5's NABLA sparse attention path is
never engaged at this resolution and is unsupported by this recipe --
attempting a larger resolution will raise loudly from `Kandinsky5Model`
rather than silently mis-scale the visual RoPE.

## Data

Kandinsky5 uses two text encoders (Qwen/Reason1 + CLIP), unlike Wan's single
encoder. The CLIP pooled projection is zero-padded into a row and prepended
to the Qwen sequence embeddings, then stored as a single `[seq+1, dim]`
tensor in the existing `text_embedding` Parquet field -- the same trick
`preprocess_hunyuan_overfit.py` uses for LLaMA+CLIP. No new Parquet schema
is needed.

Build a small Parquet dataset from raw videos + captions
(`videos2caption.json` + `videos/*.mp4`, matching the Hunyuan overfit script's
layout) with:

```bash
# from the repo root; reads data/kandinsky5_overfit and writes
# data/kandinsky5_overfit_preprocessed by default -- override with
# KANDINSKY5_OVERFIT_DATA_DIR / KANDINSKY5_OVERFIT_OUTPUT_DIR env vars
python -m fastvideo.pipelines.preprocess.preprocess_kandinsky5_overfit
```

This loads Kandinsky5's VAE (shared with HunyuanVideo) and both text
encoders via FastVideo's own loaders (`VAELoader`/`TextEncoderLoader`), and
writes `data_00000.parquet` + `validation_prompts.json` into the configured
output directory. Point `training.data.data_path` at that output directory
in the YAML configs below.

There is currently no scalable/production preprocessing pipeline (the kind
built on `BasePreprocessPipeline` that Wan/LTX2 use) for Kandinsky5 -- like
Hunyuan, its dual text encoders don't fit that base class's single-encoder
assumption, and `BasePreprocessPipeline.preprocess_video_and_text` isn't
cleanly overridable in pieces. If you need to scale past a handful of
overfit samples, extend `preprocess_kandinsky5_overfit.py`'s loop rather
than forking the production pipeline base class.

## Train stage 1 (Attn-QAT finetune)

```bash
bash examples/train/configs/fine_tuning/kandinsky5/finetune_qat.sh \
    --training.data.data_path data/kandinsky5_overfit_preprocessed
```

The YAML assigns `ATTN_QAT_TRAIN` to the student role, routing Kandinsky5's
dense/local attention through the fake-quantized straight-through-estimator
Triton kernel so the DiT learns to absorb quantization error. Weight-level
FP4/FP8 quantization is applied post-hoc at inference time (see below), not
during either training stage.

## Train stage 2 (QAT-aware DMD distillation)

Stage 1 writes a raw DCP checkpoint (`checkpoint-<N>/dcp` + metadata/RNG
state) under `training.checkpoint.output_dir`
(`outputs/kandinsky5_t2v_qat_finetune/` by default) -- `models.*.init_from`
needs a diffusers model directory instead (`model_index.json` + component
subfolders). Pass the stage-1 checkpoint to the launcher, which performs
the conversion and injects the export into all three `init_from` overrides:

```bash
bash examples/train/configs/distribution_matching/kandinsky5/distill_dmd_qat.sh \
    outputs/kandinsky5_t2v_qat_finetune/checkpoint-<N>
```

Under the hood it runs exactly:

```bash
python -m fastvideo.train.entrypoint.dcp_to_diffusers \
    --checkpoint outputs/kandinsky5_t2v_qat_finetune/checkpoint-<N> \
    --output-dir outputs/kandinsky5_t2v_qat_finetune/checkpoint-<N>-diffusers \
    --role student --overwrite --verify
```

then launches `dmd2_t2v_480p_qat.yaml` with
`--models.student/teacher/critic.init_from` all pointing at the export.
(Passing an already-exported diffusers directory to the launcher skips the
conversion and uses it directly.)

`--role student` exports the trained transformer weights; student, teacher,
and critic in stage 2 all load from the same checkpoint. `--verify` strictly
reloads the exported transformer immediately, so a key-mapping bug fails here
instead of deep inside the stage-2 launch.

Only the student is quantized (Attn-QAT); the teacher and critic stay full
precision / dense attention. The YAML assigns `ATTN_QAT_TRAIN` to the student
and `FLASH_ATTN` to teacher/critic. Their `disable_custom_init_weights: true`
setting also activates the loader's family-agnostic quant-config mask, so
teacher/critic weights remain full precision. Validation runs the distilled student through
`Kandinsky5DMDPipeline` at the step counts configured in
`callbacks.validation.sampling_steps`.

## Export the stage-2 student for inference

Stage 2 writes raw DCP checkpoints too, so the distilled student must be
exported the same way stage 1 was before anything can load it -- the
`path/to/kandinsky5_dmd_checkpoint` used below is exactly this export:

```bash
python -m fastvideo.train.entrypoint.dcp_to_diffusers \
    --checkpoint outputs/kandinsky5_t2v_dmd2_4steps_qat/checkpoint-<N> \
    --output-dir outputs/kandinsky5_t2v_dmd2_4steps_qat/checkpoint-<N>-diffusers \
    --role student --verify
```

## Inference (NVFP4 / FP8 weight quantization)

Kandinsky5's attention out-projection (`self_attention.out_layer` /
`cross_attention.out_layer`) and FFN (`feed_forward.mlp.fc_in` /
`feed_forward.mlp.fc_out`) names differ from Wan's (`to_out`, `ffn.fc_in` /
`ffn.fc_out`); `to_query`/`to_key`/`to_value` already substring-match the
`to_q`/`to_k`/`to_v` entries the quant configs use for Wan. Both
`nvfp4_qat` and `FP8` now include Kandinsky5's out-projection/FFN names in
their default layer lists (`fastvideo/layers/quantization/nvfp4_qat_config.py`,
`fastvideo/layers/quantization/fp8_config.py`), so no `target_layers`
override is needed:

`dcp_to_diffusers` records the configured validation pipeline in the exported
`model_index.json`. Stage-2 exports therefore resolve automatically to
`Kandinsky5DMDPipeline` and its four-step `Kandinsky5DMDConfig`:

```python
from fastvideo import VideoGenerator
from fastvideo.layers.quantization import get_quantization_config

gen = VideoGenerator.from_pretrained(
    "path/to/kandinsky5_dmd_checkpoint", num_gpus=1,
    transformer_quant=get_quantization_config("nvfp4_qat")(),
    use_fsdp_inference=False,
)
# Kandinsky5DmdDenoisingStage ignores request.sampling.num_inference_steps
# and guidance_scale -- it's a fixed 4-step, no-CFG sampler driven entirely
# by the exported pipeline config's default [1000, 750, 500, 250] schedule.
gen.generate(request={"prompt": "...", "output": {"save_video": True}})
```

As with Wan, combine with `FASTVIDEO_ATTENTION_BACKEND=ATTN_QAT_INFER` on an
sm_120 GPU for the full 4-bit path; on other GPUs the attention falls back
to a supported dense backend while the FP4 linear layers still run.
`flashinfer` is required for the FP4 path.

## Tests

- `fastvideo/tests/train/models/test_load_kandinsky5.py` -- loads the real
  checkpoint and runs one transformer forward pass.
- `fastvideo/tests/train/models/test_kandinsky5_qat_attention_engages.py` --
  confirms `ATTN_QAT_TRAIN` actually engages (not a silent SDPA fallback)
  and that gradients flow through a forward+backward pass.
- `fastvideo/tests/api/test_kandinsky5_dmd_pipeline_resolution.py` --
  confirms a stage-2 export resolves directly to `Kandinsky5DMDPipeline` plus
  `Kandinsky5DMDConfig`, and that rewriting a hard-linked `model_index.json`
  does not mutate the base checkpoint.
- `fastvideo/tests/nightly/test_e2e_kandinsky5_dmd_t2v_overfit.py` -- a few
  steps of both training stages on a single synthetic sample, exercising the
  whole recipe end to end: the `dcp_to_diffusers --verify` conversion
  between the stages AND of the final stage-2 student, then reloading that
  export through ordinary `VideoGenerator.from_pretrained`, generating
  deterministically (fixed seed), and comparing MS-SSIM against a committed
  reference video. The
  test fails (does not skip) if the reference is missing; record it once on
  a sanctioned GPU box with `KANDINSKY5_E2E_WRITE_REFERENCE=1`, review the
  written video, and commit it (see the test's module docstring). Nightly
  tests are not collected per-PR (`fastvideo/tests/contract/
  test_ci_test_collection.py` allowlists the directory), so run it
  explicitly:

  ```bash
  pytest fastvideo/tests/nightly/test_e2e_kandinsky5_dmd_t2v_overfit.py -vs
  ```
