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
# from the repo root; edit DATA_DIR/OUTPUT_DIR at the top of the script,
# or point KANDINSKY5_OVERFIT_DATA_DIR-equivalent env vars if you've added
# CLI plumbing for your own dataset
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

`FASTVIDEO_ATTENTION_BACKEND=ATTN_QAT_TRAIN` (set by the script) routes
Kandinsky5's dense/local attention through the fake-quantized
straight-through-estimator Triton kernel, so the DiT learns to absorb
quantization error instead of fighting it. This is purely env-var driven,
model-agnostic, and does not touch weights during training -- weight-level
FP4/FP8 quantization is applied post-hoc at inference time (see below), not
during either training stage.

## Train stage 2 (QAT-aware DMD distillation)

Stage 1 writes a raw DCP checkpoint (`checkpoint-<N>/dcp` + metadata/RNG
state) under `training.checkpoint.output_dir`
(`outputs/kandinsky5_t2v_qat_finetune/` by default) -- `models.*.init_from`
needs a diffusers model directory instead (`model_index.json` + component
subfolders), so convert it first:

```bash
python -m fastvideo.train.entrypoint.dcp_to_diffusers \
    --checkpoint outputs/kandinsky5_t2v_qat_finetune/checkpoint-<N> \
    --output-dir outputs/kandinsky5_t2v_qat_finetune/checkpoint-<N>-diffusers \
    --role student --verify
```

`--role student` exports the trained transformer weights (the only role
that matters here: student/teacher/critic in stage 2 all load from the same
checkpoint, and `_loading_teacher_critic_model` handles the full-precision
masking for teacher/critic at load time, not at export time). `--verify`
strictly reloads the exported transformer immediately, so a key-mapping bug
fails here instead of deep inside the stage-2 launch below. Then point
`models.student/teacher/critic.init_from` in `dmd2_t2v_480p_qat.yaml` at
that `checkpoint-<N>-diffusers` directory (already the placeholder value in
the checked-in config) and launch:

```bash
bash examples/train/configs/distribution_matching/kandinsky5/distill_dmd_qat.sh
```

Only the student is quantized (Attn-QAT); the teacher and critic stay full
precision / dense attention. This is enforced in the loader
(`fastvideo/models/loader/component_loader.py`, via the
`_loading_teacher_critic_model` flag), which masks `quant_config` and clears
`FASTVIDEO_ATTENTION_BACKEND` for teacher/critic -- the same global env var
reaches only the student, with no per-model flags or Kandinsky5-specific
handling. Validation runs the distilled student through
`Kandinsky5DMDPipeline` at the step counts configured in
`callbacks.validation.sampling_steps`.

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

`dcp_to_diffusers` copies the base T2V checkpoint's `model_index.json`
unchanged into every export (see "Train stage 2" above), so `_class_name`
still says the base T2V pipeline and the registry
(`fastvideo/registry.py`) has no way to auto-detect that a given directory
is actually a DMD (four-step re-noise sampler) export -- it will resolve to
`Kandinsky5T2VPipeline` and run the full-length sampler on DMD-distilled
weights. Pass `override_pipeline_cls_name` and a `Kandinsky5DMDConfig`
explicitly to select the right pipeline/sampler and get
`dmd_denoising_steps` set (`Kandinsky5T2VConfig`'s default of `None` makes
`Kandinsky5DmdDenoisingStage` raise immediately):

```python
from fastvideo import VideoGenerator
from fastvideo.configs.pipelines.kandinsky5 import Kandinsky5DMDConfig
from fastvideo.layers.quantization import get_quantization_config

gen = VideoGenerator.from_pretrained(
    "path/to/kandinsky5_dmd_checkpoint", num_gpus=1,
    override_pipeline_cls_name="Kandinsky5DMDPipeline",
    pipeline_config=Kandinsky5DMDConfig(),
    transformer_quant=get_quantization_config("nvfp4_qat")(),
    use_fsdp_inference=False,
)
# Kandinsky5DmdDenoisingStage ignores request.sampling.num_inference_steps
# and guidance_scale -- it's a fixed 4-step, no-CFG sampler driven entirely
# by pipeline_config.dmd_denoising_steps (set above). Adjust
# Kandinsky5DMDConfig(dmd_denoising_steps=[...]) instead if the checkpoint
# was distilled/validated with a different schedule than the default
# [1000, 750, 500, 250].
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
  confirms an unmodified export resolves to `Kandinsky5T2VPipeline` (the bug)
  and that `override_pipeline_cls_name="Kandinsky5DMDPipeline"` fixes it (the
  documented workaround above), plus `Kandinsky5DMDConfig`'s
  `dmd_denoising_steps` default.
- `fastvideo/tests/nightly/test_e2e_kandinsky5_dmd_t2v_overfit.py` -- a few
  steps of both training stages on a single synthetic sample, including the
  `dcp_to_diffusers --verify` conversion between them (the same command
  documented above). No golden reference video exists yet (unlike the Wan
  e2e test); once a real training run is available, snapshot one and add an
  SSIM assertion to match.
