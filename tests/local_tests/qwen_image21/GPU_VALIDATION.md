# Qwen-Image-2.1 GPU validation

Updated: 2026-10-05 UTC. This file records completed checks, their evidence,
and the current verification status. Text run records under
`outputs/qwen_image21/` are preserved in a separate Git commit: logs, JSON
metrics/specifications, XML reports, source provenance, reproducer scripts,
and the validated worktree patch. Images, tensors, model weights and the
upstream source checkout are omitted at the user's request before pod deletion.

Image paths below describe artifacts inspected on the original pod; they are
not included in the saved evidence. Historical specifications contain absolute
paths from that pod. To reproduce, clone the pinned upstream revision, download
the pinned checkpoint and original assets using `official_assets/sources.json`,
update local paths, and regenerate initial latents and parity tensor artifacts.
Diagnostic and benchmark scripts that read those tensors require this setup.

The parity sweep was paused at the user's request after the T2I / two-reference
run. Subsequent requested non-layerwise I2I and warm CUDA-KV-cache T2I
benchmarks have also completed. No further GPU runs are scheduled.

## Environment and sources

- FastVideo branch: `qwen-img-2.1`, base commit
  `17a6af49f0cc1b5d4fb1e06444deb182c7c8899c`.
- Checkout: `/root/FastVideo`, also accessible through `/root/fv`.
- Existing Python environment: `/opt/venv`, Python 3.12.14. The local fork is
  installed editable; the previous editable source was `/FastVideo`.
- GPU: one NVIDIA GeForce RTX 5090, 32,607 MiB reported VRAM, driver
  `580.159.04`. A BF16 CUDA matrix multiplication passed after updating
  PyTorch to its CUDA 13 build, which includes `sm_120`.
- Effective container host-memory limit: 91,999,997,952 bytes (85.7 GiB).
  The machine's larger physical-memory total is not the container limit.
- Installed packages: PyTorch `2.12.0+cu130`, torchvision `0.27.0+cu130`,
  torchaudio `2.11.0+cu130`, Transformers `5.18.0`, tokenizers `0.23.2`,
  Accelerate `1.15.0`. Reference model imports use the pinned source below
  in this same environment.
- Released weights: [Qwen/Qwen-Image-2.1](https://huggingface.co/Qwen/Qwen-Image-2.1),
  revision `d26bb61231c349cf6b7896fa83353113880e1ba3`; all required processor,
  encoder, transformer, VAE and scheduler files downloaded. Safetensor files
  total approximately 33.1 GB. No weight conversion or quantization was used.
- Numerical reference: [Diffusers revision
  578c9b2](https://github.com/huggingface/diffusers/tree/578c9b2c6636ab2424a0e56186268b83623656b2).
  The original pod used a separate clean checkout under
  `outputs/qwen_image21/upstream/diffusers`. Official numerical model source
  is unchanged.
- Example RGB images: [Qwen's original
  repository](https://github.com/QwenLM/Qwen-Image-2.1/tree/6627d87c6433151463ec4b48b8945a24fcf16a35/prompt_rewrite/data),
  revision `6627d87c6433151463ec4b48b8945a24fcf16a35`.
  Downloaded images and their exact source URLs are recorded in
  [sources.json](../../../outputs/qwen_image21/official_assets/sources.json).

Environment evidence: [runtime verification](../../../outputs/qwen_image21/runtime-verification.log),
[install log](../../../outputs/qwen_image21/install.log),
[before](../../../outputs/qwen_image21/environment-before.json) and
[after](../../../outputs/qwen_image21/environment-after.json) package inventories.

## Confirmed runs

| Check | Result | Evidence |
| --- | --- | --- |
| Editable installation and native import | PASS; import resolves to this fork | [Runtime log](../../../outputs/qwen_image21/runtime-verification.log) |
| CLIP tokenizer dependency regression | PASS; installed Transformers/tokenizers tokenize `hi` correctly | [Tokenizer log](../../../outputs/qwen_image21/tokenizer-regression.log) |
| Native T2I, released weights, 1024 x 1024, 40 steps, seed 42 | PASS; request latency 51.45 s including deferred component loading | [Run log](../../../outputs/qwen_image21/run.log), image (`outputs/qwen_image21/teapot.png`, original pod only) |
| T2I PNG file validation | PASS; 1024 x 1024 four-channel PNG, varied RGB pixels; visually inspected red ceramic teapot | [Validation JSON](../../../outputs/qwen_image21/validation.json) |
| DiT/VAE upstream parity suite | PASS; 9 tests, zero skips, 10.67 s | [Log](../../../outputs/qwen_image21/component-parity.log), [JUnit](../../../outputs/qwen_image21/component-parity.xml) |
| DiT gate with per-block/modality metrics | PASS; 5 tests, zero skips, 8.22 s; all 32 production block outputs and final/cached predictions exactly equal | [Log](../../../outputs/qwen_image21/dit-parity.log), [JSON](../../../outputs/qwen_image21/dit-parity.json) |
| Actual-size BF16 VAE diagnostics | PASS observed exact equality; RGB-source preprocessing, tiled posterior mean/logvar, identical-latent decode and postprocessing at 1184 x 896 | [Log](../../../outputs/qwen_image21/vae-bf16-diagnostics.log), [JSON](../../../outputs/qwen_image21/vae-bf16-diagnostics.json), [reproducer](../../../outputs/qwen_image21/diagnose_vae_bf16.py) |
| Released-weight BF16 encoder parity | PASS; 1 test covering 6 cases, zero skips, 23.59 s; all compared elements exactly equal | [Log](../../../outputs/qwen_image21/encoder-parity.log), [metrics](../../../outputs/qwen_image21/encoder-parity.json) |
| Native encoder CPU contracts after the parity correction | PASS; 10 tests, 1.14 s | [Log](../../../outputs/qwen_image21/encoder-contracts.log) |
| First end-to-end plain I2I comparison, 1024 x 1024, 4 steps | FAIL numerical parity; both pipelines generated finite images, but the original pixel tolerance failed | [Log](../../../outputs/qwen_image21/pipeline-parity4.log), [JUnit](../../../outputs/qwen_image21/pipeline-parity4.xml), [metrics](../../../outputs/qwen_image21/parity4/test_qwen_image21_modes_1024_p0/parity_metrics.json) |
| Plain I2I pipeline after reference-layout correction, 1024 x 1024, 4 steps | PASS; zero skips, 58.55 s; reference latents, all four denoising predictions/steps, final latents and all pixel channels exactly equal | [Log](../../../outputs/qwen_image21/pipeline-parity4-layoutfix.log), [pixel metrics](../../../outputs/qwen_image21/parity4-layoutfix/test_qwen_image21_modes_1024_p0/parity_metrics.json), [stage metrics](../../../outputs/qwen_image21/parity4-layoutfix/test_qwen_image21_modes_1024_p0/diagnostic_metrics.json) |
| Plain I2I and editing, 1024 x 1024, 40 steps, seed 42 | PASS; 2 tests, zero skips, 270.05 s total; each case has exact prompt/reference latents, schedules, all 40 predictions/steps, final latents and RGBA pixels | [Log](../../../outputs/qwen_image21/pipeline-parity40.log), [JUnit](../../../outputs/qwen_image21/pipeline-parity40.xml), I2I image (`outputs/qwen_image21/i2i.png`, original pod only), edit image (`outputs/qwen_image21/edit.png`, original pod only) |
| T2I and two-reference generation, 1024 x 1024, 4 steps | PASS; 2 tests, zero skips, 116.41 s total; both have exact conditioning, schedules, all four predictions/steps, final latents and RGBA pixels | [Log](../../../outputs/qwen_image21/pipeline-parity4-additional.log), [JUnit](../../../outputs/qwen_image21/pipeline-parity4-additional.xml) |
| Independent BF16 VAE input-layout regressions | PASS; 2 tests, zero skips, 7.92 s; untiled 64 x 80 and tiled 288 x 320, independent preprocessing and strict posterior/normalized packed-latent comparisons | [Log](../../../outputs/qwen_image21/vae-bf16-regression.log), [JUnit](../../../outputs/qwen_image21/vae-bf16-regression.xml) |
| Canonical scheduler CPU regressions | PASS; 7 tests, zero skips, 2.27 s; exact upstream sigmas/timesteps for 4/40 steps and 4096/16384 tokens | [Log](../../../outputs/qwen_image21/scheduler-canonical-cpu.log) |
| Stage/config/parser checks after fixes | PASS; 20 tests, zero skips, 2.30 s | [Log](../../../outputs/qwen_image21/pipeline-config-contracts-final.log) |
| Final changed-file pre-commit | PASS, including mypy on checked pipeline files; repository excludes model/test/example directories by design | [Log](../../../outputs/qwen_image21/precommit-final.log) |
| Plain I2I with layerwise offload disabled, 1024 x 1024, 40 steps | PASS; request 42.44 s; initialization plus request 47.26 s; full process startup through saved image 51.42 s; output pixels exactly match the validated upstream-parity image | [Log](../../../outputs/qwen_image21/no-layerwise-i2i.log), [metrics](../../../outputs/qwen_image21/no-layerwise-i2i/metrics.json), image (`outputs/qwen_image21/no-layerwise-i2i/image.png`, original pod only) |

The native T2I run used one GPU, TP/SP = 1, BF16 components, text CPU offload,
layerwise DiT offload, VAE CPU offload and tiling, deferred module loading,
and CPU prefix caches. The GPU worker shut down normally afterward.
Successful generation alone does not establish numerical pipeline parity.
The file-format check does not establish successful transparent generation.

### Component parity coverage

The nine passing tests above compare actual numerical outputs against pinned
Diffusers source, including:

- Tiny random-weight DiT forward: `atol=2e-6`, `rtol=2e-5`.
- Q/K RMS normalization over FP32/BF16/FP16 weight and input dtype pairs:
  exact equality.
- Tiny DiT prefix extraction and cached decode, with default and CPU cache
  storage: `atol=2e-6`, `rtol=2e-5`.
- Released-weight BF16 DiT through the production loader: all 32 block outputs,
  final prediction and cached prediction, `atol=3e-3`, `rtol=1e-2`.
  A subsequent diagnostic run recorded zero numerical differences for every
  compared production tensor. This gate uses a small spatial/token grid;
  it does not substitute for full-resolution pipeline parity.
- Tiny random-weight VAE encode/decode, tiled and untiled, plus slicing:
  `atol=2e-6`, `rtol=2e-5`.
- Released-weight VAE through the production loader in FP32: posterior mean,
  log variance and decoded pixels, `atol=1e-5`, `rtol=1e-4`.
  Additional BF16 checks used the actual RGB reference resized to 1184 x 896
  and the production 256-pixel tiles / 192-pixel strides. Preprocessing,
  posterior statistics, decode from identical latents and pixel postprocessing
  all had zero mismatched elements on native and unchanged upstream models.

Command, with `QWEN_IMAGE21_MODEL_ROOT` pointing to the complete pinned
checkpoint and `QWEN_IMAGE21_DIFFUSERS_DIR` to the pinned reference checkout:

```bash
PYTHONPATH="$QWEN_IMAGE21_DIFFUSERS_DIR/src" \
QWEN_IMAGE21_RUN_DIT_PARITY=1 QWEN_IMAGE21_RUN_VAE_PARITY=1 \
/opt/venv/bin/python -m pytest \
  tests/local_tests/transformers/test_qwen_image21_transformer_parity.py \
  tests/local_tests/vaes/test_qwen_image21_vae_parity.py \
  -v -s --junitxml=outputs/qwen_image21/component-parity.xml
```

### Encoder parity coverage and corrections

The production encoder gate covers plain text, a prompt longer than 1,024
tokens, one/two/ten images, and a left-padded two-sequence text batch. Every
case has `max_abs=mean_abs=RMSE=relative_l2=0` and zero mismatched elements.
The gate retains `atol=rtol=0.01` and compares canonical Hugging Face
Qwen3-VL execution, with its final norm bypassed solely to expose the pre-norm
conditioning features required by the official Qwen-Image-2.1 pipeline.

Two native changes were necessary for exact BF16 parity:

- Execute vision projections with the packed image batch used by upstream.
  Splitting each image into a separate projection changed BF16 GEMM rounding.
  Vision attention still isolates image grids.
- Retain padded language batches and upstream's absolute text-only positions
  throughout the decoder, then zero the returned padding. Per-sequence unpadding
  and shifted RoPE positions changed reduced-precision arithmetic.

```bash
QWEN_IMAGE21_RUN_ENCODER_PARITY=1 \
QWEN_IMAGE21_ENCODER_PARITY_REPORT=outputs/qwen_image21/encoder-parity.json \
/opt/venv/bin/python -m pytest \
  tests/local_tests/encoders/test_qwen_image21_conditioner_parity.py -q -s
```

### Initial end-to-end I2I failure

The first four-step comparison used the original robot-rabbit RGB image,
identical BF16 initial noise, explicit sigma schedules, seed 42, and cached
execution on both sides. RGB mean absolute error was `0.0028066628`, maximum
absolute error `0.29296875`, and RMSE `0.0075036953`. About `0.314%` of RGB
elements exceeded `atol=0.04, rtol=0.02`. Across all four channels, 9,941 of
4,194,304 elements failed that unchanged tolerance. The run stopped on this
first failure, so it did not execute edit or two-reference cases.

Both outputs, the exact initial noise and run specification are preserved in
`outputs/qwen_image21/parity4/test_qwen_image21_modes_1024_p0/`. At that point,
the pipeline had not passed end-to-end parity; the diagnosis and passing
reruns are documented below.

An experimental four-step rerun retained the all-valid prompt mask. It also
failed (RGB mean error `0.0026899111`, max `0.2890625`, 5,617 failing
four-channel elements). Reading the full upstream `encode_prompt` boundary
confirmed that it also converts an all-valid mask to `None`. The experiment
was reverted; it is not a production fix. Evidence: [log](../../../outputs/qwen_image21/pipeline-parity4-maskfix.log),
[metrics](../../../outputs/qwen_image21/parity4-maskfix/test_qwen_image21_modes_1024_p0/parity_metrics.json).

The existing stage/scheduler CPU suite completed with **19 passed, zero
skips, 1.39 s**. Stage-file pre-commit checks, including mypy, passed.
Evidence: [CPU log](../../../outputs/qwen_image21/pipeline-contracts.log),
[pre-commit log](../../../outputs/qwen_image21/stages-precommit.log).

### Reference-layout diagnosis and passing I2I parity

The diagnostic run reproduced the first failure exactly. Prompt embeddings,
reference pixel values, image masks, shapes, initial noise and runtime precision
flags were equal. The first divergence was the VAE's normalized packed reference
latents (`mean_abs=0.0016365425`, `max_abs=0.2294921875`). Evidence:
[diagnostic log](../../../outputs/qwen_image21/pipeline-parity4-diagnostics.log),
[stage metrics](../../../outputs/qwen_image21/parity4-diagnostics/test_qwen_image21_modes_1024_p0/diagnostic_metrics.json).

The input values matched, but their singleton batch strides differed: native
`(4, 1, 4243456, 3584, 4)` versus upstream
`(4243456, 1, 4243456, 3584, 4)` for the 1184 x 896 reference. Feeding each
saved layout independently into the corresponding BF16 VAE exactly reproduced
each pipeline's reference latents. Normalizing an identical posterior on both
sides gave exact equality. Thus the initial standalone VAE check, which fed
the same native-layout tensor to both models, did not expose this layout issue.
Evidence: [reproducer](../../../outputs/qwen_image21/diagnose_reference_latents.py),
[log](../../../outputs/qwen_image21/reference-latent-diagnostics.log),
[JSON](../../../outputs/qwen_image21/reference-latent-diagnostics.json).

`reference_pixels` now creates the batch axis before permuting into channels
first, matching upstream's effective layout. The four-step I2I rerun passed
the existing `atol=0.04, rtol=0.02` gate with **zero** pixel differences;
all reference latents, denoising predictions, step latents and final latents
also matched exactly. A remaining FP32 scheduler difference of up to
`1.192e-7` in sigma did not change this four-step output and was aligned
before the forty-step runs. No upstream source or tolerance was changed.

The scheduler's NumPy branch used `np.exp(mu)`, producing a NumPy float64
scalar which promoted otherwise FP32 sigma arithmetic under NumPy 2.5.3.
Canonical upstream uses Python's `math.exp(mu)`. Matching that operation
preserves FP32 and produces exactly equal 4/40-step schedules at both tested
token counts. The new CPU regression compares the actual pinned upstream
scheduler, not a shape-only substitute. The scheduler's Torch branch and
Euler FP32 stepping policy remain unchanged.

Changed-file pre-commit on `inputs.py` initially found an existing mypy error
in the unpacked image-size call. Explicit width/height arguments fixed it;
the repeated checks, including mypy, passed. Evidence:
[initial log](../../../outputs/qwen_image21/inputs-precommit.log),
[passing log](../../../outputs/qwen_image21/inputs-precommit-final.log).

## Status at pause and unresolved checks

- All required component gates pass against real upstream models. The
  four-step and forty-step plain I2I gates pass with exact output, as does
  forty-step editing. T2I and two-reference four-step gates also pass exactly.
- Dedicated transparency/alpha, mask/annotation and 2K verification are deferred.
- Ten-reference pipeline execution, true CFG and cache-disabled pipeline GPU
  cases have not been run in this session; corresponding component/CPU coverage
  does not establish those full pipeline results.
- The 24 GB memory target has not been validated on a 24 GB GPU.
- The branch remains `qwen-img-2.1`; the validated fixes are uncommitted local
  changes. The editable environment continues to resolve to this fork. A
  [worktree patch](../../../outputs/qwen_image21/validated-worktree.patch)
  records tracked changes used for validation.

## Forty-step functional I2I and editing confirmation

Both cases use the original RGB robot-rabbit image `1413730.png`, released
unquantized BF16 weights, output 1024 x 1024, 40 steps, seed 42, explicit
shared BF16 initial noise, shared sigmas and KV caching enabled. Native and
upstream workers run separately in the same environment. The combined run
completed **2 passed, 0 skipped in 270.05 s**.

| Case | RGB/all-channel mean / max / RMSE error | Noise predictions / sampler steps exact | Numerical evidence |
| --- | --- | --- | --- |
| Plain I2I | `0 / 0 / 0` | `40/40` predictions and `40/40` steps | [Pixel metrics](../../../outputs/qwen_image21/parity40/test_qwen_image21_modes_1024_p0/parity_metrics.json), [stage metrics](../../../outputs/qwen_image21/parity40/test_qwen_image21_modes_1024_p0/diagnostic_metrics.json) |
| Editing | `0 / 0 / 0` | `40/40` predictions and `40/40` steps | [Pixel metrics](../../../outputs/qwen_image21/parity40/test_qwen_image21_modes_1024_p1/parity_metrics.json), [stage metrics](../../../outputs/qwen_image21/parity40/test_qwen_image21_modes_1024_p1/diagnostic_metrics.json) |

The unchanged pixel assertion remains `atol=0.04, rtol=0.02`; observed equality
is stronger. Prompt embeddings, normalized reference latents, masks, shapes,
FP32 sigma/timestep schedules and final packed latents also match exactly.
Each case's full specification, initial noise and native/upstream tensor
artifacts are preserved beside its metrics.

Visual inspection confirmed:

- **Plain I2I works:** a coherent close-up of the same blue robot rabbit,
  purple vest/bow tie, gold trim and star, with a white background. The model
  regenerates details (including the exposed eye), so this is reference-guided
  generation rather than a pixel identity guarantee.
- **Editing works:** the purple vest, bow tie and matching waist fabric become
  bright red. The blue subject, pose, gold trim/star and white background remain
  recognizable and consistent with the reference.

Review input (`outputs/qwen_image21/official_assets/1413730.png`, original pod only),
native plain I2I (`outputs/qwen_image21/i2i.png`, original pod only),
native edit (`outputs/qwen_image21/edit.png`, original pod only), and the corresponding
upstream I2I (`outputs/qwen_image21/i2i-upstream.png`, original pod only) /
upstream edit (`outputs/qwen_image21/edit-upstream.png`, original pod only).

Reproduce from the repository root after exporting the model and pinned-source
environment variables described above:

```bash
PYTHONPATH="$QWEN_IMAGE21_DIFFUSERS_DIR/src" \
QWEN_IMAGE21_RUN_PIPELINE_PARITY=1 \
QWEN_IMAGE21_PARITY_CASES_JSON=outputs/qwen_image21/parity-cases.json \
QWEN_IMAGE21_PARITY_STEPS=40 QWEN_IMAGE21_PARITY_DIAGNOSTICS=1 \
/opt/venv/bin/python -m pytest \
  tests/local_tests/qwen_image21/test_pipeline_parity.py -v -s \
  -k 'modes_1024 and (i2i or edit) and not rgba' \
  --basetemp=outputs/qwen_image21/parity40-new-run
```

Use a fresh `--basetemp` for every run: pytest removes an existing base directory.

## Final T2I and two-reference checks

The last run completed **2 passed, 0 skipped in 116.41 s**, at 1024 x 1024
and four denoising steps. T2I used seed 21. Two-reference generation used seed
42 and the original-source RGB portraits `1549226_a.png` / `1549226_b.png`,
with an instruction to place both people at a modern livestream desk.

Both cases had RGB and all-channel mean/max/RMSE error of **zero**, with exact
conditioning, sigma/timestep schedules, all four noise predictions and sampler
steps, and final latents. This is numerical coverage; forty-step two-reference
output quality has not been assessed.

Evidence: [T2I pixel metrics](../../../outputs/qwen_image21/parity4-additional/test_qwen_image21_modes_1024_p0/parity_metrics.json),
[T2I stage metrics](../../../outputs/qwen_image21/parity4-additional/test_qwen_image21_modes_1024_p0/diagnostic_metrics.json),
[two-reference pixel metrics](../../../outputs/qwen_image21/parity4-additional/test_qwen_image21_modes_1024_p1/parity_metrics.json),
[two-reference stage metrics](../../../outputs/qwen_image21/parity4-additional/test_qwen_image21_modes_1024_p1/diagnostic_metrics.json).

The final selection was `-k 'modes_1024 and (t2i or ref2img)'`, using
`QWEN_IMAGE21_PARITY_STEPS=4`, the same case manifest and diagnostic flag,
and `--basetemp=outputs/qwen_image21/parity4-additional`.

## Confirmed startup and harness corrections

- The Python example specified `PipelineSelection.preset`, which the existing
  compatibility adapter rejects before loading weights. Removing that field
  allows automatic native Qwen model detection; `workload_type` is retained.
  Both YAML examples now follow the same convention; their typed parsing and
  dotted overrides pass CPU checks. CLI GPU execution remains unverified.
- The existing pipeline parity helper had the same preset error and imported
  the wrong upstream namespace. The pinned source lives under
  `diffusers.pipelines.qwenimage21`, not `diffusers.pipelines.qwenimage`.
- The helper now supplies the same cache mode and sigma sequence to both sides
  and accepts original-source images via `QWEN_IMAGE21_PARITY_CASES_JSON`.
  It retains the original pixel tolerances, `atol=0.04`, `rtol=0.02`.

Changes to numerical behavior are checked against canonical upstream model
execution. No parity tolerance has been widened.

## Layerwise-offload latency benchmark

At the user's request, the same validated plain I2I case was rerun on the RTX
5090 with **layerwise DiT offload disabled** and whole-DiT CPU offload disabled.
The complete BF16 DiT stays on GPU during denoising. Text/VAE CPU offload,
VAE tiling, lazy module loading and CPU prefix-cache storage remain enabled.
Resolution is 1024 x 1024, with one original RGB reference, 40 steps, seed 42,
and the exact saved initial latents and sigma schedule from the parity run.
No production code or example defaults were changed for this benchmark.

| Measurement | Without layerwise offload |
| --- | --- |
| Generation request through PNG save, including deferred component loading | **42.44 s** |
| Generator initialization | 4.81 s |
| Generator initialization plus generation request | **47.26 s** |
| Full benchmark-process startup, including imports, through saved image | **51.42 s** |
| Sampled peak total GPU memory use (`nvidia-smi`, 0.2 s interval) | **17,257 MiB / 16.85 GiB** |
| Worker peak Torch memory allocated / reserved | 14,602.64 MiB / 16,032 MiB |
| Output difference against the validated native/upstream-parity image | Exact; mean and maximum pixel error **0** |

The earlier layerwise-enabled plain I2I request took **52.01 s** including
deferred loading and PNG save. The new request is 9.56 s shorter (about 18.4%).
These are single-run measurements: the earlier run also captured per-step
parity tensors, while the latency benchmark performs no such captures. The
comparison includes loading and does not claim a warm, already-loaded latency.

Layerwise offload is unnecessary for this tested single-reference configuration
on the 32 GB GPU: the non-layerwise run completed with substantial VRAM headroom.
Text/VAE offload and CPU prefix caching were retained, so this is not an
all-components-resident memory or latency measurement.

Evidence: [complete log](../../../outputs/qwen_image21/no-layerwise-i2i.log),
[timing and effective runtime flags](../../../outputs/qwen_image21/no-layerwise-i2i/metrics.json),
[GPU-memory samples](../../../outputs/qwen_image21/no-layerwise-i2i/gpu-memory.json),
[benchmark reproducer](../../../outputs/qwen_image21/benchmark_offload.py).

```bash
PYTHONPATH="$QWEN_IMAGE21_DIFFUSERS_DIR/src" \
/opt/venv/bin/python outputs/qwen_image21/benchmark_offload.py \
  --output-dir outputs/qwen_image21/no-layerwise-i2i-new-run
```

The effective runtime flags were confirmed by worker RPC:
`dit_layerwise_offload=False`, `dit_cpu_offload=False`,
`use_fsdp_inference=False`. The benchmark uses public
`OffloadConfig(dit=False, dit_layerwise=False, text_encoder=True, vae=True,
lazy_module_load=True)`.

## Warm T2I with CUDA prefix KV cache

The existing successful benchmark records confirm two warm T2I requests at
1024 x 1024, 40 steps, seed 21, after a four-step warmup. Models were loaded
before measurement (`lazy_module_load=False`); both whole-DiT and layerwise
DiT offload were disabled. Text encoder and VAE CPU offload remained enabled,
and `use_kv_cache=True` stored prefix tensors on CUDA. Stage logging
synchronizes CUDA before and after each measured stage.

| Measurement | Run 1 | Run 2 | Mean |
| --- | --- | --- | --- |
| Text encoding | 0.56177 s | 0.56227 s | 0.56202 s |
| DiT denoising | 14.34195 s | 14.41838 s | 14.38017 s |
| VAE decoding | 0.42786 s | 0.45731 s | 0.44259 s |
| Encoder + DiT + decoder | 15.33159 s | 15.43796 s | **15.38477 s** |
| Complete request including PNG saving | 15.58954 s | 15.69059 s | **15.64006 s** |

Generator initialization took 22.76 s, and the four-step warmup took 3.01 s;
both are outside these warm-request timings. The earlier 42.44 s plain-I2I
benchmark included deferred loading, a reference image and CPU KV cache, so
the difference does not isolate KV placement alone.

The observer confirmed 32 cache stores and 1,248 cache gets per 40-step run,
all on CUDA with BF16 shape `[1, 26, 32, 128]` per layer. Cache gets reused the
stored tensor allocations, the total peak prefix cache was **13 MiB**, and
the request cleared it afterward. Worker peak allocated/reserved memory was
16,005.34 / 17,190 MiB; sampled total GPU usage peaked at 18,535 MiB (18.10 GiB).

The four-step warmup pixels exactly matched the earlier upstream-validated
T2I baseline (mean/max error zero). The two 40-step outputs exactly matched
each other. These records do not establish a new 40-step T2I comparison
against upstream. The first benchmark attempt failed before any measurement
because it omitted a valid `reference_resolution`; the corrected attempt
succeeded, and both attempts' logs and metrics are retained.

The T2I YAML now sets `experimental.kv_cache_device: cuda`. It still enables
layerwise DiT offload and lazy loading, so the benchmark reproducer's complete
configuration is required to reproduce the measured warm latency.

Evidence: [successful log](../../../outputs/qwen_image21/t2i-gpu-kv-side-20261002/run.log),
[metrics and runtime flags](../../../outputs/qwen_image21/t2i-gpu-kv-side-20261002/metrics.json),
[memory samples](../../../outputs/qwen_image21/t2i-gpu-kv-side-20261002/gpu-memory.json),
[benchmark source](../../../outputs/qwen_image21/t2i-gpu-kv-side-20261002/benchmark.py),
[first-attempt log](../../../outputs/qwen_image21/t2i-gpu-kv-side-20261002/attempt-1-run.log),
and [first-attempt metrics](../../../outputs/qwen_image21/t2i-gpu-kv-side-20261002/attempt-1-metrics.json).

## torch.compile T2I benchmark (2026-10-05)

**PASS: one 1024 x 1024, 40-step warmup followed by one timed 40-step T2I
request.** The successful retry started with no other GPU compute processes.
It used the pinned Qwen checkpoint, seed 42, BF16, default SDPA, and the prompt
`A red ceramic teapot on a white studio background, product photography.`

Compilation used the existing DiT block path with `backend=inductor` and
`fullgraph=False`. Whole/layerwise DiT offload and lazy loading were disabled;
CUDA prefix KV caching, text/VAE CPU offload, VAE tiling and CPU memory pinning
remained enabled. Only the DiT was compiled; text encoding and VAE decoding
remained eager. Stage timers synchronize CUDA before and after each stage.

| Measurement | Successful retry |
| --- | --- |
| Generator/model initialization, excluded from request timings | 58.29764 s |
| Full 40-step warmup, excluded from measured request | **19.57909 s** |
| Timed request including PNG saving | **15.19754 s** |
| Encoder + DiT + decoder | **14.56447 s** |
| Text encoding | 0.57930 s |
| DiT denoising | **13.15578 s** |
| VAE decoding | 0.82938 s |
| PNG save stage | 0.61373 s |

The successful retry reused Inductor disk-cache artifacts created by the
earlier attempt. Its 19.58 s warmup therefore does not measure compilation
from an empty compiler cache. Warmup and measurement used the same model
instance, prompt, seed, dimensions and step count, covering both prefix-cache
extraction and cached denoising. The worker reported all **32 blocks wrapped**,
**3 captured graphs**, **329 captured operations**, and **zero graph breaks**.
The graph/operation counts were unchanged after the timed request, confirming no
additional graph capture during that measurement.

Both requests produced finite four-channel pixels. Warmup and measured pixel
tensors were exactly equal (mean/max error zero), and the saved 1024 x 1024
RGBA image was visually confirmed as a red ceramic teapot. This establishes
compiled execution and repeatability, not a new numerical comparison against
eager or upstream. One timed request without a matched eager measurement does
not establish a controlled speedup or a general latency guarantee.

### Earlier attempts and final configuration

The first attempt failed during text-encoder loading at a pinned-memory
allocation, before generation or compiled execution. Its cause was not
established. CPU pinning remained enabled on the final successful retry.

The second attempt **completed its 40-step warmup** in 28.19745 s, then failed
with CUDA OOM during the timed request. The error recorded another process
holding 13.88 GiB, this worker using about 16.29 GiB, and only 479 MiB free
when a further 1.16 GiB allocation was requested. This shared-GPU failure
does not demonstrate that the compiled model alone exceeds the GPU capacity.
The subsequent retry on the empty GPU completed successfully.

After the benchmark, the worker shut down and the saved
[T2I YAML](../../../examples/inference/basic/qwen_image21_t2i.yaml) was verified
with **`generator.engine.compile.enabled: false`**. The benchmark enables
compilation through temporary typed overrides; ordinary example execution
keeps compilation disabled. The text run records are archived with this
validation update, and the YAML change is in a separate commit. These changes
have not been pushed.

Evidence: [successful run log](../../../outputs/qwen_image21/compile-20261005/run.log),
[timings, compile counters and final flag](../../../outputs/qwen_image21/compile-20261005/metrics.json),
[benchmark source](../../../outputs/qwen_image21/compile-20261005/benchmark_compile.py),
[initial loading failure](../../../outputs/qwen_image21/compile-20261005/attempt-1-run.log),
[shared-GPU attempt log](../../../outputs/qwen_image21/compile-20261005/attempt-2-run.log),
and [shared-GPU attempt metrics](../../../outputs/qwen_image21/compile-20261005/attempt-2-metrics.json).
