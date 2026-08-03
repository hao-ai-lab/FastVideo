# LingBot-Video Local Tests

This directory is the developer runbook for validating the LingBot-Video port in
FastVideo. The current port covers text-to-video (T2V) and text-and-image-to-video
(TI2V) inference for both released checkpoints:

- Dense: one Dense transformer generates the T2V or TI2V base video.
- MoE/refiner: one MoE transformer generates the base video, then a second MoE
  transformer optionally refines it at a larger resolution.

These are local tests, which means they are run manually in this worktree and are
not part of FastVideo's default CI suite. For TI2V, the input image is used by
Qwen3-VL for prompt conditioning and by the VAE as the clean first-frame latent.
Prompt rewriting, automatic negative-prompt generation, and T2I are outside the
current port.

## Current Testing Status

The results below compare the pinned official implementation with FastVideo.
They use three distinct meanings of "pass":

- **Exact parity:** `torch.equal` is true and zero tensor values differ.
- **Tolerance parity:** values are close within the test's stated threshold.
- **Smoke pass:** the production path completes without error; this does not
  compare its tensors with the official implementation.

A pytest summary such as `2 passed` means that two test functions satisfied
their own assertions. Use the table's **Test type** column to determine whether
those assertions checked exact parity, tolerance parity, or only execution.

| Scope                                       | Test type           | Current result                                                        |
| ------------------------------------------- | ------------------- | --------------------------------------------------------------------- |
| CPU/API contracts                           | Unit and API        | 87 passed; 6 GPU-only cases skipped in the CPU run                    |
| T2V encoder, scheduler, and transformers    | Exact parity        | Exact at the tested component and controlled-pipeline boundaries      |
| T2V Wan VAE                                 | Tolerance parity    | Passes with `atol=0.05` and `rtol=0.05`                               |
| T2V MoE production batched CFG              | Tolerance parity    | Full-size, 40-step decoded frames have mean SSIM above 0.78           |
| T2V MoE base sequential CFG                 | Exact parity        | Full-size, 40-step decoded pixels match exactly                       |
| T2V production base and refiner workflows   | Smoke               | Pass                                                                  |
| TI2V Qwen3-VL text and vision branches      | Exact parity        | Zero differing hidden-state values                                    |
| TI2V clean-frame VAE encoding               | Exact parity        | Zero differing posterior, sample, and condition-latent values         |
| TI2V Dense base generation                  | Exact parity        | Exact conditioning, 40-step trajectory, final latent, and pixels      |
| TI2V MoE base generation                    | Exact parity        | Exact conditioning, 40-step trajectory, final latent, and pixels      |
| TI2V refiner preparation from one MP4       | Exact parity        | Exact resized pixels, VAE latents, seeded noise, and initial latent   |
| TI2V MoE plus in-memory refiner production  | Smoke               | Pass with batched CFG, FSDP, and sequence parallelism                 |
| Complete cross-repo refiner workflow        | Not exact by design | Not claimed; the production handoff inputs intentionally differ       |

The T2V production batched-CFG comparison uses the original repository's
packed FlashAttention-3 path and FastVideo's padded Torch SDPA path. Because
those kernels have different bfloat16 rounding, the test compares decoded
frames with a mean SSIM threshold of 0.78 instead of requiring identical
pixels. Like-for-like sequential component and pipeline parity remains
bit-exact.

### How Parity Is Interpreted

The exact TI2V base test uses official example 4 at 480x832, 121 frames, 40
denoising steps, guidance 3, and seed 42. Both implementations run the positive
and negative classifier-free-guidance branches as separate passes so their
operation order is identical. The test compares conditioning, the clean-frame
condition latent, initial noise, every denoising step, the final latent, decoded
float pixels. Dense and MoE both match exactly. MP4 encoding is outside this
strict comparison.

FastVideo's released presets use batched CFG for speed. That production setting
is covered by a smoke test, not by the exact base comparison.

The full refiner workflows cannot receive identical inputs because they cross
the base-to-refiner boundary differently:

```text
Official repo: base decode -> save MP4 -> load MP4 -> resize and VAE-encode -> refine
FastVideo:     base decode -> keep tensor in memory -> resize and VAE-encode -> refine
```

MP4 encoding is lossy, so the official production workflow changes the pixels
before refinement while FastVideo intentionally preserves the decoded tensor.
The exact refiner test removes that input difference by loading one shared
official base MP4 in both implementations. From that shared input, resizing,
clean-frame preparation, VAE encoding, seeded noise, and the initial refiner
latent match exactly. The separate production smoke test verifies FastVideo's
intended in-memory base-to-refiner workflow end to end.

## Setup: Required Workspace And Environment

Complete the following steps before running the local tests. The example uses a
directory named `fv-hub`, but that directory can live anywhere:

```bash
mkdir -p fv-hub
cd fv-hub
export FV_HUB="$PWD"
```

Run the remaining commands in the same shell so `FV_HUB` remains defined. Place
the FastVideo checkout being tested at
`$FV_HUB/fastvideo-port-lingbot-video`.

### Step 1: Clone The Official Reference Implementation

The parity tests import the official LingBot-Video implementation as the Python
package `lingbot_video`. Clone the official repository at the validated commit:

```bash
cd "$FV_HUB/fastvideo-port-lingbot-video"
python3 \
  .agents/skills/add-model-01-prep/scripts/clone_reference_repo.py \
  https://github.com/robbyant/lingbot-video.git \
  "$FV_HUB/lingbot-video-reference" \
  --commit a638721cf2271804d02738b69f2ad788c4a559fc
```

### Step 2: Prepare The Virtual Environment

The validated environment used Python 3.12, PyTorch 2.11.0+cu128,
Transformers 5.13.0, and Diffusers 0.39.0. Create the virtual environment, install
the normal FastVideo dependencies for the local platform, then register both
checkouts without changing dependency versions:

```bash
python3.12 -m venv "$FV_HUB/.venv"
PY="$FV_HUB/.venv/bin/python"

cd "$FV_HUB/fastvideo-port-lingbot-video"
$PY -m pip install --no-deps -e .
$PY -m pip install --no-deps -e "$FV_HUB/lingbot-video-reference"
```

The `--no-deps` commands do not populate an empty virtual environment; they only
register the two checkouts after the FastVideo dependencies are available. The
tests import the installed `lingbot_video` package; they do not modify
`sys.path`. Do not build FlashAttention as part of this setup.

### Step 3: Configure The Hugging Face Cache

The tests download their required model components from public Hugging Face
repositories after the corresponding opt-in test flag is set. No Hugging Face
token and no manually converted checkpoint directory are required.

| Role                  | Repository                                                   | Pinned revision                            |
| --------------------- | ------------------------------------------------------------ | ------------------------------------------ |
| Official Dense        | `robbyant/lingbot-video-dense-1.3b`                          | `f9789a7d9b4772a47aba62d4eb5282ddefd1da21` |
| FastVideo Dense       | `FastVideo/LingBot-Video-Dense-1.3B-Diffusers`               | `efca07f906aa17a9b03380e2fc58ef17089b4e91` |
| Official MoE/refiner  | `robbyant/lingbot-video-moe-30b-a3b`                         | `f2e538f64afe00cc4ae674db2aeb52e2945edfd5` |
| FastVideo MoE/refiner | `FastVideo/LingBot-Video-MoE-30B-A3B-Diffusers`              | `575e01b56b39f8fc31a8029ab1789339de078663` |

Keep the cache inside the shared `fv-hub` workspace by setting `HF_HOME` before
running pytest:

```bash
export HF_HOME="$FV_HUB/.cache/huggingface"
mkdir -p "$HF_HOME"
```

The helper in `tests/local_tests/lingbot_video/hf_assets.py` pins the revisions
above and requests only the components used by each test. A complete run of both
MoE transformer variants and the refiner pipeline can retain roughly 260 GB of
official and FastVideo weights in the shared cache.

### Step 4: Understand Checkpoint Conversion

Conversion is already reflected in the two public `FastVideo/*-Diffusers`
repositories and is not a test setup step. Maintainers reproducing those
packages can use
`scripts/checkpoint_conversion/lingbot_video_to_diffusers.py`. The script:

- Keeps the released transformer, VAE, and scheduler tensors unchanged.
- Fuses the Qwen3-VL language projections into FastVideo's native layout.
- Preserves the Qwen3-VL vision tower in the same text-encoder component.
- Copies the official multimodal processor into `tokenizer/`.
- Maps the official MoE `refiner/` component to FastVideo's `transformer_2/`.

T2V and TI2V therefore use the same Dense or MoE Hugging Face repository. T2V
uses only the language branch; TI2V also uses the vision branch. There are no
separate TI2V checkpoint repositories.

### Setup Result

| Purpose                       | Location or identifier                                         |
| ----------------------------- | -------------------------------------------------------------- |
| FastVideo checkout            | `$FV_HUB/fastvideo-port-lingbot-video`                         |
| Official implementation       | `$FV_HUB/lingbot-video-reference` at the pinned commit         |
| Python executable             | `$FV_HUB/.venv/bin/python`                                     |
| Official Dense weights        | `robbyant/lingbot-video-dense-1.3b`                            |
| Official MoE/refiner weights  | `robbyant/lingbot-video-moe-30b-a3b`                           |
| FastVideo Dense weights       | `FastVideo/LingBot-Video-Dense-1.3B-Diffusers`                 |
| FastVideo MoE/refiner weights | `FastVideo/LingBot-Video-MoE-30B-A3B-Diffusers`                |

### Compute Requirements For Tests

Run GPU tests only inside an allocated GPU job. The activation flags documented
below are safety gates: without the required flag, pytest skips the expensive
test instead of consuming a GPU accidentally.

| Test group                            | Required compute                                                    |
| ------------------------------------- | ------------------------------------------------------------------- |
| Routing, layout, and refiner logic    | CPU                                                                 |
| Scheduler parity                      | CPU; its small pinned scheduler component downloads automatically   |
| Dense component and base-pipeline     | 1 allocated CUDA GPU; acceptance runs used an H200                  |
| Qwen3-VL and clean-frame VAE parity   | 1 allocated CUDA GPU; acceptance runs used an H200                  |
| Dense sequence-parallel test          | 2 allocated CUDA GPUs                                               |
| Full MoE DiT and base pipeline        | 1 H200; official and FastVideo models are loaded sequentially       |
| Production MoE TI2V plus refiner test | 8 H200 GPUs by default; FSDP and sequence parallelism are exercised |

Minimum GPU memory has not been established for the Dense tests. The H200 counts
above describe the configuration used for the retained acceptance results.

---

## Test Catalog

Tests are organized beside the FastVideo component they validate, so most files
are outside this `lingbot_video/` directory.

### CPU And Preflight Tests

| Test path                                                             | What it checks                                                      |
| --------------------------------------------------------------------- | ------------------------------------------------------------------- |
| `tests/local_tests/lingbot_video/test_conversion_layout.py`           | Dense versus MoE converted directory and model-index layout         |
| `tests/local_tests/transformers/test_lingbot_video_moe.py`            | MoE routing, expert weighting, dtype policy, and checkpoint surface |
| `tests/local_tests/schedulers/test_lingbot_video_scheduler_parity.py` | Official and FastVideo scheduler timesteps, sigmas, and updates     |
| `tests/local_tests/pipelines/test_lingbot_video_refiner_stages.py`    | Refiner resize, VAE encode/noise preparation, schedule, and loading |
| `tests/local_tests/pipelines/test_lingbot_video_pipeline_smoke.py`    | T2V/TI2V registry, processor, validation, and latent contracts      |
| `tests/local_tests/models/test_fsdp_load_mixed_dtype.py`              | Checkpoint dtype restoration; its nested-FSDP case is GPU-gated     |

Run them from the FastVideo worktree:

```bash
cd "$FV_HUB/fastvideo-port-lingbot-video"
PY="$FV_HUB/.venv/bin/python"

$PY -m pytest -q \
  tests/local_tests/lingbot_video/test_conversion_layout.py \
  tests/local_tests/transformers/test_lingbot_video_moe.py \
  tests/local_tests/schedulers/test_lingbot_video_scheduler_parity.py \
  tests/local_tests/pipelines/test_lingbot_video_refiner_stages.py \
  tests/local_tests/pipelines/test_lingbot_video_pipeline_smoke.py \
  tests/local_tests/models/test_fsdp_load_mixed_dtype.py
```

### Dense GPU Tests

| Test path                                                                    | What it checks                                             |
| ---------------------------------------------------------------------------- | ---------------------------------------------------------- |
| `tests/local_tests/encoders/test_lingbot_video_dense_text_encoder_parity.py` | Text-only Qwen3-VL construction and official output parity |
| `tests/local_tests/transformers/test_lingbot_video_dense_dit_parity.py`      | Full Dense transformer output parity                       |
| `tests/local_tests/vaes/test_lingbot_video_vae_parity.py`                    | Wan VAE encode and decode parity within tolerance          |
| `tests/local_tests/pipelines/test_lingbot_video_pipeline_parity.py`          | Controlled official and FastVideo denoised-latent parity   |
| `tests/local_tests/pipelines/test_lingbot_video_pipeline_smoke.py`           | Registry, input contracts, model loading, and generation   |
| `tests/local_tests/models/test_fsdp_load_mixed_dtype.py`                     | Mixed-bf16/fp32 checkpoint loading under FSDP              |

Run the files individually so a large model is released before the next test:

```bash
cd "$FV_HUB/fastvideo-port-lingbot-video"
PY="$FV_HUB/.venv/bin/python"

LINGBOT_VIDEO_RUN_GPU_TESTS=1 $PY -m pytest -v -s \
  tests/local_tests/encoders/test_lingbot_video_dense_text_encoder_parity.py
LINGBOT_VIDEO_RUN_GPU_TESTS=1 $PY -m pytest -v -s \
  tests/local_tests/transformers/test_lingbot_video_dense_dit_parity.py
LINGBOT_VIDEO_RUN_GPU_TESTS=1 $PY -m pytest -v -s \
  tests/local_tests/vaes/test_lingbot_video_vae_parity.py
LINGBOT_VIDEO_RUN_GPU_TESTS=1 $PY -m pytest -v -s \
  tests/local_tests/pipelines/test_lingbot_video_pipeline_parity.py
LINGBOT_VIDEO_RUN_GPU_TESTS=1 $PY -m pytest -v -s \
  tests/local_tests/pipelines/test_lingbot_video_pipeline_smoke.py
LINGBOT_VIDEO_RUN_GPU_TESTS=1 $PY -m pytest -v -s \
  tests/local_tests/models/test_fsdp_load_mixed_dtype.py
```

The pipeline parity defaults to a small one-step fixture. The following optional
variables select its shape and distributed mode:

```text
LINGBOT_VIDEO_PARITY_HEIGHT
LINGBOT_VIDEO_PARITY_WIDTH
LINGBOT_VIDEO_PARITY_NUM_FRAMES
LINGBOT_VIDEO_PARITY_NUM_INFERENCE_STEPS
LINGBOT_VIDEO_PARITY_NUM_GPUS
LINGBOT_VIDEO_PARITY_SP_SIZE
LINGBOT_VIDEO_PARITY_USE_FSDP=1
LINGBOT_VIDEO_PARITY_FORCE_MATH_SDPA=1
LINGBOT_VIDEO_PARITY_VARIANT=dense|moe
LINGBOT_VIDEO_PARITY_BATCH_CFG=1
LINGBOT_VIDEO_PARITY_OUTPUT_TYPE=latent|np
LINGBOT_VIDEO_PARITY_DETERMINISTIC=1
```

`LINGBOT_VIDEO_PARITY_FORCE_MATH_SDPA=1` is required for an exact two-GPU
sequence-parallel comparison. With optimized SDPA, head sharding can select a
different bf16 attention kernel; that kernel-choice difference creates small
numerical drift even when the sequence-parallel data movement is correct.

Run the full-size MoE batched-CFG semantic gate with:

```bash
LINGBOT_VIDEO_RUN_GPU_TESTS=1 \
LINGBOT_VIDEO_PARITY_VARIANT=moe \
LINGBOT_VIDEO_PARITY_BATCH_CFG=1 \
LINGBOT_VIDEO_PARITY_HEIGHT=480 \
LINGBOT_VIDEO_PARITY_WIDTH=832 \
LINGBOT_VIDEO_PARITY_NUM_FRAMES=121 \
LINGBOT_VIDEO_PARITY_NUM_INFERENCE_STEPS=40 \
LINGBOT_VIDEO_PARITY_OUTPUT_TYPE=np \
LINGBOT_VIDEO_PARITY_DETERMINISTIC=1 \
  $PY -m pytest -v -s \
  tests/local_tests/pipelines/test_lingbot_video_pipeline_parity.py
```

The original repository implements batched CFG with the optional
`flash_attn_interface.flash_attn_varlen_func` kernel. When that kernel is
available, both sides run batched CFG and the test compares decoded frames with
SSIM. When it is unavailable, the test prints the reason and uses the original
repository's sequential-CFG result as the numerical oracle while FastVideo
still runs batched CFG; that dependency-free fallback requires
`LINGBOT_VIDEO_PARITY_FORCE_MATH_SDPA=1` and uses tight pixel-drift thresholds.
The fallback exists only in the parity test and does not change either
production pipeline.

For a bit-exact, full-size MoE base comparison, keep both sides on sequential
CFG:

```bash
LINGBOT_VIDEO_RUN_GPU_TESTS=1 \
LINGBOT_VIDEO_PARITY_VARIANT=moe \
LINGBOT_VIDEO_PARITY_HEIGHT=480 \
LINGBOT_VIDEO_PARITY_WIDTH=832 \
LINGBOT_VIDEO_PARITY_NUM_FRAMES=121 \
LINGBOT_VIDEO_PARITY_NUM_INFERENCE_STEPS=40 \
LINGBOT_VIDEO_PARITY_OUTPUT_TYPE=np \
LINGBOT_VIDEO_PARITY_DETERMINISTIC=1 \
  $PY -m pytest -v -s \
  tests/local_tests/pipelines/test_lingbot_video_pipeline_parity.py
```

### TI2V GPU Tests

| Test path                                                                           | What it checks                                                         |
| ----------------------------------------------------------------------------------- | ---------------------------------------------------------------------- |
| `tests/local_tests/encoders/test_lingbot_video_ti2v_text_encoder_parity.py`         | Exact text-only and image-conditioned Qwen3-VL hidden states           |
| `tests/local_tests/vaes/test_lingbot_video_ti2v_vae_encoder_parity.py`              | Exact clean-frame VAE posterior parameters and seeded sample           |
| `tests/local_tests/pipelines/test_lingbot_video_ti2v_pipeline_parity.py`            | Dense/MoE exact one-step, 40-step, and decoded-pixel parity            |
| `tests/local_tests/vaes/test_lingbot_video_ti2v_refiner_vae_parity.py`              | Exact refiner VAE, clean-frame, noise, and initial latent from one MP4 |
| `tests/local_tests/pipelines/test_lingbot_video_ti2v_refiner_pipeline_smoke.py`     | Production MoE TI2V, batched CFG, in-memory refinement, and decode     |
| `tests/local_tests/pipelines/test_lingbot_video_pipeline_smoke.py`                  | TI2V routing, processor loading, image-latent contract, and T2V smoke  |

Run the single-GPU component tests separately:

```bash
cd "$FV_HUB/fastvideo-port-lingbot-video"
PY="$FV_HUB/.venv/bin/python"

LINGBOT_VIDEO_RUN_GPU_TESTS=1 $PY -m pytest -v -s \
  tests/local_tests/encoders/test_lingbot_video_ti2v_text_encoder_parity.py
LINGBOT_VIDEO_RUN_GPU_TESTS=1 $PY -m pytest -v -s \
  tests/local_tests/vaes/test_lingbot_video_ti2v_vae_encoder_parity.py
```

The exact base-pipeline test first records the official output, then compares
FastVideo with it. Run one variant at a time and set `VARIANT` to `dense` or
`moe`:

```bash
export CASE_DIR="$FV_HUB/lingbot-video-reference/assets/cases/ti2v/example_4"
export PARITY_ROOT="$FV_HUB/fastvideo-port-lingbot-video/outputs/lingbot-video/validation/ti2v_exact_parity"
export LINGBOT_QWEN_ATTN_IMPLEMENTATION=sdpa
export FASTVIDEO_ATTENTION_BACKEND=TORCH_SDPA
VARIANT=dense

$PY tests/local_tests/lingbot_video/generate_ti2v_reference_baseline.py \
  "$VARIANT" "$CASE_DIR" "$PARITY_ROOT/reference/$VARIANT"

LINGBOT_VIDEO_RUN_GPU_TESTS=1 \
LINGBOT_VIDEO_TI2V_VARIANT="$VARIANT" \
LINGBOT_VIDEO_TI2V_BASELINE_DIR="$PARITY_ROOT/reference/$VARIANT" \
LINGBOT_VIDEO_TI2V_CASE_DIR="$CASE_DIR" \
LINGBOT_VIDEO_TI2V_RESULT_DIR="$PARITY_ROOT/fastvideo/$VARIANT" \
  $PY -m pytest -v -s \
  tests/local_tests/pipelines/test_lingbot_video_ti2v_pipeline_parity.py
```

The SDPA settings keep the Qwen3-VL attention backend the same in both
implementations. This removes kernel-rounding differences from the exact test.

The refiner VAE parity test requires one shared 121-frame base MP4. Both VAE
implementations reload that same file, so MP4 compression is held constant:

```bash
LINGBOT_VIDEO_RUN_GPU_TESTS=1 \
LINGBOT_VIDEO_TI2V_SHARED_BASE_MP4=/path/to/original_base.mp4 \
LINGBOT_VIDEO_TI2V_CONDITION_IMAGE="$FV_HUB/lingbot-video-reference/assets/cases/ti2v/example_4/first_frame.png" \
  $PY -m pytest -v -s \
  tests/local_tests/vaes/test_lingbot_video_ti2v_refiner_vae_parity.py
```

Run the production base-plus-refiner smoke inside one eight-H200 allocation:

```bash
LINGBOT_VIDEO_RUN_REFINER_PIPELINE_TESTS=1 \
LINGBOT_VIDEO_REFINER_NUM_GPUS=8 \
  $PY -m pytest -v -s \
  tests/local_tests/pipelines/test_lingbot_video_ti2v_refiner_pipeline_smoke.py
```

### MoE And Refiner GPU Tests

| Test path                                                                  | What it checks                                             |
| -------------------------------------------------------------------------- | ---------------------------------------------------------- |
| `tests/local_tests/transformers/test_lingbot_video_moe_block_parity.py`    | One real-checkpoint MoE block against the official block   |
| `tests/local_tests/transformers/test_lingbot_video_moe_dit_parity.py`      | Full base or refiner 48-block transformer parity           |
| `tests/local_tests/pipelines/test_lingbot_video_moe_pipeline_smoke.py`     | Five-frame, two-step batched-CFG base-MoE generation       |
| `tests/local_tests/pipelines/test_lingbot_video_refiner_pipeline_smoke.py` | Five-frame, two-step base-to-refiner in-memory workflow    |

Run each expensive test separately:

```bash
cd "$FV_HUB/fastvideo-port-lingbot-video"
PY="$FV_HUB/.venv/bin/python"

LINGBOT_VIDEO_RUN_GPU_TESTS=1 $PY -m pytest -v -s \
  tests/local_tests/transformers/test_lingbot_video_moe_block_parity.py

LINGBOT_VIDEO_RUN_FULL_MOE_TESTS=1 \
LINGBOT_VIDEO_MOE_PARITY_VARIANT=base \
  $PY -m pytest -v -s \
  tests/local_tests/transformers/test_lingbot_video_moe_dit_parity.py

LINGBOT_VIDEO_RUN_FULL_MOE_TESTS=1 \
LINGBOT_VIDEO_MOE_PARITY_VARIANT=refiner \
  $PY -m pytest -v -s \
  tests/local_tests/transformers/test_lingbot_video_moe_dit_parity.py

LINGBOT_VIDEO_RUN_MOE_PIPELINE_TESTS=1 $PY -m pytest -v -s \
  tests/local_tests/pipelines/test_lingbot_video_moe_pipeline_smoke.py

LINGBOT_VIDEO_RUN_REFINER_PIPELINE_TESTS=1 \
LINGBOT_VIDEO_REFINER_NUM_GPUS=8 \
  $PY -m pytest -v -s \
  tests/local_tests/pipelines/test_lingbot_video_refiner_pipeline_smoke.py
```

### Two-GPU Sequence-Parallel Test

`fastvideo/tests/distributed/test_sp_lingbot_video.py` starts its own two-process
`torchrun` worker and compares sequence-parallel output with the single-rank
reference. Run pytest once from a two-GPU allocation; do not wrap this command in
another `torchrun`:

```bash
cd "$FV_HUB/fastvideo-port-lingbot-video"
"$FV_HUB/.venv/bin/python" -m pytest -v -s \
  fastvideo/tests/distributed/test_sp_lingbot_video.py
```

## Interpreting Skips And Failures

- A skipped GPU test means no validation occurred. Re-run it in the required GPU
  allocation with the documented activation flag.
- An exact-parity failure means the paired official and FastVideo tensors differ;
  inspect the first reported mismatch rather than treating successful generation
  as a substitute.
- A T2V VAE tolerance failure means the error exceeded `atol=0.05` or
  `rtol=0.05`. The TI2V VAE encoder tests instead require exact equality.
- A smoke-test failure means FastVideo could not complete that production path; it
  does not by itself identify which component lost numerical parity.
- Missing files under the required workspace paths are setup failures, not model
  correctness failures.

This README intentionally uses direct pytest commands. The acceptance Slurm
scripts, exact verdicts, and paired TI2V videos from the current validation are
retained under `outputs/lingbot-video/quality_comparison_ti2v/`; they are
evidence, not prerequisites for running the local tests.
