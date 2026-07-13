# LingBot-Video Local Tests

This directory is the developer runbook for validating the LingBot-Video port in
FastVideo. The current port covers text-to-video (T2V) inference for both released
checkpoints:

- Dense: a single Dense transformer generates the video.
- MoE/refiner: an MoE transformer generates a base video, then a second MoE
  transformer refines it at a larger resolution.

These are local tests, which means they are run manually in this worktree and are
not part of FastVideo's default CI suite. Prompt rewriting, automatic negative-prompt
 generation, T2I, TI2V are outside the current port.

## Current Testing Status

Comparing official Lingbot-Video impl vs. FastVideo impl.

| Area                         | Validation performed                                       | Current result                              |
| ---------------------------- | ---------------------------------------------------------- | ------------------------------------------- |
| Qwen3-VL text-only encoder   | Official versus FastVideo component comparison             | Exact parity                                |
| Flow-UniPC scheduler         | Timesteps, sigmas, and deterministic update comparison     | Exact parity                                |
| Dense transformer            | Full official versus FastVideo transformer comparison      | Exact parity                                |
| Wan VAE                      | Official versus FastVideo encode and decode comparison     | Passes with `atol=0.05`, `rtol=0.05`        |
| Dense controlled pipeline    | One- and two-step latent comparison                        | Exact parity                                |
| Dense sequence parallelism   | Two-GPU pipeline latent comparison with math SDPA          | Exact parity                                |
| MoE transformer block        | Official versus FastVideo real-checkpoint block comparison | Exact parity                                |
| MoE base transformer         | Full 48-block sequential comparison                        | Exact parity                                |
| MoE refiner transformer      | Full 48-block sequential comparison                        | Exact parity                                |
| Dense production pipeline    | Batched-CFG loading and generation                         | Smoke pass; not an official CFG comparison  |
| MoE base pipeline            | Production loading and one denoising step                  | Smoke pass                                  |
| MoE plus refiner pipeline    | Base decode, in-memory handoff, refine, and final decode   | Smoke pass                                  |
| Final generated MP4          | Production generation and decode                           | Generation pass; no exact end-to-end parity |
| T2I, TI2V, and vision branch | Not implemented by this T2V port                           | Outside current scope                       |

The Dense official-pipeline comparison runs classifier-free guidance (CFG) as
separate conditional and unconditional passes because that is how the official
repo runs it. FastVideo's production path combines those inputs into one batch
for efficiency; that batched-CFG path is covered by smoke tests, not by an
official numerical comparison.

### One NOTE:
Parity is established by pairing the official and FastVideo implementations of
the text encoder, scheduler, Dense or MoE transformer, and VAE. A controlled
Dense pipeline test also compares the latent tensor after denoising. There is no
exact final-MP4 comparison for the complete MoE/refiner workflow because the two
pipelines hand the base video to the refiner differently:

```text
Official repo: base decode -> save MP4 -> load MP4 -> resize and VAE-encode -> refine
FastVideo:     base decode -> keep tensor in memory -> resize and VAE-encode -> refine
```

MP4 encoding is lossy, so the official workflow changes the pixels before the
refiner receives them. FastVideo intentionally avoids that loss. The component
parity tests therefore provide the meaningful numerical comparison, while the
complete MoE/refiner test is a production smoke test.

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
| FastVideo Dense       | `FastVideo/LingBot-Video-Dense-1.3B-Diffusers`               | `743ed04b96d77150d952eb08a59a56ee61b9bc95` |
| Official MoE/refiner  | `robbyant/lingbot-video-moe-30b-a3b`                         | `f2e538f64afe00cc4ae674db2aeb52e2945edfd5` |
| FastVideo MoE/refiner | `FastVideo/LingBot-Video-MoE-30B-A3B-Diffusers`              | `401dce84db5897cb950969e766410c8eadd4fbdf` |

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
`scripts/checkpoint_conversion/lingbot_video_to_diffusers.py`. The script keeps
the released transformer, VAE, and scheduler tensors; converts the text-only
Qwen3-VL weights to FastVideo's fused layout; and maps the official MoE
`refiner/` component to FastVideo's `transformer_2/` component.

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

| Test group                         | Required compute                                                    |
| ---------------------------------- | ------------------------------------------------------------------- |
| Routing, layout, and refiner logic | CPU                                                                 |
| Scheduler parity                   | CPU; its small pinned scheduler component downloads automatically   |
| Dense component and pipeline tests | 1 allocated CUDA GPU; prior acceptance runs used an H200            |
| Dense sequence-parallel test       | 2 allocated CUDA GPUs                                               |
| Full MoE DiT and base pipeline     | 1 H200; official and FastVideo models are loaded sequentially       |
| Complete MoE/refiner pipeline      | 8 H200 GPUs by default; FSDP and sequence parallelism are exercised |

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
```

`LINGBOT_VIDEO_PARITY_FORCE_MATH_SDPA=1` is required for an exact two-GPU
sequence-parallel comparison. With optimized SDPA, head sharding can select a
different bf16 attention kernel; that kernel-choice difference creates small
numerical drift even when the sequence-parallel data movement is correct.

### MoE And Refiner GPU Tests

| Test path                                                                  | What it checks                                           |
| -------------------------------------------------------------------------- | -------------------------------------------------------- |
| `tests/local_tests/transformers/test_lingbot_video_moe_block_parity.py`    | One real-checkpoint MoE block against the official block |
| `tests/local_tests/transformers/test_lingbot_video_moe_dit_parity.py`      | Full base or refiner 48-block transformer parity         |
| `tests/local_tests/pipelines/test_lingbot_video_moe_pipeline_smoke.py`     | Production base-MoE loading and denoising                |
| `tests/local_tests/pipelines/test_lingbot_video_refiner_pipeline_smoke.py` | Production base-to-refiner in-memory workflow            |

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
- A VAE failure means the error exceeded the explicit `0.05` absolute or relative
  tolerance used by the VAE tests.
- A smoke-test failure means FastVideo could not complete that production path; it
  does not by itself identify which component lost numerical parity.
- Missing files under the required workspace paths are setup failures, not model
  correctness failures.

This README intentionally uses direct pytest commands. There are no retained
LingBot-Video `launch_env.sh` or Slurm wrapper scripts in this worktree.
