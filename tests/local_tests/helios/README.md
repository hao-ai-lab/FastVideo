# Helios Local Verification

Reviewer-facing setup and verification record for the native
`BestWishYsh/Helios-Distilled` text-to-video port. These tests compare FastVideo
against the exact Diffusers Helios implementation and require local assets for
non-skip author verification.

Port state and resolved issues live in
`tests/local_tests/helios/PORT_STATUS.md`.

## Pinned Sources

| Field | Value |
| --- | --- |
| Model family | `helios` |
| Public scope | Helios-Distilled T2V |
| Architecture source | `PKU-YuanGroup/Helios@8f2a2faab3298c8a7630a2c73aea37c01b5bab01` |
| Executable parity reference | Diffusers `0.39.0` `HeliosPyramidPipeline` |
| HF checkpoint | `BestWishYsh/Helios-Distilled` |
| HF revision | `1999182614cb08d3bdcc46b9827504af2914b87b` |
| Local assets | `official_weights/helios` |
| Source layout | Diffusers; no conversion required |
| Integration base | `upstream/main@a159b63c67a1a283ce55813b694524909ea67b15` |

The official T2V model index loads one `transformer`; `transformer_ode` is not
declared or used and remains outside this PR. Base/Mid, training, ODE, I2V, and
other variants are not supported by this first pipeline.

## Environment And Assets

Run from the FastVideo repository root:

```bash
uv venv --python 3.12 --seed
UV_TORCH_BACKEND=cu130 uv pip install -e ".[dev]"

.venv/bin/python -c "from diffusers import AutoencoderKLWan, HeliosDMDScheduler, HeliosPyramidPipeline, HeliosTransformer3DModel; from transformers import AutoTokenizer, UMT5EncoderModel; print('imports ok')"
```

The verified environment uses Python 3.12.3, PyTorch `2.12.0+cu130`, CUDA 13,
Diffusers 0.39.0, Transformers 5.13.1, and `flash-attn==2.8.1`.

Download the pinned inference snapshot without the unused ODE transformer:

```bash
.venv/bin/python \
  .agents/skills/add-model-01-prep/scripts/download_hf_weights.py \
  BestWishYsh/Helios-Distilled \
  official_weights/helios \
  --revision 1999182614cb08d3bdcc46b9827504af2914b87b \
  --ignore-pattern "transformer_ode/*"
```

Use only `HF_TOKEN`, `HUGGINGFACE_HUB_TOKEN`, or `HF_API_KEY` when auth is
needed. Never put a token value in this file or a command log.

## Component Matrix

| Component | FastVideo target | Test | Current author result |
| --- | --- | --- | --- |
| Transformer | `fastvideo/models/dits/helios.py` | `tests/local_tests/transformers/test_helios_transformer_parity.py` | Non-skip PASS: 1101/1101 strict load, tiny exact normal/pyramid, full BF16 normal/pyramid, SP=2, FlashAttention-vs-SDPA |
| Scheduler | `fastvideo/models/schedulers/scheduling_helios_dmd.py` | `tests/local_tests/schedulers/test_helios_dmd_scheduler_parity.py` | 10 passed: registry resolution plus bit-exact three-stage/amplify/step parity |
| VAE | Existing native Wan VAE | `tests/local_tests/vaes/test_helios_vae_parity.py` | Non-skip PASS; decode max/mean diff `0/0` |
| UMT5 | Existing native UMT5 | `tests/local_tests/encoders/test_helios_umt5_parity.py` | FP32 max/mean `1.25e-6/1.0e-7`; BF16 `0.0234375/0.00160135` |
| Tokenizer | Production tokenizer loader | `tests/local_tests/encoders/test_helios_umt5_parity.py` | Exact IDs and attention masks |
| Pipeline | `fastvideo/pipelines/basic/helios/` | `tests/local_tests/pipelines/test_helios_pipeline_parity.py` | Non-skip final-latent PASS |

## Mandatory Commands

Run component gates before pipeline gates:

```bash
PYTHONPATH="$PWD" .venv/bin/pytest \
  tests/local_tests/transformers/test_helios_transformer_parity.py -v -s

PYTHONPATH="$PWD" .venv/bin/pytest \
  tests/local_tests/vaes/test_helios_vae_parity.py -v -s

PYTHONPATH="$PWD" .venv/bin/pytest \
  tests/local_tests/encoders/test_helios_umt5_parity.py -v -s

PYTHONPATH="$PWD" .venv/bin/pytest \
  tests/local_tests/schedulers/test_helios_dmd_scheduler_parity.py -v -s

PYTHONPATH="$PWD" .venv/bin/pytest \
  tests/local_tests/pipelines/test_helios_pipeline_math.py \
  tests/local_tests/pipelines/test_helios_pipeline_stages.py \
  tests/local_tests/pipelines/test_helios_pipeline_smoke.py -v -s

DISABLE_SP=1 PYTHONPATH="$PWD" .venv/bin/pytest \
  tests/local_tests/pipelines/test_helios_pipeline_parity.py -v -s
```

## End-To-End Latent Parity

Both implementations load the same checkpoint from scratch and use CPU seed
42, 128×192, 33 pixel frames, guidance 1.0, three `[2,2,2]` DMD stages,
history `[16,2,1]`, and nine latent frames per chunk.

```text
shape                    [1, 16, 9, 16, 24]
official abs mean        0.95911187
FastVideo abs mean       0.96272093
abs-mean relative drift  0.3763%
max absolute diff        1.89453125
mean absolute diff       0.17001711
RMSE                     0.24650776
cosine                   0.97970295
result                   PASS
```

Fixed gates are shape equality, cosine ≥0.95, abs-mean drift ≤5%, mean
absolute error ≤0.30, and RMSE ≤0.40. The managed current-head run completed in
82.85 seconds with one passed test and no skip.

## Typed Public Example

```bash
HELIOS_MODEL_PATH=official_weights/helios \
HELIOS_OUTPUT_PATH=outputs/helios/helios_distilled_t2v_384x640_33f.mp4 \
PYTHONPATH="$PWD" .venv/bin/python \
  examples/inference/basic/basic_helios_distilled_t2v.py
```

The example uses `VideoGenerator.from_config`, `GeneratorConfig`,
`GenerationRequest`, and typed sampling/output configs. The current-head run
completed generation in 29.43 seconds and produced:

```text
codec       H.264
resolution  640x384
fps         24
frames      33
duration    1.375 s
SHA-256     2e6d5099715256b16b865cb300479b328b2835f9f8792cb5427a98c17eb0b038
```

Full ffmpeg decode and the non-black-frame/container quality gate pass. The
contact sheet contains the requested fish, coral, and underwater scene.

Run the committed local quality gate with:

```bash
HELIOS_QUALITY_CANDIDATE=outputs/helios/helios_distilled_t2v_384x640_33f.mp4 \
  PYTHONPATH="$PWD" .venv/bin/pytest \
  tests/local_tests/helios/test_helios_quality_regression.py -v -s
```

Expected author result is one passed integrity test and one skipped reference
comparison. A CI SSIM reference is deliberately `deferred_with_reason`: its
publication requires separate upload approval, which this PR does not have.

## Review Notes

- Production code does not import Diffusers or Transformers model classes;
  those imports exist only in parity tests. The existing tokenizer boundary is
  reused.
- The dedicated pyramid stage is required by the official nine-frame
  autoregressive chunks, three spatial levels, history geometry, block-noise
  covariance, stage-local DMD schedules, and chunk decode contract.
- `use_zero_init` and `zero_steps` remain in the public call surface for
  Diffusers signature compatibility. The pinned Distilled model declares
  `is_cfg_zero_star=false`, so both official and FastVideo standard-CFG paths
  intentionally ignore zero-star initialization for this checkpoint.
- Final decoded and latent outputs are moved to CPU inside the worker before
  multiprocessing return. A regression test prevents CUDA-IPC OOM from
  returning a GPU tensor while the 14.31B transformer remains resident.
- Generated media, tensor artifacts, weights, reference clones, and private
  reports remain ignored and must not be committed.

## Latest-Main Unit-Lane Baseline

The branch is rebased onto `a159b63c`. The current
`.buildkite/scripts/unit_test.sh` completes with `1047 passed` and 21 warnings
in 28.70 seconds. `pre-commit run --all-files` also passes completely.
