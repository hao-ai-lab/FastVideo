# MAGI-2 Preview Local Port Validation

This directory contains the reproducible local validation suite for the MAGI-2
Preview text-to-video (T2V) and image-to-video (I2V) port. The validation target
is exact tensor equality with the official SandAI implementation for model
components, pipeline stages, terminal video tensors, and terminal audio tensors.

## Sources

| Source                    | Location                                                                                     | Revision                                   |
| ------------------------- | -------------------------------------------------------------------------------------------- | ------------------------------------------ |
| Official inference code   | `https://github.com/SandAI-org/MAGI-2-preview`                                               | `073c84f2102ec3c9287623113a103c14402770ad` |
| Official local checkout   | `/mnt/weka/shrd/wm/junda/fv-hub/MAGI-2-preview`                                              | `073c84f2102ec3c9287623113a103c14402770ad` |
| Published checkpoints     | `https://huggingface.co/sand-ai/MAGI-2-preview`                                              | `2dea51b64db47ee5b4402d36fd90829a0c58913b` |
| Local checkpoint snapshot | `official_weights/magi2`                                                                     | `2dea51b64db47ee5b4402d36fd90829a0c58913b` |
| MagiAttention             | `fastvideo/third_party/magi_attention`                                                       | `2c6413571c2cac6a80d1f85a434c6713fe0f5286` |
| MagiCompiler              | `fastvideo/third_party/magi_compiler`                                                        | `5950612ddf1205f9ba9c3238a8f02a078023e15c` |
| Flash Attention Hopper    | `https://github.com/Dao-AILab/flash-attention/tree/b613d9e2c8475945baff3fd68f2030af1b890acf` | `b613d9e2c8475945baff3fd68f2030af1b890acf` |

The Hugging Face repository is public and does not require an access token. No
Hugging Face token environment variable is used.

## Checkpoint Layout

The checkpoint repository has a custom mixed layout without a root
`model_index.json` file. The converter maps the official component directories
into a FastVideo model repository and writes the component metadata. The tensor
files remain byte-identical hard links.

| Checkpoint path          | Pipeline role                                                      |
| ------------------------ | ------------------------------------------------------------------ |
| `preview/`               | Joint video-audio preview transformer.                             |
| `refiner/`               | 1080p video refiner transformer.                                   |
| `text_encoder/`          | Qwen3.5-27B prompt encoder and tokenizer.                          |
| `vae/Wan2.2_VAE.pth`     | Wan 2.2 reference-image encoder for I2V.                           |
| `turbo_vae/`             | Default distilled decoder; first window 7 latents, step 7 latents. |
| `stable-audio-open-1.0/` | Stable Audio variational autoencoder (VAE) audio decoder.          |

The checkpoint snapshot contains 306,721,986,476 bytes. The converted repository
uses hard links, so the source and converted layouts share the same file data on
the local filesystem.

Run the converter with:

```bash
venv-port-magi-2/bin/python \
  scripts/checkpoint_conversion/convert_magi2_to_fastvideo.py \
  --source official_weights/magi2 \
  --output converted_weights/magi2
```

## Environment

The Python environment is `venv-port-magi-2/` at the FastVideo worktree root.
It uses Python 3.12, PyTorch 2.11 with CUDA 12.8, and one numeric software stack
for the official implementation and the FastVideo implementation.

Setup commands:

```bash
uv venv --python /mnt/weka/home/junda.su/.local/bin/python3.12 venv-port-magi-2
UV_TORCH_BACKEND=cu128 uv pip install \
  --python venv-port-magi-2/bin/python \
  pip setuptools wheel ninja torch==2.11.0 torchvision torchaudio
```

The official code imports Flash Attention 3 through `flash_attn_interface`.
There is no wheel for the official pinned revision and this Python, CUDA, and
PyTorch combination. Build revision
`b613d9e2c8475945baff3fd68f2030af1b890acf` with `MAX_JOBS=16`. Do not increase
the build job count.

Initialize the pinned MagiAttention dependency before installation:

```bash
git -C fastvideo/third_party/magi_attention submodule update --init --recursive
```

The environment setup record and command logs are stored under
`archived/magi2_parity/env_setup/`.

## Official Execution Contract

Use eight Hopper GPUs. Set both deterministic environment variables for strict
official parity because the official `--deterministic` implementation does not
set the environment variable that controls deterministic mixture-of-experts
(MoE) scatter:

```bash
MAGI2_DETERMINISTIC=1 \
MAGI_ATTENTION_DETERMINISTIC_MODE=1 \
MAGI2_SAVE_LATENT_PATH=archived/magi2_parity/validation/reference_latents \
torchrun --nproc_per_node=8 inference/pipeline/entry.py \
  --resolution 1080p \
  --seconds 10 \
  --seed 42 \
  --prompt-file assets/sample_000.txt \
  --output archived/magi2_parity/validation/reference_t2v
```

Add `--image assets/sample_000.jpeg` for I2V. The FastVideo deterministic option
sets fixed Python, NumPy, PyTorch, and CUDA seeds and activates both MAGI-2
kernel determinism controls.

## Required Parity Coverage

Each component test loads the official component and the production FastVideo
component. A reused FastVideo component requires a non-skipped parity pass.
Strict parity compares shape, dtype, stride, and every tensor value.

| Scope                       | Test                                       |
| --------------------------- | ------------------------------------------ |
| Prompt and Qwen3.5 encoding | `test_magi2_text_encoder_parity.py`        |
| I2V Wan VAE encoding        | `test_magi2_image_encoder_parity.py`       |
| Preview token packing       | `test_magi2_preview_data_proxy_parity.py`  |
| Preview transformer         | `test_magi2_preview_transformer_parity.py` |
| Refiner token packing       | `test_magi2_refiner_data_proxy_parity.py`  |
| Refiner transformer         | `test_magi2_refiner_transformer_parity.py` |
| Flow UniPC scheduler        | `test_magi2_scheduler_parity.py`           |
| Turbo VAE decoding          | `test_magi2_turbo_vae_parity.py`           |
| Stable Audio VAE decoding   | `test_magi2_audio_vae_parity.py`           |
| Runtime controls            | `test_magi2_runtime_controls.py`           |
| Checkpoint conversion       | `test_magi2_checkpoint_conversion.py`      |
| Registry and metadata       | `test_magi2_registry_and_metadata.py`      |
| T2V and I2V pipeline stages | `test_magi2_pipeline_parity.py`            |
| Decoded video and audio     | `test_magi2_pipeline_parity.py`            |

Run the local suite with:

```bash
venv-port-magi-2/bin/python -m pytest tests/local_tests/magi2 -sv
```

The component tests compare prompt normalization, tokenization, image encoding,
scheduler arrays, packed model inputs, every preview and refiner transformer
layer boundary, all three Turbo VAE temporal-window roles, audio decoding before
resampling, and resampled audio. The pipeline test compares stage-boundary
digests for conditioned prompts, reference-image latents, text embeddings,
initial video and audio noise, preview outputs, the noise-injected refiner input,
the post-refiner latent, and decoded outputs. End-to-end parity compares the
decoded `uint8 [T, H, W, 3]` video array and the sample-major stereo audio array
before media encoding.

## Release-Profile Result

The official and FastVideo workers ran both T2V and I2V with seed 42, 100
preview steps, 5 refiner steps, and deterministic MAGI-2 and MagiAttention
kernels. The manifest comparison found one expected metadata difference: the
`implementation` label. All captured tensor metadata and SHA-256 digests match.

| Artifact            | Path                                                                    |
| ------------------- | ----------------------------------------------------------------------- |
| Official manifest   | `archived/magi2_parity/validation/pipeline/official/capture.json`       |
| FastVideo manifest  | `archived/magi2_parity/validation/pipeline/fastvideo/capture.json`      |
| Completion report   | `tests/local_tests/magi2/PORT_STATUS.md`                                |
| Environment record  | `archived/magi2_parity/env_setup/SETUP_SUMMARY.md`                      |

## Review Requirements

The port is ready for review when all required component tests and the combined
T2V/I2V pipeline test report non-skipped strict parity passes. Review also
verifies strict checkpoint loading, Turbo VAE default selection, one-rank-per-
video decoding, deterministic controls, post-refiner latent saving through
`MAGI2_SAVE_LATENT_PATH`, T2V behavior, I2V behavior, and audio output.

Generated tensors, checkpoints, converted weights, compiler caches, and run
logs remain outside version control.
