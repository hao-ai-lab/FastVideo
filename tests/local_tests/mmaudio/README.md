# MMAudio Local Tests

Local-only parity and smoke tests for the native MMAudio FastVideo port. These
tests compare FastVideo against the official reference implementation and are
not expected to run in CI unless explicitly promoted later.

Port progress, open questions, issues, and handoff notes live in
`tests/local_tests/mmaudio/PORT_STATUS.md`.

## Reference Assets

| Field | Value |
|---|---|
| Model family | `mmaudio` |
| Workload types | `V2A`, `T2A` (new native workload values; no T2V compatibility shim) |
| Official reference | `https://github.com/hkchengrex/MMAudio` |
| Local reference dir | `../MMAudio` relative to the FastVideo repository |
| Official commit/version | `974010a026c731054592d8f777218bd9d85a6c24` |
| HF weights | `hkchengrex/MMAudio` plus canonical DFN5B CLIP and BigVGAN component repos |
| HF revision | default; pin immutable revisions before publishing converted artifacts |
| Local weights dir | `official_weights/mmaudio` (downloaded locally and gitignored) |
| Source layout | mixed raw official checkpoints and external pretrained components |
| Needs conversion | yes |

The public reference assets have been downloaded locally. Never write token
values in this file; use `HF_TOKEN`, `HUGGINGFACE_HUB_TOKEN`, or `HF_API_KEY`
only in the shell when a future gated asset requires one.

## First Implementation Scope

- Inference parity target: `large_44k_v2`, 44.1 kHz, V2A and T2A.
- Training parity target: a v1 44.1 kHz checkpoint because the official
  repository states that `_v2` training is unsupported.
- Inputs: video plus optional text for V2A, or text-only for T2A.
- Output: mono waveform and sample rate through FastVideo's audio-only result
  contract. Source-video muxing is deferred.
- Duration: 8 seconds is the published training/default duration, but inference
  uses dynamic sequence lengths and accepts shorter or longer clips. As in the
  official demo, quality can fall when moving far away from 8 seconds.
- All five official architecture variants now have FastVideo configs, presets,
  registry entries, and conversion support. Real-weight parity is complete for
  `large_44k_v2`; the other four remain local follow-up gates. Sequence/tensor
  parallel optimization and quality-reference publication are deferred.

## Shared Environment Setup

Run from the FastVideo repository root in the same environment used for
FastVideo. Do not create a separate upstream-only environment: both sides of a
parity test must share the same PyTorch/CUDA numeric stack.

The shared environment is `FastVideo/.venv`, created with uv-managed CPython
3.12.13. FastVideo is installed editable with its development dependencies;
MMAudio is installed editable with `--no-deps`, followed by the missing
reference dependencies. NumPy is pinned to 2.0.2, satisfying both MMAudio's
`numpy<2.1` constraint and FastVideo's installed OpenCV/SciPy constraints.

Recreate the environment with:

```bash
uv venv --python 3.12 --seed
source .venv/bin/activate
UV_TORCH_BACKEND=cu126 uv pip install -e ".[dev]"
uv pip install --no-deps -e ../MMAudio
uv pip install cython "gitpython>=3.1" "hydra-core>=1.3.2" \
  "torchdiffeq>=0.2.5" "librosa>=0.8.1" nitrous-ema hydra_colorlog \
  "tensordict>=0.6.1" colorlog "open_clip_torch>=2.29.0" "numpy==2.0.2"
```

## Official Environment Status

```text
dependency_changes: installed no-deps editable plus missing official dependencies in current env
official_env_status: imports_ok
private_dep_stubs: none planned
blocked_on: none
```

Run the FastVideo demo with an explicit duration:

```bash
MMAUDIO_MODEL_PATH=converted_weights/mmaudio/large_44k_v2 \
python examples/inference/basic/basic_mmaudio.py \
  --video-path /path/to/video.mp4 \
  --duration-seconds 10 \
  --output-path outputs_audio/mmaudio_10s.wav
```

## Weight Setup

The approved reference set is present: `mmaudio_large_44k_v2.pth`,
`v1-44.pth`, Synchformer, DFN5B CLIP, and canonical 44.1 kHz BigVGAN-v2.

Converted artifacts will remain untracked under:

```text
official_weights/mmaudio/
converted_weights/mmaudio/
```

## Prototype And Conversion Artifacts

```text
official_key_dumps:
  transformer: converted_weights/mmaudio/_mapping/transformer_official_keys.json
  audio_vae: converted_weights/mmaudio/_mapping/audio_vae_official_keys.json
  synchformer: converted_weights/mmaudio/_mapping/synchformer_official_keys.json
fastvideo_key_dumps:
  transformer: converted_weights/mmaudio/_mapping/transformer_fastvideo_keys.json
  audio_vae: converted_weights/mmaudio/_mapping/audio_vae_fastvideo_keys.json
  synchformer: converted_weights/mmaudio/_mapping/synchformer_fastvideo_keys.json
conversion_script: scripts/checkpoint_conversion/convert_mmaudio_to_diffusers.py
conversion_source_layout: mixed
converted_weights_dir: converted_weights/mmaudio
strict_load_status: pass for all production components
```

## Expected Parity Tests

| Component | Official files / args | Test | Concerns | Status |
|---|---|---|---|---|
| Transformer | `mmaudio/model/networks.py`; `large_44k_v2()` | `tests/local_tests/mmaudio/test_mmaudio_transformer_parity.py` | RoPE scaling, `nearest-exact`, official final-layer `global_c` behavior | exact real-weight parity |
| DFN5B conditioner | `mmaudio/model/utils/features_utils.py`; `apple/DFN5B-CLIP-ViT-H-14-384` | `tests/local_tests/mmaudio/test_mmaudio_clip_parity.py` | patched tokenwise text output and normalized vision projection | exact real-weight parity |
| Synchformer | `mmaudio/ext/synchformer/`; 16-frame windows with stride 8 | `tests/local_tests/mmaudio/test_mmaudio_synchformer_parity.py` | shared backbone must stay outside eval-only namespace | exact real-weight model and usage parity |
| Audio VAE | `mmaudio/ext/autoencoder/`; `v1-44.pth` | `tests/local_tests/mmaudio/test_mmaudio_audio_vae_parity.py` | latent transpose and normalization statistics | exact real-weight parity |
| BigVGAN | canonical `nvidia/bigvgan_v2_44khz_128band_512x` instantiation used by `AutoEncoderModule` | `tests/local_tests/mmaudio/test_mmaudio_vocoder_parity.py` | exact resampling and weight-norm behavior | exact real-weight parity |
| Scheduler | `mmaudio/model/flow_matching.py::FlowMatching` | `tests/local_tests/mmaudio/test_mmaudio_scheduler_parity.py` | forward-time Euler convention | existing FastVideo scheduler parity passed |
| Pipeline | `mmaudio/eval_utils.py::generate` and `demo.py` | `tests/local_tests/mmaudio/test_mmaudio_pipeline_parity.py` | dual-FPS frame sampling, CFG order, Euler integration, waveform output | exact real-weight parity passed |

Planned commands:

```bash
pytest tests/local_tests/mmaudio/test_mmaudio_transformer_parity.py -v -s
pytest tests/local_tests/mmaudio/test_mmaudio_clip_parity.py -v -s
pytest tests/local_tests/mmaudio/test_mmaudio_synchformer_parity.py -v -s
pytest tests/local_tests/mmaudio/test_mmaudio_audio_vae_parity.py -v -s
pytest tests/local_tests/mmaudio/test_mmaudio_vocoder_parity.py -v -s
pytest tests/local_tests/mmaudio/test_mmaudio_scheduler_parity.py -v -s
pytest tests/local_tests/mmaudio/test_mmaudio_pipeline_parity.py -v -s
```

## Review Notes

- MMAudio must be a FastVideo-native component/pipeline port; production code
  must not import `mmaudio.*`.
- Converted weights and large reference assets must remain ignored.
- Every stateful component, including reused DFN5B/BigVGAN implementations,
  requires non-skip numerical parity before final handoff.
- Pipeline smoke and waveform parity are required before the new model is
  presented as runnable.
- MMAudio checkpoints are documented upstream as CC-BY-NC 4.0; converted model
  publishing must preserve the applicable license and attribution.

## Native VGGSound Feature Extraction

FastVideo consumes the same `id<TAB>label` split manifests as the reference
`mmaudio.data.extraction.vgg_sound.VGGSound` adapter. `label` is a full caption,
not the original VGGSound category. The reference checkout is used only for
local parity review and is not imported by production preprocessing.

Training video frames are resampled with one PyAV-hosted FFmpeg filter graph:
8 FPS for DFN5B and 25 FPS for Synchformer. This matches the reference
`StreamingMediaDecoder.add_basic_video_stream(frame_rate=...)` behavior while
remaining compatible with FastVideo's current PyTorch environment. The
inference timestamp sampler is a separate default path and is not changed.

```bash
DATASET_PATH=/path/to/VGGSound SPLIT=train GPU_NUM=4 \
  bash examples/training/finetune/mmaudio/preprocess_vggsound.sh
```

The launcher supports `train`, `val`, and `test` through the same FastVideo
pipeline. The corresponding default manifests are
`sets/filtered_caption/vgg-{split}-filtered-caption.tsv`, and caches are written
under `mmaudio_features/{split}`. The cache is resumable: rerunning the command
skips IDs already written and retries pending/failed IDs. Failure logs are
append-only, so deduplicate by ID when auditing multiple runs.

Caches produced with the earlier timestamp-based training sampler are valid
TensorDict data but are not preprocessing-compatible with the FFmpeg sampler.
Do not resume into such a cache. Rebuild into a new output root instead:

```bash
OUTPUT_ROOT=/path/to/VGGSound/mmaudio_features_ffmpeg \
  ./examples/training/finetune/mmaudio/run_preprocess_vggsound.sh
```

### Optional official-compatible torio reader

For audits that require the original training media reader, FastVideo also
provides an opt-in backend in
`fastvideo/pipelines/preprocess/mmaudio/torio_media_reader.py`. It reproduces
the official `StreamingMediaDecoder` streams, audio normalization/resampling,
and both torchvision transform chains. The rest of the operation remains the
native FastVideo preprocessing pipeline: VAE, DFN5B, Synchformer, text encoder,
distributed workflow, progress bar, resumable writer, and failure logging.
Production code does not import `mmaudio.*` or execute the reference checkout.

Torio was removed from current PyTorch releases, so keep its PyTorch 2.7 CUDA
12.8 stack in a sibling environment rather than changing FastVideo's `.venv`:

```text
/path/to/workspace/FastVideo
/path/to/workspace/envs/fastvideo-mmaudio-torio
```

The verified environment uses Python 3.12, PyTorch 2.7.0+cu128,
torchvision/torchaudio 0.22.0/2.7.0, NumPy 1.26.4, and an editable
`fastvideo --no-deps` install. Select it only through the dedicated launcher:

```bash
DATASET_PATH=/path/to/VGGSound \
TORIO_VENV_PATH=/path/to/workspace/envs/fastvideo-mmaudio-torio \
CUDA_VISIBLE_DEVICES=0,1,2,3 GPU_NUM=4 BATCH_SIZE=16 \
DATALOADER_NUM_WORKERS=4 \
  ./examples/training/finetune/mmaudio/run_preprocess_vggsound_torio.sh
```

By default this processes `train val test` with their existing split manifests
and writes a separate cache under `mmaudio_features_torio/{split}`. Override
`SPLITS`, `OUTPUT_ROOT`, or the usual batching variables exactly as with the
standard launcher. Never resume a PyAV/FFmpeg cache into the torio output root.

For torio, `DATALOADER_NUM_WORKERS` is the number of background decode threads
per GPU rank, not DataLoader subprocesses. The default is 4 based on local
VGGSound measurements. FastVideo keeps two batches queued, so CPU decoding of
the next batch overlaps GPU feature extraction without sending large tensors
through `/dev/shm`. IDs already present in the feature cache are filtered before
decode and are not read again on resume.

Exact real-media parity against the official VGGSound dataset adapter is
covered by an opt-in local test:

```bash
PYTHONNOUSERSITE=1 \
MMAUDIO_TORIO_TEST_VIDEO=/path/to/an-8s-or-longer-video.mp4 \
/path/to/workspace/envs/fastvideo-mmaudio-torio/bin/pytest -q \
  tests/local_tests/mmaudio/test_mmaudio_torio_preprocessing.py -s
```

This test imports the official dataset adapter only on the test side; it is not
a runtime dependency of FastVideo.
