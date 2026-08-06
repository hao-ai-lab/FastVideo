# MAGI-2 Preview Port Status

## Completion

The FastVideo MAGI-2 Preview port supports text-to-video (T2V), image-to-video
(I2V), and joint audio generation. Strict release-profile parity passes against
the official SandAI implementation for both modalities.

| Field                  | Value                                                         |
| ---------------------- | ------------------------------------------------------------- |
| Branch                 | `model/port-magi-2`                                           |
| Base commit            | `1c04ace57351f7340e8d1a2e8b8f62180856ed16`                    |
| Official code revision | `073c84f2102ec3c9287623113a103c14402770ad`                    |
| Checkpoint revision    | `2dea51b64db47ee5b4402d36fd90829a0c58913b`                    |
| Workloads              | T2V and I2V with generated audio                              |
| Release profile        | 100 preview steps, 5 refiner steps, seed 42                   |
| Distributed topology   | 8 Hopper GPUs; sequence, context, and expert parallel width 8 |
| Numerical requirement  | Exact shape, dtype, stride, and tensor bytes                  |
| Full pipeline result   | Exact for every captured stage and both decoded outputs       |
| Video output           | `uint8 [249, 1088, 1920, 3]` at 25 frames per second          |
| Audio output           | `float32 [441000, 2]` at 44.1 kHz                             |

## Component Matrix

| Component                | FastVideo implementation                                         | Strict parity result                         |
| ------------------------ | ---------------------------------------------------------------- | -------------------------------------------- |
| Qwen3.5 prompt encoder   | `fastvideo/models/encoders/qwen3_5.py`                           | Exact                                        |
| Wan 2.2 image encoder    | `fastvideo/models/vaes/magi2_wan_loader.py`                      | Exact                                        |
| Preview input packing    | `fastvideo/pipelines/basic/magi2/stages/preview_data_proxy.py`   | Exact                                        |
| Preview transformer      | `fastvideo/models/dits/magi2.py`                                 | Exact across 40 layers                       |
| Refiner input packing    | `fastvideo/pipelines/basic/magi2/stages/refiner_data_proxy.py`   | Exact                                        |
| Refiner transformer      | `fastvideo/models/dits/magi2_refiner.py`                         | Exact across 30 layers                       |
| Flow UniPC scheduler     | `fastvideo/models/schedulers/scheduling_flow_unipc_multistep.py` | Exact                                        |
| Turbo VAE decoder        | `fastvideo/models/vaes/magi2_turbo_vae.py`                       | Exact across first, middle, and last windows |
| Stable Audio VAE decoder | `fastvideo/models/vaes/magi2_audio_vae.py`                       | Exact before and after resampling            |
| T2V pipeline             | `fastvideo/pipelines/basic/magi2/magi2_pipeline.py`              | Exact, 100 + 5 steps                         |
| I2V pipeline             | `fastvideo/pipelines/basic/magi2/magi2_pipeline.py`              | Exact, 100 + 5 steps                         |

## Architecture Fidelity

The preview `Transformer` preserves the 40-layer hierarchical head-parallel
design described in sections 3.1 and 3.2 of the SandAI MAGI-2 report. The
implementation exchanges fixed head activations between ranks, shards each
layer's experts across the local high-bandwidth interconnect, and uses the
12-head, 256-expert, top-6 MagiMoE routing layout. Routing runs in FP32, expert
computation runs in BF16, and each expert uses the SwiGLU7 feed-forward layout.
The strict preview parity test compares every layer boundary across eight ranks.

The refiner preserves the 30-layer local-attention design, including the exact
temporal and spatial attention ranges for the 1080p latent grid. The strict
refiner parity test compares every layer boundary across eight ranks.

## Full-Profile Output Digests

The official and FastVideo manifests differ only in the `implementation`
label. Every stage-boundary digest and output digest is identical.

| Workload | Tensor        | Shape                  | SHA-256                                                            |
| -------- | ------------- | ---------------------- | ------------------------------------------------------------------ |
| T2V      | Decoded video | `[249, 1088, 1920, 3]` | `0326570a07353cc78d488117f265b881ef6c681a3d660dc83acce2222b62e9a3` |
| T2V      | Decoded audio | `[441000, 2]`          | `fa62b45da7cfd055829ba79d1fe24d09736ed38e1075098aca71c2d0baad1937` |
| I2V      | Decoded video | `[249, 1088, 1920, 3]` | `fd211e0f647e2b3304f636202940b19ec138b520abdfd4bfabba74678097edc9` |
| I2V      | Decoded audio | `[441000, 2]`          | `32d578e901d9443d21b3659b068992e112e6430c97e1a56231a99499f6fa3db0` |

The full-profile manifests are stored at:

- `archived/magi2_parity/validation/pipeline/official/capture.json`
- `archived/magi2_parity/validation/pipeline/fastvideo/capture.json`

## Compiled-Path Verification

The official and FastVideo pipelines also ran from separate empty node-local
MagiCompiler caches with one preview step and one refiner step. Their T2V and
I2V manifests match exactly at every captured stage and decoded output. This
reduced schedule verifies compilation and production wiring; the 100 + 5 step
comparison provides the full release-profile numerical result.

| Workload | Tensor        | SHA-256                                                            |
| -------- | ------------- | ------------------------------------------------------------------ |
| T2V      | Decoded video | `ca379741f7af306cdc2d2fbbd38dfabfa35f215b613b8c7fe8b9193dcef154f5` |
| T2V      | Decoded audio | `2e44652e3c4047560255fac37b606ecc26e44d140ce500ee0a7f6328cf30c22e` |
| I2V      | Decoded video | `354a56ace8504a61487f299867c4a31beee2fd60fcce7f5bc1eb6acefdf9f532` |
| I2V      | Decoded audio | `622420e1a0777b16804ec1a8e1b53ff5a2ac2dad7edae7823b8d39a8377f67aa` |

The compiled manifests are stored under
`archived/magi2_parity/validation/pipeline_compiled/`.

## Checkpoint Conversion

`scripts/checkpoint_conversion/convert_magi2_to_fastvideo.py` converts the
official mixed checkpoint layout into the component layout consumed by the
FastVideo registry. The converter validates every indexed shard and uses hard
links for tensor files so that both layouts share the same file data.

| Official component       | FastVideo component | Role                                     |
| ------------------------ | ------------------- | ---------------------------------------- |
| `preview/`               | `transformer/`      | Joint video-audio preview transformer    |
| `refiner/`               | `transformer_2/`    | 1080p video refiner                      |
| `text_encoder/`          | `text_encoder/`     | Qwen3.5 prompt encoder and tokenizer     |
| `vae/Wan2.2_VAE.pth`     | `image_encoder/`    | Wan reference-image encoder              |
| `turbo_vae/`             | `vae/`              | Default distilled sliding-window decoder |
| `stable-audio-open-1.0/` | `audio_vae/`        | Stable Audio VAE decoder                 |

The preview, refiner, Turbo VAE, Wan VAE, and audio VAE production loaders
reject missing tensors, unexpected tensors, and shape mismatches. The Qwen3.5
loader selects the text backbone from the multimodal checkpoint and leaves the
visual tower and language-model head unused, matching the release pipeline.

## Runtime Contract

- Turbo VAE decoding is the default video-decoding path. The decoder uses a
  seven-latent first window and seven-latent subsequent windows on one context-
  parallel leader rank per video.
- `--deterministic` and `MAGI2_DETERMINISTIC=1` set fixed Python, NumPy,
  PyTorch, and CUDA seeds and activate deterministic MAGI-2 and MagiAttention
  kernels.
- `MAGI2_SAVE_LATENT_PATH` writes each sample's post-refiner latent from the
  leader rank. An empty environment value disables latent saving.
- I2V conditioning preserves the official BF16 image round trip before the Wan
  encoder performs FP32 encoding.
- Random tensors are drawn in the official order: preview video noise, preview
  audio noise, and refiner video noise.
- The pipeline returns decoded video and generated audio tensors before media
  container encoding.

## Pinned Dependencies

| Dependency             | Repository                                     | Revision                                   |
| ---------------------- | ---------------------------------------------- | ------------------------------------------ |
| MagiAttention          | `https://github.com/SandAI-org/MagiAttention`  | `2c6413571c2cac6a80d1f85a434c6713fe0f5286` |
| MagiCompiler           | `https://github.com/SandAI-org/MagiCompiler`   | `5950612ddf1205f9ba9c3238a8f02a078023e15c` |
| Flash Attention Hopper | `https://github.com/Dao-AILab/flash-attention` | `b613d9e2c8475945baff3fd68f2030af1b890acf` |

The revisions match the build arguments in the published MAGI-2 Dockerfile.
The dependencies live under `fastvideo/third_party/` and are declared in the
repository's `.gitmodules` file.

## Validation Coverage

| Validation file                            | Coverage                                                   |
| ------------------------------------------ | ---------------------------------------------------------- |
| `test_magi2_text_encoder_parity.py`        | Prompt formatting, CJK splitting, tokenization, embeddings |
| `test_magi2_image_encoder_parity.py`       | I2V resize, BF16 round trip, Wan posterior mean            |
| `test_magi2_preview_data_proxy_parity.py`  | Preview packing, coordinates, and context padding          |
| `test_magi2_preview_transformer_parity.py` | T2V and I2V boundaries across all 40 layers                |
| `test_magi2_refiner_data_proxy_parity.py`  | Local-attention ranges and refiner packing                 |
| `test_magi2_refiner_transformer_parity.py` | Boundaries across all 30 refiner layers                    |
| `test_magi2_scheduler_parity.py`           | Preview and refiner scheduler arrays and updates           |
| `test_magi2_turbo_vae_parity.py`           | First, middle, and last temporal decode windows            |
| `test_magi2_audio_vae_parity.py`           | Stable Audio VAE decoding and 44.1 kHz resampling          |
| `test_magi2_runtime_controls.py`           | Determinism and leader-only latent saving                  |
| `test_magi2_checkpoint_conversion.py`      | Component mapping and indexed-shard validation             |
| `test_magi2_registry_and_metadata.py`      | T2V and I2V registry routing                               |
| `test_magi2_pipeline_parity.py`            | Full T2V and I2V stage and decoded-output parity           |

The validation suite and environment setup records are described in
`tests/local_tests/magi2/README.md`.

## Decisions

| Decision                                 | Reason                                                      |
| ---------------------------------------- | ----------------------------------------------------------- |
| Branch from local `main`                 | The requested base is local `main` commit `1c04ace...`.     |
| Preserve the official dependency pins    | Kernel and compiler revisions affect strict parity.         |
| Use Turbo VAE by default                 | The official release and port contract select this decoder. |
| Compare tensors before MP4/AAC encoding  | Media codecs operate outside model inference.               |
| Use eager mode for full byte-parity runs | Eager execution isolates model math from compiler caching.  |
| Keep MagiCompiler enabled in production  | The pinned compiler provides the official optimized path.   |
