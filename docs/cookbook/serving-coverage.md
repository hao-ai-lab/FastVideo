# Serving coverage and hardware evidence

This ledger tracks what the repository can actually launch today. A checked-in
configuration means the model path, workload and request fields resolve; it does
not mean the recipe has been run on a particular GPU. Hardware catalog entries
are owned by the shared cookbook catalog, not by this page.

Last source audit: 2026-10-19.

## Runtime surface

| Runtime | Interface | Workloads accepted by the server | Output |
| --- | --- | --- | --- |
| `fastvideo-cuda-rest` | OpenAI-compatible REST | `t2v`, `i2v`; `t2i` through `/v1/images`; `t2a`, `v2a` through the media-job routes | `.mp4`, `.png`, `.wav` |
| `fastvideo-cuda-streaming` | WebSocket streaming | `t2v` | fragmented MP4 |
| `fastvideo-mlx-rest` | MLX REST | `t2v` | `.mp4` |

Audio uses `/v1/videos` and `/v1/videos/sync` because the job store and polling
contract are already shared. The response `media_type` is `audio/wav` and the
downloaded file is WAV; clients must branch on `media_type`, not on the endpoint
name. There is no separate audio job store.

## Maintained serving baselines

| Baseline | Model | Workload | Runtime | Cookbook deployment | Evidence |
| --- | --- | --- | --- | --- | --- |
| `openai_fasth3.yaml` | FastH3 Preview v0.2 | `t2v` | CUDA REST | none (legacy catalog only) | Source-backed; H100/GB200 benchmark notes exist, not a per-GPU fit claim |
| `openai_fasth3_8step.yaml` | FastH3 8-Step V2 | `t2v` | CUDA REST | `fasth3-8step/cuda-rest` | Source-backed |
| `openai_fasth3_lora.yaml` | MiniMax-H3 + LoRA | `t2v` | CUDA REST | none | Source-backed |
| `openai_fasth3_spark.yaml` | FastH3 V1 | `t2v` | CUDA REST | none (legacy catalog only) | Source-backed; one GB10 |
| `openai_fasth3_trim_spark.yaml` | FastH3 Trim | `t2v` | CUDA REST | `fasth3-trim/cuda-rest` | Source-backed; baseline says not GPU-validated |
| `openai_compacth3_rtx5090.yaml` | CompactH3 NVFP4 | `t2v` | CUDA REST | none | Source-backed; one RTX 5090 |
| `openai_compacth3_rtx_pro6000.yaml` | CompactH3 NVFP4 | `t2v` | CUDA REST | none | Source-backed; one RTX PRO 6000 |
| `openai_fastwan21_1_3b.yaml` | FastWan2.1 1.3B | `t2v` | CUDA REST | `fastwan21` | Source-backed |
| `openai_wan21_i2v_14b.yaml` | Wan2.1 I2V 14B | `i2v` | CUDA REST | `wan21-i2v` | Source-backed |
| `openai_wan22_t2v_a14b.yaml` | Wan2.2 T2V A14B | `t2v` | CUDA REST | none | Source-backed |
| `openai_wan22_i2v_a14b.yaml` | Wan2.2 I2V A14B | `i2v` | CUDA REST | `wan22-i2v-a14b` | Source-backed |
| `openai_wan22_ti2v_5b.yaml` | Wan2.2 TI2V 5B | `t2v` | CUDA REST | none | Source-backed |
| `openai_hunyuan15_t2v_480p.yaml` | HunyuanVideo 1.5 T2V 480p | `t2v` | CUDA REST | `hunyuan15-t2v-480p` | Source-backed |
| `openai_hunyuan15_i2v_480p.yaml` | HunyuanVideo 1.5 I2V 480p | `i2v` | CUDA REST | `hunyuan15-i2v-480p` | Source-backed |
| `openai_kandinsky6_ti2va.yaml` | Kandinsky 6 Pro TI2VA | `t2v` | CUDA REST | `kandinsky6-ti2va` | Source-backed |
| `openai_kandinsky6_ti2va_distilled.yaml` | Kandinsky 6 Pro-distill TI2VA | `t2v` | CUDA REST | `kandinsky6-ti2va-distilled` | Source-backed |
| `openai_ltx2_distilled.yaml` | LTX2 Distilled | `t2v` | CUDA REST | `ltx2-distilled/cuda-rest` | Source-backed |
| `streaming_demo.yaml` | LTX2 Distilled | `t2v` | CUDA streaming | `ltx2-distilled/cuda-streaming` | Source-backed |
| `mlx_fasth3.yaml` | FastH3 V1 converted | `t2v` | MLX REST | none (legacy catalog only) | Source runtime; T2VA only |
| `mlx_fasth3_8step.yaml` | FastH3 V2 converted | `t2v` | MLX REST | `fasth3-8step/mlx-rest` | Source runtime; T2VA only |
| `openai_flux1_dev.yaml` | FLUX.1-dev | `t2i` | CUDA REST `/v1/images` | none | Source-backed; no image manifest yet |
| `openai_flux2_dev.yaml` | FLUX.2-dev | `t2i` | CUDA REST `/v1/images` | none | Source-backed; no image manifest yet |
| `openai_flux2_klein.yaml` | FLUX.2-klein-4B | `t2i` | CUDA REST `/v1/images` | none | Source-backed; no image manifest yet |
| `openai_sd35.yaml` | SD3.5 Medium | `t2i` | CUDA REST `/v1/images` | none | Source-backed; no image manifest yet |
| `openai_stable_audio.yaml` | Stable Audio Open 1.0 | `t2a` | CUDA REST media jobs | none | Source-backed; optional audio dependencies |
| `openai_mmaudio.yaml` | MMAudio large-44k-v2 | `v2a` | CUDA REST media jobs | none | Source-backed; video reference required |

## Registry families without a maintained baseline

The registry recognizes more model IDs than this table lists. These families
still resolve through the typed API, but they do not have a checked-in serving
configuration in this checkout:

- Kandinsky 5 T2V/I2V, Kandinsky 6 VSR, Kandinsky 5 SR.
- Cosmos, Cosmos 2.5, GEN3C, Helios, HyWorld, LingBot Video/World/World2.
- MatrixGame 2/3 interactive generation.
- Wan Fun InP/Control, Wan Animate, Wan S2V, Lucy Edit, Self-Forcing variants.
- MMAudio small/medium variants, Stable Audio Open Small.
- MagiHuman pipelines have no registry entry.

Adding a baseline for one of these is a code and evidence task, not a catalog
edit. The next reviewable batches are image manifests (after the shared runtime
workload contract admits `t2i`), audio manifests, then the interactive and
super-resolution families.

## Hardware evidence found in recipes

| Device identity in source | Memory type | Capacity stated by source | Recipe or test evidence | Source |
| --- | --- | --- | --- | --- |
| NVIDIA GB200 | Dedicated HBM | Not stated per device | FastH3 four-device benchmarks and measured latency table | `examples/inference/basic/README.md`, `docs/inference/support_matrix.md` |
| NVIDIA H100 (sm_90) | Dedicated HBM | 80 GB per device | FastH3 four- and eight-device benchmark rows | `examples/inference/basic/basic_fasth3_h100.yaml`, `examples/inference/basic/README.md` |
| NVIDIA B200 | Dedicated HBM | Not stated per device | Kandinsky 5 and Stable Audio recorded single-GPU runs | `docs/cookbook/kandinsky5.md`, `docs/cookbook/stable-audio.md` |
| NVIDIA RTX 5090 | Dedicated GDDR | 32 GB | CompactH3 and OmniRef profiles | `examples/serving/openai_compacth3_rtx5090.yaml`, `examples/inference/basic/basic_fasth3_omniref_rtx5090.yaml` |
| NVIDIA RTX PRO 6000 Blackwell | Dedicated GDDR | 96 GB | CompactH3, FastH3 OmniRef profiles | `examples/serving/openai_compacth3_rtx_pro6000.yaml`, `examples/inference/basic/basic_fasth3_omniref_rtx_pro6000.yaml` |
| NVIDIA RTX 4090 | Dedicated GDDR | 24 GB | FastH3 OmniRef FP8 profile, no FP4 | `examples/inference/basic/basic_fasth3_omniref_rtx4090.yaml` |
| NVIDIA DGX Spark (GB10) | Unified LPDDR | 128 GB system memory | FastH3 and OmniRef one- and two-Spark recipes | `examples/serving/openai_fasth3_spark.yaml`, `examples/inference/basic/basic_fasth3_omniref_pdd.py` |
| Apple M4 Max | Unified | 16 GB or 36 GB system memory | FastMetal and MLX support rows | `docs/inference/support_matrix.md` |
| Apple M1 Pro | Unified | 16 GB system memory | PR #1802 real 1.3B run, 256x256, 17 frames | PR #1802, not merged into this checkout |
| Apple M2 (MacBook Air) | Unified | 8 GB system memory | Local harness host only; no successful 1.3B run recorded | Session hardware, not a recipe |

The source often records a GPU count but not a device model. Wan2.1 I2V, Wan2.2
A14B, HunyuanVideo 1.5, Kandinsky 6, FLUX and SD3.5 currently fall in that
category. A GPU-count match is not evidence that a device fits.

## Hardware handoff to the catalog and UI

- Exact identities above need canonical catalog variants. `RTX 5090 32 GB`,
  `RTX PRO 6000 Blackwell 96 GB`, `H100 80 GB`, `GB10 128 GB unified` and
  `M4 Max 16/36 GB unified` must not collapse to marketing names without
  capacity or memory type.
- Dedicated VRAM and unified memory are separate estimator inputs. The Spark
  and Apple rows are system memory shared with the OS; none of them is a
  dedicated VRAM budget. A usable unified budget needs a platform policy or a
  user-entered application budget.
- Host RAM is a third input. FastH3 and CompactH3 offload text/image encoders
  and pin CPU memory; no recipe records the host RAM used, so offload host
  requirements remain unknown.
- Multi-GPU rows are topology, not pooled memory. FastH3 runs four or eight
  devices with sequence parallelism; one device's capacity is the relevant
  per-device fact.
- Restrictions such as MIG, vGPU or a user-set memory limit must not inherit
  the full card capacity.

## Verification

The declaration checks run without model weights:

```bash
pytest fastvideo/tests/entrypoints/test_openai_serving_engine.py \
  fastvideo/tests/entrypoints/test_serving_family_baselines.py
cd docs && npm run test:cookbook
```

Audio clients are in `examples/serving/clients/audio.py`, `audio.mjs` and
`audio.sh`; they read `media_type` before choosing the `.wav` extension.

Runtime evidence is recorded only when a generation actually completes with
the exact configuration, request, environment, revisions, hardware and metric
scope. A source-backed row above stays unverified until then.
