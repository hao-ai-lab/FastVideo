# MMAudio Port Status

## Summary

- model_family: `mmaudio`
- workload_types: `V2A`, `T2A`
- official_ref: `https://github.com/hkchengrex/MMAudio`
- official_ref_dir: `../MMAudio`
- hf_weights_path: `hkchengrex/MMAudio` plus canonical component repositories
- local_weights_dir: `official_weights/mmaudio` (not downloaded)
- source_layout: `mixed`
- local_tests_readme: `tests/local_tests/mmaudio/README.md`

## Current Phase

- phase: `component_implementation_and_conversion`
- status: `in_progress`
- owner: `parity`
- last_updated: `2026-07-20`

## Component Matrix

| Component | Type | Reuse/Port | Official Definition | Official Instantiation | FastVideo Target | Prototype | Conversion | Parity | Open Issues |
|---|---|---|---|---|---|---|---|---|---|
| MMAudio transformer | dit | port | `../MMAudio/mmaudio/model/{networks,transformer_layers,low_level,embeddings}.py` | `large_44k_v2()` | `fastvideo/models/dits/mmaudio.py`, `fastvideo/configs/models/dits/mmaudio.py` | done | implemented | implementation_pass | I002 |
| DFN5B CLIP conditioner | conditioner | native_reuse | `../MMAudio/mmaudio/model/utils/features_utils.py::FeaturesUtils` | `apple/DFN5B-CLIP-ViT-H-14-384`, patched text encoder | `fastvideo/models/encoders/mmaudio_clip.py` | done | implemented | implementation_and_tokenizer_pass | I002 |
| Synchformer visual encoder | encoder | shared_core | `../MMAudio/mmaudio/ext/synchformer/` | `Synchformer()` plus released state dict | `fastvideo/models/encoders/mmaudio_synchformer.py`, `fastvideo/third_party/synchformer/` | done | implemented | state_structure_pass | I002 |
| 44.1 kHz audio VAE | vae | port | `../MMAudio/mmaudio/ext/autoencoder/` | `AutoEncoderModule(..., mode="44k")` | `fastvideo/models/audio/mmaudio_vae.py` | done | implemented | implementation_pass | I002 |
| BigVGAN v2 vocoder | vocoder | shared_native | canonical BigVGAN implementation used by `AutoEncoderModule` | `nvidia/bigvgan_v2_44khz_128band_512x` | `fastvideo/models/audio/bigvgan.py` | done | implemented | implementation_pass | I002 |
| Flow-matching scheduler | scheduler | reuse | `../MMAudio/mmaudio/model/flow_matching.py::FlowMatching` | Euler, 25 steps, min sigma 0 | `FlowMatchEulerDiscreteScheduler` with inverted reference schedule | done | passthrough | pass | none |

## Conversion State

- conversion_script: `scripts/checkpoint_conversion/convert_mmaudio_to_diffusers.py`
- converted_weights_dir: `converted_weights/mmaudio`
- source_layout: `mixed`
- strict_load_status: `implementation-level strict loads pass; published weights not run`
- passthrough_components: `scheduler only; all weight-owning model components are native`
- retry_history: `none`

## Parity Commands

| Scope | Command | Last Result | Notes |
|---|---|---|---|
| transformer | `pytest tests/local_tests/mmaudio/test_mmaudio_transformer_parity.py -v -s` | implementation_pass / real_skip | Real checkpoint absent |
| conditioner | `pytest tests/local_tests/mmaudio/test_mmaudio_clip_parity.py -v -s` | implementation_pass / tokenizer_pass / real_skip | DFN5B snapshot absent |
| sync encoder | `pytest tests/local_tests/mmaudio/test_mmaudio_synchformer_parity.py -v -s` | structure_pass / real_skip | Released checkpoint absent |
| audio VAE | `pytest tests/local_tests/mmaudio/test_mmaudio_audio_vae_parity.py -v -s` | implementation_pass / real_skip | `v1-44.pth` absent |
| vocoder | `pytest tests/local_tests/mmaudio/test_mmaudio_vocoder_parity.py -v -s` | implementation_pass / real_skip | Canonical snapshot absent |
| scheduler | `pytest tests/local_tests/mmaudio/test_mmaudio_scheduler_parity.py -v -s` | pass | Exact 25-step schedule reuse |
| pipeline | `pytest tests/local_tests/mmaudio/test_mmaudio_pipeline_parity.py -v -s` | gated | Component real-weight parity required first |

## Open Questions

| ID | Question | Owner | Needed By Phase | Status | Resolution |
|---|---|---|---|---|---|
| Q001 | Which existing FastVideo Python/conda environment should be activated, or may a project environment be installed? | user | prep | resolved | User approved installation; created `FastVideo/.venv` with uv-managed Python 3.12.13. |
| Q002 | May the recommended 44.1 kHz reference weights and external components be downloaded for strict-load and numerical parity? | user | prep/parity | open | |

## Issues And Blockers

| ID | Phase | Component | Severity | Issue | Evidence | Owner | Status | Resolution |
|---|---|---|---|---|---|---|---|---|
| I001 | prep | all | blocker | Current shell had no usable FastVideo Python environment | `/usr/bin/python3` could not import torch, torchaudio, torchvision, einops, av, safetensors, or huggingface_hub | user/prep | resolved | Installed FastVideo and MMAudio into `FastVideo/.venv`; key imports and `uv pip check` pass, and PyTorch sees 8 GPUs. |
| I002 | prep | all stateful components | blocker | No official weights are present and no supported HF token env is set | `../MMAudio/{weights,ext_weights}` contains no files; `HF_TOKEN`, `HUGGINGFACE_HUB_TOKEN`, `HF_API_KEY` are unset | user/prep | open | Public assets may not need auth, but download size still requires approval. |

## Escape Hatches

| ID | Phase | Decision Type | Question | Recommended Option | Status | Resolution |
|---|---|---|---|---|---|---|
| E001 | prep | dependency | Which Python environment should be used, and may parity dependencies be installed? | Create a project-local uv environment and preserve FastVideo's Torch/CUDA pins. | resolved | User approved installation; shared environment is healthy. |
| E002 | prep/parity | cost | May the recommended multi-gigabyte 44.1 kHz reference weight set be downloaded? | Download only the `large_44k_v2` inference parity assets first. | open | |

## Decisions

| Date | Decision | Rationale | Impact |
|---|---|---|---|
| 2026-07-20 | First inference scope is `large_44k_v2` with V2A and T2A. | It is the official recommended inference model and exercises the complete V2A path. | Transformer/pipeline/config/conversion scope. |
| 2026-07-20 | Training parity will target a v1 44.1 kHz model first. | The official repository explicitly says `_v2` training is unsupported. | Training adapter will not claim v2 reproduction. |
| 2026-07-20 | Port components natively; do not import `mmaudio.*` in production. | Required by FastVideo's add-model production boundary. | All model, pipeline, and conversion work. |
| 2026-07-20 | Use a uv-managed Python 3.12 environment with PyTorch 2.12.0+cu126 and NumPy 2.0.2. | Matches FastVideo installation guidance while satisfying MMAudio's NumPy constraint. | Shared official/FastVideo parity environment. |
| 2026-07-20 | Split DFN5B into native FastVideo text and vision encoder components. | FastVideo's existing CLIP transformer is numerically compatible and avoids an OpenCLIP model-class runtime dependency. | Shared loader/config/conversion path for future V2A models. |
| 2026-07-20 | Extend the component loader with optional indexed image encoders. | MMAudio requires DFN5B and Synchformer simultaneously. | Existing pipelines retain singular image-encoder fields; multimodal pipelines can use `image_encoder_2`. |
| 2026-07-20 | Reuse FastVideo's flow scheduler with an inverted reference schedule. | A 25-step numerical parity test matches official forward-time Euler integration. | No MMAudio-only scheduler implementation. |

## Handoff Notes

- The official source is already available at sibling path `../MMAudio`, commit
  `974010a026c731054592d8f777218bd9d85a6c24`; no clone is required.
- The FastVideo worktree was clean at prep start (`main`, `191fcbf4`).
- Environment verification passed: FastVideo and official MMAudio import in the
  same `.venv`, `uv pip check` is clean, and PyTorch sees all 8 RTX 6000 Ada
  GPUs. Continue with early parity scaffolds before native prototypes.
