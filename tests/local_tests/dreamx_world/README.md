# DreamX World Local Tests

Local-only parity and smoke tests for the `dreamx_world` FastVideo port. These
tests compare FastVideo against the official DreamX-World reference
implementation and are not expected to run in CI unless explicitly promoted
later.

Port progress, open questions, issues, and handoff notes live in
`tests/local_tests/dreamx_world/PORT_STATUS.md`.

## Reference Assets

| Field | Value |
|---|---|
| Model family | `dreamx_world` |
| First-PR scope | `DreamX-World-5B-Cam` only |
| Out-of-scope variants | `DreamX-World-5B` autoregressive long-horizon forcing |
| Workload types | I2V camera-control compatibility shim: image + prompt + action sequence to video |
| Official reference | `https://github.com/AMAP-ML/DreamX-World` |
| Local reference dir | `DreamX-World/` |
| Official commit/version | `221875811ba31f7eac6c3025b215c09ad2cefd1d` |
| HF weights | `GD-ML/DreamX-World-5B-Cam` |
| HF revision | default |
| Local weights dir | `official_weights/dreamx_world` |
| Source layout | `raw_official` |
| Needs conversion | `yes` |

Do not write token values in this file. Current token env var detected during
prep: `none`.

## Shared Environment Setup

Run from the FastVideo repo root in the same conda/env used for FastVideo. Do
not create a separate upstream environment for parity tests.

```bash
python ".agents/skills/add-model-01-prep/scripts/clone_reference_repo.py" \
    "https://github.com/AMAP-ML/DreamX-World.git" \
    "DreamX-World" \
    --commit "221875811ba31f7eac6c3025b215c09ad2cefd1d" \
    --update-gitignore
```

DreamX-World does not expose a packaging file for editable install. During prep
the official import check used `sys.path.insert(0, "DreamX-World")`.

Additional official deps installed into the current environment for imports:

```bash
uv pip install xfuser==0.4.1
uv pip install opencv-python-headless
```

These packages are for running the official DreamX reference during local
parity only. They must not become FastVideo production/runtime dependencies for
the native `dreamx_world` pipeline.

Do not install the full `DreamX-World/requirements.txt` without explicit
approval. It pins core FastVideo stack packages including `torch`, `torchvision`,
`triton`, `flash_attn`, and `diffusers`.

## Official Environment Status

```text
dependency_changes: installed official deps in current env
official_env_status: imports_ok
private_dep_stubs: none
blocked_on: none
```

Import check used during prep:

```bash
python -c "import sys; sys.path.insert(0, 'DreamX-World'); import inference_dreamx5b; print('imports_ok')"
```

## Weight Setup

HF layout inspection found no root `model_index.json`; the repo contains
`config.json`, a safetensors index, and three transformer safetensors shards.
This is a raw official transformer layout and requires conversion before
FastVideo can load it through `VideoGenerator.from_pretrained`.

```bash
python ".agents/skills/add-model-01-prep/scripts/inspect_hf_layout.py" \
    "GD-ML/DreamX-World-5B-Cam" \
    --json
```

Weights have been staged workspace-locally. The raw DreamX transformer repo lives at `official_weights/dreamx_world`; Wan2.2 raw base artifacts live at `official_weights/Wan2.2-TI2V-5B`; Wan2.2 Diffusers reusable components live at `official_weights/Wan2.2-TI2V-5B-Diffusers`. To reproduce the DreamX download:

```bash
python ".agents/skills/add-model-01-prep/scripts/download_hf_weights.py" \
    "GD-ML/DreamX-World-5B-Cam" \
    "official_weights/dreamx_world"
```

## Prototype And Conversion Artifacts

State-dict key/shape dumps are generated after FastVideo native prototypes exist
and are used to build the conversion mapping.

```text
official_key_dumps:
  transformer: converted_weights/dreamx_world/_mapping/transformer_official_keys.json
fastvideo_key_dumps:
  transformer: converted_weights/dreamx_world/_mapping/transformer_fastvideo_keys.json
conversion_script: scripts/checkpoint_conversion/dreamx_world_to_diffusers.py
conversion_script_status: transformer_model_index_and_config_consistency_smoke_pass
conversion_source_layout: raw_official
converted_weights_dir: converted_weights/dreamx_world
model_index_status: smoke_pass
strict_load_status: pass
```

The converter writes `transformer/` from raw DreamX shards. To create a full
FastVideo-loadable diffusers-style root, pass a Wan2.2 Diffusers directory as the
component source so reusable components are copied or symlinked before
`model_index.json` is emitted:

```bash
python scripts/checkpoint_conversion/dreamx_world_to_diffusers.py \
    --source official_weights/dreamx_world \
    --output converted_weights/dreamx_world \
    --component-source /path/to/Wan2.2-TI2V-5B-Diffusers \
    --symlink-components
```

## Expected Parity Tests

Planned local tests for this family:

| Component | Official files / args | Test | Concerns | Status |
|---|---|---|---|---|
| transformer | `DreamX-World/models/wan_transformer3d.py`; instantiated in `DreamX-World/inference_dreamx5b.py` with `cam_method=prope`, `add_control_adapter=True` | `tests/local_tests/dreamx_world/test_dreamx_world_transformer_parity.py` | FastVideo uses dedicated `DreamXWorldTransformer3DModel`/`DreamXWorldConfig` files for DreamX 5B-Cam config, PRoPE, conversion mapping, real converted 5B strict-load PASS, and official-vs-FastVideo small-input forward parity PASS on CUDA. Wan DiT/config have no DreamX-specific adapter fields. | strict_load_and_forward_parity_pass |
| vae | `DreamX-World/models/wan_vae3_8.py`; `vae_type=AutoencoderKLWan3_8`, `vae_subpath=Wan2.2_VAE.pth` | `tests/local_tests/dreamx_world/test_dreamx_world_vae_parity.py` | DreamX raw `Wan2.2_VAE.pth` maps 196/196 keys into FastVideo Wan VAE; encode parity passes after applying official latent normalization. | encode_parity_pass |
| text_encoder/tokenizer | `DreamX-World/models/wan_text_encoder.py`; T5 path from Wan2.2 base model | `tests/local_tests/dreamx_world/test_dreamx_world_text_encoder_parity.py` | Official `WanT5EncoderModel` vs FastVideo `UMT5EncoderModel` hidden-state parity passes on CUDA with staged Wan2.2 weights/tokenizer. | hidden_state_parity_pass |
| scheduler | Diffusers `FlowMatchEulerDiscreteScheduler`; selected by default `sampler_name=Flow` | `tests/local_tests/dreamx_world/test_dreamx_world_scheduler_parity.py` | First PR can support official default `Flow`; optional `Flow_Unipc` and `Flow_DPM++` are out of first-PR scope. | non_skip_pass |
| camera_conditioning | `DreamX-World/utils/inference_utils.py`, `models/prope_utils.py`, `wan/modules/camera_prope.py` | `tests/local_tests/dreamx_world/test_dreamx_world_camera_conditioning_parity.py` | Action sequence to PRoPE/control input must match official tensor shapes and values. | non_skip_pass |
| pipeline | `DreamX-World/pipeline/pipeline_dreamxworld.py`; call path in `DreamX-World/inference_dreamx5b.py` | `tests/local_tests/dreamx_world/test_dreamx_world_pipeline_config.py`; `tests/local_tests/pipelines/test_dreamx_world_pipeline_smoke.py`; `tests/local_tests/pipelines/test_dreamx_world_pipeline_parity.py` | DreamX PipelineConfig wires first-scope DiT/VAE/UMT5/Flow/TI2V settings, official `shift=3.0`, default preset values, FlowMatch scheduler initialization, and local `model_index.json` resolution; independent pipeline smoke covers real CUDA local load + latent generation, and parity compares public API output to worker-side explicit ForwardBatch execution. | pipeline_smoke_and_parity_pass |

Include reused components in parity. Reuse is accepted only after the FastVideo
component definition and official instantiation arguments have both been checked
and the component parity test passes non-skip.

Run the relevant tests with:

```bash
python -m pytest tests/local_tests/dreamx_world/test_dreamx_world_camera_conditioning_parity.py -v -s
python -m pytest tests/local_tests/dreamx_world/test_dreamx_world_scheduler_parity.py -v -s
python -m pytest tests/local_tests/dreamx_world/test_dreamx_world_transformer_parity.py -v -s
python -m pytest tests/local_tests/dreamx_world/test_dreamx_world_vae_parity.py -v -s
python -m pytest tests/local_tests/dreamx_world/test_dreamx_world_text_encoder_parity.py -v -s
python -m pytest tests/local_tests/dreamx_world/test_dreamx_world_pipeline_config.py -v -s
python -m pytest tests/local_tests/dreamx_world/test_dreamx_world_conversion.py -v -s
python -m pytest tests/local_tests/pipelines/test_dreamx_world_pipeline_smoke.py tests/local_tests/pipelines/test_dreamx_world_pipeline_parity.py -q -rs
```

## Current Local Results

```bash
python -m pytest tests/local_tests/dreamx_world/ -v -s
# 2026-07-01: 26 passed, 0 skipped

python -m pytest tests/local_tests/pipelines/test_dreamx_world_pipeline_smoke.py tests/local_tests/pipelines/test_dreamx_world_pipeline_parity.py -q -rs
# 2026-06-30: 4 passed, 0 skipped
# 2026-07-01: smoke image-path TI2V coverage passed separately with 3 passed, 0 skipped

PYTHONPATH=/workspace/FastVideo FASTVIDEO_SSIM_MODEL_ID=DreamX-World-5B-Cam python -m pytest fastvideo/tests/ssim/test_dreamx_world_similarity.py -q -rs
# 2026-07-01: 1 passed, 0 skipped
```

`camera_conditioning`, default `Flow` scheduler, DreamX component/pipeline configs, default preset, pipeline entry/registry, FlowMatch scheduler initialization, DreamX camera stage, denoising `y_camera` pass-through, conversion model-index/config-consistency checks, real converted 5B dedicated DreamX transformer strict-load and forward parity on CUDA, VAE encode parity on CUDA, text hidden-state parity on CUDA, and the native DreamX PRoPE branch structure smoke are non-skip PASS results. The full local DreamX component suite, independent pipeline smoke/parity suite, image-path TI2V smoke, and SSIM quality regression currently have zero skips. The basic example was also run against `converted_weights/dreamx_world` with a 64x64/9-frame/1-step saved-video smoke, and imageio decoded the generated MP4 successfully.

## Review Notes

- Required before handoff: non-skip PASS for each required component parity
  test, including reused components that own weights or numerical behavior.
- First PR scope is `DreamX-World-5B-Cam`; do not pull in
  `inference_ar_forcing.py` or `pipeline_causal_camera.py` unless the scope is
  explicitly expanded.
- AR long-horizon support is out of first-PR scope because it needs a separate causal/KV cache
  forcing pipeline and likely separate parity gates.
- FastVideo production code must not require `xfuser` or OpenCV just because the
  official reference import needed them. Port camera/action preprocessing and
  sequence-parallel behavior into existing FastVideo-native utilities or keep
  reference-only imports inside local parity tests.
- Review agents should verify setup commands still match the PR, then run the
  listed parity tests or report the exact blocker.


## Quality Regression

Quality regression is added in `fastvideo/tests/ssim/test_dreamx_world_similarity.py`. The default test uses the workspace converted model root, `TORCH_SDPA`, a deterministic generated conditioning image, 9 frames, 1 denoise step, seed 1024, and min SSIM 0.98. A local A40 reference was seeded under `fastvideo/tests/ssim/reference_videos/default/A40_reference_videos/DreamX-World-5B-Cam/TORCH_SDPA/`, and the test passed non-skip locally on 2026-07-01. Full-quality params are present for 480x832, 161 frames, and 30 steps; publishing an L40S/HF reference remains a release operation requiring a write-capable HF token env var, never a raw token value.
