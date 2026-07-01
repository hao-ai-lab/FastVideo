# DreamX World Port Status

## Summary

- model_family: `dreamx_world`
- workload_types: `I2V camera-control compatibility shim`
- official_ref: `https://github.com/AMAP-ML/DreamX-World`
- official_ref_dir: `DreamX-World/`
- hf_weights_path: `GD-ML/DreamX-World-5B-Cam`
- local_weights_dir: `official_weights/dreamx_world`
- source_layout: `raw_official`
- local_tests_readme: `tests/local_tests/dreamx_world/README.md`

## Current Phase

- phase: `phase_11_post_parity_handoff`
- status: `complete`
- owner: `orchestrator`
- last_updated: `2026-07-01`

## Component Matrix

| Component | Type | Reuse/Port | Official Definition | Official Instantiation | FastVideo Target | Prototype | Conversion | Parity | Open Issues |
|---|---|---|---|---|---|---|---|---|---|
| transformer | dit | ported_dedicated | `DreamX-World/models/wan_transformer3d.py`; PRoPE helpers in `DreamX-World/models/prope_utils.py` | `DreamX-World/inference_dreamx5b.py::setup_models`, `Wan2_2Transformer3DModel.from_pretrained(... cam_method=prope, add_control_adapter=True)` | `fastvideo/models/dits/dreamx_world.py`; `fastvideo/configs/models/dits/dreamx_world.py`; DreamX pipeline config helper | native_prope_pass | real_conversion_pass | strict_load_and_forward_parity_pass | none |
| vae | vae | reuse_pending | `DreamX-World/models/wan_vae3_8.py` | `DreamX-World/inference_dreamx5b.py::setup_models`, `AutoencoderKLWan3_8.from_pretrained(Wan2.2_VAE.pth)` | `fastvideo/models/vaes/wanvae.py`; DreamX VAE config helper | config_smoke_pass | raw_key_mapping_pass | encode_parity_pass | none |
| text_encoder/tokenizer | encoder | reuse_pending | `DreamX-World/models/wan_text_encoder.py`; tokenizer via Wan2.2 base model | `DreamX-World/inference_dreamx5b.py::setup_models`, `WanT5EncoderModel` + tokenizer subpaths | `fastvideo/models/encoders/t5.py::UMT5EncoderModel`; DreamX UMT5 config helper | config_smoke_pass | staged_weight_load_pass | hidden_state_parity_pass | none |
| scheduler | generic | reuse_proven | Diffusers `FlowMatchEulerDiscreteScheduler` | `DreamX-World/inference_dreamx5b.py::setup_models`, default `sampler_name=Flow` | `fastvideo/models/schedulers/scheduling_flow_match_euler_discrete.py` | pass | not_required | non_skip_pass | Q003 |
| camera_conditioning | generic | port_pending | `DreamX-World/utils/inference_utils.py`, `DreamX-World/models/prope_utils.py`, `DreamX-World/wan/modules/camera_prope.py` | `DreamX-World/inference_dreamx5b.py::get_camera_sequence`, `pipeline(... control_camera_video=...)` | `fastvideo/pipelines/basic/dreamx_world/camera_conditioning.py` | pass | not_required | non_skip_pass | none |
| pipeline | pipeline | port_complete | `DreamX-World/pipeline/pipeline_dreamxworld.py` | `DreamX-World/inference_dreamx5b.py::process_inference_from_json` | `fastvideo/pipelines/basic/dreamx_world/` plus config/preset/registry | pipeline_load_generate_smoke_pass | model_index_and_config_consistency_smoke_pass | pipeline_api_vs_worker_forward_parity_pass | none |

## Conversion State

- conversion_script: `scripts/checkpoint_conversion/dreamx_world_to_diffusers.py`
- converted_weights_dir: `converted_weights/dreamx_world`
- source_layout: `raw_official`
- strict_load_status: `pass`
- conversion_script_status: `transformer_model_index_and_config_consistency_smoke_pass`
- model_index_status: `smoke_pass`
- passthrough_components: `Wan2.2 Diffusers scheduler, tokenizer, and text encoder are symlinked from official_weights/Wan2.2-TI2V-5B-Diffusers; VAE parity uses raw Wan2.2_VAE.pth with an explicit DreamX raw-to-FastVideo key mapper because official encode returns normalized latents.`
- retry_history: `none`

## Parity Commands

| Scope | Command | Last Result | Notes |
|---|---|---|---|
| transformer | `python -m pytest tests/local_tests/dreamx_world/test_dreamx_world_transformer_parity.py -v -s` | strict_load_and_forward_parity_pass | 2026-07-01: converted real 5B-Cam transformer shards strict-load into dedicated `DreamXWorldTransformer3DModel` with 0 shape mismatches; official-vs-FastVideo small-input fp32 forward parity passes on CUDA (`diff_max=0.072533`, `diff_mean=0.008014`). |
| vae | `python -m pytest tests/local_tests/dreamx_world/test_dreamx_world_vae_parity.py -v -s` | encode_parity_pass | 2026-06-30: official DreamX raw `Wan2.2_VAE.pth` maps 196/196 keys into FastVideo Wan VAE and encode parity passes after applying the same official latent normalization (`(mu - mean) / std`). |
| text_encoder | `python -m pytest tests/local_tests/dreamx_world/test_dreamx_world_text_encoder_parity.py -v -s` | hidden_state_parity_pass | 2026-06-30: official `WanT5EncoderModel` vs FastVideo `UMT5EncoderModel` hidden-state parity passes on CUDA using staged Wan2.2 text encoder/tokenizer weights and reference-only `xfuser` stubs. |
| scheduler | `python -m pytest tests/local_tests/dreamx_world/test_dreamx_world_scheduler_parity.py -v -s` | non_skip_pass | 2026-06-30: FastVideo FlowMatch scheduler matches official Diffusers timesteps and step output for DreamX default Flow sampler; `DreamXWorldPipeline` initializes FlowMatch with official `shift=3.0`. |
| camera_conditioning | `python -m pytest tests/local_tests/dreamx_world/test_dreamx_world_camera_conditioning_parity.py -v -s` | non_skip_pass | 2026-06-29: 3 parameterized cases passed against official reference on CPU. |
| pipeline_config | `python -m pytest tests/local_tests/dreamx_world/test_dreamx_world_pipeline_config.py -v -s` | pipeline_entry_preset_scheduler_modelinfo_and_camera_stage_smoke_pass | DreamX 5B-Cam PipelineConfig wires DiT/VAE/UMT5/Flow/TI2V settings and official `shift=3.0`; default preset is registered for `GD-ML/DreamX-World-5B-Cam`; local converted-style `model_index.json` resolves to `DreamXWorldPipeline`; the pipeline initializes FlowMatch, camera conditioning writes `batch.extra["dreamx_y_camera"]`, and generic denoising can pass it as `y_camera`. |
| pipeline_smoke | `python -m pytest tests/local_tests/pipelines/test_dreamx_world_pipeline_smoke.py tests/local_tests/pipelines/test_dreamx_world_pipeline_parity.py -q -rs` | 4_passed_0_skipped | 2026-06-30: combined smoke/parity passed. 2026-07-01: smoke alone passed with real `image_path` TI2V coverage (`3 passed`), validating image load, TI2V preprocessing, VAE first-frame encode under CPU offload, camera conditioning, and 1-step latent generation from `converted_weights/dreamx_world`. Tests force `FASTVIDEO_ATTENTION_BACKEND=TORCH_SDPA` to avoid the local FlashAttention-4 cute ABI mismatch. |
| basic_example | `PYTHONPATH=/workspace/FastVideo FASTVIDEO_ATTENTION_BACKEND=TORCH_SDPA DREAMX_WORLD_MODEL_DIR=converted_weights/dreamx_world DREAMX_WORLD_IMAGE_PATH= DREAMX_WORLD_HEIGHT=64 DREAMX_WORLD_WIDTH=64 DREAMX_WORLD_NUM_FRAMES=9 DREAMX_WORLD_STEPS=1 DREAMX_WORLD_GUIDANCE=1.0 DREAMX_WORLD_OUTPUT_PATH=outputs_video/dreamx_world_example_smoke python examples/inference/basic/basic_dreamx_world.py` | generated_video_pass | 2026-06-30: example saved an MP4 under `outputs_video/dreamx_world_example_smoke`; imageio/ffmpeg decoded frame 0 as `(64, 64, 3)` uint8, fps 16, duration 0.56s. |

## Open Questions

| ID | Question | Owner | Needed By Phase | Status | Resolution |
|---|---|---|---|---|---|
| Q001 | Should first PR expose only the `DreamX-World-5B-Cam` 5s camera-control mode and exclude AR long-horizon forcing? | user | prep | resolved | User approved starting with `DreamX-World-5B-Cam`; AR long-horizon is out of first-PR scope. |
| Q002 | Does FastVideo's existing Wan2.2 TI2V transformer support DreamX PRoPE/control adapter with a small extension, or is a DreamX-specific DiT required? | component:transformer | Phase 3 | resolved | Project guidance prefers a separate DreamX DiT for maintainability. DreamX PRoPE/control adapter now lives in `fastvideo/models/dits/dreamx_world.py`; Wan DiT/config have no DreamX-specific fields or `y_camera` signature. |
| Q003 | Which sampler is in first-PR scope: official default `Flow` only, or also `Flow_Unipc` and `Flow_DPM++`? | orchestrator | Phase 3 | resolved | First PR should support official default `Flow` only. FastVideo FlowMatch scheduler parity is non-skip PASS; optional `Flow_Unipc` and `Flow_DPM++` are out of first-PR scope. |
| Q004 | Which HF token env var should be used if rate limits or gated Wan2.2 base weights require auth? | user | Phase 5 | resolved | No auth was required for the completed local downloads; keep using env var names only if future gated repos require auth. |
| Q005 | Should native FastVideo production code depend on DreamX reference-only packages such as `xfuser` or OpenCV? | user | Phase 3 | resolved | No. These packages may be used only for official reference/local parity setup; native FastVideo integration must remove that runtime requirement. |

## Issues And Blockers

| ID | Phase | Component | Severity | Issue | Evidence | Owner | Status | Resolution |
|---|---|---|---|---|---|---|---|---|
| I001 | prep | official_env | medium | Official import initially failed because `xfuser` was missing. | `ModuleNotFoundError: No module named 'xfuser'` from `python -c "import sys; sys.path.insert(0, 'DreamX-World'); import inference_dreamx5b"` | prep | resolved | Installed `xfuser==0.4.1`; import progressed. |
| I002 | prep | official_env | medium | Official import then failed because GUI OpenCV required missing system `libxcb.so.1`. | `ImportError: libxcb.so.1: cannot open shared object file` through `cv2` import in Diffusers ConsisID path. | prep | resolved | Installed `opencv-python-headless`; `import inference_dreamx5b` passed. |
| I003 | prep | weights | medium | HF repo has raw official transformer shards and no Diffusers `model_index.json`. | `inspect_hf_layout.py GD-ML/DreamX-World-5B-Cam --json` returned `source_layout=raw_official`, `needs_conversion=yes`, `model_index_class=null`. | conversion | resolved | Downloaded raw DreamX shards to `official_weights/dreamx_world`; converted transformer to `converted_weights/dreamx_world/transformer`; symlinked reusable Wan2.2 Diffusers components; real 5B transformer strict-load passes. |
| I004 | prep | dependencies | high | Official reference import required extra packages in the local environment, but FastVideo native runtime should not inherit those dependencies. | `xfuser==0.4.1` and `opencv-python-headless` were installed only to make `DreamX-World/inference_dreamx5b.py` import for reference/parity. | pipeline | resolved | Production DreamX FastVideo code uses native camera/image/video utilities and has no runtime `xfuser` or OpenCV import requirement; those packages remain reference-only local parity dependencies. |
| I005 | parity | transformer | medium | Transformer full forward parity initially failed in bf16 official harness. | Official CUDA bf16 LayerNorm path was unstable; fp32 small-input harness avoids that dtype issue and compares against FastVideo with single-process SP identity patches. | component:transformer | resolved | Official-vs-FastVideo forward parity now passes on CUDA with `diff_max=0.072533`, `diff_mean=0.008014`. |
| I006 | parity | vae/text_encoder | medium | VAE/text parity initially remained skipped after weights were staged. | Text official import needed a reference-only `xfuser` stub; VAE comparison initially used raw official normalized latents against FastVideo raw mu. | component:vae,component:text_encoder | resolved | Text hidden-state parity passes. VAE encode parity passes after raw key mapping and applying the official latent normalization to FastVideo output. |
| I007 | quality | pipeline_ti2v | medium | Real image-path TI2V smoke initially failed when `vae_cpu_offload=True` because DenoisingStage encoded the first frame while the VAE weights remained on CPU. | DreamX SSIM first run failed with `RuntimeError: Input type (torch.cuda.FloatTensor) and weight type (torch.FloatTensor) should be the same` at `fastvideo/pipelines/stages/denoising.py` VAE encode. | pipeline | resolved | DenoisingStage now moves the VAE to `local_device` before TI2V first-frame encode; image-path pipeline smoke and DreamX SSIM both pass. |

## Escape Hatches

| ID | Phase | Decision Type | Question | Recommended Option | Status | Resolution |
|---|---|---|---|---|---|---|

## Decisions

| Date | Decision | Rationale | Impact |
|---|---|---|---|
| 2026-06-29 | First PR scope is `DreamX-World-5B-Cam` only. | Cam mode is closest to existing Wan2.2 TI2V support; AR forcing needs separate causal/KV pipeline work. | Component inventory and parity focus on `inference_dreamx5b.py` and `pipeline_dreamxworld.py`. |
| 2026-06-29 | Do not install full DreamX requirements during prep. | Full requirements pin core FastVideo stack packages. | Installed only `xfuser==0.4.1` and `opencv-python-headless` to make official imports work. |
| 2026-06-29 | Treat HF DreamX-World-5B-Cam weights as raw official transformer layout requiring conversion. | HF inspection found no `model_index.json`. | Phase 5 must create `scripts/checkpoint_conversion/dreamx_world_to_diffusers.py` after component prototype/key dumps. |
| 2026-06-29 | Do not add DreamX reference-only dependencies to FastVideo production requirements. | The current environment should remain the FastVideo environment; extra packages are only for official reference parity. | Native DreamX integration must avoid runtime `xfuser` and OpenCV requirements unless explicitly approved later. |
| 2026-06-29 | Implement DreamX camera conditioning as native FastVideo utility. | It is weightless and removes the need to import DreamX reference utilities at production runtime. | `fastvideo/pipelines/basic/dreamx_world/camera_conditioning.py` now has non-skip parity against official action-to-PRoPE tensors. |
| 2026-06-29 | First PR supports DreamX default `Flow` sampler only. | FastVideo FlowMatch Euler scheduler matches the official Diffusers scheduler for DreamX defaults. | Pipeline work can use FastVideo native FlowMatch scheduler; UniPC and DPM++ are out of first-PR scope. |
| 2026-07-01 | Keep DreamX PRoPE/control adapter in a dedicated DreamX DiT class. | Project guidance is that putting too much DreamX behavior into Wan makes the model hard to manage. | `fastvideo/models/dits/dreamx_world.py` defines `DreamXWorldTransformer3DModel`, `DreamXWorldTransformerBlock`, and `DreamXPropeSelfAttention`; `fastvideo/configs/models/dits/dreamx_world.py` owns DreamX adapter config fields; Wan DiT/config are unchanged from DreamX. |
| 2026-06-30 | Camera parity test loads official camera functions by file instead of importing the official `utils` package. | Official package initialization pulls unrelated dependencies that can require GUI OpenCV system libraries. | Camera parity remains non-skip without adding DreamX reference-only dependencies to FastVideo production requirements. |
| 2026-06-30 | Add DreamX-World-5B-Cam model and pipeline config helpers plus a conversion script. | Official HF DreamX 5B-Cam transformer config is 30 layers, hidden size 3072, 24 heads, 48 latent channels, plus Wan2.2 48-channel VAE and UMT5-XXL text encoder. | DreamX helpers wire DiT/VAE/UMT5/Flow/TI2V settings; `dreamx_world_to_diffusers.py` writes a FastVideo-loadable transformer config plus renamed safetensors; strict-load smoke passes on a tiny official DreamX transformer and the real 5B converted shards. |
| 2026-06-30 | Pass DreamX camera PRoPE condition through the FastVideo batch/denoising path. | DreamX transformer expects `y_camera={"viewmats", "K"}` at denoising time. | `DreamXWorldPipeline` is registered as a basic pipeline entry and initializes the official default FlowMatch scheduler; `dreamx_world_5b_cam` preset mirrors official 5B-Cam defaults; `DreamXWorldCameraConditioningStage` writes `batch.extra["dreamx_y_camera"]`; generic denoising filters and forwards it as `y_camera` only for compatible transformers. |

## Handoff Notes

- Prep, component parity, pipeline smoke/parity, and the basic example validation are complete for `DreamX-World-5B-Cam`.
- Official reference clone is staged at `DreamX-World/` and ignored by git.
- Workspace-local weights are staged: DreamX raw transformer shards under `official_weights/dreamx_world`, Wan2.2 raw base artifacts under `official_weights/Wan2.2-TI2V-5B`, and Wan2.2 Diffusers reusable components under `official_weights/Wan2.2-TI2V-5B-Diffusers`.
- Camera conditioning parity is active and passing without weights.
- Default Flow scheduler parity is active and passing without weights.
- Transformer has corrected official 5B-Cam architecture in dedicated DreamX DiT/config files, native PRoPE/control-adapter, conversion mapping, real converted 5B strict-load, and official-vs-FastVideo forward parity passing on CUDA. VAE encode parity and text hidden-state parity pass on CUDA. Pipeline entry/registry, local model_info resolution, preset, config, FlowMatch scheduler init, camera stage, denoising `y_camera` kwarg smokes, independent CUDA pipeline smoke/parity, and a small saved-video basic example pass.
- Full local DreamX component suite is non-skip PASS: `python -m pytest tests/local_tests/dreamx_world/ -q -rs` returned `26 passed` on 2026-07-01. Pipeline smoke/parity and SSIM quality regression are also non-skip PASS locally.
- Keep `xfuser` and OpenCV as reference-only parity dependencies. Do not add them
  to FastVideo requirements or production imports.


## Quality Regression

- status: `added`
- test: `fastvideo/tests/ssim/test_dreamx_world_similarity.py`
- command: `PYTHONPATH=/workspace/FastVideo FASTVIDEO_SSIM_MODEL_ID=DreamX-World-5B-Cam python -m pytest fastvideo/tests/ssim/test_dreamx_world_similarity.py -q -rs`
- result: `1 passed, 0 skipped` on 2026-07-01
- reference: Local A40/TORCH_SDPA reference seeded from the generated candidate under `fastvideo/tests/ssim/reference_videos/default/A40_reference_videos/DreamX-World-5B-Cam/TORCH_SDPA/`. The test uses a deterministic generated input image, 64x64 request dimensions, 9 frames, 1 denoise step, seed 1024, and min SSIM 0.98. Full-quality params are present for 480x832/161 frames/30 steps.
- note: HF upload/Modal L40S seeding remains a release operation requiring `HF_API_KEY`, `HUGGINGFACE_HUB_TOKEN`, or `HF_TOKEN` with write access; no token values were used or recorded.

## Final Handoff

```text
final_handoff:
  prep_handoff_complete: yes
  conversion_status: pass
  components:
    - name: transformer
      reuse_or_port: ported_dedicated_dit
      parity_test: tests/local_tests/dreamx_world/test_dreamx_world_transformer_parity.py
      parity_status: non_skip_pass
      concerns_or_unknowns: none
    - name: vae
      reuse_or_port: reused
      parity_test: tests/local_tests/dreamx_world/test_dreamx_world_vae_parity.py
      parity_status: non_skip_pass
      concerns_or_unknowns: none
    - name: text_encoder_tokenizer
      reuse_or_port: reused
      parity_test: tests/local_tests/dreamx_world/test_dreamx_world_text_encoder_parity.py
      parity_status: non_skip_pass
      concerns_or_unknowns: none
    - name: scheduler
      reuse_or_port: reused
      parity_test: tests/local_tests/dreamx_world/test_dreamx_world_scheduler_parity.py
      parity_status: non_skip_pass
      concerns_or_unknowns: none
    - name: camera_conditioning
      reuse_or_port: ported
      parity_test: tests/local_tests/dreamx_world/test_dreamx_world_camera_conditioning_parity.py
      parity_status: non_skip_pass
      concerns_or_unknowns: none
  pipeline_smoke: pass
  pipeline_parity: pass
  example_status: pass
  quality_regression: added
  local_tests_readme: tests/local_tests/dreamx_world/README.md
  port_state_file: tests/local_tests/dreamx_world/PORT_STATUS.md
  token_values_committed: no
  runtime_third_party_model_imports: none
  blockers: none
  escape_hatch: none
```
