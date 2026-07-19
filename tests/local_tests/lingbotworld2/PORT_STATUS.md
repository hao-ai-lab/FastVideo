# LingBot World 2 Port Status

## Summary

- model_family: `lingbotworld2`
- workload_types: I2V causal-fast
- official_ref: `/mnt/weka/shrd/wm/junda/fv-hub/lingbot-world-v2`
- official_ref_dir: `/mnt/weka/shrd/wm/junda/fv-hub/lingbot-world-v2`
- hf_weights_path: `robbyant/lingbot-world-v2-14b-causal-fast`
- local_weights_dir: `/mnt/weka/shrd/wm/junda/fv-hub/lingbot-world-v2/ckpts/lingbot-world-v2-14b-causal-fast`
- source_layout: `raw_official`
- local_tests_readme: `/mnt/weka/shrd/wm/junda/fv-hub/fastvideo-port-lingbot/tests/local_tests/lingbotworld2/README.md`

## Current Phase

- phase: port verified
- status: complete
- owner: orchestrator
- last_updated: 2026-07-10

## Component Matrix

| Component      | Type      | Reuse/Port | Official Definition                                                                                                                 | Official Instantiation                                                                               | FastVideo Target                                                                                                             | Prototype        | Conversion                       | Parity                    | Open Issues |
| -------------- | --------- | ---------- | ---------------------------------------------------------------------------------------------------                                 | ---------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------- | ---------------- | -------------------------------  | ------------------------- | ----------- |
| Transformer    | DiT       | ported     | `/mnt/weka/shrd/wm/junda/fv-hub/lingbot-world-v2/wan/modules/model_fast.py`                                                         | `/mnt/weka/shrd/wm/junda/fv-hub/lingbot-world-v2/wan/image2video.py`                                 | `/mnt/weka/shrd/wm/junda/fv-hub/fastvideo-port-lingbot/fastvideo/models/dits/lingbotworld2/causal_fast.py`                   | import/meta pass | symlink raw shards               | exact generated parity    | none        |
| VAE            | VAE       | reused     | `/mnt/weka/shrd/wm/junda/fv-hub/lingbot-world-v2/wan/modules/vae2_1.py`                                                             | `/mnt/weka/shrd/wm/junda/fv-hub/lingbot-world-v2/wan/image2video.py`                                 | `/mnt/weka/shrd/wm/junda/fv-hub/fastvideo-port-lingbot/fastvideo/models/vaes/lingbotworld2_wanvae.py`                        | reuse proven     | symlink raw `.pth`               | exact encode/decode pass  | none        |
| Text encoder   | encoder   | ported     | `/mnt/weka/shrd/wm/junda/fv-hub/lingbot-world-v2/wan/modules/t5.py`                                                                 | `/mnt/weka/shrd/wm/junda/fv-hub/lingbot-world-v2/wan/image2video.py`                                 | `/mnt/weka/shrd/wm/junda/fv-hub/fastvideo-port-lingbot/fastvideo/models/encoders/lingbotworld2_t5.py`                        | import pass      | symlink official `.pth` as `.pt` | strict-load pass          | none        |
| Camera utility | generic   | ported     | `/mnt/weka/shrd/wm/junda/fv-hub/lingbot-world-v2/wan/utils/cam_utils.py`                                                            | `/mnt/weka/shrd/wm/junda/fv-hub/lingbot-world-v2/wan/image2video.py`                                 | `/mnt/weka/shrd/wm/junda/fv-hub/fastvideo-port-lingbot/fastvideo/models/dits/lingbotworld2/cam_utils.py`                     | import pass      | stateless                        | covered by pipeline smoke | none        |
| Scheduler      | scheduler | reused     | `FlowUniPCMultistepScheduler` in official sampling loop                                                                             | `/mnt/weka/shrd/wm/junda/fv-hub/lingbot-world-v2/wan/image2video.py`                                 | `/mnt/weka/shrd/wm/junda/fv-hub/fastvideo-port-lingbot/fastvideo/models/schedulers/scheduling_flow_unipc_multistep.py`       | reuse proven     | config emitted                   | exact generated parity    | none        |
| Pipeline       | pipeline  | ported     | `/mnt/weka/shrd/wm/junda/fv-hub/lingbot-world-v2/generate.py`; `/mnt/weka/shrd/wm/junda/fv-hub/lingbot-world-v2/wan/image2video.py` | `/mnt/weka/shrd/wm/junda/fv-hub/lingbot-world-v2/run_fast.sh`                                        | `/mnt/weka/shrd/wm/junda/fv-hub/fastvideo-port-lingbot/fastvideo/pipelines/basic/lingbotworld2/causal_fast_pipeline.py`      | import pass      | model_index emitted              | exact generated parity    | none        |

## Conversion State

- conversion_script: `/mnt/weka/shrd/wm/junda/fv-hub/fastvideo-port-lingbot/scripts/checkpoint_conversion/convert_lingbotworld2_causal_fast.py`
- converted_weights_dir: `/mnt/weka/shrd/wm/junda/fv-hub/lingbot-world-v2/ckpts/lingbot-world-v2-14b-causal-fast-fastvideo`
- source_layout: `raw_official`
- strict_load_status: VAE pass; text encoder pass; transformer full 14B load pass
- passthrough_components: transformer shards, text encoder checkpoint, tokenizer files, raw VAE checkpoint
- retry_history: generic `AutoencoderKLWan` VAE conversion strict-loaded but was not numerically exact; final bundle uses `LingBotWorld2WanVAE`.

## Parity Commands

| Scope                    | Command                                                                                                                                                                                                                                                                                                   | Last Result | Notes                                                                         |
| ------------------------ | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------                                                                              | ----------- | -----------------------------------------------                               |
| Preflight and parity     | `/mnt/weka/shrd/wm/junda/fv-hub/.venv/bin/python -m pytest tests/local_tests/lingbotworld2/test_lingbotworld2_causal_fast_preflight.py tests/local_tests/pipelines/test_lingbotworld2_causal_fast_pipeline_smoke.py tests/local_tests/pipelines/test_lingbotworld2_causal_fast_pipeline_parity.py -q -rs` | pass        | 2026-07-10: 5 passed, 1 skipped; skip is opt-in 14B heavy smoke.              |
| Reference matrix         | `/mnt/weka/shrd/wm/junda/fv-hub/lingbot-world-v2/scripts/run_reference_matrix.sh`                                                                                                                                                                                                                         | pass        | `5` and `9` frame cases recorded as official_unsupported.                     |
| FastVideo matrix         | `/mnt/weka/shrd/wm/junda/fv-hub/.venv/bin/python /mnt/weka/shrd/wm/junda/fv-hub/lingbot-world-v2/scripts/run_fastvideo_matrix.py`                                                                                                                                                                         | pass        | Matches official-supported cases and records `5`/`9` as official_unsupported. |
| Generated video parity   | `/mnt/weka/shrd/wm/junda/fv-hub/.venv/bin/python /mnt/weka/shrd/wm/junda/fv-hub/lingbot-world-v2/scripts/compare_matrix_outputs.py`                                                                                                                                                                       | pass        | 2026-07-09: all 9 supported rows exact; max abs 0, mean abs 0.0.              |
| Heavy FastVideo smoke    | `LINGBOTWORLD2_RUN_HEAVY_SMOKE=1 /mnt/weka/shrd/wm/junda/fv-hub/.venv/bin/python -m pytest tests/local_tests/pipelines/test_lingbotworld2_causal_fast_pipeline_smoke.py -q -rs`                                                                                                                           | pass        | 2026-07-10: 2 passed; 8-GPU 14B latent generation passed in 3:15.             |

## Open Questions

| ID   | Question                                                                                   | Owner        | Needed By Phase | Status   | Resolution                                                             |
| ---- | ------------------------------------------------------------------------------------------ | ------------ | --------------- | -------- | ----------                                                             |
| Q001 | Should final reported matrix include source-impossible requested `5` and `9` frame cases?  | user/author  | final parity    | resolved | Record them as official-source failures; do not change `chunk_size=4`. |

## Issues And Blockers

| ID   | Phase              | Component | Severity | Issue                                                                  | Evidence                                                                                                           | Owner        | Status | Resolution                                                                  |
| ---- | ------------------ | --------- | -------- | ---------------------------------------------------------------------  | ------------------------------------------------------------------------------------------------------------------ | ------------ | ------ | ----------                                                                  |
| I001 | pipeline parity    | pipeline  | high     | Requested `5` and `9` frames cannot produce official reference videos. | `/mnt/weka/shrd/wm/junda/fv-hub/lingbot-world-v2/wan/image2video.py` sets `lat_f=0`, then `F=-3`.                  | orchestrator | closed | Matrix scripts record unsupported cases; supported cases remain `17/33/65`. |

## Escape Hatches

| ID | Phase | Decision Type | Question | Recommended Option | Status | Resolution |
| -- | ----- | ------------- | -------- | ------------------ | ------ | ---------- |

## Decisions

| Date       | Decision                                                                                  | Rationale                                                                                  | Impact                                                                                                                   |
| ---------- | ------------------------------------------------------------------------                  | ------------------------------------------------------------------------------------------ | ------------------------------------------------------------                                                             |
| 2026-07-09 | Keep the FastVideo causal-fast port native and do not import source repo.                 | User required a FastVideo-native model/pipeline implementation.                            | Production code owns DiT, T5 wrapper, camera utilities, and pipeline loop.                                               |
| 2026-07-09 | Preserve official `chunk_size=4` behavior.                                                | `run_fast.sh` does not override chunk size and `generate.py` defaults to `4`.              | Requested `5` and `9` frame cases are recorded as official failures.                                                     |
| 2026-07-09 | Store generated outputs under the LingBot World 2 source tree.                            | User required outputs to stay within the named directories.                                | Matrix outputs live under `/mnt/weka/shrd/wm/junda/fv-hub/lingbot-world-v2/outputs`.                                     |
| 2026-07-09 | Use the exact LingBot World 2 Wan VAE implementation in FastVideo.                        | Generic `AutoencoderKLWan` strict-loaded but decoded output was not numerically exact.     | `LingBotWorld2WanVAE` loads the original `Wan2.1_VAE.pth` and restores exact video parity.                               |
| 2026-07-10 | Use `lingbotworld2` for the causal-fast model family and `LingBotWorld2` for its classes. | FastVideo already has a separate LingBot World v1 family under `lingbotworld`.             | V2 modules, configs, presets, registry entries, tests, converter, dataset, and example no longer share the v1 namespace. |

## Handoff Notes

- The final comparison report is `/mnt/weka/shrd/wm/junda/fv-hub/lingbot-world-v2/outputs/verification/matrix_compare.md`.
- Exact generated-output parity is only meaningful for cases where the official reference writes a video.
- Do not loosen parity by changing `chunk_size` unless the user explicitly changes the reference config.
