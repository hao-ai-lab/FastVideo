# LingBot-Video Port Status

## Summary

- model_family: `lingbot_video`
- workload_types: T2V first, then T2I and TI2V (`WorkloadType.I2V`)
- official_ref: `https://github.com/robbyant/lingbot-video` at `a638721cf2271804d02738b69f2ad788c4a559fc`
- official_weights: `robbyant/lingbot-video-dense-1.3b`; `robbyant/lingbot-video-moe-30b-a3b`
- fastvideo_weights: `FastVideo/LingBot-Video-Dense-1.3B-Diffusers`; `FastVideo/LingBot-Video-MoE-30B-A3B-Diffusers`
- source_layout: `official custom-code Diffusers; converted FastVideo packages published on Hugging Face`
- converted_model_index_classes: `LingBotVideoDensePipeline`; `LingBotVideoMoePipeline`
- local_tests_readme: `tests/local_tests/lingbot_video/README.md`

## Current Phase

- phase: `t2v_complete`
- status: `complete`
- owner: `pipeline`
- last_updated: `2026-07-13`

## Component Matrix

| Component        | Type      | Reuse/Port | FastVideo Target                                      | Prototype | Conversion  | Parity                         | Open Issues            |
| ---------------- | --------- | ---------- | ----------------------------------------------------- | --------- | ----------- | ------------------------------ | ---------------------- |
| Dense DiT        | DiT       | port       | `fastvideo/models/dits/lingbot_video.py`              | pass      | pass        | production exact pass          | none                   |
| MoE DiT          | DiT       | port       | `fastvideo/models/dits/lingbot_video.py`              | pass      | pass        | full 48-block exact pass       | none                   |
| MoE DiT refiner  | DiT       | port       | `fastvideo/models/dits/lingbot_video.py`              | pass      | pass        | full 48-block exact pass       | none                   |
| Qwen3-VL text    | encoder   | reuse/port | `fastvideo/models/encoders/lingbot_video.py`          | pass      | pass        | production exact pass          | none                   |
| Qwen3-VL vision  | encoder   | deferred   | `fastvideo/models/encoders/lingbot_video.py`          | deferred  | not_started | outside current T2V scope      | TI2V phase             |
| Wan VAE          | VAE       | reuse      | `fastvideo/models/vaes/wanvae.py`                     | pass      | passthrough | non-skip pass (`2/2`)          | none                   |
| Flow-UniPC       | scheduler | reuse      | `fastvideo/models/schedulers/`                        | pass      | not_needed  | exact schedule/update pass     | none                   |
| Base pipeline    | pipeline  | port       | `fastvideo/pipelines/basic/lingbot_video/`            | pass      | pass        | Dense exact; MoE smoke pass    | none                   |
| Refiner pipeline | pipeline  | port       | `fastvideo/pipelines/basic/lingbot_video/`            | pass      | pass        | FSDP plus SP=8 smoke pass      | visual caveat accepted |

## Checkpoint Conversion

The official checkpoints cannot be loaded directly by native FastVideo because
their folder layout and text-encoder format are different. The conversion
script creates FastVideo-compatible packages and does not modify the official
downloads. Local tests use the published packages below; developers do not need
to run conversion before testing.

- Script: `scripts/checkpoint_conversion/lingbot_video_to_diffusers.py`

| Variant       | Official input repository                  | Published FastVideo repository                              |
| ------------- | ------------------------------------------ | ----------------------------------------------------------- |
| Dense 1.3B    | `robbyant/lingbot-video-dense-1.3b`        | `FastVideo/LingBot-Video-Dense-1.3B-Diffusers`              |
| MoE + refiner | `robbyant/lingbot-video-moe-30b-a3b`       | `FastVideo/LingBot-Video-MoE-30B-A3B-Diffusers`             |

What changes:

- Text encoder conversion: `scripts/checkpoint_conversion/lingbot_video_to_diffusers.py`
  converts the official Qwen3-VL checkpoint into the format loaded by FastVideo's
  `LingBotVideoQwen3VLTextModel`. The official checkpoint is multimodal: it
  contains both a vision encoder and a language model. The current port supports
  text-to-video generation, so the converter keeps the language-model weights and
  omits the unused vision-encoder weights. Within each language-model layer,
  "projection weights" are the learned matrices used by the attention and MLP
  computations. Qwen3-VL stores the attention matrices separately as `q_proj`,
  `k_proj`, and `v_proj`; FastVideo reads the same values stacked into one
  `qkv_proj` tensor. Qwen3-VL likewise stores `gate_proj` and `up_proj` separately;
  FastVideo reads them stacked into one `gate_up_proj` tensor. This conversion
  changes only the names and grouping of the retained weights. It does not
  quantize, retrain, approximate, or otherwise change their numerical values.

- The official `processor/` files are copied to `tokenizer/`.
- For MoE, the official `refiner/` is exposed as FastVideo's `transformer_2/`.
- A new `model_index.json` identifies the package as Dense or MoE/refiner. Both
  packages use the shared `LingBotVideoPipeline` implementation.

What does not change:

- The Dense, MoE, and refiner transformer weights are reused without changing
  their tensor values.
- The Wan VAE and scheduler are reused without weight conversion.
- Safetensors files are hard-linked where possible; configuration and tokenizer
  files are copied.

Related files:

- Published package metadata: each `FastVideo/*-Diffusers` repository's `model_index.json`
- Dense key mappings: `checkpoints/lingbot-video/mapping/dense/`
- Conversion layout test: `tests/local_tests/lingbot_video/test_conversion_layout.py`
- Numerical loading and output checks: see **Validation Results** below.

## Validation Results

The results below separate three questions:

1. Did the test run successfully?
2. Did FastVideo exactly match the original LingBot-Video implementation?
3. Which command and Slurm job provide the evidence?

"Not tested" means that the check did not compare FastVideo with the original
implementation. It does not mean that the check failed.

### Numerical parity with the original implementation

These tests ran the same inputs through FastVideo and the original
LingBot-Video implementation.

| What was tested               | Successful?    | Exact match with original?                          | Evidence                                                                                        |
| ----------------------------- | -------------- | --------------------------------------------------- | ----------------------------------------------------------------------------------------------- |
| Dense denoising model         | Yes            | Yes                                                 | `scripts/slurm/lingbot_video_dense_dit_parity.sbatch`, Slurm `1859061`                          |
| Dense VAE encode and decode   | Yes: 2 tests   | No exact claim; passed with `atol=rtol=0.05`        | `scripts/slurm/lingbot_video_dense_vae_parity.sbatch`, Slurm `1859017`                          |
| Dense text encoder            | Yes: 2 tests   | Yes for encoder output; the other test checks setup | `scripts/slurm/lingbot_video_dense_text_encoder_parity.sbatch`, Slurm `1859116`                 |
| Scheduler                     | Yes: 2 tests   | Yes                                                 | `tests/local_tests/schedulers/test_lingbot_video_scheduler_parity.py`                           |
| Dense pipeline, one step      | Yes            | Yes                                                 | `scripts/slurm/lingbot_video_dense_pipeline_parity.sbatch`, Slurm `1859069`                     |
| Dense pipeline, two steps     | Yes            | Yes                                                 | `scripts/slurm/lingbot_video_dense_pipeline_multistep_parity.sbatch`, Slurm `1859076`           |
| Dense pipeline on two GPUs    | Yes            | Yes, using the same mathematical attention backend  | `scripts/slurm/lingbot_video_dense_pipeline_sp_parity.sbatch`, Slurm `1859099`                  |
| One MoE transformer block     | Yes            | Yes                                                 | `scripts/slurm/lingbot_video_moe_block_parity.sbatch`, Slurm `1859113`                          |
| Complete MoE base transformer | Yes            | Yes                                                 | `scripts/slurm/lingbot_video_moe_dit_parity.sbatch`, Slurm `1859114`                            |
| Complete MoE refiner          | Yes            | Yes                                                 | `scripts/slurm/lingbot_video_moe_refiner_dit_parity.sbatch`, Slurm `1859145`                    |

### Functional tests without an original-output comparison

These tests checked that FastVideo could load and run the port correctly. They
were not numerical comparisons with the original implementation.

| What was tested                         | Successful?  |  Evidence                                                                                   |
| --------------------------------------- | ------------ |  ------------------------------------------------------------------------------------------ |
| Dense pipeline on one GPU               | Yes: 4 tests |  `scripts/slurm/lingbot_video_dense_pipeline_smoke.sbatch`, Slurm `1859073`                 |
| Dense pipeline with sharded model state | Yes          |  `scripts/slurm/lingbot_video_dense_pipeline_fsdp_smoke.sbatch`, Slurm `1859075`            |
| Mixed-precision sharded model loading   | Yes: 2 tests |  `scripts/slurm/lingbot_video_mixed_dtype_fsdp_test.sbatch`, Slurm `1859072`                |
| Dense pipeline on two GPUs              | Yes          |  `scripts/slurm/lingbot_video_dense_pipeline_sp_smoke.sbatch`, Slurm `1859089`              |
| MoE base pipeline on one GPU            | Yes          |  `scripts/slurm/lingbot_video_moe_pipeline_smoke.sbatch`, Slurm `1859121`                   |
| MoE base plus refiner on eight GPUs     | Yes          |  `scripts/slurm/lingbot_video_refiner_pipeline_smoke.sbatch`, Slurm `1859122`               |

### Local API and CPU regression tests

This was a selected local test suite, not an original-versus-FastVideo parity
test. It checked API parsing and presets, checkpoint conversion layout,
mixed-precision loading, pipeline setup, refiner stages, scheduler behavior,
and MoE routing. The result was `127 passed, 3 skipped`.

The three skipped tests require allocated GPUs: Dense pipeline generation, MoE
pipeline generation, and mixed-precision sharded execution. Their corresponding
GPU runs are recorded in the functional-test table above.

The exact local command was:

```bash
python -m pytest -q \
  fastvideo/tests/api/test_compat_translation.py \
  fastvideo/tests/api/test_parser.py \
  fastvideo/tests/api/test_presets.py \
  fastvideo/tests/entrypoints/test_video_generator.py \
  tests/local_tests/lingbot_video/test_conversion_layout.py \
  tests/local_tests/models/test_fsdp_load_mixed_dtype.py \
  tests/local_tests/pipelines/test_lingbot_video_moe_pipeline_smoke.py \
  tests/local_tests/pipelines/test_lingbot_video_pipeline_smoke.py \
  tests/local_tests/pipelines/test_lingbot_video_refiner_stages.py \
  tests/local_tests/schedulers/test_lingbot_video_scheduler_parity.py \
  tests/local_tests/transformers/test_lingbot_video_moe.py
```

### Generation and performance runs

These runs produced final videos or performance measurements. They are not
original-versus-FastVideo numerical parity tests.

| What was run                 | Successful? |  Evidence                                                                                        |
| ---------------------------- | ----------- |  ----------------------------------------------------------------------------------------------- |
| Dense final video            | Yes         |  `scripts/slurm/lingbot_video_dense_final_generation.sbatch`, Slurm `1859166`                    |
| MoE plus refiner final video | Yes         |  `scripts/slurm/lingbot_video_moe_refiner_final_generation.sbatch`, Slurm `1859167`              |
| Dense performance benchmark  | Yes         |  `scripts/slurm/lingbot_video_dense_benchmark.sbatch`, Slurm `1859117`                           |
| Dense performance profile    | Yes         |  `scripts/slurm/lingbot_video_dense_nsys_profile.sbatch`, Slurm `1859106`                        |

## Open Questions

| ID   | Question                                                                                 | Owner            | Needed By Phase | Status   | Resolution                                                  |
| ---- | ---------------------------------------------------------------------------------------- | ---------------- | --------------- | -------- | ----------------------------------------------------------- |
| Q001 | Do existing FastVideo Wan VAE and Flow-UniPC classes match exact official instantiation? | component owners | reuse gate      | resolved | VAE encode/decode GPU parity and scheduler CPU parity pass. |

## Issues And Blockers

| ID   | Phase     | Component | Severity | Issue                                                                                        | Evidence                                                                                          | Owner      | Status   | Resolution                                                                                                                     |
| ---- | --------- | --------- | -------- | -------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------- | ---------- | -------- | ------------------------------------------------------------------------------------------------------------------------------ |
| I001 | prep      | runtime   | medium   | Official recommends newer Torch/CUDA than shared venv.                                       | All parity and production smoke jobs pass on Torch 2.11/cu128.                                    | prep       | resolved | No dependency changes were required.                                                                                           |
| I002 | parity    | Dense DiT | medium   | GPU numerical parity had not run.                                                            | Initial implementation had an exact 377-key/shape surface.                                        | parity     | resolved | Production-loaded output is exact in Slurm `1859061`.                                                                          |
| I003 | parity    | Dense DiT | medium   | An all-true mask selected a different bf16 SDPA path.                                        | Slurm `1859029` first drifted at block-1 attention.                                               | components | resolved | Only build a mask for real padding; production parity is exact in Slurm `1859061`.                                             |
| I004 | parity    | Qwen3-VL  | medium   | Generic Qwen3 RoPE and causal-mask arithmetic differed from Qwen3-VL.                        | Slurm `1859030` first drifted at layer-0 `o_proj`.                                                | components | resolved | LingBot specialization matches bf16 RoPE and causal GQA; Slurm `1859068` is exact.                                             |
| I005 | pipeline  | SDPA      | medium   | Official and native imports could select different cuDNN SDPA state.                         | The one-step full-pipeline comparison drifted before alignment.                                   | pipeline   | resolved | Match the backend state for both paths and restore it; Slurm `1859069` is exact.                                               |
| I006 | inference | FSDP      | medium   | Mixed fp32/bf16 parameters cannot share one FSDP parameter group.                            | Nested child/root inference coverage passes in Slurm `1859072`.                                   | loader     | resolved | Preserve mixed dtypes and exclude fp32 parameters from inference FSDP; reject training use.                                    |
| I007 | parity    | SP SDPA   | medium   | Head sharding selected a different optimized bf16 Torch SDPA kernel.                         | Jobs `1859090`/`1859098`: drift `0.01412875`/`0.10981119`.                                        | pipeline   | resolved | Math SDPA on both paths is exactly zero in Slurm `1859099`, validating SP movement/masking.                                    |
| I008 | benchmark | text      | high     | Temporary Qwen replacement retained 300 unreachable parameters totaling 7,424,306,688 bytes. | Slurm `1859109`: native steady allocation was 18,747,731,456 bytes, 6,451,255,808 above official. | encoder    | resolved | Construct final layers directly; Slurm `1859117` native steady allocation is 11,319,211,520 bytes, 977,264,128 below official. |
| I009 | registry  | discovery | low      | One shared model-index identity made converted local MoE discovery ambiguous.                | Pre-fix jobs `1859121`/`1859122` logged both registry IDs.                                        | registry   | resolved | Emit distinct Dense/MoE `_class_name` identities and map both to the shared native runtime.                                    |
| I010 | refiner   | workflow  | low      | Official CLI round-trips base pixels through MP4 before refinement.                          | `lingbot_video/runner.py` saves, reloads, then VAE-encodes.                                       | pipeline   | accepted | Native composition refines the decoded tensor in memory; document that pixels need not match.                                  |
| I011 | final     | visual    | low      | Canonical MoE/refiner output has pronounced blurred pillarbox-like side regions.             | Slurm `1859167` contact sheet; bytes match Slurm `1859144`.                                       | model      | accepted | Record the composition caveat; geometry, decode, stability, and component correctness pass.                                    |

> FastVideo disables PyTorch’s cuDNN attention backend during CUDA-platform initialization, while the original implementation may leave it enabled. The two implementations therefore used different bfloat16 attention kernels during the initial pipeline comparison,
> creating small rounding differences. The parity test now applies the same backend settings to both implementations and restores the previous process settings afterward.

> The original implementation omits the attention mask when every token is valid. FastVideo initially passed an all-true mask, which is mathematically equivalent but caused PyTorch to choose a different optimized bfloat16 attention kernel. The different rounding broke
> exact parity. FastVideo now creates a mask only when the input contains real padding.

## Escape Hatches

An escape hatch is a documented exception used when the planned implementation
or validation cannot be completed as intended. Each entry records the blocker,
the proposed alternative or reduced scope, whether that alternative was
approved, and its consequences. An empty table means that this port did not use
any escape hatches.

| ID  | Phase | Decision Type | Question | Recommended Option | Status | Resolution |
| --- | ----- | ------------- | -------- | ------------------ | ------ | ---------- |

## Decisions

| Date       | Decision                                                    | Rationale                                                      | Impact                                      |
| ---------- | ----------------------------------------------------------- | -------------------------------------------------------------- | ------------------------------------------- |
| 2026-07-11 | Isolate this port from both prior worktrees.                | User requirement.                                              | Only current main and pinned official code. |
| 2026-07-11 | Port Dense completely before beginning MoE implementation.  | User-requested order.                                          | Dense parity is the MoE prerequisite.       |
| 2026-07-11 | Exclude prompt rewriter and auto-negative inference.        | User-selected scope.                                           | Use fixed structured prompts/negatives.     |
| 2026-07-11 | Keep every runtime write below the shared workspace root.   | Home quota is small and can cause disk failures.               | Launches source `launch_env.sh`.            |
| 2026-07-11 | Reuse native Qwen3 for Dense/MoE text-only conditioning.    | Pure-text MRoPE axes are identical; vision is not used by T2V. | TI2V vision remains separately gated.       |
| 2026-07-11 | Use native grouped-mm for the released MoE expert layout.   | It preserves the checkpoint surface and matches official CUDA. | Base and refiner DiTs are output-exact.     |
| 2026-07-11 | Pass decoded base tensor directly to the refiner.           | Original reloads an MP4; FastVideo keeps the tensor in memory. | No MP4 loss; not exact end-to-end parity.   |
| 2026-07-11 | Give converted Dense and MoE layouts distinct identities.   | Model discovery must not depend on an ambiguous shared name.   | Both identities use one native runtime.     |

NOTE: Component-level numerical parity was achieved. Full MoE/refiner end-to-end parity was not tested or claimed because FastVideo intentionally omits the original MP4 round trip.
    - Provenance: See fastvideo/pipelines/basic/lingbot_video/lingbot_video_pipeline.py and lingbot_video/runner.py for the difference. 

“Grouped-mm” means grouped matrix multiplication.
- Keep the checkpoint’s experts as separate expert weights.
- Preserve the original router’s expert selections.
- Use PyTorch’s built-in grouped-matrix operation to execute them.
- Do not merge the experts into a Dense model or implement a separate custom GPU kernel.
- Reference: fastvideo/models/dits/lingbot_video.py

## Handoff Notes

- Worktree base is `19a51a1fe630bcbeaf9fb6d864ad5ed3f31a3536` on branch `model/port-lingbot-video`.
- Official source imports successfully in the shared venv with no dependency changes.
- Dense official and FastVideo weights are pinned in `tests/local_tests/lingbot_video/hf_assets.py`.
- Dense DiT key manifests are under `checkpoints/lingbot-video/mapping/dense`; official and native surfaces both have 377 tensors with exact shapes.
- Dense Qwen3-VL text-only native config/class and parity test are present; conversion filters `model.language_model.*` and fuses QKV and gate/up through the existing Qwen3 loader.
- Scheduler, Dense DiT, VAE, text, and one/two-step sequential-CFG pipeline parity all pass without skips.
- Both official `robbyant` model repositories require conversion before native FastVideo use; the public `FastVideo` repositories provide that converted layout.
- Dense conversion is published at `FastVideo/LingBot-Video-Dense-1.3B-Diffusers`; its text surface has 290 fused tensors with exact native shapes, and its model index uses `_class_name: LingBotVideoDensePipeline`.
- Production inference preserves the Dense checkpoint's mixed fp32/bf16 parameter policy. Nested FSDP inference passes, while mixed-dtype FSDP training is intentionally rejected because the replicated fp32 parameters would not have synchronized gradients.
- Native batched CFG passes the non-FSDP and FSDP smoke tests. Official numerical parity covers sequential CFG; the smoke tests are not official batched-CFG parity evidence.
- Sequence-parallel validation is complete: SP=1 Dense DiT and full-pipeline regressions are exact in Slurm `1859083`/`1859084`; two-GPU B=2 unequal-mask odd-padding coverage passes in Slurm `1859085` and the official CP-order rerun `1859097`; production batched-CFG smoke passes in Slurm `1859089`.
- Official/native SP=2 parity is exactly zero with math SDPA at 192x320, 9 frames, and two steps in Slurm `1859099`. Normal optimized Torch SDPA is not exact: diagnostic Slurm jobs `1859090`/`1859098` produced mean/max latent drift of `0.01412875`/`0.10981119` because head sharding selected a different bf16 kernel.
- Canonical Dense final generation Slurm `1859166` produced a visually coherent 832x480, 121-frame, 24-FPS artifact in 330.31 seconds with a 26,738.52 MB rank-zero worker lifetime peak; its bytes exactly match Slurm `1859101`.
- The corrected Dense benchmark is under `validation/lingbot-video/benchmarks/job-1859117`: native/official mean latency is 0.214387/0.201173 seconds; native steady/peak allocation is 977,264,128/977,289,728 bytes lower.
- Nsight Systems artifacts are under `validation/lingbot-video/profiles/job-1859106`; launch and synchronization overhead dominate CUDA API time. Its pre-fix steady allocation is superseded by Slurm `1859117`.
- MoE conversion is published at `FastVideo/LingBot-Video-MoE-30B-A3B-Diffusers`; official base and refiner shards map to native `transformer/` and `transformer_2/` with exact 977-key surfaces, and its model index uses `_class_name: LingBotVideoMoePipeline`.
- Base and refiner 30.08B DiTs are exact across every block and final output in Slurm `1859114` and `1859145`. Production base-only and FSDP/SP=8 refiner smokes pass in Slurm `1859121` and `1859122`.
- Canonical Slurm `1859167` completed a 1920x1088, 121-frame, 24-FPS generation in 341.71 FastVideo seconds with a 63,450.65 MB rank-zero worker lifetime peak. Its bytes match superseded Slurm `1859144`; visual review records blurred pillarbox-like side regions as an accepted composition caveat.
- The final targeted CPU/API regression passed 127 tests with 3 expected GPU-gated skips; targeted Ruff and tracked/untracked whitespace validation pass.

## Remaining T2V Gates

- None.
- T2I/TI2V and Qwen3-VL vision conditioning remain outside the completed T2V scope.
