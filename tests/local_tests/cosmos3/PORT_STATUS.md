# Cosmos3 Port Status

## Summary

- model_family: `cosmos3`
- workload_types: `T2V, I2V, T2I` supported by `WorkloadType` today; full-omni target also needs audio (AV), VLM reasoning, and action-conditioning, which require framework extensions (Q002, Q003).
- official_ref: `https://github.com/NVIDIA/cosmos-framework` — diffusers backend `diffusers_cosmos3.pipeline.Cosmos3OmniDiffusersPipeline`; HF `nvidia/Cosmos3-Nano`.
- official_ref_dir: `cosmos-framework` (symlink -> `/home/william5lin/FastVideo/cosmos-framework`, commit `003d66d4`)
- hf_weights_path: `nvidia/Cosmos3-Nano`
- local_weights_dir: `official_weights/cosmos3` (symlink -> `/home/william5lin/FastVideo/official_weights/cosmos3`, 33 GiB / 67 files)
- source_layout: `diffusers`
- local_tests_readme: `tests/local_tests/cosmos3/README.md`

## Current Phase

- phase: `PR1 (video core) + PR2 (audio/t2vs) COMPLETE, real-weights verified. Video core (T2V/I2V/T2I) + audio (AVAE decode / DiT sound pathway / sound packing / t2vs CFG velocity) all framework-parity verified bit-exact (suite 130 passed, 0 skipped). t2vs real-weights on B200 produces coherent video + real stereo 48kHz audio. Branch chain: feat/cosmos3-tier-a-port (T2V) -> feat/cosmos3-i2v (I2V+T2I) -> feat/cosmos3-audio (t2vs). Next: PR3 action / PR4 reasoning.`
- status: `in_progress`
- owner: `orchestrator`
- last_updated: `2026-06-07`
- env: `fv-cosmos3` (conda clone of fv-main; `fastvideo` editable repointed to this worktree). Run tests from the worktree cwd with this env's python.
- branch: rebased onto `origin/main` @ `1c627a3f9` (was 33 behind, merge-base 2026-05-22); now 6 commits ahead; `fastvideo` imports clean; Tier-A `13 passed, 2 skipped`.

## Component Matrix

| Component | Type | Reuse/Port | Official Definition | Official Instantiation | FastVideo Target | Prototype | Conversion | Parity | Open Issues |
|---|---|---|---|---|---|---|---|---|---|
| transformer | dit | port | `diffusers_cosmos3/transformer.py:Cosmos3OmniTransformer` (model_type `qwen3_vl_text`, MoT + MRoPE) | `model_index.json: transformer`; `cosmos_framework/model/vfm/mot/cosmos3_vfm_network.py`, `omni_mot_model.py` | `fastvideo/models/dits/cosmos3.py` (branch: `Cosmos3VFMTransformer`+`Cosmos3LanguageModel` — reconcile to `Cosmos3OmniTransformer`) | skeleton | not_started | scaffold_skip | I001 |
| vae | vae | reuse | diffusers `AutoencoderKLWan` | `model_index.json: vae` | reuse Wan VAE (`fastvideo/models/vaes/`, cf. `cosmos25wanvae.py`) | not_started | passthrough? | not_started | Q001 |
| scheduler | generic | reuse (flow-coerced) | framework `FlowUniPCMultistepScheduler` (`cosmos_framework/.../fm_solvers_unipc.py`; checkpoint ships diffusers-style config) | `model_index.json: scheduler`; `cosmos_framework/.../samplers/unipc.py:UniPCSampler` | FastVideo-native `UniPCMultistepScheduler` (flow config), coerced in `initialize_pipeline` | done | n/a | framework-parity DONE (`test_cosmos3_scheduler_parity`: timesteps bit-exact, sigmas ~1e-8, trajectory <~1e-6) | I003 (resolved) |
| text_tokenizer | tokenizer | reuse | transformers `Qwen2TokenizerFast` | `model_index.json: text_tokenizer` | reuse (tokenizer = allowed third-party) | not_started | passthrough | scaffold_skip (`test_cosmos3_tokenizer_chat_template`) | - |
| vision_encoder | encoder | port | transformers `Qwen3VLVisionModel` | `model_index.json: vision_encoder` | new encoder bucket OR documented lazy-wrapper | not_started | not_started | not_started | Q002 |
| sound_tokenizer | generic/vae | port (decode) | framework AVAE `LatentAutoEncoderV2` (`avae_utils`); checkpoint is decoder-only AutoencoderOobleck-named w/ SnakeBeta | `model_index.json: sound_tokenizer` | reuse FastVideo native `OobleckVAE` decoder + `Cosmos3SoundVAE` wrapper (`models/audio/cosmos3_avae.py`) | done (decode) | n/a | DECODE bit-exact vs framework (`test_cosmos3_avae_parity`); real ckpt strict-loads | PR2 (branch feat/cosmos3-audio) |

## Conversion State

- conversion_script: `scripts/checkpoint_conversion/cosmos3_convert.py` (branch has it, 246 lines, built vs vllm-omni — repoint/verify vs diffusers checkpoint)
- converted_weights_dir: `converted_weights/cosmos3` (n/a while needs_conversion=no)
- source_layout: `diffusers`
- needs_conversion: `no` (HF already diffusers-format; verify FastVideo loaders consume directly)
- strict_load_status: `not_run`
- passthrough_components: `vae (AutoencoderKLWan), scheduler (UniPC), text_tokenizer (Qwen2)` likely passthrough
- retry_history: `none`

## Parity Commands

| Scope | Command | Last Result | Notes |
|---|---|---|---|
| Tier-A scaffold | `cd <worktree> && <fv-cosmos3 python> -m pytest tests/local_tests/cosmos3/ -q` | `13 passed, 2 skipped` (2026-06-06, post-rebase) | 2 skips: Cosmos3 tokenizer/_tokenize_prompt not yet wired on pipeline |
| component | `pytest tests/local_tests/<bucket>/test_cosmos3_<component>_parity.py -v -s` | `not_run` | after env activation + native prototypes |
| pipeline | `pytest tests/local_tests/pipelines/test_cosmos3_pipeline_parity.py -v -s` | `not_run` | |

## Open Questions

| ID | Question | Owner | Needed By Phase | Status | Resolution |
|---|---|---|---|---|---|
| Q001 | Does Cosmos3 VAE (`AutoencoderKLWan`) match FastVideo's existing Wan VAE config/instantiation exactly (z_dim, scale factors, latents_mean/std)? | orchestrator | 3 (reuse gate) | open | |
| Q002 | `vision_encoder` (`Qwen3VLVisionModel`): native port vs documented lazy-wrapper exception? Needed for I2V/reasoning. | user/orchestrator | 3 | open | |
| Q003 | `sound_tokenizer` (`Cosmos3AVAEAudioTokenizer`) + audio output requires `WorkloadType` AV + audio regression metric. | user | 0/10 | open | full-omni scope chosen 2026-06-06; infra extensions pending |

## Issues And Blockers

| ID | Phase | Component | Severity | Issue | Evidence | Owner | Status | Resolution |
|---|---|---|---|---|---|---|---|---|
| I001 | port | transformer | high | Branch DiT (`Cosmos3VFMTransformer`+`Cosmos3LanguageModel`) built vs vllm-omni #3454; official checkpoint loads `Cosmos3OmniTransformer` (diffusers shim). Class/structure reconciliation required. | `model_index.json`; `diffusers_cosmos3/transformer.py`; branch commit `52bb65f49` | orchestrator | resolved | DiT rewritten to checkpoint layout (single `layers` dual-pathway, BaseDiT-conformant); bit-identical framework parity (3d_rope + unified_3d_mrope), commits 59a4a571c/7c4633295 |
| I002 | all | tests | medium | Tier-A conftest+tests mirror vllm-omni line-by-line (stubs, `vllm_omni...guardrails`). Must be repointed to `diffusers_cosmos3` / official structures. | `tests/local_tests/cosmos3/conftest.py` | orchestrator | open | |
| I003 | inference | scheduler | high | First real-weights T2V was all-black: checkpoint `scheduler_config.json` sets `use_karras_sigmas=true`; vendored UniPC checks karras before `use_flow_sigmas` -> diffusion (beta) sigmas -> `scheduler.step` -> NaN latents. DiT/CFG velocity was clean. The scheduler had never been parity-tested vs the framework (`test_cosmos3_denoise_cfg_parity` used diffusers UniPC on both sides). | `result_latent` NaN at denoise step 0 (v_pred clean); ffprobe 3 KB black mp4 | orchestrator | resolved | Coerce loaded config to flow setup in `initialize_pipeline`; switch pipeline+tests to native UniPC (no diffusers at runtime); add `test_cosmos3_scheduler_parity` vs framework `FlowUniPCMultistepScheduler`; repoint denoise_cfg oracle to the framework scheduler. Commit 255311cf2 |

## Escape Hatches

| ID | Phase | Decision Type | Question | Recommended Option | Status | Resolution |
|---|---|---|---|---|---|---|
| E001 | prep | dependency/env | Shared `fv-main` env has `fastvideo` editable-installed from the MAIN worktree; the cosmos3 worktree's `fastvideo` is not importable (PEP660 finder overrides PYTHONPATH), so Tier-A tests skip. How to activate the worktree's `fastvideo` for verification without disrupting ~24 other worktrees sharing the env? | Dedicated conda env for the cosmos3 worktree | resolved | Created fv-cosmos3 (clone of fv-main); repointed fastvideo editable to worktree; run from worktree cwd. Branch also rebased onto origin/main to fix stale import. |

## Decisions

| Date | Decision | Rationale | Impact |
|---|---|---|---|
| 2026-06-06 | Reference source of truth = official diffusers (`Cosmos3OmniDiffusersPipeline` + `cosmos-framework`/`diffusers-cosmos3`), not vllm-omni #3454 | Official weights now public & diffusers-format; the artifact users actually load | Repoint DiT/pipeline/conversion/tests off vllm-omni (I001, I002) |
| 2026-06-06 | Resume in worktree `/home/william5lin/FastVideo_cosmos3_port`; weights+reference symlinked (no copy) | Preserve 2,492 lines of Tier-A work; avoid 33 GB duplication | Verification needs worktree `fastvideo` active (E001) |
| 2026-06-06 | Scope = full omni (video + audio + reasoning + action) | User choice (revised from branch's original video-only scope) | Adds `vision_encoder`, `sound_tokenizer` ports + `WorkloadType` AV + audio metric |
| 2026-06-06 | Downloaded full 34.9 GB (33 GiB) `nvidia/Cosmos3-Nano` | Unblocks May-22 `PENDING` weight status (HF was 401, now public) | Real parity now possible |
| 2026-06-06 | Rebased branch onto origin/main (33 commits); resolved registry.py conflict by reconstructing from main + cosmos3 import/entry | Branch was stale; fastvideo failed to import (main removed MatrixGameI2V480PConfig) | Branch imports clean; Tier-A 13 passed/2 skipped |
| 2026-06-06 | Reference = cosmos_framework ONLY (full omni); diffusers shim dropped even for video | User directive (Phase 1 found diffusers __call__ is video-only; sound/action/reasoning live only in the framework) | Larger port; ref DiT = `Cosmos3VFMNetwork`/`Cosmos3VFMNetworkConfig` (not diffusers `Cosmos3OmniTransformer`); core model imports in fv-cosmos3 with light deps; TE only in optional dot_product_attention |

## Handoff Notes

- Prep (weights/reference/env editable installs) done in MAIN worktree; symlinked into this worktree. Env installs (`diffusers-cosmos3`, `cosmos-framework`) are in shared `fv-main`.
- Next: resolve E001 (env), then Phase 1 reference study of `diffusers_cosmos3` pipeline/transformer, then Phase 3 reuse gate (VAE/scheduler/tokenizer) + component dispatch (transformer, vision_encoder, sound_tokenizer).
- diffusers 0.36.0 imports the shim OK; checkpoint saved with 0.37.1 — watch `from_pretrained` needs (bump within FastVideo's `diffusers>=0.33.1` pin if required).

### PR1 (video core) progress — 2026-06-06
- Arch config 1:1 with checkpoint, committed `9567efdf0`.
- Framework parity-reference harness committed `dd97efda3`: `tests/local_tests/cosmos3/test_cosmos3_reference_forward.py` builds a tiny `Cosmos3VFMNetwork` on CPU/float32 (SDPA monkeypatch; flash2/3/natten are CUDA-only) and forwards `packed_seq -> {last_hidden_state, preds_vision}`. 23 tests pass in fv-cosmos3. This is the ground-truth side for DiT parity. Run: `cd <worktree> && <fv-cosmos3 py> -m pytest tests/local_tests/cosmos3/test_cosmos3_reference_forward.py -q`.
- THREE naming conventions to bridge:
  1. framework-native (`Cosmos3VFMNetwork`): `language_model.model.layers.{i}.self_attn.{q,k,v,o}_proj(+ _moe_gen)`, `{q,k}_norm(+_moe_gen)`, `mlp(+_moe_gen)`, `vae2llm`/`llm2vae`, `time_embedder.mlp.{0,2}`.
  2. diffusers checkpoint (on disk, what we load): `layers.{i}.self_attn.{to_q,to_k,to_v,to_out}` + `{add_q,add_k,add_v}_proj`/`to_add_out`, `{norm_q,norm_k,norm_added_q,norm_added_k}`, `mlp`/`mlp_moe_gen`, `proj_in`/`proj_out`, `time_embedder.linear_{1,2}`.
  3. FastVideo DiT (our choice). Conversion maps (2)->(3); the DiT parity test copies (1)->(3).
- BaseDiT signature is `__init__(self, config: DiTConfig, hf_config: dict)`; the branch `Cosmos3VFMTransformer` uses `fastvideo_args`/SimpleNamespace and does NOT conform — rewrite to conform + match the checkpoint key surface (single `layers` dual-pathway, not split language_model/gen_layers).
- Native layers (per cosmos2_5): `ReplicatedLinear`/`MLP`/`RMSNorm` (fastvideo.layers.*), `LocalAttention`/`DistributedAttention` (fastvideo.attention), `apply_rotary_emb` (use_real_unbind_dim=-2 for Cosmos). EntryClass at module bottom; class attrs bound from config; 3D-MRoPE has no reusable util — adapt Cosmos25RotaryPosEmbed.
- NEXT: write native `fastvideo/models/dits/cosmos3.py` + fastvideo-vs-framework forward parity test (copy framework weights into the FastVideo DiT, compare outputs), then conversion script (diffusers checkpoint -> FastVideo) + strict-load, then video pipeline/packing.

### PR1 (video core) acceptance — real-weights E2E — 2026-06-07
- First real-weights T2V (`examples/inference/basic/basic_cosmos3_new_api.py`, `COSMOS3_MODEL_PATH=official_weights/cosmos3`) ran mechanically but produced an all-black 3 KB mp4. Instrumenting the denoise loop showed `v_pred` clean at step 0 but `scheduler.step` -> NaN. Root cause I003: checkpoint `scheduler_config.json` is diffusers-style (`use_karras_sigmas=true`), and the vendored UniPC checks karras before `use_flow_sigmas` -> diffusion (beta) sigmas instead of flow sigmas -> NaN. The framework actually samples with `FlowUniPCMultistepScheduler` (pure flow: `shift` + `num_train_timesteps`).
- Fix (commit `255311cf2`): coerce the loaded scheduler to the flow setup in `Cosmos3OmniDiffusersPipeline.initialize_pipeline`; use FastVideo's native UniPC (not diffusers) in pipeline + tests. Added `test_cosmos3_scheduler_parity.py` (native UniPC flow-config vs framework `FlowUniPCMultistepScheduler`: timesteps bit-exact, sigmas ~1e-8, full trajectory <~1e-6 over shift in {10,3}, steps in {4,10,35}). Repointed `test_cosmos3_denoise_cfg_parity` oracle to the framework scheduler (it previously compared diffusers-vs-diffusers, so the scheduler was never checked against the framework).
- Also wired the remaining integration glue (registry alias `Cosmos3OmniTransformer`->`Cosmos3VFMTransformer`; `text_tokenizer`->TokenizerLoader; scheduler config param-filtering; DiT `materialize_non_persistent_buffers` + compute-dtype casts; packing device-move in `to_dit_kwargs`; empty text-preprocess).
- Verified: 1280x704, 29 frames, 35 steps on a single B200 -> coherent golden-retriever-in-meadow video matching the prompt (no NaNs; per-frame pixel std ~58; visible temporal motion). Full cosmos3 suite: 95 passed, 0 skipped.
- NEXT: PR2 audio (`sound_tokenizer` AVAE) / PR3 action / PR4 reasoning. Optional: I2V/T2I real-weights spot-checks; force-push branch (needs explicit OK).

### PR1 (video core) — I2V real-weights — 2026-06-07 (branch feat/cosmos3-i2v)
- Forked `feat/cosmos3-i2v` off `feat/cosmos3-tier-a-port` (stacked, includes the T2V + scheduler fix).
- Studied the framework I2V path: `cosmos_framework.inference.vision.load_conditioning_image` (aspect-preserving resize + center crop + uint8 quantize -> `/127.5-1`) + `build_conditioned_video_batch` (frame 0 = image, remaining frames REPEAT the last conditioning frame -> static video), then VAE-encode; `condition_frame_indexes=[0]` (latent). Condition frames kept clean during sampling exactly as FastVideo already does: init noise `cond_mask*x0 + (1-cond_mask)*noise` (`omni_mot_model._prepare_inference_data`) + velocity zeroed `pred*(1-cond_mask)` each step (`_get_velocity`), no re-injection.
- Bug found + fixed (commit `bd8d604fb`): FastVideo's `_image_to_video_tensor` ZERO-filled the non-condition frames; the temporal Wan VAE (4x) makes latent frame 0 depend on several pixel frames, so zero-fill -> wrong conditioning latent. Rewrote it to repeat-fill + framework resize/crop/quantize.
- Parity: `test_cosmos3_i2v_conditioning_parity.py` vs framework `load_conditioning_image` + repeat-fill — bit-exact (max abs diff 0.0) across aspect/size/frame cases. Existing `test_cosmos3_denoise_cfg_parity` already covers the I2V cond-mask + velocity math (i2v case).
- Example: `examples/inference/basic/basic_cosmos3_i2v_new_api.py` (`InputConfig(image_path=...)`, default `assets/images/cyclist.jpg`).
- Verified on B200 (1280x704, 29f, 35 steps, real weights): output frame 0 reproduces the conditioning cyclist image; later frames show coherent forward motion down the trail following the prompt. Full suite 98 passed, 0 skipped.
- NEXT: optional T2I real-weights spot-check; then PR2 audio / PR3 action / PR4 reasoning.

### PR1 (video core) — T2I real-weights + resolution-based flow_shift — 2026-06-07 (branch feat/cosmos3-i2v)
- Studied framework T2I: tokenization uses `vlm_config.use_system_prompt` which is `false` in the checkpoint (config.json:199) — matches FastVideo's hardcoded `use_system_prompt=False` for all modes (no divergence). Canonical T2I is 960x960 (inputs/omni/t2i.json), single-frame (num_frames=1).
- Bug found + fixed (commit `604dc2637`): the stage chose `flow_shift` by task (`3.0 if is_t2i else 10.0`), but the framework picks it purely by the named resolution bucket (`OmniSampleArgs._RESOLUTION_SHIFT_DEFAULTS`, 8B backbone: 256->3.0, 480->5.0, 720/768->10.0; model default resolution "720"). Task-based only matched T2V@720 / T2I@256 by luck; canonical T2I@960x960 is the "720" bucket -> 10.0, so `is_t2i->3.0` was wrong. Replaced with `_flow_shift_for_resolution(h,w)` (longest-side bucketing), applied to all tasks.
- Parity: `test_cosmos3_flow_shift_parity.py` checks the mapping vs framework `{VIDEO,IMAGE}_RES_SIZE_INFO` x `_RESOLUTION_SHIFT_DEFAULTS` (8B rows, 20 cases). Also hardened `_image_to_video_tensor` tensor branch to respect the [-1,1] convention (PIL path stays framework-exact).
- Example: `examples/inference/basic/basic_cosmos3_t2i_new_api.py` (num_frames=1, 960x960).
- Verified on B200 (real weights, 35 steps): coherent red-panda image matching the prompt, flow_shift=10.0. Full suite 118 passed, 0 skipped.
- Video core (T2V/I2V/T2I) is now complete and real-weights verified. NEXT: PR2 audio (sound_tokenizer AVAE + audio output) on a new stacked branch.
