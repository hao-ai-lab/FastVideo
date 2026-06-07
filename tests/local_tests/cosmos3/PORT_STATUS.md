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

- phase: `Phase 1 done (reference = cosmos_framework, full omni); feasibility = framework runnable for component parity with light deps`
- status: `in_progress`
- owner: `orchestrator`
- last_updated: `2026-06-06`
- env: `fv-cosmos3` (conda clone of fv-main; `fastvideo` editable repointed to this worktree). Run tests from the worktree cwd with this env's python.
- branch: rebased onto `origin/main` @ `1c627a3f9` (was 33 behind, merge-base 2026-05-22); now 6 commits ahead; `fastvideo` imports clean; Tier-A `13 passed, 2 skipped`.

## Component Matrix

| Component | Type | Reuse/Port | Official Definition | Official Instantiation | FastVideo Target | Prototype | Conversion | Parity | Open Issues |
|---|---|---|---|---|---|---|---|---|---|
| transformer | dit | port | `diffusers_cosmos3/transformer.py:Cosmos3OmniTransformer` (model_type `qwen3_vl_text`, MoT + MRoPE) | `model_index.json: transformer`; `cosmos_framework/model/vfm/mot/cosmos3_vfm_network.py`, `omni_mot_model.py` | `fastvideo/models/dits/cosmos3.py` (branch: `Cosmos3VFMTransformer`+`Cosmos3LanguageModel` — reconcile to `Cosmos3OmniTransformer`) | skeleton | not_started | scaffold_skip | I001 |
| vae | vae | reuse | diffusers `AutoencoderKLWan` | `model_index.json: vae` | reuse Wan VAE (`fastvideo/models/vaes/`, cf. `cosmos25wanvae.py`) | not_started | passthrough? | not_started | Q001 |
| scheduler | generic | reuse | diffusers `UniPCMultistepScheduler` | `model_index.json: scheduler` | reuse | not_started | n/a | scaffold_skip (`test_cosmos3_scheduler_default_parity`) | - |
| text_tokenizer | tokenizer | reuse | transformers `Qwen2TokenizerFast` | `model_index.json: text_tokenizer` | reuse (tokenizer = allowed third-party) | not_started | passthrough | scaffold_skip (`test_cosmos3_tokenizer_chat_template`) | - |
| vision_encoder | encoder | port | transformers `Qwen3VLVisionModel` | `model_index.json: vision_encoder` | new encoder bucket OR documented lazy-wrapper | not_started | not_started | not_started | Q002 |
| sound_tokenizer | generic/vae | port | `Cosmos3AVAEAudioTokenizer` (model_type `autoencoder_v2`) | `model_index.json: sound_tokenizer` | new audio component | not_started | not_started | not_started | Q003 |

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
| I001 | port | transformer | high | Branch DiT (`Cosmos3VFMTransformer`+`Cosmos3LanguageModel`) built vs vllm-omni #3454; official checkpoint loads `Cosmos3OmniTransformer` (diffusers shim). Class/structure reconciliation required. | `model_index.json`; `diffusers_cosmos3/transformer.py`; branch commit `52bb65f49` | orchestrator | open | |
| I002 | all | tests | medium | Tier-A conftest+tests mirror vllm-omni line-by-line (stubs, `vllm_omni...guardrails`). Must be repointed to `diffusers_cosmos3` / official structures. | `tests/local_tests/cosmos3/conftest.py` | orchestrator | open | |

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
