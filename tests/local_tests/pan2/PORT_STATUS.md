# PAN2 Port Status

## Summary
- model_family: pan2
- workload_types: T2V, I2V (one pipeline; I2V when the request carries a first frame)
- official_ref: vllm-project/vllm-omni#8602 at `e0c4c73158a5e4f44eda8eaa8879f7fd9cedbd74` (the diffusers integration is private)
- official_ref_dir: none; rebuilt in `tests/local_tests/pan2/reference.py`
- hf_weights_path: `IFM/PAN2` (not public); tests use `wuqing157/tiny-pan2-modular-pipe@3ff3130b`
- local_weights_dir: HF cache, or `PAN2_TINY_MODEL_DIR`
- source_layout: Modular Diffusers (`modular_model_index.json`), diffusers component subfolders
- local_tests_readme: `tests/local_tests/pan2/README.md`

## Current Phase
- phase: 11 (post-parity review and handoff)
- status: in_progress (GPU and real-weight validation pending, see I001 and I002)
- owner: orchestrator
- last_updated: 2026-10-20

## Component Matrix
| Component | Type | Reuse/Port | Official Definition | Official Instantiation | FastVideo Target | Prototype | Conversion | Parity | Open Issues |
|---|---|---|---|---|---|---|---|---|---|
| transformer | DiT | port | vLLM-Omni `pan2_transformer.py` (diffusers `PAN2Transformer3DModel`) | `transformer/config.json` | `fastvideo/models/dits/pan2.py`, `fastvideo/configs/models/dits/pan2.py` | done | load-time key mapping (Q/K/V fused) | non_skip_pass | I002 |
| text_encoder | encoder | port | transformers `Qwen3_5TextModel` | `text_encoder/config.json` | `fastvideo/models/encoders/qwen3_5.py`, `fastvideo/configs/models/encoders/qwen3_5.py` | done | passthrough | non_skip_pass | I002 |
| tokenizer | tokenizer | reuse | `Qwen2Tokenizer` | `tokenizer/` | AutoTokenizer | n/a | passthrough | covered by pipeline parity | — |
| vae | VAE | reuse | diffusers `AutoencoderKLWan` (Wan2.2, z_dim 48) | `vae/config.json` | `fastvideo/models/wan/vae.py` | done | passthrough | non_skip_pass | — |
| scheduler | scheduler | reuse | diffusers `FlowMatchEulerDiscreteScheduler` | `scheduler/scheduler_config.json` (shift 7) | `fastvideo/models/schedulers/scheduling_flow_match_euler_discrete.py` | done | passthrough | non_skip_pass | — |

## Conversion State
- conversion_script: none; the transformer maps diffusers keys at load time (`PAN2ArchConfig.param_names_mapping`)
- converted_weights_dir: n/a
- source_layout: diffusers
- strict_load_status: tiny checkpoint loads strictly (no missing or unexpected keys)
- passthrough_components: text_encoder, tokenizer, vae, scheduler
- retry_history: none

## Parity Commands
| Scope | Command | Last Result | Notes |
|---|---|---|---|
| DiT | `pytest tests/local_tests/transformers/test_pan2_transformer_parity.py` | 13 passed (CPU) | SP=2/4 vs one rank over gloo, T2V/I2V, one/two pad tokens; max abs 7.7e-6 |
| Text encoder | `pytest tests/local_tests/encoders/test_pan2_qwen3_5_parity.py` | 5 passed (CPU) | exact |
| VAE | `pytest tests/local_tests/vaes/test_pan2_vae_parity.py` | 9 passed (CPU) | exact |
| Scheduler | `pytest tests/local_tests/schedulers/test_pan2_scheduler_parity.py` | 9 passed (CPU) | exact |
| Pipeline | `pytest tests/local_tests/pipelines/test_pan2_pipeline_parity.py` | 3 passed (CPU) | T2V and I2V latents exact |

## Open Questions
| ID | Question | Owner | Needed By Phase | Status | Resolution |
|---|---|---|---|---|---|
| Q001 | Which reference, with the diffusers integration private? | orchestrator | 2 | resolved | vLLM-Omni #8602, rebuilt from diffusers/transformers blocks in `reference.py` |
| Q002 | Port the `pan2-guardrail` prompt/video guardrails? | orchestrator | 7 | resolved | Ported: `PAN2PromptGuardrailStage` and `PAN2VideoGuardrailStage` (`guardrails.py`), on by default (`enable_guardrails`). |
| Q003 | Port Cache-DiT `quality` and CFG-parallel? | orchestrator | 7 | resolved | Not ported: vLLM-Omni serving features. Ulysses SP is supported through `DistributedAttention`. |

## Issues And Blockers
| ID | Phase | Component | Severity | Issue | Evidence | Owner | Status | Resolution |
|---|---|---|---|---|---|---|---|---|
| I001 | 9-10 | all | medium | No real-weight parity or SSIM regression | `IFM/PAN2` is not public | orchestrator | open | Add `fastvideo/tests/ssim/test_pan2_similarity.py` once weights are public. |
| I002 | 9 | transformer, text_encoder | medium | No GPU parity run yet (bf16 + FlashAttention, NCCL SP, 9B encoder speed) | Local parity tests ran on a CPU login node | orchestrator | open | Run the same tests on a GPU node. |
| I003 | 7 | pipeline | low | The shared denoiser casts latents to the model-output dtype in the scheduler step; PAN2 samples on fp32 latents | pipeline parity fails (8e-2) without the fix | orchestrator | resolved | `DenoisingStage.prepare_scheduler_model_output` hook; `PAN2DenoisingStage` returns `noise_pred.float()`. |

## Escape Hatches
| ID | Phase | Decision Type | Question | Recommended Option | Status | Resolution |
|---|---|---|---|---|---|---|

## Decisions
| Date | Decision | Rationale | Impact |
|---|---|---|---|
| 2026-10-15 | Native Qwen3.5 encoder with a PyTorch chunked gated delta rule | No `fla`/`causal_conv1d` dependency; encoder runs once per prompt | Slower than fused kernels for long prompts |
| 2026-10-15 | Fuse the per-stream Q/K/V projections at load time | One GEMM per stream instead of three | Mapping in `PAN2ArchConfig.param_names_mapping` |
| 2026-10-15 | Text embeddings kept in fp32 after encoding | The shared latent stage draws noise in the embedding dtype; PAN2 uses fp32 noise and latents | Denoiser casts embeddings to the DiT dtype (`cast_prompt_embeds_to_dit_dtype`) |

## Handoff Notes
- The remaining work is I001 and I002. Both need hardware or weights, not code changes.
