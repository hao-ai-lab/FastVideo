# InterleaveThinker Integration Design

This page summarizes the InterleaveThinker integration branch for reviewers.
The detailed execution log remains in
`.agents/exploration/interleavethinker-fastvideo-integration.md`.

## Scope

The branch adds FastVideo-native support for InterleaveThinker-style workflows
without adding new `fastvideo` CLI commands or HTTP API routes:

- Qwen3-VL planner and critic model wrappers;
- planner and critic SFT configs;
- planner and critic GRPO configs with optional reference-policy KL;
- InterleaveThinker reward parsing and scoring utilities;
- Gemini and Nano Banana wrappers for optional network-backed rewards;
- Python orchestration helpers for planner -> generator -> critic traces.

It does not vendor InterleaveThinker, EasyR1, Verl, LLaMA-Factory, or training
framework internals from those projects.

## Integrated Surfaces

### Training

| Surface | Purpose |
|---------|---------|
| `fastvideo.train.models.interleave_thinker.Qwen3VLActorBase` | Shared Transformers Qwen3-VL runtime for planner and critic actors. |
| `InterleaveThinkerPlannerModel` | FastVideo `RoleModelBase` actor wrapper for `InterleaveThinker/InterleaveThinker-Planner-8B`. |
| `InterleaveThinkerCriticModel` | FastVideo `RoleModelBase` actor wrapper for `InterleaveThinker/Critic-SFT-8B` and `InterleaveThinker/InterleaveThinker-Critic-8B`. |
| `InterleaveThinkerSFTMethod` | Response-token supervised fine-tuning method for planner and critic actors. |
| `InterleaveThinkerRLMethod` | Managed GRPO-style loop for planner and critic actors. |
| `fastvideo.train.methods.rl.common.grpo` | Shared GRPO math helpers. |
| `fastvideo.train.methods.rl.rewards.interleave_thinker` | Format, critic, and planner reward utilities. |
| `fastvideo.train.methods.rl.rewards.interleave_api` | Optional Gemini and Nano Banana API-backed reward wrappers. |

Training examples live under:

- `examples/train/configs/interleave_thinker/planner_sft_lora.yaml`;
- `examples/train/configs/interleave_thinker/critic_sft_lora.yaml`;
- `examples/train/configs/interleave_thinker/planner_smoke.yaml`;
- `examples/train/configs/rl/interleave_thinker/critic_grpo.yaml`;
- `examples/train/configs/rl/interleave_thinker/planner_grpo.yaml`.

### Orchestration Helpers

The Python helper layer under `fastvideo.workflow.interleave_thinker` is intentionally
not registered as a CLI or server contract. It provides reusable dataclasses,
provider adapters, image-backend adapters, trace serialization, prompt-set
execution helpers, and saved-trace metrics for tests, examples, and downstream
integration code that already imports FastVideo as a library.

The runnable example is:

- `examples/interleave/interleave_single_prompt.py`.

## Architecture Boundaries

| Layer | Owner | Notes |
|-------|-------|-------|
| Planner and critic actors | `fastvideo/train/models/interleave_thinker/` | Wrap Transformers Qwen3-VL checkpoints. They are training actors, not diffusion pipeline components. |
| RL/SFT algorithms | `fastvideo/train/methods/` | Own loss, reward aggregation, advantage computation, KL, and optimizer cadence. |
| Rewards and API clients | `fastvideo/train/methods/rl/rewards/` | Offline reward aggregation is separate from network-backed Gemini/Nano Banana clients. |
| Image generation/editing helpers | `fastvideo/workflow/interleave_thinker/generator.py` | Presents a small image backend protocol for FastVideo, Nano Banana, and fake backends. |
| Runtime orchestration helpers | `fastvideo/workflow/interleave_thinker/` | Plans steps, calls generator/edit backends, calls critic providers, records traces. |
| Evaluation helpers | `fastvideo/workflow/interleave_thinker/evaluation.py` and `trace_eval.py` | Prompt-set execution and saved-trace reporting remain outside training methods. |

This keeps the Qwen actor implementation reusable by SFT, planner GRPO, critic
GRPO, and inference providers without coupling those paths to a specific
generator service.

## Validation Matrix

All GPU/model validation below ran on Modal L40S through
`fastvideo/tests/modal/launch_l40s_job.py`.

| Area | Evidence | Modal app |
|------|----------|-----------|
| API-backed model/reward wrappers | `27 passed, 14 warnings`; final pre-commit passed. | `ap-QOKlzapm5bSAo3c21lprwv` |
| Critic backend hardening | `30 passed, 14 warnings`; pre-commit passed. | `ap-DplMFq23YYfBx34e6TcsRc` |
| Real critic checkpoint smoke | Loaded `InterleaveThinker/Critic-SFT-8B`; generated one rollout; printed `SMOKE_OK`. | `ap-hDxj5MhLgdnGq22mRLjgIK` |
| FastVideo RL loop skeleton | `22 passed, 14 warnings`; pre-commit passed. | `ap-2Z2sH2UfhMoPmKolG0KY6t` |
| Shared Qwen actor and planner wrapper | `36 passed, 14 warnings`; pre-commit passed. | `ap-ZapOKZPOmhyMZFxZ0X1fQm` |
| Real planner checkpoint smoke | Loaded `InterleaveThinker/InterleaveThinker-Planner-8B`; parsed 3 steps; printed `PLANNER_SMOKE_OK`. | `ap-BzH7QxVXoc5XFXBah5cJ2H` |
| Real critic refactor smoke | Loaded critic wrapper after Qwen base refactor; printed `CRITIC_REFACTOR_SMOKE_OK`. | `ap-NGxUDBNJFiU30Wef0yAQN1` |
| Planner/critic provider adapters | `20 passed, 14 warnings`; pre-commit passed. | `ap-wfRX2DCt30DN903gETbDpj` |
| Real provider loop smoke | Real planner and critic with fake generator; printed `INTERLEAVE_PROVIDER_REAL_LOOP_SMOKE_OK`. | `ap-ZABadeyKBuGVcfy67LqmXt` |
| Dataset normalization | `17 passed, 14 warnings`; pre-commit passed. | `ap-ISuDU2lwc6Pl5NYDZnnBEb` |
| Planner and critic SFT | `20 passed, 14 warnings`; final pre-commit passed. | `ap-1jmIczO3KwZoP3WtLYOIxc` |
| Critic GRPO policy loss | Broad InterleaveThinker test set: `28 passed, 14 warnings`; pre-commit passed. | `ap-aYBz0F0ZiQ2nTGndudnGaH` |
| Real critic RL smoke | Loaded LoRA critic student; generated rollouts; completed one GRPO update; printed `INTERLEAVE_CRITIC_RL_SMOKE_OK`. | `ap-eXMO3I81OcCyxj53XbPWj9` |
| Reference-policy KL | `16 passed, 14 warnings`; real reference smoke printed `INTERLEAVE_CRITIC_RL_REFERENCE_SMOKE_OK`. | `ap-UQ38OTnymREO9bz0L1QzC5` |
| Planner GRPO | `37 passed, 14 warnings`; real planner GRPO smoke printed `INTERLEAVE_PLANNER_RL_SMOKE_OK`. | `ap-PDBijC8opxsMiMU0Uc064A` |
| Prompt-set evaluation helpers | `15 passed, 14 warnings`; pre-commit passed. | `ap-eeQpAgNQvQGi2H8MB0kJCU` |
| Trace-level evaluation helpers | `19 passed, 14 warnings`; pre-commit passed. | `ap-s7ewT9rDZSTdPNhyYrEYO7` |

## Recommended PR Stack

1. **Python orchestration shell**
   - schema and trace dataclasses;
   - generator backend protocol;
   - provider adapters;
   - fake-backend tests.
2. **Qwen3-VL actor wrappers**
   - shared Qwen actor base;
   - planner and critic wrappers;
   - data normalization helpers;
   - real checkpoint load smokes.
3. **SFT path**
   - `InterleaveThinkerSFTMethod`;
   - planner and critic SFT configs;
   - response-token masking tests.
4. **Reward and API backend path**
   - InterleaveThinker reward parser/scorers;
   - Gemini and Nano Banana wrappers;
   - fake-client tests.
5. **GRPO path**
   - shared GRPO helpers;
   - `InterleaveThinkerRLMethod`;
   - critic GRPO, reference KL, planner GRPO;
   - real one-step LoRA smokes.
6. **Evaluation and docs**
   - prompt-set runner;
   - trace evaluator and HTML report helpers;
   - examples and design docs.

Each PR should keep the handoff updated until it lands or is superseded.

## Remaining Risks

- **Full 8B training memory:** Real one-step LoRA smokes passed. Full-parameter
  8B optimizer training and longer distributed runs still need dedicated
  hardware validation.
- **Checkpoint/resume:** Configs include checkpoint settings, but planner/critic
  SFT and GRPO checkpoint/resume smokes are not yet recorded.
- **Closed-source API drift:** Gemini and Nano Banana wrappers are unit-tested
  with fake clients. Live API outputs can change and should not be deterministic
  CI baselines.
- **EasyR1 parity:** The FastVideo GRPO path matches the important objective
  pieces used here, but it is not a wholesale EasyR1/Verl port. Distributed
  rollout semantics and memory strategy should remain explicit in docs.
- **Native Qwen3-VL port:** The branch uses Transformers Qwen3-VL wrappers. A
  FastVideo-native Qwen3-VL port should only be considered if conversion,
  performance, or distributed execution needs justify it.
- **End-to-end real generator cost:** Real planner/critic and real FastVideo
  generator pieces have smoke coverage, but large prompt-set runs with all real
  components can be expensive and should be scheduled intentionally.

## Review Checklist

- Confirm no training code imports from the legacy `fastvideo/training/` stack.
- Confirm API clients import optional dependencies lazily.
- Confirm fake-provider tests cover planner, critic, generator, reward, and
  trace-evaluation behavior without credentials.
- Confirm real-checkpoint smoke commands document whether they used a pushed
  commit or an explicitly approved Modal patch upload.
- Confirm public YAML configs are parseable and clearly state credential,
  dataset, and hardware assumptions.
