---
name: add-model-02-parity
description: Use during /add-model after reference/architecture study to scaffold and later activate local FastVideo component parity tests. Emphasizes early test creation, official-reference loading, standardized FastVideo loading, and non-skip handoff gates.
---

# Add Model Parity

## Goal

Create parity tests as early as possible in a FastVideo port. The first pass can
land before conversion or component implementation as an executable scaffold;
handoff is blocked until the same tests become non-skip PASS with real weights.

## When To Run

Follow `../add-model/shared/common_rules.md` for token/auth safety, state files,
escape hatches, and skip/pass semantics.

Run immediately after `/add-model` Phase 1 has identified:

- official component classes and call signatures;
- FastVideo target component buckets/classes/configs;
- local reference clone or import path from `add-model-01-prep`;
- local raw or Diffusers weight path;
- `official_env_status=imports_ok`, or private deps that will be stubbed
  locally in tests;
- `local_tests_readme` documenting setup and planned review/test commands;
- expected component inputs and output tensors.

Do not wait for all FastVideo components to be implemented. Write the tests
first, then let component-porting subagents make them pass.

## Outputs

- One component parity test per required component, including reused components:
  `tests/local_tests/<bucket>/test_<family>_<component>_parity.py`.
- Optional helper for upstream private deps:
  `tests/local_tests/helpers/<family>_upstream.py`.
- Pipeline parity is owned later by `../add-model-09-pipeline/SKILL.md` after all
  component parity tests pass non-skip.
- A parity status block for the `/add-model` parity verification phase.

## Early Scaffold Rules

- A scaffold may skip while the FastVideo class, converted weights, or official
  import is missing.
- A scaffold must already encode the real official load path, FastVideo load
  path, deterministic inputs, expected output extraction, and tolerance target.
- Each parity test must declare its coverage scope in the file docstring or a
  module constant: `production_loader`, `implementation_subcomponent`, or `both`.
  Implementation/subcomponent parity may bypass production loaders deliberately,
  but final handoff still needs production-loader coverage somewhere before the
  pipeline depends on that component.
- Official reference imports must run in the current FastVideo environment; do
  not create or assume a separate upstream venv/conda env.
- A scaffold is not evidence of correctness. It becomes evidence only after a
  local non-skip PASS.
- Prefer env-var path overrides with repo-relative defaults.
- Keep tests local-only under `tests/local_tests/`; package/CI quality tests are
  added later.
- Update shared state files as described in
  `../add-model/shared/common_rules.md` whenever adding or activating parity
  tests.

## Component Template

Copy `templates/component_parity_test.py` and fill every `TODO` marker. The
template is distilled from:

- `tests/local_tests/transformers/test_ltx2.py`
- `tests/local_tests/transformers/test_gamecraft_parity.py`
- `tests/local_tests/encoders/test_ltx2_gemma_parity.py`
- `tests/local_tests/vaes/test_oobleck_vae_parity.py`
- `tests/local_tests/sd35/test_sd35_component_parity.py`

The template supports three states:

| State | Meaning |
|---|---|
| Scaffold skip | Test is committed early, but official import, FastVideo class, or weights are not available yet. |
| Debug red | Both sides load and the test fails numerically. This is useful: porting can chase the first drift. |
| Non-skip pass | Required before `/add-model` handoff. |

## Subagent Dispatch Pattern

After Phase 1, dispatch one parity subagent per component before or alongside
component implementation:

```text
Create a local parity test scaffold for <family> <component>.

Use the prep handoff:
- official_ref_dir/import: <...>
- local_weights_dir: <...>
- source_layout: <...>
- needs_conversion: <yes/no>
- official_env_status: <imports_ok | private_deps_need_stubs>
- local_tests_readme: tests/local_tests/<model_family>/README.md
- port_state_file: tests/local_tests/<model_family>/PORT_STATUS.md
- official_definition_files: <paths + classes/functions>
- official_instantiation_files: <paths + factory/pipeline/config call sites + args>
- concerns_or_unknowns: <known ambiguous inputs, outputs, deps, or args>

The complete per-component packet must match
`../add-model/contracts/component_context.md`.

Read the official component call path and the planned FastVideo component API.
Add tests/local_tests/<bucket>/test_<family>_<component>_parity.py based on
add-model-02-parity/templates/component_parity_test.py.

The scaffold must load the official model with real weights when available,
load the FastVideo model through the standardized config/class/loader path when
available, create deterministic inputs, compare concrete outputs, and skip only
when a dependency is genuinely missing. Do not make an unconditional skip or a
shape-only test.
```

## FastVideo Load Patterns

Pick the narrowest load path that matches the component:

| Component | Preferred FastVideo load path |
|---|---|
| DiT / transformer | Bucket config + model class, or `TransformerLoader` when testing converted Diffusers component dirs. |
| VAE | VAE class `from_pretrained(...)` when implemented, or bucket config + class for local converted dirs. |
| Text/image encoder | Bucket config + model class; pass HF subpaths from `local_weights_dir` or converted component dirs. |
| Scheduler/conditioner | Native class/config plus exact official kwargs. |

For early scaffolds, an import of the planned FastVideo class may be inside a
helper that calls `pytest.skip` if the class does not exist yet. Replace that
skip with a real import once the component PR adds the class.

Direct class/config construction is allowed for implementation or subcomponent
parity, such as connector-only encoder checks or official monolithic-checkpoint
mapping tests. Label that scope explicitly and add separate production-loader
coverage when converted component dirs are available.

## Official Load Patterns

- Clone/reference repo path: add its source dir to `sys.path` before imports.
- HF/Diffusers reference: import only inside the test, not production code.
- Private deps: add a helper under `tests/local_tests/helpers/` to install
  stubs before importing upstream modules; do not rely on an external upstream
  environment.
- Gated HF repos: resolve `HF_TOKEN`, `HUGGINGFACE_HUB_TOKEN`, or `HF_API_KEY`
  under the token rules in `../add-model/shared/common_rules.md`.

## Non-Skip Activation Checklist

Before `/add-model` handoff, each scaffolded test must be activated:

```text
[ ] Official side imports and loads real weights.
[ ] FastVideo side imports and loads the converted or original weights.
[ ] Test executes at least one real forward call on both sides.
[ ] Test compares output tensors, not only shapes or state-dict keys.
[ ] Local pytest output contains PASSED, not SKIPPED or XFAIL.
[ ] Tolerance is justified for the component scope and kernel alignment.
```

## Component Parity Details

Reference imports:

- Import from `official_ref_dir` or the recorded package/import path.
- If upstream has private deps, add a helper under
  `tests/local_tests/helpers/<family>_upstream.py` that installs minimal stubs
  before importing upstream modules.
- Common stubs: identity compile/op-registration decorators, CP world size set to
  1, identity scatter/gather, and test-friendly custom-op kernels.
- Stub decorators that register `torch.ops.<ns>.<op>` must preserve the
  `torch.library` registration side-effect. Identity decorators alone are not
  enough.
- Delete stub helpers and every `install_stubs()` call as soon as the real deps
  become required installs. No-op shims are dead code.

Kernel and wrapper pitfalls:

- If parity routes flash-attn GQA through SDPA, expand KV heads manually on the
  SDPA side with `repeat_interleave` along the head axis.
- If upstream VAE `decode()` denormalizes internally but FastVideo/Diffusers
  expects pre-denormalized latents, apply `z = z * std + mean` only on the
  FastVideo side in the parity test.
- Per-channel VAE `latents_mean` / `latents_std` must be reshaped explicitly,
  e.g. `.view(1, z_dim, 1, 1, 1)` for 5D video latents.

Tolerance guide:

| Scope | Start `atol` / `rtol` | Notes |
|---|---|---|
| Single block, same kernel | `1e-4` / `1e-4` | Tight default. |
| Full DiT, aligned kernels | `1e-2` / `1e-2` | Cross-layer accumulation. |
| Full DiT, cross-kernel bf16 | `0.1` / `0.1` | Also require abs-mean drift below 5% and per-modality diagnostics. |
| VAE decode fp32 | `5e-2` / `5e-2` | After normalization alignment. |
| Encoder wrapper around same HF class | `1e-3` / `1e-3` | Should be near-zero. |

Element-wise `assert_close` alone is not enough for deep full-DiT parity. Also
log global abs-mean drift and per-modality summaries.

Useful local commands:

```bash
pytest tests/local_tests/<bucket>/test_<family>_*parity*.py -v -s
pytest tests/local_tests -k "<family> and parity" -v -s
```

## Escape Hatches

Follow `../add-model/shared/common_rules.md`. Parity-specific ask cases include
private dependency approval, choosing between incompatible official references,
accepting a shape-only substitute, or loosening required tolerances.

## Pipeline Parity

Pipeline parity is later than component parity because it needs stages, presets,
registry wiring, converted weights, and green component parity. Record official
pipeline call notes in `local_tests_readme`, but do not treat pipeline parity as
owned by this skill.

Use `../add-model-09-pipeline/SKILL.md` and its
`templates/pipeline_parity_test.py` for pipeline parity scaffolding and
debugging. Compare denoised latents or decoded media, not just successful
generation.

## Handoff Status Block

Return `../add-model/contracts/parity_status.md` to `/add-model` and update the
shared state files before handoff.
