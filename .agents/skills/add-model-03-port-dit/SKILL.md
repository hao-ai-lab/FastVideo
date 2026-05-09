---
name: add-model-03-port-dit
description: Use during /add-model Phase 4 or Phase 6 to prototype or parity-debug one FastVideo-native DiT/transformer component.
---

# Add Model Port DiT

## Goal

Prototype or parity-debug one diffusion transformer in FastVideo-native code.
This skill is for one component only; do not work on the VAE, encoders,
pipeline, or unrelated conversion code unless the current component cannot load
without a minimal fix there.

## Inputs

Follow `../add-model/shared/component_skill_common.md` and require the complete
packet from `../add-model/contracts/component_context.md`.

DiT-specific packet fields:

- `component`: transformer or DiT name.
- `parity_test`: `tests/local_tests/<bucket>/test_<family>_<component>_parity.py`.
- `weights`: converted transformer dir or local official path.
- `target_files`: `fastvideo/models/dits/<family>.py` and
  `fastvideo/configs/models/dits/<family>.py`.

## Modes

Use the common prototype and parity-debug modes from
`../add-model/shared/component_skill_common.md`.

DiT-specific prototype concerns include ambiguous official flags, shape
mismatches, missing FastVideo layer equivalents, and dedicated output heads.

## Reuse Proof

Apply the shared reuse proof. DiT-specific comparison must include attention
algorithm, positional embeddings, RoPE/patching, timestep/guidance embeddings,
scaling constants, dtype casts, state-dict names, and every output head.

## Existing FastVideo Patterns

- Base class: `fastvideo/models/dits/base.py::BaseDiT`.
- Config bases: `DiTConfig` and `DiTArchConfig` in
  `fastvideo/configs/models/dits/base.py`.
- Use the matching DiT config bucket. Wrong bucket inheritance can typecheck but
  fail during pipeline wiring.
- Config export: add the config to
  `fastvideo/configs/models/dits/__init__.py`.
- Registry discovery: set `EntryClass = <ClassName>` in the model file.
- Loader path: `TransformerLoader` reads `transformer/config.json`, calls
  `dit_config.update_model_arch(config)`, resolves `_class_name` through
  `ModelRegistry`, and constructs the class with `config` and `hf_config`.
- Reference examples: `stable_audio.py`, `wanvideo.py`, `sd3.py`, `longcat.py`,
  and `ltx2.py`.
- Layer guidance: `fastvideo/layers/AGENTS.md`.

## Implementation Rules

- Use FastVideo-native layers by default: `ReplicatedLinear` for DiT hot-path
  linears, `DistributedAttention` for standard full-sequence attention, and
  `LocalAttention` for local/window attention or simple single-GPU parity paths.
- Raw SDPA is acceptable for cross-modality flat streams when no FastVideo
  distributed equivalent exists; document the SP gap in the module docstring.
- Mirror official tensor contracts exactly: latent packing, patch ordering,
  timestep embedding scale, RoPE/positional embedding, guidance embedding,
  cross-attention context order, output head order, and dtype casts.
- Preserve all output heads that the official DiT emits. Do not silently drop
  audio, depth, pose, mask, or auxiliary heads.
- Put architecture fields on `DiTArchConfig`; keep inference steps, CFG scales,
  FPS, flow shift, and sampling defaults out of the arch config.
- Define `_fsdp_shard_conditions`, `_compile_conditions`,
  `param_names_mapping`, and `reverse_param_names_mapping` where needed.
- Follow the production import boundary in
  `../add-model/shared/common_rules.md`.

## Prototype Checks

Follow the shared prototype success criteria. A useful one-off check is:

```bash
python - <<'PY'
# Import the target config/class, instantiate with random weights, and print
# state_dict names/shapes for the conversion mapping.
PY
```

## Parity-Debug Loop

Run the shared parity-debug loop. The component test command is:

```bash
pytest <parity_test> -v -s
```

For numerical drift, narrow the first divergent block with per-block hooks or
intermediate tensor comparisons before changing layers.

## Escape Hatches

Follow `../add-model/shared/common_rules.md` and the component-specific guidance
in `../add-model/shared/component_skill_common.md`. DiT-specific ask cases include
dropping an output head/modality, accepting an unsupported kernel/private op, or
choosing between incompatible official transformer definitions.

## Handoff

Return `../add-model/contracts/component_skill_handoff.md` following the common
handoff rules in `../add-model/shared/component_skill_common.md`.
