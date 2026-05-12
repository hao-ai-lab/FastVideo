---
name: add-model-05-port-encoder
description: Use during /add-model Phase 4 or Phase 6 to prototype or parity-debug one FastVideo-native text, image, audio, or compound encoder component.
---

# Add Model Port Encoder

## Goal

Prototype or parity-debug one encoder or encoder-like conditioner in
FastVideo-native code. Use this for text encoders, image encoders, audio
encoders, and compound conditioners that fit the encoder config/loader bucket.

## Inputs

Follow `../add-model/shared/component_skill_common.md` and require the complete
packet from `../add-model/contracts/component_context.md`.

Encoder-specific packet fields:

- `component`: encoder or encoder-like conditioner name.
- `parity_test`: `tests/local_tests/encoders/test_<family>_<component>_parity.py`.
- `weights`: converted encoder dir, HF subfolder, or external HF id.
- `target_files`: `fastvideo/models/encoders/<arch_or_family>.py` and
  `fastvideo/configs/models/encoders/<arch_or_family>.py`.

## Modes

Use the common prototype and parity-debug modes from
`../add-model/shared/component_skill_common.md`.

Encoder-specific prototype concerns include tokenizer kwargs, hidden-state
extraction, output packing, connector order, and external/passthrough weight
needs.

## Reuse Proof

Apply the shared reuse proof. Encoder-specific comparison must include tokenizer
contracts, hidden-state extraction, masks, positional IDs, output packing,
connector/projection ordering, passthrough paths, and returned dataclass shape.

## Existing FastVideo Patterns

- Base classes: `TextEncoder` and `ImageEncoder` in
  `fastvideo/models/encoders/base.py`.
- Output type: `BaseEncoderOutput`.
- Config bases: `TextEncoderConfig`, `ImageEncoderConfig`,
  `TextEncoderArchConfig`, and `ImageEncoderArchConfig` in
  `fastvideo/configs/models/encoders/base.py`.
- Use the matching encoder config bucket. Wrong bucket inheritance can typecheck
  but fail during pipeline wiring.
- Config export: add the config to
  `fastvideo/configs/models/encoders/__init__.py`.
- Registry discovery: set `EntryClass = <ClassName>` or a list of class names in
  the model file.
- Reference examples: native `t5.py`, `clip.py`, `siglip.py`, `llama.py`,
  `qwen2_5.py`, `gemma.py`, and compound `stable_audio_conditioner.py`.
- Layer guidance: `fastvideo/layers/AGENTS.md`.

## Implementation Rules

- Reuse tokenizers and pure data utilities when needed, but do not add runtime
  third-party model-class imports as a placeholder for a component that owns
  weights or numerical behavior.
- For LLM-style encoders, follow existing tensor-parallel patterns such as
  `QKVParallelLinear`, `MergedColumnParallelLinear`, `RowParallelLinear`,
  `VocabParallelEmbedding`, and `RMSNorm` when matching native examples.
- Match official hidden-state extraction exactly: layer index, pooled output,
  attention mask dtype, padding side, truncation, special tokens, final norm,
  output_hidden_states, and returned tuple/dataclass shape.
- For connector or conditioner modules, preserve sub-conditioner order and the
  exact packing of cross-attention tokens, masks, and global conditioning.
- Put tokenizer kwargs and architecture constants on the arch config when they
  affect numerical behavior.
- If an external HF encoder is explicitly accepted as a lazy wrapper, keep it
  isolated, document why it is not a native port, and still require parity for
  the wrapper's output contract.

Hybrid external-HF encoder checklist:

- Put external model folders in passthrough subfolders such as
  `text_encoder/<external_name>/`, or record a root `model_index.json` path field
  that the loader resolves to a local directory.
- Keep external model parameters out of the FastVideo-owned state-dict surface
  when the external model is loaded lazily from its own HF files.
- Convert and strict-check only the FastVideo-owned connector/projection weights;
  document external model weights as passthrough.
- Add parity for the wrapper's final output contract and, when useful, a narrower
  connector-only parity test that labels its scope as
  `implementation_subcomponent`.
- Verify the production loader resolves the same external path used by the
  pipeline, not just the direct class used in the parity test.

## Prototype Checks

Follow the shared prototype success criteria.

## Parity-Debug Loop

Run the shared parity-debug loop. The component test command is:

```bash
pytest <parity_test> -v -s
```

For numerical drift, check tokenization, masks, hidden-state selection,
positional IDs, dtype/autocast, and output packing before changing layers.

## Escape Hatches

Follow `../add-model/shared/common_rules.md` and the component-specific guidance
in `../add-model/shared/component_skill_common.md`. Encoder-specific ask cases
include accepting private model-code execution, choosing between incompatible
tokenizer/encoder references, or dropping a required conditioning stream.

## Handoff

Return `../add-model/contracts/component_skill_handoff.md` following the common
handoff rules in `../add-model/shared/component_skill_common.md`.
