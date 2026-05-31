---
name: add-model-07-conversion
description: Use during /add-model Phase 5 to write and verify a FastVideo checkpoint conversion script after native component prototypes expose FastVideo state-dict keys/shapes.
---

# Add Model Conversion

## Goal

Convert official weights into a FastVideo-loadable component layout after Phase 4
native prototypes exist. The conversion script owns parameter mapping, component
splitting, passthrough assets, config emission, and strict-load verification.

## Inputs

Follow `../add-model/shared/common_rules.md` for token/auth safety, state files,
escape hatches, production boundaries, and skip/pass semantics.

Require the initial request from
`../add-model/contracts/conversion_request.md`.

If the FastVideo key/shape dump is missing, return to `/add-model` Phase 4. Do
not write a final mapping against an unimplemented component.

For Phase 6 retry requests from component skills, also require the retry shape
from `../add-model/contracts/conversion_request.md`.

## Output

- `scripts/checkpoint_conversion/<family>_to_diffusers.py`.
- `converted_weights/<family>/` with `model_index.json` and per-component
  subfolders.
- Updated `tests/local_tests/<model_family>/README.md` with conversion command,
  source layout, output path, and strict-load status.
- Updated `tests/local_tests/<model_family>/PORT_STATUS.md` with conversion
  state, retry history, open questions, and issues/blockers.

## Reference Scripts

- `scripts/checkpoint_conversion/convert_ltx2_weights.py`: component prefix
  splitting, metadata config extraction, passthrough Gemma/tokenizer assets, and
  optional component-only output.
- `scripts/checkpoint_conversion/stable_audio_to_diffusers.py`: monolithic
  `model.safetensors` split into transformer/VAE/conditioner, plus copied
  passthrough subfolders. Use this shape for single-checkpoint official repos.
- `scripts/checkpoint_conversion/convert_gamecraft_full.py`: separate official
  sources for transformer, VAE, text encoders, tokenizers, scheduler, and root
  `model_index.json`.
- `scripts/checkpoint_conversion/longcat_to_fastvideo.py`: fused QKV/KV split,
  renamed native transformer weights, and copied existing Diffusers components.
- `scripts/checkpoint_conversion/pt_to_safetensors.py`: simple `.pt` extraction
  helper for nested checkpoint dictionaries.

## Source Layout Decision

Choose exactly one primary layout:

| Layout | Conversion behavior |
|---|---|
| `diffusers` | Usually no tensor remap; verify configs/classes and copy or update `_class_name` only when needed. |
| `raw_official` | Convert a raw official checkpoint file or directory. Choose explicit component ownership before writing output. |
| `separate_components` | Convert/copy each component from its own file or directory. |
| `monolithic` | Load one model checkpoint and split state dict by authoritative prefixes into component buckets. |
| `mixed` | Convert some components and copy passthrough components such as tokenizers, text encoders, schedulers, or already-Diffusers VAE dirs. |
| `custom` | Document why none of the above fits before writing conversion code. |

Monolithic checkpoints need explicit prefix ownership. For example, Stable Audio
uses one `model.safetensors` with DiT, pretransform/VAE, and conditioner keys;
the converter splits those keys into FastVideo component subfolders and writes
per-component configs.

## Script Shape

Start from `templates/family_to_diffusers.py` or the closest reference script.
Keep the script explicit and reviewable:

- `COMPONENT_SPECS` or `COMPONENT_PREFIXES` declares component ownership.
- `PARAM_NAME_MAP` declares key renames.
- `SKIP_PATTERNS` declares intentionally dropped training-only keys.
- tensor split/fuse helpers are named by operation, e.g. `split_qkv`.
- `build_component_configs(...)` writes loader-compatible config files. Most
  model components use `config.json`; schedulers use `scheduler_config.json`.
- `build_model_index(...)` writes a root `model_index.json` matching the target
  FastVideo pipeline and component classes.
- verification reports missing, unexpected, skipped, unchanged, renamed, and
  shape-mismatched keys.

`model_index.json` library tokens must match FastVideo loaders:

- standard native DiT/VAE/audio/vocoder/upsampler components loaded by existing
  Diffusers-style loaders usually use `"diffusers"` with a FastVideo
  `_class_name` in the component `config.json`;
- text encoders, tokenizers, image encoders, processors, and feature extractors
  usually use `"transformers"`;
- `conditioner` currently expects `"fastvideo"`;
- use fully qualified `"fastvideo.<module>"` only when intentionally relying on
  the custom fastvideo-library escape path;
- do not write bare `"fastvideo"` for transformer, VAE, or other loaders that
  expect `"diffusers"` unless the loader explicitly expects it.

## Mapping Rules

- Use Phase 4 key/shape dumps to derive mappings. Do not guess from official key
  names alone.
- Preserve each component's official file paths, parity test path, and prototype
  concerns in comments or structured constants near the mapping that uses them.
- Every official inference parameter should be mapped, copied through, or listed
  as intentionally skipped with a reason.
- Every FastVideo prototype parameter should receive a tensor or be listed as an
  intentional external/passthrough parameter.
- Shape matches are necessary but not sufficient; check semantic pairing for
  Q/K/V, gate/up/down, norm scale/bias, LoRA/base, and modality-specific heads.
- If official and FastVideo fuse or split tensors differently, convert tensors in
  the script rather than changing production code to match checkpoint quirks.

## Verification

Run conversion locally, then verify before returning to Phase 6. For retry
requests, update the mapping, rerun conversion, and refresh only the implicated
converted component when safe; otherwise rerun the full conversion.

```bash
python scripts/checkpoint_conversion/<family>_to_diffusers.py \
    --src <official_weights> \
    --revision <hf_revision> \
    --dst converted_weights/<model_family>
```

Omit `--revision` for local sources or when prep recorded `default` / `none`.

Minimum output layout:

```text
converted_weights/<family>/
  model_index.json
  transformer/config.json
  transformer/*.safetensors
  vae/config.json
  vae/*.safetensors
  scheduler/scheduler_config.json as needed
  text_encoder/... as needed
```

Required checks:

- `model_index.json` exists and lists every required component.
- Each converted component has the config filename its loader expects and
  safetensors weights when it owns weights. Scheduler dirs require
  `scheduler_config.json`; most other native model dirs use `config.json`.
- Weight filenames may vary by loader: transformer and VAE loaders glob all
  `*.safetensors`; text encoders may load `*.safetensors`, `*.bin`, and
  sometimes `*.pt`; `conditioner` currently expects
  `diffusion_pytorch_model.safetensors`. Use the loader's actual accepted layout
  rather than assuming one global filename.
- Passthrough components are copied or referenced deliberately.
- Each emitted component config validates through the same path production
  loaders use. Instantiate the relevant config and call `update_model_arch(...)`
  or `update_model_config(...)` with the emitted JSON so unknown keys fail during
  conversion, not at pipeline load time.
- Record production loader strictness for every stateful component. If the loader
  intentionally uses non-strict loading, add explicit missing/unexpected-key
  assertions in the parity test and document exactly which keys are allowed.
- Each new FastVideo component strict-loads converted weights where its production
  loader is strict. If strict loading is impossible, record the exact allowed
  missing/unexpected keys and why they are not inference weights.
- Retry fixes include the original component parity evidence and the new
  strict-load result in `local_tests_readme` so the component subagent can resume
  without rediscovering context.
- `local_tests_readme` records the command, output directory, and strict-load
  result.

Do not chase numerical parity in this skill except to identify a conversion
mapping bug. Long parity-debug loops belong to `/add-model` Phase 6.

## Escape Hatches

Follow `../add-model/shared/common_rules.md`. Conversion-specific ask cases
include selecting between incompatible official checkpoints, publishing/uploading
weights, overwriting an existing converted repo not created by this run,
accepting non-strict missing inference weights, or dropping a component/output
from scope.

## Handoff

Return `../add-model/contracts/conversion_handoff.md` and update the shared state
files before handoff.
