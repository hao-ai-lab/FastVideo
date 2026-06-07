# Cosmos3 local parity workspace

## Overview

This workspace tracks the FastVideo Cosmos3 port. Live port state, component matrix,
decisions, and blockers live in `PORT_STATUS.md`.

- **Reference (2026-06-06): official NVIDIA `cosmos-framework` diffusers backend** —
  `Cosmos3OmniDiffusersPipeline` from the `diffusers-cosmos3` shim — loading the
  now-public `nvidia/Cosmos3-Nano` checkpoint.
- **Scope: full omni** — T2V / I2V / T2I, audio (sound generation), VLM reasoning,
  and action-conditioning.
- The original Tier-A scaffold was written against vllm-omni PR #3454 before official
  weights were public; it is being repointed to the diffusers reference (see I001/I002
  in `PORT_STATUS.md`).

## Reference code

Primary (official):

- Local: `cosmos-framework/` (symlink -> `/home/william5lin/FastVideo/cosmos-framework`,
  commit `003d66d4`); GitHub <https://github.com/NVIDIA/cosmos-framework>
- diffusers shim `cosmos-framework/packages/diffusers-cosmos3/diffusers_cosmos3/`:
  - `pipeline.py` — `Cosmos3OmniDiffusersPipeline`
  - `transformer.py` — `Cosmos3OmniTransformer`
  - `sequence_packing.py`
- framework model code: `cosmos_framework/model/vfm/mot/cosmos3_vfm_network.py`,
  `cosmos_framework/model/vfm/omni_mot_model.py`
- Installed editable in shared `fv-main`: `diffusers-cosmos3`, `cosmos-framework`
  (both `--no-deps`).

Original Tier-A reference (superseded, kept for diffing during repoint):

- vllm-omni PR #3454 <https://github.com/vllm-project/vllm-omni/pull/3454>, pinned
  `8536f5b1`, checkout `/home/william5lin/cosmos3-reference`.
- The current `conftest.py` + tests still mirror this suite line-by-line.

## Weight status

DOWNLOADED (2026-06-06). `nvidia/Cosmos3-Nano` is now public and diffusers-format
(the 2026-05-22 `401` is resolved).

- Local: `official_weights/cosmos3/` (symlink -> main worktree; 33 GiB, 67 files,
  `model_index.json` present)
- Source: `nvidia/Cosmos3-Nano`, default revision; `source_layout=diffusers`,
  `needs_conversion=no`
- `model_index` class: `Cosmos3OmniDiffusersPipeline` (diffusers 0.37.1)
- Token: not required (public repo)

Components (from `model_index.json`): `transformer` (`Cosmos3OmniTransformer`),
`vae` (`AutoencoderKLWan`), `scheduler` (`UniPCMultistepScheduler`),
`text_tokenizer` (`Qwen2TokenizerFast`), `vision_encoder` (`Qwen3VLVisionModel`),
`sound_tokenizer` (`Cosmos3AVAEAudioTokenizer`).

## Running the Tier-A scaffold

```bash
PYTHONPATH=/home/william5lin/FastVideo_cosmos3_port \
  python -m pytest tests/local_tests/cosmos3/ -q
```

NOTE: as of 2026-06-06 these report `15 skipped` because the shared `fv-main` env's
editable `fastvideo` resolves to the MAIN worktree (a PEP660 finder overrides
`PYTHONPATH`), so the worktree's cosmos3 modules are not importable. Tracked as E001
in `PORT_STATUS.md`.

## SSIM placeholder

No SSIM references seeded yet. Add SSIM coverage only after a FastVideo inference path
can load the Cosmos3 weights and generate stable T2V/I2V/T2I outputs. Audio quality
uses a separate metric (not SSIM); see `PORT_STATUS.md` Q003.
