# Cosmos3 local parity workspace

## Overview

This workspace tracks the FastVideo Cosmos3 port. Live port state, component matrix,
decisions, and blockers live in `PORT_STATUS.md`.

- **Reference: official NVIDIA `cosmos-framework`** — the framework is the parity
  oracle for every modality; the public `nvidia/Cosmos3-Nano` repository supplies
  the diffusers-layout checkpoint only.
- **Public scope:** registered T2V/I2V, plus a T2I path/preset and opt-in T2VS.
  Action and reasoning are native component-parity surfaces, not first-class
  public pipeline workloads.
- The original Tier-A scaffold was written against vllm-omni PR #3454 before official
  weights were public; it was superseded by framework-based parity coverage.

## Reference code

Primary (official):

- GitHub: <https://github.com/NVIDIA/cosmos-framework>
- The transformer fingerprint seed pins framework revision
  `ed8287fd7477113f8ac4f6b84290514d55cf0cdc`.
- framework model code: `cosmos_framework/model/vfm/mot/cosmos3_vfm_network.py`,
  `cosmos_framework/model/vfm/omni_mot_model.py`

Original Tier-A reference (superseded, kept for diffing during repoint):

- vllm-omni PR #3454 <https://github.com/vllm-project/vllm-omni/pull/3454>, pinned
  `8536f5b1`, checkout `/home/william5lin/cosmos3-reference`.
- The remaining reference-specific helpers are retained only for local parity
  tests; production code does not import that project.

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
python -m pytest tests/local_tests/cosmos3/ -q
```

The recorded component-parity suite result is `150 passed, 0 skipped`; see
`PORT_STATUS.md` for the date, scope, and limitations of that evidence.

## SSIM placeholder

No Cosmos3 SSIM references are seeded yet. Select and seed public T2V/I2V/T2I
cases before treating generated-media quality as CI-covered. Audio quality needs
a separate metric because SSIM cannot cover it.

## Transformer CI fingerprint

The routine transformer gate runs separate depth-one FastVideo T2V, T2VS,
action2world, and deepstack-reasoning forwards with production hidden, head,
MLP, latent, sound, and action dimensions. It reads 44 tensors or slices
(408,486,912 parameters, about 779.13 MiB in BF16) from the pinned
`nvidia/Cosmos3-Nano` checkpoint cache. It hashes the production/fingerprint
config contract, fixed inputs, selected weights, and every named output.

The golden was seeded and verified on Modal L40S using the pinned CI image. Seed
mode also proved the captured FastVideo decoder-layer outputs for all three
generation cases bit-exact against the pinned NVIDIA framework layer. Normal
mode reran every case twice from the committed hashes and passed. No extracted
fixture or separate weights repository is required; SSIM remains the
end-to-end media gate for uncovered surfaces.
