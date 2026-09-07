# Wan model family

This package owns the dense and causal Wan transformers (`transformer.py`,
`causal_transformer.py`), their architecture,
FSDP predicates, and checkpoint/LoRA mappings (`config.py`), and the Wan VAE
(`vae.py`) with its config (`vae_config.py`). The VAE includes both the video
encoder and decoder. `pipeline_config.py` owns component and pipeline defaults;
`definition.py` links supported variants to those configs and existing sampling
presets. Shared text/image encoders, VAE utilities, pipelines, sampling presets,
and training adapters remain in their existing directories.
Other families reuse these classes through canonical imports. Old paths remain
explicit compatibility exports.

## Invariants

- Keep `__init__.py` lightweight: no eager model or pipeline imports.
- Keep `definition.py` data-only. Config and preset names reference their
  existing owners; do not copy defaults into the catalog. Its registration
  groups preserve first-match detector ordering around DreamX. Checkpoint
  manifests and explicit pipeline overrides still choose the executable
  pipeline; a definition must not silently pin that choice.
- Keep `configs.pipelines.wan` as explicit compatibility aliases, including
  `t5_postprocess_text` for old serialized configs. Shared encoder classes
  remain in `configs.models.encoders` and `models.encoders`.
- Import configs directly from `fastvideo.models.wan.config` or
  `fastvideo.models.wan.vae_config` in the matching component.
- Preserve the old `models.dits.wanvideo` and `configs.models.dits.wanvideo`
  modules as explicit aliases, not subclasses or duplicate implementations.
- Keep `models.vaes.wanvae` and `configs.models.vaes.wanvae` as explicit aliases
  too, including the VAE's cache context variables and compile predicate.
- Keep `EntryClass = WanTransformer3DModel` in `transformer.py` only. Registry
  architecture names, state-dict keys, layer names, and mappings are compatibility
  contracts.
- Keep `EntryClass = CausalWanTransformer3DModel` in `causal_transformer.py`
  only; `dits/causal_wanvideo.py` is an alias. Preserve cache layout, sink
  eviction, absolute/relativistic RoPE, and the global-attention window limit.
- Keep `EntryClass = AutoencoderKLWan` in `vae.py` only. Preserve latent
  normalization, first-frame handling, cache reset, streaming, tiling, and
  encoder/decoder compile conditions. Do not merge the separate Cosmos25,
  Gen3C, or LingBotWorld2 VAE adapters into this implementation.
- `config.py`, `vae_config.py`, `pipeline_config.py`, `definition.py`, and
  `__init__.py` are pre-commit checked;
  `transformer.py` and `vae.py` retain the existing model-code exclusion.
  Avoid unrelated reformatting.

## Focused checks

Follow the [testing guide](../../../docs/contributing/testing.md): cheap
compatibility checks first, then the smallest golden covering the changed
component, before default or full-quality renders. Imports may require GPU
dependencies even when a check needs no weights.

```bash
pytest fastvideo/tests/api/test_wan_definitions.py -q
pytest fastvideo/tests/loader/test_wan_family_imports.py -q
pytest fastvideo/tests/contract/test_merge_ci_plan.py -q
pytest fastvideo/tests/vaes/test_wan_vae_compile.py -q
```

Use `bash scripts/validate_wan.sh <vae|dense|causal|all> [golden|parity|default]`
from the repo root; it stops at the first failed boundary. The default is the
small golden tier. The four Wan gates cover dense block 0, VAE encode/decode
and streaming, causal block cache updates, and a three-step dense trajectory.
The independent Diffusers component checks and focused SSIM remain available
at higher tiers. A relocation needs unchanged-parent numerical evidence;
aliases alone are not proof. See `basic/wan/AGENTS.md` under pipelines for
sampling ownership; shared T5/UMT5 and CLIP stay shared.
