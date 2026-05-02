# Layer Guidance For Model Ports

Use this file when adding FastVideo-native model components. Keep it generic:
model-specific parameter mappings belong in `scripts/checkpoint_conversion/`, not
in this directory.

## Linear Layers

- Use `ReplicatedLinear` for DiT and VAE hot paths when the layer is not tensor
  parallel and should expose a normal `weight`/`bias` state-dict surface.
- Use `QKVParallelLinear` for LLM-style fused query/key/value projections when
  the existing encoder pattern already expects tensor parallel loading.
- Use `MergedColumnParallelLinear` for fused MLP gate/up projections that are
  loaded as packed column shards.
- Use `ColumnParallelLinear` and `RowParallelLinear` for tensor-parallel encoder
  blocks that follow existing `t5.py`, `clip.py`, `llama.py`, or `qwen2_5.py`
  patterns.
- Do not replace a simple official layer with a fused FastVideo layer unless the
  conversion script explicitly handles the resulting key and tensor layout.

## Attention Layers

- Use `DistributedAttention` for standard DiT full-sequence attention when the
  model should participate in sequence parallel execution.
- Use `LocalAttention` for local/window attention or narrow single-GPU parity
  paths that match existing component style.
- Raw `torch.nn.functional.scaled_dot_product_attention` is acceptable for
  unusual cross-modality flat streams when no FastVideo distributed primitive
  matches yet. Document the sequence-parallel gap in the owning model file.

## State-Dict Surface

- Prototype the native component before writing conversion mappings. The
  prototype's `state_dict()` is the source of truth for FastVideo target keys and
  shapes.
- Conversion scripts should map official keys into the native state-dict surface;
  production model code should not be contorted to match checkpoint naming.
- Fused and packed FastVideo layers may require tensor split/fuse logic in the
  converter, especially QKV/KV projections and gated MLP projections.
- Record intentional skipped keys in the conversion script with a reason, such
  as training-only EMA/logvar/optimizer state or dynamically computed buffers.

## Porting Discipline

- Match the official layer definition and the official instantiation arguments.
  A reusable class with different constructor args is not reused.
- Keep architecture constants on the component arch config. Runtime sampling,
  guidance, precision, and pipeline defaults belong on pipeline config or
  presets.
- Prefer small, direct implementations until parity passes. Add helpers only
  when they serve multiple call sites or make the mapping clearer.
