# Direct loading of a packed MiniMax-H3 VAE decoder

Status: design only. The implementation and measurements below are acceptance
work, except for the explicitly identified CPU reproduction.

## Decision and scope

The opportunity exists: a packed decoder currently replaces a fully constructed
and loaded dense CPU decoder. Add a specialized loader for
`AutoencoderKLMiniMaxH3` with an explicitly selected
`fastvideo_h3_decoder_nvfp4_packed_v1` safetensors export. Build topology on
`meta`, materialize the encoder and `quant_conv` from the base checkpoint, and
materialize the decoder and `post_quant_conv` from the export.

This targets host/unified-memory startup pressure, initialization, checkpoint
payload access, and copies. The current swap already keeps the discarded dense
decoder off the GPU. Do not advertise a reduction in its steady-state GPU
storage, denoising latency, or warm decode latency. No quantitative startup
savings have been measured. Safetensors mapping, page faults, filesystem cache,
and allocator reuse prevent equating checkpoint bytes with peak RSS savings.

Keep dense VAE loading, `.pt` deploy exports, bf16-master safetensors exports,
other VAEs, and inference kernels on their existing paths. No format migration,
new user option, general loader rewrite, or encoder removal is needed.

## Verified current path

Evidence is pinned to upstream commit
[`cb6a9a13e1cc54810da19747f57501ef9b06d86f`](https://github.com/hao-ai-lab/FastVideo/commit/cb6a9a13e1cc54810da19747f57501ef9b06d86f).
A live upstream-main read on 2026-11-01 returned the same loader blob
`8cea9dde7949768a86eea987510998e39f4ec478` as this checkout.

1. [`VAELoader.load`](https://github.com/hao-ai-lab/FastVideo/blob/cb6a9a13e1cc54810da19747f57501ef9b06d86f/fastvideo/models/loader/component_loader.py#L975-L1045)
   constructs the complete VAE. NVFP4 selection postpones its device move.
   It merges every dense safetensors shard into `loaded`, strictly copies the
   complete state, deletes `loaded`, then applies the selected export.
2. [`AutoencoderKLMiniMaxH3.__init__`](https://github.com/hao-ai-lab/FastVideo/blob/cb6a9a13e1cc54810da19747f57501ef9b06d86f/fastvideo/models/vaes/minimax_h3_video.py#L565-L655)
   builds encoder, both convolutions, and the dense decoder, then calls
   `self.float()`. The release geometry therefore has FP32 dense parameters
   even under the loader's lower-precision default-dtype context.
3. [`_apply_h3_nvfp4_decoder`](https://github.com/hao-ai-lab/FastVideo/blob/cb6a9a13e1cc54810da19747f57501ef9b06d86f/fastvideo/models/loader/component_loader.py#L830-L861)
   loads the export where the VAE was built, applies it, moves the VAE, and
   freezes CUDA linears. A packed export is already frozen and needs no packing.
4. [`apply_nvfp4_decoder_checkpoint`](https://github.com/hao-ai-lab/FastVideo/blob/cb6a9a13e1cc54810da19747f57501ef9b06d86f/fastvideo/models/vaes/minimax_h3_nvfp4_decoder.py#L444-L498)
   depth-cuts before selecting six block linears per non-skipped block. The
   exported keys address the resulting reindexed block list. Norms, register
   tokens, `proj_in`, `proj_out`, and skipped blocks remain dense.
5. [`apply_packed_nvfp4_decoder`](https://github.com/hao-ai-lab/FastVideo/blob/cb6a9a13e1cc54810da19747f57501ef9b06d86f/fastvideo/models/vaes/minimax_h3_nvfp4_checkpoint.py#L115-L172)
   constructs frozen layers using `from_packed`, pops their packed buffers from
   its working dictionary, then strictly loads remaining decoder state and
   `post_quant_conv`. The quantized master weight is absent. The packed buffers
   are nonpersistent; bias and `input_amax` remain persistent.

## CPU evidence and the meta trap

A weight-free reproduction used the existing `_tiny_vae_arch(3)`,
`_tiny_h3_vae`, and `_fake_packed_checkpoint` test helpers, Python 3.12 and
PyTorch 2.12 on macOS CPU. It loaded no external weights and ran no NVFP4 GEMM.
The real tiny VAE allocated 858,512 parameter bytes, including 154,368 decoder
bytes. These are fixture sizes, not production memory estimates.

Constructing the same class under `with torch.device("meta")`, then applying
`load_state_dict(real.state_dict(), strict=True, assign=True)`, left exactly:

```text
latents_mean, latents_std, pixel_mean, pixel_std, decoder.rope.inv_freq
```

All five are nonpersistent buffers. `.to("cpu")` then raised
`NotImplementedError: Cannot copy out of meta tensor`. Calling the existing
packed apply function directly on a meta VAE left 19 decoder parameters meta
and produced 21 copy-to-meta warnings; strict success alone is insufficient.
The frozen packed layer's bias remained bf16 and its four buffers were real
CPU tensors. This verifies the failure mechanism, not the proposed loader.

## Proposed control flow and ownership

`VAELoader` remains the dispatcher. A new H3-only helper, for example
`load_h3_packed_vae(model_path, vae_config, decoder_path, target_device)`, lives
in `models/vaes/minimax_h3_packed_loader.py`. It owns checkpoint routing,
manifest validation, and materialization, returning an eval VAE.

1. Resolve the decoder through the existing argument-before-environment rule.
   Validate its existence and inspect safetensors headers before constructing
   the VAE. Dispatch only for the packed format; malformed selected exports
   must raise, never silently fall back. Non-packed valid exports retain their
   existing path. Preserve `model_paths["vae"]` and config-update behavior.
2. Enter `torch.inference_mode(False)` and `torch.no_grad()` while creating
   parameters/buffers, so later fused-state version checks remain usable even
   when the caller is in inference mode. Build the unchanged H3 class on meta.
   This reuses the actual encoder, decoder geometry, temporal planning, and
   tiling defaults rather than maintaining another architecture definition.
3. Before depth-cutting, retain the dense `state_dict` key/shape/dtype manifest.
   Discover base shards with the existing glob and ConvRot-overlay exclusion.
   Inspect all effective base keys and shapes using `safe_open`/`get_slice`,
   including decoder keys whose payload will be skipped. Preserve strict base
   missing/unexpected/shape rejection and the current last-shard-wins behavior
   for duplicates; do not silently sort shards or introduce a new merge rule.
4. Read only `encoder.*` and `quant_conv.*` base payloads with `get_tensor`.
   Cast each to the old destination dtype (FP32), then assign into the meta
   module. A strict load per complete owned submodule is straightforward;
   header validation provides whole-base strictness. Release checkpoint-only
   references as ownership transfers. Do not call `load_file` on base shards.
5. Read the packed export onto CPU once. Reuse the existing metadata-driven
   depth cut, linears selection, `skip_blocks`, rotation, activation mode,
   `num_linears`, and `from_packed` construction. Extend the packed apply API
   with an internal `assign=False` default; the specialized loader uses
   `assign=True` for remaining decoder state and `post_quant_conv`.
   Preserve packed-buffer popping and strict residual loads. Validate the
   residual key/shape manifest before assignment. Never unpack or re-quantize.
6. Assignment retains source dtype, unlike today's copy. Explicitly match the
   old destination dtype for each persistent residual tensor: dense decoder
   parameters and `post_quant_conv` are FP32; a frozen packed layer's bias
   retains its exported compute dtype; `input_amax` is FP32. Keep packed codes,
   scales, and global scales unchanged. Do not call `.float()` after replacing
   linears, which would also cast frozen BF16 biases and scale buffers.
7. Add a small H3-owned runtime-buffer initialization helper, shared by the
   normal constructor and meta materialization. Recreate `latents_mean/std`
   and `pixel_mean/std` from the config/constants on CPU in FP32. Recreate
   `decoder.rope.inv_freq` with the existing rotary class and config's head
   dimension, rope ratio, and theta. Preserve `persistent=False`; do not read
   these buffers from state files or fill them with uninitialized `to_empty`.
8. Assert every named parameter and buffer is real, including nonpersistent
   buffers. Verify selected linears have `weight is None` and real packed
   buffers. Disable gradients, set BF16 decode autocast, retain fused-block
   enablement, move with the existing device-only `.to(target_device)`, and
   return `.eval()`. Keep offload, compile preparation, and parallel decode
   integration in their existing callers.

The packed checkpoint module owns assignment support; the VAE class owns
deterministic runtime buffers. The specialized helper must not reach into
generic FSDP/DiT loading or change the export writer.

## Compatibility and failure contract

- Keep the encoder for all tasks: T2V does not establish that I2V, Omni,
  conditioning, or later calls to `encode` can omit it. Both encode branches
  use the original encoder plus `quant_conv` and FP32 normalization buffers.
- Preserve untiled/tiled encode, spatial tile batching, temporal decode,
  output-buffer decode, and latent normalization geometry. Config remains the
  base geometry; metadata controls the decoder's depth and quantized blocks.
- At the pinned main, a deeper export than the base VAE raises `ValueError`;
  keep that behavior initially. Shallower students keep the same evenly
  spaced indices, then consume reindexed exported weights. `skip_blocks`
  addresses that resulting list; its dense weights come from the export.
- Preserve the INT8-overlay skip when NVFP4 is selected. Preserve missing-base
  `ValueError`, missing-export `FileNotFoundError`, invalid format/header
  errors, architecture-mismatch `ValueError`, and strict missing/unexpected
  residual keys. Dynamic/unit mode may omit `input_amax` and receives zero;
  static mode must provide it. Do not turn optional metadata into required
  fields or invent broader metadata validation in this optimization.
- Match current malformed packed shape and input-width checks. Stronger scale
  layout/dtype validation is separate unless needed to prevent assignment
  from bypassing an existing check. CPU synthetic scales prove loading only;
  they are not valid FlashInfer-format correctness evidence.

## Validation required before implementation can ship

CPU tests should extend the existing H3 VAE loader/packed-export fixtures:

- Compare legacy and specialized loaded keys, shapes, dtypes, and exact tensor
  contents, including frozen packed buffers and all five runtime buffers.
  Assert no meta parameters/buffers, no copy-to-meta warnings, eval mode,
  disabled gradients, BF16 decode autocast, and unchanged encode geometry.
- Trace payload requests: zero `get_tensor` calls for base `decoder.*` and
  `post_quant_conv.*`; fail if base `load_file` is called. Instrument tensor
  allocations during topology construction, not just final parameter counts,
  to prove there is no real dense decoder/master initialization or copy.
- Cover full/light bases, full/shallower exports, depth reindexing, skipped
  dense blocks, static/dynamic/unit modes, optional `num_linears`, wrong
  width/shape, missing/extra base and export keys, duplicate shards, ConvRot
  exclusion, missing files, non-packed fallback, and argument/env precedence.
- Test the caller inside inference mode and verify version-tracked buffers.
  Exercise tiled and untiled CPU encode against legacy, with fixed inputs and
  separate identically seeded posterior generators. Do not run fake packed
  codes as a CPU decode or claim numerical decoder proof from them.

On supported Blackwell hardware with real packed exports and FlashInfer:

- Load legacy and specialized paths in separate processes with identical
  weights, config, precision, device/offload policy, and seeded inputs. Require
  bitwise tensor-state equality and matched encoder posterior statistics;
  compare decode tensors with the established deterministic baseline on the
  same hardware/backend. Any tolerance needs an explained backend source.
- Exercise quality-36L and fast-8L, spatial tiled/untiled and temporal edges,
  available activation modes/rotations/skips, compile on/off, CPU offload then
  repeated load/unload, and supported parallel VAE/SP configurations. Check
  fused and eager paths separately. Report unsupported combinations as gaps.
- Measure fresh-process VAE load time, peak host RSS/PSS where available,
  unified-memory pressure on GB10, major faults/read bytes, and CUDA peak
  allocated/reserved memory. Run cold-cache and warm-cache trials separately;
  record hardware, export/base sizes, versions, repeats, and variability.
  Check warm decode and end-to-end output for regression, not claimed speedup.

## Upstream overlap, risks, and go/no-go

Live PR review on 2026-11-01 found:

- [#2015](https://github.com/hao-ai-lab/FastVideo/pull/2015), open draft at
  `be8a13b7498c8dd0b72d263ca028d5eb5801a9ff`: same loader integration area;
  adds deeper-export replacement of light decoders and optional runtime dense
  quantization. Its loader still constructs/loads the base before replacement.
  Rebase its deeper-topology rule into meta construction if it lands; do not
  use the base's shallower decoder manifest to validate the export topology.
- [#1945](https://github.com/hao-ai-lab/FastVideo/pull/1945), open at
  `579e160660ff7abd078e244680c64756eddb91a6`: streams DiT/FSDP conversion in
  `fsdp_load.py` and `loader/utils.py`; it does not fix this VAE path.
- [#2036](https://github.com/hao-ai-lab/FastVideo/pull/2036), open at
  `96cf7227f8f93da3b348e4eb97be6e6d684e2a6e`: enables existing VAE dispatch
  for Regen2K and edits shared tests. Reuse its integration coverage; its
  reported warm decoder speedups do not measure this startup proposal.

Go to a narrowly scoped implementation: the redundant allocation/read/copy is
confirmed and the API boundary is small. Do not enable it until CPU allocation,
strictness, dtype/buffer checks and real-GPU parity pass. Hold shipment if peak
startup pressure does not improve in matched fresh-process trials, if startup
latency materially regresses, or if offload/compile integration leaves meta or
stale device tensors. Do not infer an additive memory saving from independent
allocation counts. No implementation, GPU validation, or performance result is
part of this design commit.
