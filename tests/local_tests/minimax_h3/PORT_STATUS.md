# MiniMax H3 port status

## Status

- workloads: T2VA, FL2VA, Ref2VA joint video/audio generation
- component parity: complete
- FastVideo runtime acceptance: complete
- official end-to-end pipeline parity: complete

## Coverage

| Scope | Evidence | State |
|---|---|---|
| Qwen3-VL encoder | exact text/image/video hidden states through the production loader; bounded exact vision interpolation | complete |
| FL2VA and Ref2VA DiTs | exact video/audio heads for both model partitions | complete |
| Video VAE | exact encode, normalization, and decode through the production loader | complete |
| Video VAE streaming | exact chunked encode/decode and output-rank-only distributed decode | complete |
| Audio VAE | exact encode and normalization; decode maximum absolute drift `2.4e-7` | complete |
| Video/audio schedulers | pinned `12/3` schedule parity | complete |
| FL2VA packing | pinned row, position, tag, timestep, and RNG parity | complete |
| Ref2VA media and packing | pinned media and packing parity | complete |
| Public surface | manifest resolution, pipeline registration, and three presets | complete |
| FastVideo distributed runtime | valid joint AV outputs; SP=1/SP=4 latent consistency | complete |
| Official end-to-end pipeline | exact T2VA, FL2VA, and Ref2VA video/audio latents | complete |

## Current validation

T2VA, FL2VA, and Ref2VA match the official video/audio latents exactly. On 2026-08-26, the Qwen3-VL production loader
also matched Transformers 5.15.1 exactly for text, image, and video layer-50 hidden states on GB10. Its production
`[15, 42, 74] x 1152` vision interpolation matched the official helper bit-exactly while reducing incremental CUDA
allocation from `1,291,986,432` to `284,866,048` bytes relative to the unbounded float32 implementation. A packed
production image/video grid reduced the same metric from `2,086,086,144` to `418,362,880` bytes.

## Decisions

- Preserve each H3 scheduler's configured shift; global `flow_shift` is invalid.
- Do not wrap the H3 DiT in global autocast; its FP32 projections must stay FP32.
- Let FSDP move CPU-offloaded Qwen parameters; do not move the wrapped conditioner as a whole.
- Load `transformer/` for T2VA/FL2VA and `transformer_ref/` for Ref2VA.
- Keep `last_image`, `references`, and `audio_latents` on the typed request path.
- Treat the published component folders as the loading boundary.
- Keep reference videos on CPU between VAE clips and decode final pixels only on the executor's output rank.
- Preserve Qwen3-VL's float32 vision interpolation through accumulation, bound its four-tap temporary workspace, and
  cast the completed positions only at the vision residual-add boundary.

## Evidence boundary

Completed rows summarize recorded component and FastVideo runtime runs. Registry smoke, generated media, and
FastVideo SP consistency are supporting checks, not substitutes for the recorded official comparisons.
