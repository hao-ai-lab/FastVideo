# MiniMax H3 port status

## Status

- workloads: T2VA, FL2VA, Ref2VA joint video/audio generation
- component parity: complete
- FastVideo runtime acceptance: complete
- official end-to-end pipeline parity: T2VA complete; FL2VA/Ref2VA pending

## Coverage

| Scope | Evidence | State |
|---|---|---|
| Qwen3-VL encoder | exact text/image/video hidden states through the production loader | complete |
| FL2VA and Ref2VA DiTs | exact video/audio heads for both model partitions | complete |
| Video VAE | exact encode, normalization, and decode through the production loader | complete |
| Audio VAE | exact encode and normalization; decode maximum absolute drift `2.4e-7` | complete |
| Video/audio schedulers | pinned `12/3` schedule parity | complete |
| FL2VA packing | pinned row, position, tag, timestep, and RNG parity | complete |
| Ref2VA media and packing | pinned media and packing parity | complete |
| Public surface | manifest resolution, pipeline registration, and three presets | complete |
| FastVideo distributed runtime | valid joint AV outputs; SP=1/SP=4 latent consistency | complete |
| Official end-to-end pipeline | T2VA video/audio latents are exact; FL2VA/Ref2VA remain pending | partial |

## Current validation

T2VA matches the official video/audio latents exactly. Validate FL2VA and Ref2VA only after deciding to expand scope.

## Decisions

- Preserve each H3 scheduler's configured shift; global `flow_shift` is invalid.
- Do not wrap the H3 DiT in global autocast; its FP32 projections must stay FP32.
- Let FSDP move CPU-offloaded Qwen parameters; do not move the wrapped conditioner as a whole.
- Load `transformer/` for T2VA/FL2VA and `transformer_ref/` for Ref2VA.
- Keep `last_image`, `references`, and `audio_latents` on the typed request path.
- Treat the published component folders as the loading boundary.

## Evidence boundary

Completed rows summarize recorded component and FastVideo runtime runs. T2VA parity does not establish FL2VA or
Ref2VA parity; registry smoke, generated media, and FastVideo SP consistency are not substitutes for either.
