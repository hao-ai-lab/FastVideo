# MiniMax H3 port status

## Status

- workloads: T2VA, FL2VA, Ref2VA joint video/audio generation
- component parity: complete
- FastVideo runtime acceptance: complete
- official pipeline parity: pending

## Coverage

| Scope | Evidence | State |
|---|---|---|
| Qwen3-VL encoder | text/image/video production-loader parity | complete |
| FL2VA and Ref2VA DiTs | both model partitions | complete |
| Video VAE | production-loader parity | complete |
| Audio VAE | production-loader parity | complete |
| Video/audio schedulers | pinned `12/3` schedule parity | complete |
| FL2VA packing | pinned row, position, tag, timestep, and RNG parity | complete |
| Ref2VA media and packing | pinned media and packing parity | complete |
| Public surface | manifest resolution and pipeline registration/preset smoke | complete |
| FastVideo distributed runtime | valid joint AV outputs; SP=1/SP=4 latent consistency | complete |
| Official end-to-end pipeline | Diffusers-vs-FastVideo FL2VA/Ref2VA latent comparison | pending |

## Open item

Add one gated pipeline parity test comparing official and FastVideo video/audio latents for representative FL2VA and
Ref2VA requests.

## Decisions

- Preserve each H3 scheduler's configured shift; global `flow_shift` is invalid.
- Let FSDP move CPU-offloaded Qwen parameters; do not move the wrapped conditioner as a whole.
- Load `transformer/` for T2VA/FL2VA and `transformer_ref/` for Ref2VA.
- Keep `last_image`, `references`, and `audio_latents` on the typed request path.
- Treat the published component folders as the loading boundary.
