# LingBot-World-Fast port status

## Summary

- model family: `lingbotworld_fast`
- workload: causal I2V with camera poses
- official source: `https://github.com/Robbyant/lingbot-world`
- official checkpoint: `robbyant/lingbot-world-fast`
- FastVideo bundle: `FastVideo/LingBot-World-Fast-Diffusers`
- source layout: official DiT checkpoint plus shared Wan components
- current phase: SSIM seeding

## Reused implementation evidence

The FastVideo variant reuses
`LingBotWorld2CausalFastTransformer3DModel`. The released transformer index has
1,421 parameter keys, and the PR's initial validation found all names and
shapes identical to the reused implementation.

Variant-specific runtime values:

| Setting | Value | Source |
|---|---:|---|
| `local_attn_size` | `-1` | released checkpoint config / `generate_fast.py` default |
| `chunk_size` | `3` | official `generate_fast.py` call |
| fixed timestep indices | `(0, 179, 358, 679)` | released fast sampling path |
| `sample_shift` | `10.0` | official I2V config |

## Verification matrix

| Scope | Command | Latest result |
|---|---|---|
| Official-vs-FastVideo full DiT | See `README.md` | 2026-08-17: 2 passed; max/mean/relative-mean error all `0.0` |
| Backend selection | `pytest fastvideo/tests/attention/test_lingbotworld_fast_backend.py -v` | 2026-08-17: passed |
| Preset and exact pipeline resolution | `pytest fastvideo/tests/api/test_presets.py::TestLingBotWorldFastPresets -v` | 2026-08-17: 2 passed |
| SSIM collection on this GB10 host | `pytest fastvideo/tests/ssim/test_lingbot_fast_similarity.py -v -s` | 2026-08-17: 1 skipped; unsupported GB10 |
| L40S reference generation and SSIM | `seed-ssim-references` workflow | blocked: Modal CLI and HF write token unavailable locally |

## Acceptance requirements

- [x] Official and FastVideo implementations load real released weights.
- [x] Both execute a real full-transformer forward.
- [x] Full-transformer numerical comparison passes without skip.
- [x] Backend and preset/registry tests pass.
- [ ] L40S SSIM reference is generated and visually approved.
- [ ] SSIM reference is uploaded and the package test passes against it.

## Environment notes

Local verification host:

- GPU: NVIDIA GB10
- detected GPU count: 1
- PyTorch: 2.12.0 + CUDA 13.0
- FlashAttention: unavailable; parity aligns both implementations on SDPA

The SSIM suite does not accept GB10 references and the test requires two GPUs.
This machine can validate transformer parity and CPU contract tests, but it
cannot produce the required L40S SSIM acceptance evidence.

## Open blockers

1. Run the new SSIM test on Modal L40S, inspect the generated video, and seed
   `FastVideo/ssim-reference-videos`.
