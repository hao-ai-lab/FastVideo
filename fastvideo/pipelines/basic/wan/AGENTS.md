# Wan pipelines

Pipeline files assemble existing stages. Sampling defaults stay in `presets.py`.
Pipeline configs and variant definitions live in `fastvideo/models/wan/` as
`pipeline_config.py` and `definition.py`. The shared registry consumes these
definitions; old pipeline config imports remain compatibility aliases. Shared text/image
encoders stay shared; the Wan VAE video encoder and decoder live together in
`fastvideo/models/wan/vae.py`.

## Sampling boundaries

- `stages/conditioning.py`: encode and normalize `first_frame_latent` before
  dense TI2V or causal DMD sampling. Clear it for each request. Preserve VAE
  precision, normalization order, and device restoration.
- `stages/denoising.py`: Wan input packing, expert boundaries, guidance choice,
  Lucy timesteps, and TI2V first-frame preservation around the shared loop.
  Request state belongs in `WanDenoisingState`, never on a reusable stage.
- `stages/dmd.py`: dense DMD sampling. The caller passes a full training-noise
  `FlowMatchEulerDiscreteScheduler` with `DMD_TRAINING_NOISE_SHIFT` from the
  family definition (8.0). Do not share the scheduler
  mutated by `TimestepPreparationStage`: DMD indexes the full timestep/sigma
  table when converting noise to video and adding noise.
- `stages/causal_denoising.py`: standard and DMD causal samplers share cache
  allocation, not a sampling-loop inheritance chain. Reset UniPC per block and
  caches per request; preserve clean-context writes and RNG ordering.
- `stages/vace_input.py`: VACE-specific mask loading, reference-image
  preprocessing, and zero-pixel synthesis whenever source video is absent. Runs after
  shared validation and text encoding.
- `stages/vace_conditioning.py`: VAE-encodes video/mask/reference into 96-channel
  `control_hidden_states`. Control signal goes through `control_hidden_states`,
  not channel concat in denoising.
- `stages/vace_latent_preparation.py`: appends reference-frame temporal padding
  to latent shape; restores `num_frames` after preparation. Runs after
  `vace_context_stage`, matching Diffusers `WanVACEPipeline` ordering.
- `stages/vace_decoding.py`: strips reference-frame latents before VAE decode.

Do not change scheduler arithmetic, autocast placement, step counts, offload,
or model math as incidental cleanup. Keep legacy sampler exports as aliases;
`CausalDMDDenosingStage` retains its historical spelling for compatibility.

The first-frame stage has its own timing. Its encode time is no longer inside
`dit_time_s`; total request latency still includes it. Do not reseed performance
baselines automatically because this boundary moved.

## Checks

See [the testing guide](../../../../docs/contributing/testing.md). Start with
`bash scripts/validate_wan.sh all` from the repo root in a GPU environment.
It runs weight-free contracts before small device/runtime-matched goldens.
Pass `parity` or `default` as the second argument for independent component
parity or focused SSIM after those gates. Cache goldens do not establish
long-rollout quality, distilled-model quality, or complete pipeline parity.
