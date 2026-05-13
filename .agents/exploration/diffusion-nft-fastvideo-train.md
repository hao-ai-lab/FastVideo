# Exploration Log: DiffusionNFT in fastvideo/train

## Status: draft

## Context
DiffusionNFT is an online diffusion reinforcement method with a training loop
that alternates policy sampling, reward scoring, and forward-process updates.
FastVideo's new trainer treats `max_train_steps` as optimizer updates, so the
DiffusionNFT collection epoch needs to live inside the method without changing
the trainer loop semantics.

## Progress
- [x] Read local `DiffusionNFT/` reference implementation and configs.
- [x] Mapped text-only Wan data loading to the existing DMD2 text-only parquet path.
- [x] Added a `DiffusionNFTMethod` that keeps trainer steps as optimizer updates and logs collection/reward metrics when a rollout buffer is refreshed.
- [x] Added a Wan 1.3B text-to-image config with `num_latent_t=1` and `num_frames=1`.
- [x] Added a runnable shell launcher that can reuse or auto-create text-only parquets.

## Findings
The cleanest integration is a method-local collection round. A trainer step
consumes queued NFT training batches; when the queue is empty, the method samples
new clean latents, decodes one frame, computes rewards, computes prompt-grouped
advantages, and precomputes old-policy predictions for the queued forward-process
training batches. This avoids adding a teacher or critic role.

For Wan 1.3B on a 32 GB GPU, loading the T5 text encoder for negative prompt
conditioning after the trainable transformer and VAE can OOM. DiffusionNFT's
default no-CFG sampling (`sample_guidance_scale=1.0`) does not need negative
conditioning, so the method disables negative prompt loading unless CFG sampling
is explicitly requested.

## Mistakes / Dead Ends
- Initial smoke attempt OOMed in `ensure_negative_conditioning()` because
  negative prompt embeddings were loaded unconditionally.

## Proposed Standardization
If this method becomes a maintained training path, promote the collection-round
pattern into a documented `fastvideo/train` recipe for online RL methods whose
sampling epoch differs from optimizer-step accounting.
