# Cosmos3 → FastVideo port — feedback (pitfalls, issues, difficulties)

Retrospective on porting the full **NVIDIA Cosmos3-Nano** omni world model
(video / audio / action generation + text & image reasoning) into FastVideo.
Methodology: framework-only reference, native FastVideo port, a bit-exact
framework-parity test per component, then real-weights verification. Every
modality landed bit-exact (see `PORT_STATUS.md` "Full-omni parity summary").

This doc records what bit, so the next omni/world-model port (and the `/add-model`
skill) can avoid the same traps.

---

## 1. The checkpoint's config does NOT describe the runtime — verify against the framework

The single biggest time sink. The HF checkpoint is "diffusers format", which led
to two silent traps:

- **Scheduler (caused an all-black video).** `scheduler/scheduler_config.json`
  is a diffusers `UniPCMultistepScheduler` config carrying
  `use_karras_sigmas=true`, `sigma_min/sigma_max`, a beta schedule, etc. But the
  framework actually samples with a *flow-matching* `FlowUniPCMultistepScheduler`
  (shift + num_train_timesteps only). FastVideo's vendored UniPC checks
  `use_karras_sigmas` **before** `use_flow_sigmas`, so it built diffusion (beta)
  sigmas → `scheduler.step` → **NaN latents → 3 KB black mp4**. The DiT/CFG
  velocity was perfectly clean; only the scheduler diverged.
  - Fix: coerce the loaded config to the flow setup in `initialize_pipeline`.
  - **Lesson:** treat the checkpoint's generic-format config as *lossy*. Find how
    the framework actually instantiates the component and match THAT, not the JSON.

- **`flow_shift` is resolution-based, not task-based.** Natural assumption:
  "T2I uses a small shift, video a large one." Reality: the framework keys the
  UniPC shift purely off the named resolution bucket
  (`_RESOLUTION_SHIFT_DEFAULTS`, 8B backbone: 256→3, 480→5, 720/768→10). The
  task-based heuristic only *coincidentally* matched (T2V@720, T2I@256); canonical
  T2I is 960×960 (the "720" bucket → 10), so `is_t2i→3.0` was wrong.

## 2. A "parity test" that compares two copies of the wrong thing proves nothing

The original denoise/CFG test imported **diffusers** `UniPCMultistepScheduler` and
used it on BOTH the "oracle" and "FastVideo" sides. So the scheduler was never
actually compared against the framework — which is exactly why the black-video
scheduler bug sailed through a green test suite.
- **Lesson:** the oracle side of every parity test MUST be the official framework
  object, never a second instance of the unit under test. After writing a parity
  test, ask: "if the framework were wrong here, would this test fail?"

## 3. Temporal-VAE conditioning: static-repeat vs zero-fill (silent corruption)

I2V/T2I condition on the input image. The framework
(`build_conditioned_video_batch`) fills frame 0 with the image and **repeats the
last conditioning frame across the whole clip** (a static video) before
VAE-encoding. The first native cut **zero-filled** the non-condition frames.
Because the Wan VAE is temporal (4× compression), latent-frame-0 (the kept-clean
condition frame) depends on several *pixel* frames — so zero-filling produced a
*wrong* conditioning latent. This is the kind of bug that doesn't crash and can
even look plausible at a glance.
- **Lesson:** when a conditioning latent feeds a temporal autoencoder, trace the
  temporal receptive field; "only frame 0 matters" is false under temporal conv.

## 4. Checkpoint param names ≠ framework module structure (three naming conventions)

For the DiT there were **three** namings to bridge: framework-native
(`Cosmos3VFMNetwork`: `language_model.model.layers.*`, `vae2llm`, `q_proj_moe_gen`,
…), the diffusers checkpoint on disk (`layers.*.to_q`, `add_q_proj`, `proj_in`,
…), and the FastVideo DiT. The weight map crosses (framework)→(FastVideo) for
parity and (checkpoint)→(FastVideo) for loading.

The **sound tokenizer** was the sharpest example: the checkpoint is **decoder-only**
in diffusers `AutoencoderOobleck` naming (`decoder.conv1`, `block.N.conv_t1`,
`res_unitM`, `snake1`) — but with **`SnakeBeta`** (learned alpha *and* beta,
logscale), NOT diffusers' alpha-only `Snake1d`. So neither "use diffusers
AutoencoderOobleck" nor "port the framework `LatentAutoEncoderV2` Sequential
module" matched the on-disk keys.
- **Lesson:** dump `safetensors` keys + shapes for every sub-checkpoint *first*.
  The naming reveals which existing native module (if any) already matches.

## 5. A "matching" native module can still differ on an untested config path

FastVideo already had a native `OobleckVAE` (Stable Audio) whose decoder matched
the Cosmos3 sound decoder bit-for-bit — except `OobleckDecoderBlock.conv_t1`
omitted `output_padding = stride % 2`. That omission is a **no-op for Stable
Audio's even strides** [2,4,4,8,8], so it had never mattered; Cosmos3 has an
**odd** stride (5), where the framework's `output_padding=1` makes the decode one
sample longer per odd-stride block (parity diverged 60 vs 59 samples).
- **Lesson:** reusing a native module is great, but re-run parity on the *new*
  model's config — shared code can hide config-specific divergences.

## 6. Loader / registry plumbing the checkpoint format forces

- **DiT class alias.** `model_index.json` names the DiT `Cosmos3OmniTransformer`
  (the diffusers shim class); the registry normalized unknown classes to a
  generic `TransformersModel`. Needed an explicit registry alias
  `Cosmos3OmniTransformer → Cosmos3VFMTransformer`.
- **Tokenizer module name.** `model_index.json` calls the Qwen2 tokenizer
  `text_tokenizer` (not `tokenizer`); the component loader had no mapping for that
  key and tried to load it as a model.
- **Scheduler config schema drift.** The vendored UniPC predates
  `shift_terminal` / `sigma_min` / `sigma_max`; constructing it with the raw
  checkpoint config crashes on the unexpected kwargs. Filter to the class's
  `__init__` params (mirroring diffusers `from_config`).
- **Meta-device load + non-persistent buffers.** `rotary_emb.inv_freq` is derived
  from `rope_theta` and is non-persistent (absent from the checkpoint), so after
  the meta-device FSDP load it stays on the `meta` device → needs a
  `materialize_non_persistent_buffers` hook to recompute it on the real device.
- **dtype boundaries.** Noise/VAE latents arrive fp32; the model runs bf16.
  Needed explicit casts at `proj_in` and the timestep embedder (no-ops in the
  fp32 parity tests, required at inference).
- **device in packing.** The packer builds ids/positions on CPU; `to_dit_kwargs`
  must move every tensor to the model device before the forward.

## 7. The omni model is a Mixture-of-Transformers — modality bookkeeping is the work

The backbone is a dual-pathway MoT: **und** (causal text) + **gen** (full-attention
vision/sound/action). Once the video path worked, each extra modality was the same
*shape* of work (a proj-in + modality embed + timestep-scatter encode, a proj-out
decode, packing, a CFG-velocity slice) but with per-modality quirks:
- sound/action **share the vision "full" split** (preserving the causal+full
  2-split invariant); the combined flat latent is `[vision | action | sound]` in
  that order (must match the framework's per-sample concat).
- sound MRoPE uses `start_frame_offset=0` (parallel to vision); action uses
  `start_frame_offset=1`; both at the vision temporal offset, tcf=1, and do NOT
  advance the offset.
- action is **domain-aware** (`DomainAwareLinear`: per-embodiment weight/bias via
  `nn.Embedding`, indexed by a per-token domain id).
- the unpack already zeros clean frames, so the per-step velocity masking is
  defensive (but kept, to mirror the framework exactly).
- **Lesson:** build the first modality (vision) with clean seams for "a modality"
  and the rest fall out; spend the care on the packing layout + MRoPE offsets,
  which are the only per-modality novelties.

## 8. Reasoning reused more than expected; the encoder is just transformers

- **Text reasoning** needed *no new model code*: it's the und (causal) pathway +
  `embed_tokens`/`norm`/`lm_head`, all already in the DiT. A text-only forward +
  `lm_head` is token-for-token identical to the framework reasoner.
- **vision_encoder** is a stock `transformers.Qwen3VLVisionModel` (the framework
  ships its own *copy* of the same class); reusing transformers' (like the Qwen2
  tokenizer) is bit-exact vs the framework — re-porting a 27-layer ViT would have
  been wasted effort.
- **deepstack** (image-conditioned reasoning) is the one new native piece: inject
  the 3 vision-encoder deepstack features into the first 3 text layers at the
  image-token positions.
- **Lesson:** before porting a big sub-model, check whether it's literally a
  stock library class — and whether an existing in-repo module already implements
  it (audio decoder + vision encoder were both "already there").

## 9. Running the framework as a CPU parity oracle

The framework's attention path is flash/natten (CUDA-only). Parity tests run on
CPU/float32 via an SDPA monkey-patch (`test_cosmos3_reference_forward._apply_sdpa_patches`).
A couple of framework helpers also can't import headless (`cosmos_framework.inference.args`
pulls `multistorageclient`), so a constant or two is mirrored in the test with a
cited source rather than imported.
- **Lesson:** budget for "make the oracle runnable on CPU" — monkeypatch attention,
  build tiny configs, and accept a small amount of mirrored constants when a
  framework module won't import in isolation.

## 10. What made it tractable

- Tiny CPU/fp32 models + copy-framework-weights-in + bit-exact compare, per
  component, is a fast and decisive loop (max=mean=0.0 or it's wrong).
- A persistent `PORT_STATUS.md` (resumable state, issues, decisions) survived
  several context resets.
- Stacked PR branches (one per modality) kept each parity-verified increment
  reviewable and the chain bisectable.
