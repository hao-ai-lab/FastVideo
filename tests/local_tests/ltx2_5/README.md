# LTX-2.5 Local Tests

Local parity, conversion, and smoke tests for the native LTX-2.5 FastVideo
port. The launch scope is BF16 dev and distilled text/image-conditioned joint
video-audio inference. Training, QAD/NVFP4, DiffVAE, DFR, HDR, temporal
upsampling, and automatic duration prediction are follow-up work.

Port progress and unresolved questions live in
`tests/local_tests/ltx2_5/PORT_STATUS.md`.

## Reference Assets

| Field | Value |
|---|---|
| Model family | `ltx2_5` (config-gated extension of native `ltx2`) |
| Workload types | T2AV and I2AV through the existing T2V/I2V compatibility surface; output is joint video and audio |
| Official reference | `https://github.com/Lightricks/LTX-2` |
| Local reference dir | `LTX-2-Reference/` (gitignored) |
| Official commit/version | `v1.2.0`, `d151147788a9284cca791edc6ce898007e727fe6` |
| HF weights | `Lightricks/LTX-2.5` |
| HF revision | `ce298b1259d61ce6c87e05154b9ad339b16f32a0` |
| Local weights dir | Modal-backed Hugging Face cache; no local weight download |
| Source layout | Separate component safetensors with checkpoint metadata; no root `model_index.json` |
| Needs conversion | Yes |

Never place token values in this file. GPU validation resolves `HF_TOKEN` from
an approved secret in the HAO AI Lab Modal workspace.

## Shared Environment Setup

The source clone is reproducible from the repository root:

```bash
python3 .agents/skills/add-model-01-prep/scripts/clone_reference_repo.py \
  https://github.com/Lightricks/LTX-2.git \
  LTX-2-Reference \
  --branch v1.2.0 \
  --commit d151147788a9284cca791edc6ce898007e727fe6 \
  --update-gitignore
```

The local FastVideo `.venv` does not currently contain PyTorch. Do not change
FastVideo's core pins on this Mac solely for parity. Numeric checks reuse one
warm HAO AI Lab Modal L40S shell with FastVideo and the pinned official source
installed together.

```text
dependency_changes: transformers>=5.8.0,<5.15 for Gemma 4
official_env_status: source parity passing in one shared Modal environment
private_dep_stubs: none planned
gated_access_status: approved and verified at the pinned revision
```

## Weight Layout

The published gated repository exposes these inference components:

```text
diffusion_models/ltx-2.5-22b-dev-transformer-bf16.safetensors
diffusion_models/ltx-2.5-22b-distilled-transformer-bf16.safetensors
text_encoders/gemma4-12b-with-proj-ltx-2.5-bf16.safetensors
vae/ltx-2.5-video-vae-conv-bf16.safetensors
vae/ltx-2.5-audio-vae-bf16.safetensors
```

The first PR deliberately uses the official convolutional video VAE. The
diffusion VAE is a separate iterative neighborhood-attention decoder and is not
required for authentic LTX-2.5 launch inference.

## Prototype And Conversion Artifacts

```text
official_key_dumps:
  transformer: converted_weights/ltx2_5/_mapping/transformer_official_keys.json
  text_encoder: converted_weights/ltx2_5/_mapping/text_encoder_official_keys.json
  video_vae: converted_weights/ltx2_5/_mapping/video_vae_official_keys.json
  audio_vae: converted_weights/ltx2_5/_mapping/audio_vae_official_keys.json
fastvideo_key_dumps:
  transformer: converted_weights/ltx2_5/_mapping/transformer_fastvideo_keys.json
  text_encoder: converted_weights/ltx2_5/_mapping/text_encoder_fastvideo_keys.json
  video_vae: converted_weights/ltx2_5/_mapping/video_vae_fastvideo_keys.json
  audio_vae: converted_weights/ltx2_5/_mapping/audio_vae_fastvideo_keys.json
conversion_script: scripts/checkpoint_conversion/convert_ltx2_weights.py
conversion_source_layout: separate_components
converted_weights_dir: converted_weights/ltx2_5
strict_load_status: dev_and_distilled_pass
```

## Expected Parity Tests

| Component | Official files / args | Test | Concerns | Status |
|---|---|---|---|---|
| Transformer | `ltx_core.model.transformer.model_configurator.LTXModelConfigurator`; checkpoint metadata | `tests/local_tests/transformers/test_ltx2_5_transformer_parity.py` | Official random FFN/keyframe/prompt-AdaLN parity plus strict dev/distilled production loading | real strict-load pass |
| Gemma 4 encoder and connector | `ltx_core.text_encoders.gemma`; unified Gemma 4 checkpoint | `tests/local_tests/ltx2_5/test_ltx2_5_gemma.py` | BOS behavior, connector bias surface, packed conversion | focused pass |
| Convolutional video VAE | `ltx_core.model.video_vae` configurator; conv checkpoint | `tests/local_tests/ltx2_5/test_ltx2_5_vae.py` | checkpoint-derived decoder width | config pass |
| Audio VAE/vocoder | `ltx_core.model.audio_vae`; audio VAE checkpoint | `tests/local_tests/ltx2_5/test_ltx2_5_conversion.py` | split routing and BWE topology reuse | real pipeline pass |
| Ancestral sampler | `ltx_core.components.diffusion_steps` and distilled pipeline selection | `tests/local_tests/ltx2_5/test_ltx2_5_sampler.py` | terminal x0, eta=0, reproducibility, seed offset, video-before-audio RNG | unit pass |
| T2AV/I2AV pipeline | official one-stage and distilled pipeline call paths | `tests/local_tests/ltx2_5/test_ltx2_5_registry.py` | dev/distilled presets, guidance, conditioning CRF, workloads | structural and real T2AV pass |

Planned verification order:

```bash
pytest -q tests/local_tests/ltx2_5 \
  tests/local_tests/transformers/test_ltx2_5_transformer_parity.py \
  -k 'not real_weights'

pytest -q -rs \
  tests/local_tests/transformers/test_ltx2_5_transformer_parity.py \
  -k 'not real_weights'
```

The transformer prototype extends the existing native LTX-2 graph rather than
forking it. It adds the exact upstream v1.2.0 gates
`use_prompt_adaln_single=True`, `ff_bias=True`, `audio_ff_bias=True`, and
`use_keyframes_abs_pos_embedding=False`; these defaults preserve LTX-2/2.3.
Both published 2.5 BF16 variants disable video FFN bias and enable keyframe
absolute embeddings while retaining the prompt-AdaLN and audio-FFN-bias
defaults. The keyframe marker mask is sharded with video tokens on FastVideo's
SP path.

The same warm Modal L40S shell ran the complete lightweight launch suite plus
legacy LTX preset preflight after the official source clone: 24 passed with the
two real-weight cases deselected. It then strict-loaded both 42 GB BF16
transformers through the official builder and FastVideo's production loader:
development passed in 28.76s and distilled passed in 27.06s.

Real pipeline validation then passed both distilled execution modes. The
one-stage smoke ran on 2xH100 and produced a joint H.264/AAC MP4 with 9 video
frames and 17 audio frames in 11.77s after load. The full two-stage smoke ran
on 2xB200: base ancestral denoising, spatial upsampling, the published
1,632-layer refinement LoRA, all three refinement steps, and joint decode/save
completed in 76.17s. The verified output contains 9 H.264 frames and 17 AAC
frames. W&B records the launch run at
`https://wandb.ai/aryan5v-san-jose-state-university/fastvideo-ltx2-5/runs/o9ixwewk`.

CPU/reference parity coverage includes random-weight FFN output parity, exact
keyframe-marker functional parity, and official-versus-FastVideo parameter
surface checks for both legacy defaults and the 2.5 gates. The real-weight test
strict-loads dev and distilled checkpoints through the official builder and
FastVideo's production `TransformerLoader` when the gated assets are present.

## Review Notes

- Existing LTX-2 and LTX-2.3 checkpoints must remain behavior-identical when
  all 2.5 configuration flags retain their backward-compatible defaults.
- Native FastVideo loading, compilation, offload, and sequence-parallel paths
  are production requirements, not follow-up wrappers.
- A scaffold skip is not a parity pass. Real-weight claims require non-skipped
  Modal results with exact commands and hardware recorded here.
- Before every push after the first PR publication, inspect new CodeRabbit
  feedback and address actionable findings before updating the branch.
