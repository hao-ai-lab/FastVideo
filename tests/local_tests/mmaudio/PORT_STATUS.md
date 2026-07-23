# MMAudio Port Status

## Summary

- model_family: `mmaudio`
- workload_types: `V2A`, `T2A`
- official_ref: `../MMAudio` at `974010a026c731054592d8f777218bd9d85a6c24`
- first_variant: `large_44k_v2`
- phase: `native_inference_v1_training_and_preprocessing_complete`
- status: `inference_training_and_vggsound_manifest_parity_pass`
- last_updated: `2026-07-23`

## Native Components

| Component | FastVideo implementation | Reuse/port decision | Real-weight result |
|---|---|---|---|
| MMAudio transformer | `fastvideo/models/dits/mmaudio.py` | Native 1D multimodal DiT | exact |
| DFN5B text/vision | `fastvideo/models/encoders/mmaudio_clip.py` | Shared native CLIP core, MMAudio adapters | exact |
| Synchformer visual encoder | `fastvideo/models/encoders/mmaudio_synchformer.py` | Shared backbone under `fastvideo/third_party/synchformer` | exact, including 16-frame/stride-8 usage contract |
| 44.1 kHz VAE | `fastvideo/models/audio/mmaudio_vae.py` | Native audio component | exact |
| BigVGAN-v2 | `fastvideo/models/audio/bigvgan.py` | Shared native vocoder | exact |
| Euler flow schedule | shared `FlowMatchEulerDiscreteScheduler` | Reuse schedule; preserve official BF16 scalar update in MMAudio stage | exact |

## Pipeline Integration

- Pipeline: `fastvideo/pipelines/basic/mmaudio/MMAudioPipeline`
- Config: `fastvideo/configs/pipelines/mmaudio.py::MMAudioV2AConfig`
- Preset: `mmaudio_large_44k_v2`
- Registry: resolves both `WorkloadType.V2A` and `WorkloadType.T2A`
- Required production components: `transformer`, `scheduler`, `text_encoder`,
  `tokenizer`, `image_encoder`, `image_encoder_2`, `audio_vae`, `vocoder`
- Output: mono `[B,1,samples]`, 44.1 kHz, exposed through FastVideo's
  audio-only result contract and saved as WAV by `VideoGenerator`
- Duration: dynamic sequence lengths, with 8 seconds retained only as the
  published training/default duration; longer and shorter inference is accepted
- Existing T2V/I2V/T2I pipelines are not routed through these stages.

The V2A preprocessing contract is identical to official MMAudio:

1. timestamp sampling at 8 FPS for DFN5B and 25 FPS for Synchformer;
2. DFN path: bicubic resize to `384x384`, float `[0,1]`, CLIP normalization;
3. sync path: bicubic short-side resize to 224, center crop, normalize to `[-1,1]`;
4. Synchformer: 16-frame windows, stride 8, `(segment,time)` token flattening.

## Converted Checkpoint

- Converter: `scripts/checkpoint_conversion/convert_mmaudio_to_diffusers.py`
- Local artifact: `converted_weights/mmaudio/large_44k_v2`
- Production strict load: pass for all eight components
- Total local artifact size: about 9.1 GB
- Converted weights and official assets remain ignored/untracked.

The v1 training smoke additionally uses a transformer-only component tree at
`converted_weights/mmaudio/small_44k` (about 601 MB). The converter supports
`small_44k`, `medium_44k`, `large_44k`, and the inference-default
`large_44k_v2`; `--transformer-only` avoids duplicating inference-only assets.

## Training Integration

- Model adapter: `fastvideo/train/models/mmaudio/MMAudioModel`
- Shared method: shape-agnostic
  `fastvideo/train/methods/fine_tuning/flow_matching.py`
- Data: official-compatible TensorDict memmaps through
  `fastvideo/dataset/mmaudio_feature_dataset.py`
- Config: `examples/train/configs/fine_tuning/mmaudio/small_44k.yaml`
- Numerical recipe: posterior sampling, latent normalization, logit-normal
  time, linear flow interpolation, independent video/text CFG dropout, and
  velocity MSE match upstream `Runner.train_fn`
- Distributed contract: real single-GPU FSDP-wrapped step passed; DCP save,
  optimizer/dataloader/RNG restore, and resumed step passed
- Production code imports no `mmaudio.*` modules.

## Native VGGSound Preprocessing

- Shared orchestration: `fastvideo/workflow/preprocess/preprocess_workflow_v2a.py`
- Dataset adapter: `fastvideo/dataset/vggsound.py`
- MMAudio feature stage: `fastvideo/pipelines/preprocess/mmaudio/stages.py`
- Sharded resumable cache: `fastvideo/dataset/v2a_feature_cache.py`
- Input manifests use the official `id<TAB>label` contract; `label` is passed as
  the full caption to the DFN5B text encoder.
- `dataset_metadata_path` explicitly selects one manifest and `dataset_split`
  preserves the official train-only peak normalization behavior.
- Train, val, and test use one pipeline class and separate output roots.
- Production preprocessing imports no modules from the reference checkout.
- The reference training loader uses the removed
  `torio.io.StreamingMediaDecoder`. FastVideo 2.11 uses `torchaudio.load` for
  audio and PyAV timestamp sampling for video; downstream transforms, resampling,
  mel conversion, and encoder outputs retain the reference contracts.

## Parity Evidence

| Scope | Result |
|---|---|
| Official/FastVideo video preprocessing | exact (`clip max_abs=0`, `sync max_abs=0`) |
| Condition features, random latent, projected conditions | exact |
| First flow prediction and 25-step final latent | exact |
| Final 2-second V2A waveform (89,088 samples) | exact (`atol=0`, `rtol=0`) |
| Real 10-second variable-duration V2A | pass (441,344 samples, 10.0078 s) |
| Default FastVideo offload path | real one-step smoke pass |
| Real `small_44k` v1 transformer forward | exact BF16 parity |
| Real `small_44k` v1 FastVideo train step | pass (forward/backward/AdamW/grad clip) |
| Real DCP checkpoint and resume to next step | pass |
| Local MMAudio parity suite | `21 passed, 1 skipped` (the skipped test is the opt-in full gate) |
| Combined parity/dataset/training regression | `33 passed, 1 skipped` |
| Real filtered-caption manifests | exact ID/order/caption/count parity: train 180,062; val 2,047; test 15,221 |
| Filtered-caption tokenizer lengths | 0/197,330 exceed the 77-token contract; maximum 74 |

Commands:

```bash
pytest -q tests/local_tests/mmaudio

MMAUDIO_RUN_PIPELINE_PARITY=1 \
MMAUDIO_PARITY_VIDEO=/path/to/video-at-least-2s.mp4 \
pytest -q tests/local_tests/mmaudio/test_mmaudio_pipeline_parity.py::test_mmaudio_real_v2a_pipeline_waveform_parity -s
```

The opt-in real pipeline gate passed on an RTX 6000 Ada with the downloaded
official `large_44k_v2`, DFN5B, Synchformer, VAE, and BigVGAN assets.

## Important Numeric Decisions

- OpenCLIP text uses the explicit additive causal mask used by
  `nn.MultiheadAttention`; SDPA's `is_causal` shortcut rounds differently in BF16.
- The Euler time/delta scalars stay on CPU, matching official MMAudio's
  `torch.linspace` loop. Moving those float32 scalars to CUDA changes BF16 promotion.
- `t_embed.freqs` is materialized in BF16 after meta loading; dynamic RoPE buffers
  are rebuilt in FP32 exactly as official `update_seq_lengths` does.
- MMAudio VAE and BigVGAN weight norm is removed on CPU in FP32 before casting to
  BF16, matching the official feature utility construction order.

## Deferred Scope

- Publishing the converted checkpoint and immutable source revisions.
- Optional source-video mux/re-encode helper; the current V2A result is WAV/audio.
- 16 kHz inference/training and small/medium v1 inference presets.
- Sequence/tensor-parallel optimization.
- Multi-GPU data-parallel training validation.
- Full-dataset 4-GPU feature extraction and loss/quality convergence validation.
- The training config intentionally keeps `data_path` empty until the train
  feature cache has been produced.
