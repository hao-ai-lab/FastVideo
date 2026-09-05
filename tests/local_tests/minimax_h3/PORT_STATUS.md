# MiniMax H3 port status

## Status

- workloads: T2VA, FL2VA, Ref2VA joint video/audio generation
- component parity: complete
- FastVideo runtime acceptance: complete
- official end-to-end pipeline parity: complete
- modular Ref2VA/LoRA training: single-rank real-checkpoint LoRA/FSDP initialization complete; full train/export/inference pending

## Coverage

| Scope | Evidence | State |
|---|---|---|
| Qwen3-VL encoder | exact text/image/video hidden states through the production loader | complete |
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
| Modular Ref2VA packing | target-only loss slicing plus pinned row/position/tag parity | complete |
| LoRA ownership | exact 312-layer coverage; two-rank FSDP gradient and DCP-resume parity | complete |
| Real checkpoint LoRA init | strict `transformer_ref` load; 312 wrappers/624 DTensor adapters; CUDA/FSDP ownership | complete on one GB10 |
| Training export | checkpoint-wrapper-safe native LoRA merge; physical `transformer_ref`; canonical Diffusers shards | CPU contract complete; real checkpoint pending |

## Current validation

T2VA, FL2VA, and Ref2VA match the official video/audio latents exactly.
The single-rank real-checkpoint gate strictly loaded the 33.30B-parameter
`transformer_ref` component on one GB10, verified 312 LoRA wrappers and 624
trainable DTensor adapters, and recorded 62.08 GiB peak CUDA allocation. It did
not run a forward pass. The added training contracts do not claim an official
full training, export, reload, or inference run.

## Decisions

- Preserve each H3 scheduler's configured shift; global `flow_shift` is invalid.
- Do not wrap the H3 DiT in global autocast; its FP32 projections must stay FP32.
- Let FSDP move CPU-offloaded Qwen parameters; do not move the wrapped conditioner as a whole.
- Load `transformer/` for T2VA/FL2VA and `transformer_ref/` for Ref2VA.
- Keep `last_image`, `references`, and `audio_latents` on the typed request path.
- Treat the published component folders as the loading boundary.
- Keep reference videos on CPU between VAE clips and decode final pixels only on the executor's output rank.
- Insert H3 LoRA modules on the meta model before FSDP so adapters share the
  transformer's ownership, gradient synchronization, and checkpoint topology.
- Export H3 training adapters merged into the native physical component;
  standalone adapter loading is not part of the inference contract.
- Remove activation-checkpoint wrapper path segments when mapping discovered
  LoRA modules to their canonical state-dict keys.
- Emit H3 weights with Diffusers' conventional safetensors names and 5 GB
  shards; keep the converter's legacy `model.safetensors` contract for model
  plugins that have not opted into this layout.

## Evidence boundary

Completed rows summarize recorded component and FastVideo runtime runs. Registry smoke, generated media, and
FastVideo SP consistency are supporting checks, not substitutes for the recorded official comparisons.
The synthetic fixture is repository-owned test input, not model-quality evidence. Full H3 checkpoint export remains
ungated because rank 0 must gather a roughly 62 GiB transformer while the live model is resident; a 121 GiB
unified-memory GB10 is not a supported target for that full-state operation.
