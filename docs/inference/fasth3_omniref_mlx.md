# FastH3 OmniRef (Ref2VA) on Apple Silicon MLX

FastH3 OmniRef is a reference-conditioned PDD student: a prompt plus ordered
image/video/audio references become one video+audio clip in the export's fused
transformer forwards. The Apple Silicon runtime runs it natively through MLX,
the same way it runs the text-to-video-audio FastH3 checkpoints.

The CUDA counterpart is [FastH3 OmniRef inference](fasth3_omniref.md); it shares
this page's packed layout and export contract, so a clip is built from the same
request on either runtime.

## What you need

1. The **FastH3 OmniRef PDD export**
   [`FastVideo/FastVideo-FastH3-Omni-8-Step-V1`](https://huggingface.co/FastVideo/FastVideo-FastH3-Omni-8-Step-V1)
   — `transformer_ref/`, the two scheduler configs and
   `fastvideo_inference.json`. Use this BF16 release: the MLX converter needs
   BF16 weights, so the packed NVFP4 CUDA export cannot be converted.
2. A **base MiniMax-H3 snapshot** — `text_encoder/`, `tokenizer/`,
   `processor/`, `vae/`, `audio_vae/`. The export does not carry these.
3. An **MLX conversion of `transformer_ref`**. The runtime reads a packed MLX
   checkpoint, not the diffusers shards. There is no FP4 on Apple Silicon, so
   convert to INT8, INT6 or INT4 - the same weight-only formats the T2VA MLX path
   uses.

Compose the first two into one directory of symlinks and convert the DiT:

```bash
python examples/inference/basic/mlx_fasth3_omniref.py --compose-only \
  --export ./FastVideo-FastH3-Omni-8-Step-V1 \
  --base-model-path ./MiniMax-H3 \
  --composed-dir ./FastH3-OmniRef-composed

python scripts/checkpoint_conversion/convert_minimax_h3_mlx.py \
  --model-root ./FastH3-OmniRef-composed/transformer_ref \
  --out ~/models/FastH3-OmniRef-MLX --formats int8 --include-vsa
```

The composed directory is not throwaway: the runtime reads the contract
(`fastvideo_inference.json`) from it to resolve the trained step ladder, and the
VAEs and text encoder from it for every request.

## Generate

```bash
python examples/inference/basic/mlx_fasth3_omniref.py \
  --model-root ./FastH3-OmniRef-composed \
  --mlx-checkpoint ~/models/FastH3-OmniRef-MLX/int8 \
  --image reference.png --audio voice.wav \
  --prompt "a person on a sofa, lit by a warm lamp, speaking to camera" \
  --output-path ~/fasth3_outputs/omniref.mp4
```

References are **ordered**: pass them in the order the model should read them.
`--video dance.mp4 --image outfit.png --audio voice.wav` reads the video first,
then the image, then the audio. At least one image or video reference is
required; audio references accompany them.

`--steps` defaults to the contract's `transformer_forwards`: one transformer
forward per fused block of the export's trained fine grid, exactly as on CUDA.
Each step fuses that block's widened output heads into their integration-
weighted mean, so the heads never all run at once. Passing a different count is
rejected, because the partition is the checkpoint's.

The prompt is cached under `--prompt-cache-dir` **for text-only requests**; a
reference-conditioned request re-encodes, because the vision tokens are part of
the conditioning.

## What this path does and does not do

The MLX runtime runs the same packed layout, the same reference conditioning
and the same contract ladder as the CUDA path. Where it differs:

- **No NVFP4.** The MLX DiT is an INT8/INT6/INT4 weight-only conversion, and
  the text encoder and VAEs are the base snapshot's rather than the NVFP4
  ones the CUDA consumer stack prefers. An NVFP4 text encoder still works
  through the resident MLX encoder path if you have one.
- **VSA follows the contract.** With VSA on (the default for a VSA contract),
  the trained tile size, sparsity and `p2_multi_region` reference policy with
  its `vsa_ref_keep_rate` are read from `fastvideo_inference.json`. VSA needs the
  checkpoint converted with `--include-vsa`, like the T2VA path; without it the
  run falls back to dense attention with a warning.
- **The prefix split has no MLX equivalent.** The CUDA consumer route runs dense
  prefix query tiles at 8-bit Q/K to keep conditioning out of FP4 rounding;
  there is no FP4 kernel to split away from on Apple Silicon.
- **Noise draws are not bit-identical.** The seeded reference-posterior draw and
  the target noise use `mx.random`, so the same request gives the same clip on a
  given machine but not the same clip as the CUDA path.
- **The VAE encoders load only for reference requests.** They are needed to
  encode the references and are released after, which is why a
  reference-conditioned request carries a transient VAE-sized allocation.
- **No fast temporal mode.** `--fast` stretches the generated video clock, and
  the reference-prefixed packing does not stretch with it. A reference request
  that asks for both is rejected rather than quietly stretched at one end.

Not yet ported, and so absent rather than approximated: FL2VA keyframes through
the MLX pipeline (`--image` on `mlx_fasth3.py`), two-pass refinement, and
`--video-decode-backend taeh3` for reference requests (the reference encode
needs the real video VAE).

## Video VAE without the base weights

`--vae-decoder` and `--vae-encoder` swap in standalone video VAE files, so the
base `vae/` shards are never read (its `config.json` is still read, unless the
encoder is a LightVAE file, which embeds the config and latent statistics).

- Decoder: FastH3 `quality-36L/decoder-nvfp4.safetensors` (1.4 GB, loads as MLX
  NVFP4 without requantizing), `decoder.safetensors` (bf16 masters; add
  `--vae-decoder-quantization nvfp4|mxfp8|int8|int4`), or a LightVAE decoder.
- Encoder: LightVAE `lightvae-lite-h3-encoder.safetensors` (block-pruned).
- Use `--vae-dtype bf16` for the decoder and `fp16` for the encoder; fp32 is
  several times slower and larger. Decode needs the tiled path (the default):
  the decoder was trained on 256-pixel tiles.

## When something goes wrong

- **`A reference-conditioned request needs the video VAE encoder`** — you passed
  `--video-decode-backend taeh3`. Use `h3-vae`.
- **`not a Ref2VA contract`** — the export is a T2VA one; the PDD ladder and the
  reference layout come from a `model_type: ref2va` contract.
- **`vsa_tile_size ... trained with`** — `--vsa-tile-size` differs from the
  contract's trained tile. Leave it unset.
- **`VSA-H3 cannot run with this packed layout`** — the tile size does not fit
  the reference prefix. Drop `--vsa` to run dense.
