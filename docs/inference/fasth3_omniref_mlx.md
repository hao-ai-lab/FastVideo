# FastH3 OmniRef (Ref2VA) on Apple Silicon MLX

FastH3 OmniRef is a reference-conditioned PDD student: a prompt plus ordered
image/video/audio references become one video+audio clip in the export's fused
transformer forwards. The Apple Silicon runtime runs it natively through MLX,
the same way it runs the text-to-video-audio FastH3 checkpoints.

The CUDA counterpart is the FastH3 OmniRef consumer-devices page; it shares this
one's packed layout and export contract, so a clip is built from the same
request on either runtime. (Its page lands with the CUDA consumer stack, so the
link resolves once that is in.)

## What you need

1. A **FastH3 OmniRef PDD export** — `transformer_ref/`, the two scheduler
   configs and `fastvideo_inference.json`.
2. A **base MiniMax-H3 snapshot** — `text_encoder/`, `tokenizer/`,
   `processor/`, `vae/`, `audio_vae/`. The export does not carry these.
3. An **MLX conversion of `transformer_ref`**. The runtime reads a packed MLX
   checkpoint, not the diffusers shards.

Compose the first two into one directory of symlinks and convert the DiT:

```bash
python examples/inference/basic/mlx_fasth3_omniref.py \
  --export ./FastH3-OmniRef-QAD120 \
  --base-model-path ./MiniMax-H3 \
  --composed-dir ./FastH3-OmniRef-composed \
  --mlx-checkpoint ./placeholder --prompt x --image reference.png   # composes, then fails on the DiT

python scripts/checkpoint_conversion/convert_minimax_h3_mlx.py \
  --model-root ./FastH3-OmniRef-composed/transformer_ref \
  --out ~/models/FastH3-OmniRef-MLX --formats bf16 --include-vsa
```

The composed directory is not throwaway: the runtime reads the contract
(`fastvideo_inference.json`) from it to resolve the trained step ladder, and the
VAEs and text encoder from it for every request.

## Generate

```bash
python examples/inference/basic/mlx_fasth3_omniref.py \
  --model-root ./FastH3-OmniRef-composed \
  --mlx-checkpoint ~/models/FastH3-OmniRef-MLX/bf16 \
  --image reference.png --audio voice.wav \
  --prompt "a person on a sofa, lit by a warm lamp, speaking to camera" \
  --output-path ~/fasth3_outputs/omniref.mp4
```

References are **ordered**: pass them in the order the model should read them.
`--video dance.mp4 --image outfit.png --audio voice.wav` reads the video first,
then the image, then the audio. At least one image or video reference is
required; audio references accompany them.

`--steps` defaults to the contract's `transformer_forwards`, which is the
fused-block count the student was distilled on. Passing a different count is
rejected unless the export supports it.

The prompt is cached under `--prompt-cache-dir` **for text-only requests**; a
reference-conditioned request re-encodes, because the vision tokens are part of
the conditioning.

## What this path does and does not do

The MLX runtime runs the same packed layout, the same reference conditioning
and the same contract ladder as the CUDA path. Where it differs:

- **No NVFP4.** The MLX DiT is a BF16/INT8/INT6/INT4 conversion. The export's
  bf16-kept blocks land where the calibration put them, but nothing is FP4.
- **VSA reference regions are dense prefix.** With `--vsa`, the reference rows
  ride as dense prefix tiles split per reference. The CUDA path's
  `p2_multi_region` policy sparsifies each reference video region at
  `vsa_ref_keep_rate` instead, so reference rows are treated more conservatively
  here. Reference-conditioned VSA also needs the checkpoint converted with
  `--include-vsa`, like the T2VA path.
- **The prefix split has no MLX equivalent.** The CUDA consumer route runs dense
  prefix query tiles at 8-bit Q/K to keep conditioning out of FP4 rounding;
  there is no FP4 kernel to split away from on Apple Silicon.
- **Noise draws are not bit-identical.** The seeded reference-posterior draw and
  the target noise use `mx.random`, so the same request gives the same clip on a
  given machine but not the same clip as the CUDA path.
- **The VAE encoders load only for reference requests.** They are needed to
  encode the references and are released after, which is why a
  reference-conditioned request carries a transient VAE-sized allocation.

Not yet ported, and so absent rather than approximated: FL2VA keyframes through
the MLX pipeline (`--image` on `mlx_fasth3.py`), two-pass refinement, and
`--video-decode-backend taeh3` for reference requests (the reference encode
needs the real video VAE).

## When something goes wrong

- **`A reference-conditioned request needs the video VAE encoder`** — you passed
  `--video-decode-backend taeh3`. Use `h3-vae`.
- **`not a Ref2VA contract`** — the export is a T2VA one; the PDD ladder and the
  reference layout come from a `model_type: ref2va` contract.
- **`VSA-H3 cannot run with this packed layout`** — the tile size does not fit
  the reference prefix. Try `--vsa-tile-size 64`, or drop `--vsa`.
