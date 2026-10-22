# Bernini-R video editing

FastVideo provides a native integration of
[Bernini-R 1.3B](https://github.com/bytedance/Bernini). It edits one source
video using a text instruction, reusing FastVideo's Wan transformer blocks,
Wan VAE and UMT5 encoder. Sampling follows the official `v2v_apg` recipe:
UniPC with flow shift 3, guidance scale 4, APG norm threshold 50, eta 0.5 and
momentum 0.

The pipeline accepts one source video and one text instruction per request
and runs on one CUDA GPU.

## Prepare the checkpoint

The official repository has Diffusers component folders but no pipeline
manifest. The preparation command adds the manifest and native transformer
class metadata while preserving the original tensors and dtypes.

```bash
hf download ByteDance/Bernini-R-1.3B-Diffusers \
  --revision ff4c5d4d2d31365c2ffeb30e9753065ee18f58ce \
  --local-dir official_weights/bernini-r-1.3b

python scripts/checkpoint_conversion/bernini_r_to_diffusers.py \
  --source official_weights/bernini-r-1.3b \
  --output converted_weights/bernini-r-1.3b
```

The output directory must not already exist. By default it links the original
weights to avoid duplicating the checkpoint; keep the source directory in
place. Add `--copy` to create a standalone portable directory. Load the
prepared directory rather than passing the unmodified official repository
directly to `VideoGenerator`.

## Run an edit

Install FastVideo using the [GPU installation guide](../getting_started/installation/gpu.md).
Then run:

```bash
python examples/inference/basic/basic_bernini_r.py \
  --model converted_weights/bernini-r-1.3b \
  --video path/to/source.mp4 \
  --prompt "Change the scene to a watercolor painting while preserving the motion." \
  --height 480 --width 832 --frames 81 --fps 16 --steps 40
```

Height and width must be multiples of 16. Set them to preserve the source's
aspect ratio. FastVideo resizes the input to these explicit dimensions;
the official CLI instead derives dimensions from its `max_image_size`.
Frame sampling follows the official timestamp policy, including duplicated
final frames for very short clips. `num_frames` is a positive `4*k+1` upper
bound on the sampled clip. The output is a silent edited video.

The pipeline always encodes the negative prompt for APG, including at guidance
scale 1: norm clipping can still affect that case. The default negative
prompt is empty. Video conditioning uses the VAE posterior mode, and both
VAE encoding and decoding run in FP32, matching the reference. UMT5 uses BF16 with 512-token
padding, while the DiT retains the reference's FP32 parameter exceptions.
The optional `max_sequence_length` accepts values from 1 to 512.

## Runtime and memory

The example uses the offline Python API with whole-component CPU offload
for the transformer, text encoder and VAE. Validation used one NVIDIA
RTX PRO 6000 Blackwell 96 GB GPU at 480x848, 81 frames and 40 steps.
Minimum VRAM is unknown. Memory requirements vary with clip geometry and
offload settings.

Serving YAML and cookbook deployment: N/A for this offline integration.
The current serving schema requires a registered serving workload;
video editing is exposed here through `GenerationRequest.inputs.video_path`.

## Validation

Reproducible component parity commands, pinned source revisions and packing
invariants are documented in
[`tests/local_tests/bernini/README.md`](https://github.com/hao-ai-lab/FastVideo/blob/main/tests/local_tests/bernini/README.md).
The unit and transformer tests live in the existing CI-collected directories.

The Slurm SSIM scheduler also discovers
`fastvideo/tests/ssim/test_bernini_similarity.py`. It fixes the official
snowman-edit instruction and input video (SHA-256 checked), the immutable
checkpoint revision, SDPA, seed 42, 480x848, 81 frames, 16 fps and 40 steps.
The MS-SSIM requirement is `0.98`; missing references fail in normal mode.

```bash
FASTVIDEO_TEST_BERNINI_MODEL_DIR=official_weights/bernini-r-1.3b \
  pytest fastvideo/tests/ssim/test_bernini_similarity.py -vs
```

The optional test-only path points to the **official** download, including
its Hub revision metadata, rather than the prepared layout. With the default setting, the test
fetches the pinned public checkpoint into the Hub cache and prepares a
temporary manifest. Reference videos must match the actual GPU, backend
and runtime. Reference preparation follows the
[SSIM testing workflow](../contributing/testing.md#reference-videos).
