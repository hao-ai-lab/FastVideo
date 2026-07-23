# WanTrack training

WanTrack extends Wan I2V with sparse point tracks. The same model wrapper and
conditioning path are used for preprocessing, training, validation, and
standalone inference. Both bidirectional and causal checkpoints are supported.

## Build an initialization checkpoint

WanTrack uses the pretrained control slot from Wan2.1-Fun Control as its track
slot. Convert a VideoX-Fun control checkpoint together with a diffusers-format
Wan2.1-Fun InP base:

```bash
python scripts/checkpoint_conversion/wan_fun_control_to_trackwan.py \
    --inp-base models/Wan2.1-Fun-1.3B-InP-Diffusers \
    --control-ckpt models/Wan2.1-Fun-1.3B-Control/diffusion_pytorch_model.safetensors \
    --out models/wantrack-control-init
```

For the causal transformer, add `--causal` and use a separate output directory:

```bash
python scripts/checkpoint_conversion/wan_fun_control_to_trackwan.py \
    --inp-base models/Wan2.1-Fun-1.3B-InP-Diffusers \
    --control-ckpt models/Wan2.1-Fun-1.3B-Control/diffusion_pytorch_model.safetensors \
    --out models/wantrack-control-causal-init \
    --causal
```

`--causal` selects `CausalTrackWanTransformer3DModel` in the generated
transformer config. Without it, the converter selects
`TrackWanTransformer3DModel`.

The converted patch input has 52 channels:

| Channels | Meaning |
|----------|---------|
| `0:16` | Noisy video latent |
| `16:20` | I2V mask |
| `20:36` | First-frame latent |
| `36:52` | Track map |

The first three entries form the usual 36-channel Wan I2V input. A
`TrackEncoder` rasterizes sparse points on the VAE grid and appends the
16-channel track map.

## Prepare point-track data

Each video needs an `.npz` sidecar referenced by `points_path` in its metadata.
The archive must contain:

- `tracks`: floating-point source-pixel coordinates with shape `[T, N, 2]`.
  The final dimension is `(x, y)`.
- `visibility`: a boolean or numeric visibility mask with shape `[T, N]`.

`T` is the source-video timeline and `N` is the number of point tracks. The
preprocessor applies the same temporal sample and center-crop/resize transform
to the sidecar as it applies to the video. Coordinates should therefore be in
the original video's pixel space, not normalized or pre-cropped. Points outside
the retained crop are marked invisible.

For example, a merged-dataset annotation can contain:

```json
[
  {
    "path": "videos/clip_0001.mp4",
    "points_path": "tracks/clip_0001.npz",
    "cap": ["A cyclist follows a winding road."],
    "resolution": {"width": 1920, "height": 1080},
    "fps": 24.0,
    "duration": 5.0,
    "num_frames": 120
  }
]
```

The paths are relative to the dataset root named in the merge file:

```text
data/wantrack/raw,data/wantrack/metadata.json
```

All examples combined into one training batch must have a stackable point
dimension. Keeping `N` fixed across the dataset is the simplest option.

Run the I2V-track preprocessor with matching pixel and latent lengths. Wan's
temporal compression is four, so 81 pixel frames produce 21 latent frames:

```bash
torchrun --nproc_per_node=1 fastvideo/pipelines/preprocess/v1_preprocess.py \
    --model_path models/wantrack-control-init \
    --data_merge_path data/wantrack/data_merge.txt \
    --output_dir data/wantrack/preprocessed \
    --preprocess_task i2v_track \
    --num_frames 81 \
    --num_latent_t 21 \
    --max_height 480 \
    --max_width 832 \
    --train_fps 16 \
    --preprocess_video_batch_size 1
```

The trainer reads the resulting
`data/wantrack/preprocessed/combined_parquet_dataset` directory with
`preprocessed_data_type: i2v_track`.

## Train the bidirectional model

The bidirectional training wrapper is
`fastvideo.train.models.wantrack.WanTrackModel`; it loads
`TrackWanTransformer3DModel`.

```bash
torchrun --nproc_per_node=8 -m fastvideo.train.entrypoint.train \
    --config examples/train/configs/fine_tuning/wantrack/bidirectional_i2v.yaml
```

The example optionally subsamples points and applies temporal track masking.
Track IDs are sampled once for a training sample and then reused by every
denoising call and its conditional/unconditional branches. Re-sampling IDs per
call would change the point embeddings even when the coordinates are
unchanged.

## Train the causal model

The causal wrapper is
`fastvideo.train.models.wantrack.WanTrackCausalModel`; it loads
`CausalTrackWanTransformer3DModel`. The example uses
`TeacherForcingSFTMethod`, so clean history and the noisy current chunk receive
the same I2V and track conditioning:

```bash
torchrun --nproc_per_node=8 -m fastvideo.train.entrypoint.train \
    --config examples/train/configs/fine_tuning/wantrack/causal_i2v.yaml
```

Track encoding is causal at the VAE temporal boundary. The implementation
left-pads the complete source track sequence at its global beginning, encodes
it, and then slices the latent track map at the current latent `start_frame`.
It must not independently pad and encode each later chunk: doing so would
reset temporal context and produce a different feature for the same point
history.

The same stable `track_ids` are reused across all chunks, denoising steps, and
CFG branches. Causal sampling follows the RobotWM streaming contract:
`predict_noise_streaming()` owns the KV caches, each denoised block is committed
once as clean context, and caches are cleared at the sample boundary.

## Validation and inference

Both example configs use
`fastvideo.train.callbacks.track_validation.TrackValidationCallback`. It loads
fixed samples from the preprocessed WanTrack parquet, builds conditions through
the student's normal `prepare_batch()` path, generates videos, overlays the
active tracks, and logs them to the configured tracker. Set
`callbacks.track_validation.val_data_path` to use a dedicated validation
parquet; otherwise it samples from the training parquet.

The callback and standalone callers share two helpers:

```python
from fastvideo.train.models.wantrack.inference import (
    prepare_wantrack_batch,
    sample_wantrack,
)

batch = prepare_wantrack_batch(
    model,
    raw_batch,
    seed=1000,
    latents_source="zeros",
)
latents = sample_wantrack(
    model,
    batch,
    num_inference_steps=30,
    seed=1000,
    text_guidance_scale=3.0,
    motion_guidance_scale=1.5,
)
video = model.decode_latents(latents)
```

`sample_wantrack()` denoises the complete clip for `WanTrackModel`. For
`WanTrackCausalModel`, it uses the existing `CausalModelBase` streaming API and
the transformer's configured block size; it does not modify or fork the common
Wan causal denoising stage.
