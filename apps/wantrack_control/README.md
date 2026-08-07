# Causal WanTrack Control

This standalone prototype prepares one image and continuously generates causal
WanTrack blocks. Handle updates received during block N are committed only at
the next block boundary, so generated frames and control history are immutable.
One GPU session may generate at a time. The SF checkpoint uses its fixed
DMD four-step schedule (`method.dmd_denoising_steps`, default
`[1000, 750, 500, 250]` with warp) without classifier-free guidance; the
server does not accept client overrides for steps or guidance.

Set a Diffusers-format causal WanTrack export and its Self-Forcing training
YAML, then launch:

```bash
export WANTRACK_MODEL_DIR=/path/to/wantrack-causal-export
export WANTRACK_YAML_PATH=/path/to/sf/config/run.yaml
export WANTRACK_TAEHV_CHECKPOINT=/path/to/taew2_1.pth
python -m apps.wantrack_control
```

Open `http://127.0.0.1:8010`. FFmpeg with `libx264` is required. Completed
blocks are preserved under `WANTRACK_OUTPUT_DIR` (or the system temporary
directory) and concatenated into a downloadable MP4 on Stop, disconnect, or a
recoverable failure.

The interactive preview uses the official TAEHV `StreamingTAEHV` decoder with
the Wan 2.1 `taew2_1.pth` weights. The full Wan VAE remains loaded only because
input preparation still needs its encoder.

The `/ws` endpoint accepts `prepare`, `start`, `control_update`, and `stop`
JSON messages. It emits phase-specific `progress`, `prepared`,
`session_started`, `block_started`, `block_encoding`, `control_applied`,
`media_init`, `media_segment_complete`, `stream_complete`, and terminal
`error` events. Binary frames carry one fMP4 initialization section followed
by ordered block fragments.
