# Causal WanTrack Control

This standalone prototype prepares one image and continuously generates causal
WanTrack blocks. Handle updates received during block N are committed only at
the next block boundary, so generated frames and control history are immutable.
One GPU session may generate at a time.

Set a Diffusers-format causal WanTrack export and its training YAML, then launch:

```bash
export WANTRACK_MODEL_DIR=/path/to/wantrack-causal-export
export WANTRACK_YAML_PATH=/path/to/causal_i2v.yaml
python -m apps.wantrack_control
```

Open `http://127.0.0.1:8010`. FFmpeg with `libx264` is required. Completed
blocks are preserved under `WANTRACK_OUTPUT_DIR` (or the system temporary
directory) and concatenated into a downloadable MP4 on Stop, disconnect, or a
recoverable failure.

The `/ws` endpoint accepts `prepare`, `start`, `control_update`, and `stop`
JSON messages. It emits `prepared`, `session_started`, `block_started`,
`control_applied`, `media_init`, `media_segment_complete`, `stream_complete`,
and terminal `error` events. Binary frames carry one fMP4 initialization
section followed by ordered block fragments.
