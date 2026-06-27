# Data Pipeline — Progress Log

Project: build a `(video, prompt, point-tracks)` dataset to train a motion-controlled
real-time interactive video model (bidirectional finetune stage first).

Branch: `shao/realtime-bidir`. This `data_pipeline/` dir holds the **Stage-0** data
generation scripts (everything *upstream* of FastVideo's `preprocess` pipeline).

## Pipeline shape (where we are)

```
vidprom prompts ──> generate_videos.py ──> videos/*.mp4 + videos2caption.json + merge.txt
                                              │
                    extract_tracks.py  ──> tracks/*.npz  (+ patch points_path into the json)
                                              │
                    [TODO] v1_preprocess.py --preprocess_task i2v_points ──> parquet
                                              │
                    [TODO] bidir I2V+points trainer
```

## Status

- [x] Located existing FastVideo preprocess stack + MatrixGame2 control-signal pattern (the template).
- [x] Confirmed env = repo `.venv` (`/mnt/weka/home/hao.zhang/shao/FastVideo/.venv/bin/python`).
- [x] Confirmed Wan2.2-T2V-A14B weights cached (`~/.cache/huggingface/hub/models--Wan-AI--Wan2.2-T2V-A14B-Diffusers`).
- [x] Wrote `generate_videos.py` (Wan2.2-14B T2V → mp4 + manifest).
- [x] Wrote `extract_tracks.py` (CoTracker v3 50×50 grid → npz, patch manifest).
- [~] Smoke-test launched on fs-mbz-gpu-538 (`run.sh --smoke`, background). Node validated:
      internet=yes (downloads OK on the node), 8 GPUs idle, `.venv` imports fastvideo + torch
      2.11.0/CUDA, cotracker3_offline loads (25.4M params). Prompts downloaded: 248,221 (140 MB).
- [ ] Scale to 50, then 5k.
- [ ] Extend preprocess with an `i2v_points` task (mirror MatrixGame2).

## Open questions / decisions pending

- Training model for the bidir stage: 1.3B/480p for plumbing bring-up vs straight to 14B (data gen is fixed at 14B/720p regardless — user confirmed).
- Confirm conditioning shape: I2V + points (input image + per-frame points → video). Assumed yes.
- Point sampling at train time: store full 2500-pt grid + visibility; sample 1–200 in the trainer.

See `DECISIONS.md` for rationale, `RUNBOOK.md` for exact commands.
