# Pika H3 Flex Ref2VA NVFP4 QAD: Slinky (GB200) run kit

Cluster scripts and resolved configs exactly as used for the first Flex Ref2VA QAD run (2026-10-30). Paths are the
cluster's (`/mnt/nfs/vlm-aryan/pika-flex-qad`); the code there is a checkout of this branch. Jobs attach to held
trays with `srun --overlap`.

| Step | Script | Notes |
|---|---|---|
| Teachers | `merge.sh` | Downloads both adapters and runs `merge_pika_adapter.py` for ref2va and fl2va. |
| Coverage rows | `build_rows.sh plan / encode / finalize` | `build_flex_ref2va_rows.py`: 480p rows at Flex reference sizing plus audio-reference, mixed image+video, 6-9 image and two-video rows, from the stored v4 latents. 1,186 rows. |
| Calibration | `calib_pipeline.sh` (uses `calib.sh`) | collect (1,133 rows, 8 GPUs) -> candidates -> MSE -> choose -> step-0 PTQ evals. Step-0 scores: max 0.33112, MSE 0.32987, balanced 0.32595 (pinned, `examples/train/calibration/pika_h3_flex/amax-ref2va-balanced.json`). |
| Warm-up trial | `theseus_trial.sh` | 8 steps each: baseline 0.34472 -> 0.32510, Theseus (4 warm-up + 4) -> 0.32943. Not adopted. |
| QAD | `qad_node.sh` (4 nodes x 4 GPUs) with `ref2va-qad.yaml` | Needs `FASTVIDEO_PG_TIMEOUT_MIN=120` (the default 10-minute collective timeout fires during the model load). |

`env.sh` sources the cluster's `secrets.env` by path; no credentials are stored here.
