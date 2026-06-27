# Research Log — Motion-Controlled Real-Time Video World Model

Lab notebook for building a **motion-controlled, real-time interactive video world
model** on top of Wan2.2, in FastVideo (branch `shao/realtime-bidir`).

## Project goal

Convert Wan2.2-14B into a real-time interactive video model driven by **point-track
motion control** (MotionStream-style), targeting **14B + 720p + 24fps**. The plan
combines MotionStream-style training with StreamDiffusion-v2-style inference.

## Stages

1. **Bidirectional finetune** — teach the bidirectional Wan DiT to take point tracks
   as a motion-control input. ← *current focus*
2. **Causal finetune** — convert to a causal (streaming-capable) model.
3. **Step distillation** — few-step generation.
4. **Self-forcing** — autoregressive rollout stability.

End state: real-time interactive generation where a user drags points / paints
trajectories and the video follows.

## Reference

- **MotionStream** (arXiv 2511.01266) — the training recipe + evaluation we follow.
  See [05-eval-methodology-motionstream.md](05-eval-methodology-motionstream.md).

## Log index

| File | Contents |
|------|----------|
| [01-data-pipeline.md](01-data-pipeline.md) | Stage-0 data: video generation + CoTracker3 track extraction |
| [02-model-architecture.md](02-model-architecture.md) | TrackWan DiT, channel layout, track encoder, training wrapper |
| [03-training-and-overfit-results.md](03-training-and-overfit-results.md) | Bidir overfit runs, loss curves, reconstruction convergence |
| [04-controllability-eval-and-findings.md](04-controllability-eval-and-findings.md) | **Key result:** EPE controllability test → content-overfit, tracks ignored |
| [05-eval-methodology-motionstream.md](05-eval-methodology-motionstream.md) | How MotionStream evaluates (EPE, quality, interactive modes) |
| [06-tooling.md](06-tooling.md) | Synthetic track authoring, EPE metric, standalone inference, Gradio app |
| [07-open-questions-next-steps.md](07-open-questions-next-steps.md) | What's blocking, the data-design fix, prioritized TODOs |

## Current status (2026-06-27)

- ✅ Stage-0 data pipeline (10-clip smoke set generated + tracked).
- ✅ TrackWan bidir I2V+text+track training pipeline built and runs end-to-end.
- ✅ Overfit on 10 clips: reconstruction works (loss 0.42→0.06; sample-0 PSNR→17.8 dB).
- ✅ Controllability eval harness (synthetic tracks + CoTracker EPE + Gradio).
- ❌ **The overfit model ignores the tracks** — it reconstructs clips from text +
  first frame and does not follow counterfactual controls. This is structural, not a
  bug (see [04](04-controllability-eval-and-findings.md)). Fix = data design where
  first-frame+text under-determine motion.

## Key paths (outside the repo)

```
data root      /mnt/weka/home/hao.zhang/shao/data/motion_pipeline/wan22_a14b_720p_24fps/
init model     /mnt/weka/home/hao.zhang/shao/data/models/trackwan_1.3b_init/
overfit ckpts  /mnt/weka/home/hao.zhang/shao/data/motion_pipeline/wantrack_overfit_out/checkpoint-{3500,4000}
export (4k)    /mnt/weka/home/hao.zhang/shao/data/models/trackwan_1.3b_overfit4k/
wandb          project `wantrack-bidir` (runs 32dlm3ce: 0–600, tcze44uf: 600–4000)
```

## Compute

- SLURM allocation `shao_wm` (job 1788946) on node `fs-mbz-gpu-538`, 8×H200 (~140 GB).
- Attach: `srun --jobid=1788946 --overlap --ntasks=1 ...`. Never run model code on login.
