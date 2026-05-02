# Align debug: FastVideo vs flow_grpo

## Branch

Align on branch **`align_debug`**. flow_grpo lives under `FastVideo/flow_grpo/`.

## Runscripts, env, and run order

- **Runscripts:**  
  - flow_grpo: `FastVideo/flow_grpo/train_wan.sh`  
  - FastVideo: `FastVideo/examples/training/rl/finetune_t2v_grpo_4gpu.sh`
- **Conda:** flow_grpo → e.g. `flow_grpo`; FastVideo → e.g. `fastvideo_shijie`.
- **`ALIGN_LOGS_ROOT`:** Set (and export) in both scripts to the directory where align logs and videos should go. Use the **same** path for both runs so FastVideo can read flow’s `debug_metrics.txt`. If unset, logs use repo-relative defaults.
- **Run order:** Run **flow_grpo first**, then **FastVideo**.

## Log and video locations

| Codebase   | Logs & videos (when `ALIGN_LOGS_ROOT` set) | Default (unset) |
|-----------|---------------------------------------------|------------------|
| FastVideo | `$ALIGN_LOGS_ROOT/align_logs/fv_logs/`       | `FastVideo/align_logs/fv_logs/` |
| flow_grpo | `$ALIGN_LOGS_ROOT/align_logs/flow_logs/`     | `FastVideo/flow_grpo/align_logs/flow_logs/` |

- **Log files:** `prompts.txt`, `debug_metrics.txt` (per batch + reward/advantage section).
- **Videos:** `debug_batch*_sample*.mp4` (FastVideo), `debug_batch*_j*_sample_*.mp4` (flow_grpo).

## Variables in logs and how to interpret

**Per-batch (both):**  
`sum_initial_latents_fp64`, `sum_intermediate_latents_fp64`, `sum_model_pred_fp64`, `sum_prompt_embeds_fp64`, `sum_negative_prompt_embeds_fp64`, `sum_decoded_fp64`; `step_k sum_model_pred_fp64`, `step_k sum_intermediate_latents_fp64`, `step_k sum_variance_noise_fp64`; `num_inference_steps`, `guidance_scale`, `height`, `width`, `num_frames`, `batch_size`.

**FastVideo only (after flow run):**  
`pct_diff_vs_flow_<key>` = `100 * (fastvideo_value - flow_value) / (|flow_value| + 1e-20)`.  
0% = match; &gt;0% = FastVideo larger; &lt;0% = FastVideo smaller. Large |%| = numerical/pipeline difference for that quantity.
