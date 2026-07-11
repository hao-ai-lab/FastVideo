# MixKit-21 causal-kernel teacher-forcing ablation

All runs use the same training and validation settings. Only the three
attention axes listed in `experiment_matrix.tsv` vary.

| ID | sink | local | RoPE | lane | primary contrast |
|---|---:|---:|---|---|---|
| A01 | 0 | 21 | absolute | node0 | rope policy control |
| A02 | 0 | 21 | relativistic | node1 | rope policy control |
| A03 | 1 | 21 | relativistic | node0 | sink at local 21 |
| A04 | 0 | 6 | relativistic | node0 | local window at sink 0 |
| A05 | 1 | 6 | relativistic | node0 | sink at local 6 |
| A06 | 0 | 12 | relativistic | node1 | local window at sink 0 |
| A07 | 1 | 12 | relativistic | node1 | sink at local 12 |
| A08 | 3 | 12 | relativistic | node1 | sink size at local 12 |

Fixed invariants:

- MixKit precomputed data at 480x832 with 21 stored latent frames.
- Teacher forcing for 2,000 steps, batch size 1, chunk size 3.
- Fused training attention: `causal_train_attention: triton`.
- Validation at 249 pixel frames, equivalent to 63 Wan latent frames.
- Validation reuses the training transformer and pipeline config, so sink,
  local window, and RoPE policy stay identical between training and inference.
- Four GPUs, full gradient checkpointing, validation/checkpoint every 200/1000
  steps, and W&B online tracking in a dedicated project.

Validate the matrix and template:

```bash
python scripts/train/manage_mixkit21_tf_ablation.py validate
```

Run one lane after supplying `WANDB_API_KEY` at runtime:

```bash
SEQUENCE_ID=<timestamp> LANE=node0 \
RUN_CONDITIONS="A01 A03 A04 A05" \
bash scripts/train/run_mixkit21_tf_ablation_lane.sh
```
