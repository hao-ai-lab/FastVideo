# v2_examples/training — training & RL on the *same* loops

Runnable examples of every training method in `v2`. The load-bearing idea (designv4 §8): **a method's
rollout drives the same loop the engine serves** (in the ROLLOUT profile, with behavior capture) — there
is one numerics surface, not a second sampler. Each method differs only in loss math, roles, and capture.

CPU + numpy only; the toy `ToyDiT.mse_grad_step` / `ToyPromptRefiner.pg_step` are *real* gradient steps,
so the toy students actually learn — the structural stand-in for an FSDP/optimizer step. Run from anywhere:

```bash
python3 v2_examples/training/03_rl_diffusion_nft.py
```

| Script | Method | Kind |
|---|---|---|
| `01_finetune.py` | flow-match finetuning | supervised |
| `02_distillation_dmd2.py` | DMD2 (student + fake-score critic + teacher) | distillation |
| `03_rl_diffusion_nft.py` | DiffusionNFT (samples from the decay-blended *old* policy) | RL, likelihood-**free** (C2) |
| `04_self_forcing.py` | self-forcing on the causal `chunk_rollout` loop | causal distillation |
| `05_joint_lm_generator_rl.py` | UniRL/PromptRL: LM refiner + flow generator, one reward | joint RL, likelihood-**based** (C2) |
| `06_nway_joint_rl.py` | N refiner experts + generator, one reward | N-way joint RL |
| `07_end_to_end_workflow_rl.py` | T2I + I2V trained from one final-video reward | end-to-end RL across a workflow |

Every method exposes the same entry point: `method.train_step(batch, iteration) -> (loss_map, metrics)`.
RL methods (`manages_optimization() == True`) own their sample→score→update cadence inside that call.
See designv4 §9.3 (joint RL), §9.7 (N-way), §9.9 (workflow RL).
