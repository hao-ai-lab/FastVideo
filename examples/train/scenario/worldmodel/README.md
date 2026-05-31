# World-Model: Matrix-Game 2.0 I2V

Three training scenarios for the Matrix-Game 2.0 I2V world model on the
new YAML-driven trainer (`fastvideo/train/entrypoint/train.py`).

| Config | Method | Student | Notes |
|---|---|---|---|
| `finetune_i2v.yaml` | `FineTuneMethod` | `MatrixGame2Model` (bidirectional) | Multi-step SFT from `mg_bidirectional_Solaris`. |
| `dfsft_causal_i2v.yaml` | `DiffusionForcingSFTMethod` | `MatrixGame2CausalModel` | Diffusion-Forcing SFT with chunkwise timesteps. |
| `self_forcing_causal_i2v.yaml` | `SelfForcingMethod` | `MatrixGame2CausalModel` | DMD/Self-Forcing distillation; teacher = bidirectional, critic = bidirectional. |

## Usage

```bash
bash examples/train/run.sh \
    examples/train/scenario/worldmodel/finetune_i2v.yaml

bash examples/train/run.sh \
    examples/train/scenario/worldmodel/dfsft_causal_i2v.yaml

bash examples/train/run.sh \
    examples/train/scenario/worldmodel/self_forcing_causal_i2v.yaml
```

Override any field on the command line:

```bash
bash examples/train/run.sh \
    examples/train/scenario/worldmodel/dfsft_causal_i2v.yaml \
    --training.distributed.num_gpus 8 \
    --training.optimizer.learning_rate 1e-5
```
