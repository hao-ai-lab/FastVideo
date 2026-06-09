# World-Model: Matrix-Game 2.0 I2V

Three training scenarios for the Matrix-Game 2.0 I2V world model on the
new YAML-driven trainer (`fastvideo/train/entrypoint/train.py`).

| Config | Method | Student | Notes |
|---|---|---|---|
| `finetune_i2v.yaml` | `FineTuneMethod` | `MatrixGame2Model` (bidirectional) | Multi-step SFT from `mg_bidirectional_Solaris`. |
| `dfsft_causal_i2v.yaml` | `DiffusionForcingSFTMethod` | `MatrixGame2CausalModel` | Diffusion-Forcing SFT with chunkwise timesteps. |
| `self_forcing_causal_i2v_MG2.yaml` | `SelfForcingMethod` | `MatrixGame2CausalModel` | Matrix-Game 2.0 DMD/Self-Forcing distillation; teacher = bidirectional, critic = bidirectional. |
| `self_forcing_causal_i2v_zelda.yaml` | `SelfForcingMethod` | `MatrixGame2CausalModel` | Zelda world model DMD/Self-Forcing distillation; teacher = bidirectional, critic = bidirectional. |
| `streaming_long_tuning_causal_i2v.yaml` | `StreamingLongTuningMethod` | `MatrixGame2CausalModel` | Multi-stage self-forcing, then LongLive-style streaming long tuning. |

## Zelda Validation Data

The Zelda validation configs expect a small public validation bundle under
`data/zelda_validation_data`.

Download it from Hugging Face before running either Zelda scenario:

```bash
python scripts/huggingface/download_hf.py \
    --repo_id mignonjia/zelda_validation_data \
    --local_dir data/zelda_validation_data \
    --repo_type dataset
```

The bundle contains `validation_zelda.json`, `images/`, and `actions/`. The
Zelda configs point `callbacks.validation.dataset_file` at
`data/zelda_validation_data/validation_zelda.json`.

## Usage

```bash
bash examples/train/run.sh \
    examples/train/scenario/worldmodel/finetune_i2v.yaml

bash examples/train/run.sh \
    examples/train/scenario/worldmodel/dfsft_causal_i2v.yaml

bash examples/train/run.sh \
    examples/train/scenario/worldmodel/self_forcing_causal_i2v.yaml

# Zelda world model distillation
bash examples/train/run.sh \
    examples/train/scenario/worldmodel/self_forcing_causal_i2v_zelda.yaml

bash examples/train/run.sh \
    examples/train/scenario/worldmodel/streaming_long_tuning_causal_i2v.yaml
```

Override any field on the command line:

```bash
bash examples/train/run.sh \
    examples/train/scenario/worldmodel/dfsft_causal_i2v.yaml \
    --training.distributed.num_gpus 8 \
    --training.optimizer.learning_rate 1e-5
```
