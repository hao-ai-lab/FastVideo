# World-Model: Matrix-Game 2.0 I2V

Training scenarios for the Matrix-Game 2.0 I2V world model on Minecraft and
Zelda data, using the new YAML-driven trainer
(`fastvideo/train/entrypoint/train.py`).

| Config | Method | Student | Notes |
|---|---|---|---|
| `finetune_i2v.yaml` | `FineTuneMethod` | `MatrixGame2Model` (bidirectional) | Multi-step SFT from `mg_bidirectional_Solaris`. |
| `dfsft_causal_i2v.yaml` | `DiffusionForcingSFTMethod` | `MatrixGame2CausalModel` | Diffusion-Forcing SFT with chunkwise timesteps. |
| `self_forcing_causal_i2v.yaml` | `SelfForcingMethod` | `MatrixGame2CausalModel` | Matrix-Game 2.0 DMD/Self-Forcing distillation; teacher = bidirectional, critic = bidirectional. |
| `self_forcing_causal_i2v_zelda.yaml` | `SelfForcingMethod` | `MatrixGame2CausalModel` | Zelda world model DMD/Self-Forcing distillation; teacher = bidirectional, critic = bidirectional. |
| `streaming_long_tuning_causal_i2v.yaml` | `StreamingLongTuningMethod` | `MatrixGame2CausalModel` | LongLive-style streaming long tuning from the 1k-step Zelda self-forcing checkpoint. |

Zelda world-model distillation is a two-run workflow: first run
`self_forcing_causal_i2v_zelda.yaml` to train or load the 1k-step self-forcing
checkpoint (`mignonjia/mg_sf_distilled_zelda_1k_steps`), then run
`streaming_long_tuning_causal_i2v.yaml` for the 3k-step streaming long-tuning
stage. The long-tuning YAML starts from that 1k-step checkpoint; it does not run
the short self-forcing stage inside the same config.

## Zelda Training Data

The Zelda training configs use `data/zeldam2-clean` as a suggested local path.
Download the dataset from Hugging Face before running those configs:

```bash
python scripts/huggingface/download_hf.py \
    --repo_id mignonjia/zeldam2-clean \
    --local_dir data/zeldam2-clean \
    --repo_type dataset
```

You can store the dataset elsewhere; update `training.data.data_path` in the
YAML to point at that location.

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
