# World-Model: Matrix-Game 2.0 I2V

Training scenarios for the Matrix-Game 2.0 I2V world model on Solaris (Minecraft)
data and Zelda data, using the new YAML-driven trainer
(`fastvideo/train/entrypoint/train.py`).

## Solaris Configs

| Config | Method | Student | Notes |
|---|---|---|---|
| `solaris/finetune_i2v.yaml` | `FineTuneMethod` | `MatrixGame2Model` (bidirectional) | Multi-step SFT from `mg_bidirectional_Solaris`. |
| `solaris/dfsft_causal_i2v.yaml` | `DiffusionForcingSFTMethod` | `MatrixGame2CausalModel` | Diffusion-Forcing SFT with chunkwise timesteps. |
| `solaris/self_forcing_causal_i2v.yaml` | `SelfForcingMethod` | `MatrixGame2CausalModel` | Matrix-Game 2.0 DMD/Self-Forcing distillation; teacher = bidirectional, critic = bidirectional. |

## Zelda Configs

| Config | Method | Student | Notes |
|---|---|---|---|
| `zelda/finetune_i2v.yaml` | `FineTuneMethod` | `MatrixGame2Model` (bidirectional) | Zelda bidirectional I2V finetuning from `FastVideo/Matrix-Game-2.0-Base-Diffusers`. Uses 33-frame clips and Zelda validation with action overlays. |
| `zelda/dfsft_causal_i2v.yaml` | `DiffusionForcingSFTMethod` | `MatrixGame2CausalModel` | Zelda causal Diffusion-Forcing SFT from `FastVideo/Matrix-Game-2.0-Base-Diffusers`. Uses the same Zelda data, resolution, optimizer, checkpoint, and validation defaults as the Zelda finetune config. |
| `zelda/self_forcing_causal_i2v.yaml` | `SelfForcingMethod` | `MatrixGame2CausalModel` | Zelda DMD/Self-Forcing distillation; teacher = bidirectional, critic = bidirectional. |
| `zelda/streaming_long_tuning_causal_i2v.yaml` | `StreamingLongTuningMethod` | `MatrixGame2CausalModel` | LongLive-style streaming long tuning from the 1k-step Zelda self-forcing checkpoint. |

Zelda world-model distillation is a two-run workflow: first run
`zelda/self_forcing_causal_i2v.yaml` to train or load the 1k-step
self-forcing checkpoint (`mignonjia/mg_sf_distilled_zelda_1k_steps`), then run
`zelda/streaming_long_tuning_causal_i2v.yaml` for the 3k-step streaming
long-tuning stage. The long-tuning YAML starts from that 1k-step checkpoint; it
does not run the short self-forcing stage inside the same config.

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

## Multi3D Training Data

`zelda/finetune_i2v.yaml` includes an optional, commented-out Multi3D entry.
Enable it only when you want to mix Zelda with multi-game data from
`data/multi3d_games`. You can store this dataset anywhere; before enabling it,
update the matching commented `training.data.data_path` key in the YAML to the
correct location.

To mix datasets in a training YAML, set `training.data.data_path` to a
path-to-repeat-count mapping. For example, `zelda/finetune_i2v.yaml` can use
`data/zeldam2-clean: 1` and `# data/multi3d_games: 10`; uncommenting the
Multi3D entry repeats the multi-game parquet list ten times before training
samples are shuffled.

## World Model Validation Data

The world-model validation configs expect a small public validation bundle under
`data/worldmodel_validation_data`.

Download it from Hugging Face before running the Zelda or mixed finetune
scenarios:

```bash
python scripts/huggingface/download_hf.py \
    --repo_id mignonjia/worldmodel_validation_data \
    --local_dir data/worldmodel_validation_data \
    --repo_type dataset
```

The bundle contains `validation_zelda.json`, `validation_mc_ood.json`,
`images/`, and `actions/`. The Zelda configs point
`callbacks.validation.dataset_file` at
`data/worldmodel_validation_data/validation_zelda.json`; mixed or OOD finetune
configs can point at `data/worldmodel_validation_data/validation_mc_ood.json`.

## Usage

### Solaris

```bash
bash examples/train/run.sh \
    examples/train/scenario/worldmodel/solaris/finetune_i2v.yaml

bash examples/train/run.sh \
    examples/train/scenario/worldmodel/solaris/dfsft_causal_i2v.yaml

bash examples/train/run.sh \
    examples/train/scenario/worldmodel/solaris/self_forcing_causal_i2v.yaml
```

### Zelda

```bash
# Finetuning / DFSFT
bash examples/train/run.sh \
    examples/train/scenario/worldmodel/zelda/finetune_i2v.yaml

bash examples/train/run.sh \
    examples/train/scenario/worldmodel/zelda/dfsft_causal_i2v.yaml

# Distillation / long tuning
bash examples/train/run.sh \
    examples/train/scenario/worldmodel/zelda/self_forcing_causal_i2v.yaml

bash examples/train/run.sh \
    examples/train/scenario/worldmodel/zelda/streaming_long_tuning_causal_i2v.yaml
```

Override any field on the command line:

```bash
bash examples/train/run.sh \
    examples/train/scenario/worldmodel/solaris/dfsft_causal_i2v.yaml \
    --training.distributed.num_gpus 8 \
    --training.optimizer.learning_rate 1e-5
```
