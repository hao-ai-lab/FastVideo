# Training

## Quick Start

### Single-node

```bash
bash examples/train/run.sh <config.yaml> [--dotted.key value ...]
```

The script auto-detects available GPUs. Override with `NUM_GPUS`:

```bash
NUM_GPUS=4 bash examples/train/run.sh \
    examples/train/configs/fine_tuning/wan/t2v.yaml
```

### Multi-node (Slurm)

```bash
bash examples/train/run_slurm.sh <config.yaml> <num_nodes> [--dotted.key value ...]
```

```bash
bash examples/train/run_slurm.sh \
    examples/train/configs/fine_tuning/wan/t2v.yaml 4 \
    --training.distributed.hsdp_shard_dim 8 \
    --training.distributed.hsdp_replicate_dim 4
```

Slurm environment variables:

| Variable | Default | Description |
|---|---|---|
| `PARTITION` | `main` | Slurm partition |
| `NUM_GPUS` | `8` | GPUs per node |
| `CPUS_PER_TASK` | `128` | CPUs per task |
| `MEM` | `1440G` | Memory per node |
| `EXCLUDE` | `""` | Nodes to exclude |

### Overriding config values

Any config field can be overridden from the command line using dotted keys:

```bash
bash examples/train/run.sh examples/train/configs/fine_tuning/wan/t2v.yaml \
    --training.optimizer.learning_rate 1e-5 \
    --training.loop.max_train_steps 1000 \
    --training.checkpoint.output_dir outputs/my_experiment
```

### Resuming from a checkpoint

```bash
bash examples/train/run.sh examples/train/configs/fine_tuning/wan/t2v.yaml \
    --training.checkpoint.resume_from_checkpoint outputs/my_experiment/checkpoint-500
```

## GenRL HPSv3 + VideoAlign

The GenRL reward runtime is vendored as normal FastVideo package files under
`fastvideo/train/methods/rl/reward/`; it is not a git submodule. The prompt
JSONL files and VideoReward checkpoint are runtime assets and should be
prepared before launch:

```bash
uv pip install -e .
uv pip install -r examples/train/requirements-genrl.txt
python examples/train/prepare_genrl_assets.py
```

Before launching a long run, verify that both reward models load and can score
a dummy video:

```bash
VIDEOALIGN_CHECKPOINT_PATH=.cache/VideoReward \
FORCE_QWENVL_VIDEO_READER=opencv \
python examples/train/prepare_genrl_assets.py --check-rewards
```

The helper writes:

```bash
PROMPT_DATASET_PATH=.cache/genrl_filtered_prompts
VIDEOALIGN_CHECKPOINT_PATH=.cache/VideoReward
```

Launch the 4 GPU reproduction run:

```bash
WANDB_MODE=online \
WANDB_ENTITY=<your-wandb-entity> \
NUM_GPUS=4 \
VIDEOALIGN_CHECKPOINT_PATH=.cache/VideoReward \
FORCE_QWENVL_VIDEO_READER=opencv \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
bash examples/train/run.sh \
    examples/train/configs/genrl_wan2.1_t2v_1.3B_hpsv3_videoalign.yaml \
    --training.checkpoint.output_dir outputs/genrl_longcat
```

System packages used for the reproduced Modal run were `ffmpeg`, `libgl1`,
`libglib2.0-0`, `build-essential`, `ninja-build`, `cmake`, and `git-lfs`.
On CUDA 12.8 / Python 3.12, the Modal environment also used the PyTorch cu128
wheels and the `flash_attn-2.8.3+cu128torch2.10` prebuilt wheel. If
FlashAttention-2 is unavailable, the VideoAlign wrapper falls back to SDPA.

## W&B Logging

Training metrics and validation videos are logged to
[Weights & Biases](https://wandb.ai). Set your API key before launching:

```bash
export WANDB_API_KEY=your_key_here
```

To disable W&B (e.g. for local debugging):

```bash
export WANDB_MODE=offline
```

The project name and run name are set in the config under `training.tracker`:

```yaml
training:
  tracker:
    project_name: my_project
    run_name: my_run
```

## Directory Layout

```
examples/train/
├── configs/      # Single-step training configs (by method and model)
├── scenario/     # Multi-step end-to-end training pipelines
├── run.sh        # Single-node launcher
└── run_slurm.sh  # Multi-node Slurm launcher
```

See `configs/README.md` and `scenario/README.md` for details.
