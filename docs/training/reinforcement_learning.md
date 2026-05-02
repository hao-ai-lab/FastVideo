# Reinforcement learning (GRPO)

Wan2.1 T2V 1.3B + GRPO + OCR reward; ported from [flow_grpo](https://github.com/yifan123/flow_grpo).

## Hardware

Use **8× NVIDIA H100 (80 GB)** if you can. Fewer or smaller GPUs may need you to shrink batch size, resolution, or frames in the run script.

## Weights & Biases

Wandb is used to log training metrics like reward and loss. Edit the run script and set your project and API key:

```bash
# In examples/training/rl/finetune_t2v_grpo_8gpu.sh
export WANDB_API_KEY="your-api-key"
export WANDB_PROJECT="your-project-name"
```

## Download dataset (OCR prompts)

The RL dataset is plain text: one prompt per line. Training expects `train.txt` and `test.txt` under a single directory (default: `data/ocr` at the FastVideo repo root).

Get the OCR split from flow_grpo:

```bash
cd /path/to/parent
git clone https://github.com/yifan123/flow_grpo.git
```

```bash
cd /path/to/FastVideo
mkdir -p data/ocr
cp ../flow_grpo/dataset/ocr/train.txt ../flow_grpo/dataset/ocr/test.txt data/ocr/
```

Adjust `../flow_grpo` if you cloned elsewhere.

## Run training

```bash
cd /path/to/FastVideo
bash examples/training/rl/finetune_t2v_grpo_8gpu.sh
```

## Key arguments in `examples/training/rl/finetune_t2v_grpo_8gpu.sh`

| Argument | Example in script | Role |
|----------|-------------------|------|
| `--rl_num_image_per_prompt` | `4` | GRPO group size *k*: rollouts per prompt. |
| `--train_batch_size` | `8` | Prompt batch size for the RL dataloader. |
| `--num_height` | `240` | Training height (pixels). |
| `--num_width` | `416` | Training width (pixels). |
| `--num_frames` | `33` | Frames per training clip. |
| `--num_latent_t` | `20` | Temporal latent length (lower uses less memory). |
| `--max_train_steps` | `3000` | Total training steps. |
| `--gradient_accumulation_steps` | `1` | Part of how steps per epoch are computed (see below). |
| `--sp_size` | `1` | Sequence-parallel size; same. |
| `--train_sp_batch_size` | `1` | SP micro-batch; same. |
| `--rl_num_batches_per_step` | `2` | How many sampling batches to run per logged training step (flow_grpo-style). |
| `--log_validation` | `True` | Enable validation runs. |
| `--validation_steps` | `30` | Run validation every N steps. |
| `--model_path` / `--pretrained_model_name_or_path` | `Wan-AI/Wan2.1-T2V-1.3B-Diffusers` | Base checkpoint. |
| `--reward-models` | `"{\"paddle_ocr\": 1.0}"` | OCR reward only in this setup. |

## What’s supported

- OCR reward + OCR text dataset (`train.txt` / `test.txt`) only.

**Give credit to [flow_grpo](https://github.com/yifan123/flow_grpo)** when you publish or share work that uses this RL setup; FastVideo’s GRPO path is ported from that project.
