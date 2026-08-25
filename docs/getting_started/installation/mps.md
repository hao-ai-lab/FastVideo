# MPS (Apple Silicon)

Install FastVideo on Apple Silicon and run FastMetal-QAD.

Apple Silicon uses the MLX runtime and the FastMetal-QAD INT8 checkpoints.
See the [FastMetal-QAD blog](https://haoailab.com/blogs/fastmetal/) and the
[FastMetal collection](https://huggingface.co/collections/FastVideo/fastmetal).

## Requirements

- **OS: macOS 14 or newer**
- **Python: 3.12.4**

## Set up using Python

### Create a new Python environment

#### uv
Recommended default: use [uv](https://docs.astral.sh/uv/) for faster and more stable environment setup.

Please follow the [documentation](https://docs.astral.sh/uv/#getting-started) to install `uv`. After installing `uv`, create a new environment using:

```console
# (Recommended) Create a new uv environment. Use `--seed` to install `pip` and `setuptools`.
uv venv --python 3.12 --seed
source .venv/bin/activate
```

#### Conda (alternative)

You can also create a Python environment using [Conda](https://docs.conda.io/projects/conda/en/stable/user-guide/getting-started.html).

##### 1. Install Miniconda (if not already installed)

```bash
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-MacOSX-arm64.sh
bash Miniconda3-latest-MacOSX-arm64.sh
source ~/.zshrc
```

##### 2. Create and activate a Conda environment for FastVideo

```bash
conda create -n fastvideo python=3.12.4 -y
conda activate fastvideo
```

### Dependencies

```
brew install ffmpeg
```

### Installation

FastMetal's native Apple Silicon runtime requires the `mlx` extra.

#### With uv (recommended)

```bash
uv pip install "fastvideo[mlx]"
```

#### With Conda environment (alternative)

`uv` works inside an active conda env too, so prefer `uv pip` for the actual install:

```bash
uv pip install "fastvideo[mlx]"
```

### Installation from Source

#### 1. Clone the FastVideo repository

```bash
git clone https://github.com/hao-ai-lab/FastVideo.git && cd FastVideo
```

#### 2. Install FastVideo

Basic installation:

```bash
uv pip install -e ".[mlx]"
```

Alternative with Conda environment:

```bash
uv pip install -e ".[mlx]"
```

## Run FastMetal-QAD

Each release is self-contained. Download one checkpoint and point both
`--model-root` and `--mlx-checkpoint` at it (the example also auto-detects
`mlx_dit.json` under `--model-root`).

| Checkpoint | Script | Mac tier |
| --- | --- | --- |
| [`FastVideo/FastMetal-1.3B-QAD`](https://huggingface.co/FastVideo/FastMetal-1.3B-QAD) | `mlx_wan_prompt_to_video.py` | 16 GB+ |
| [`FastVideo/FastMetal-5B-QAD`](https://huggingface.co/FastVideo/FastMetal-5B-QAD) | `mlx_wan22_generate.py` | 16 GB+ |
| [`FastVideo/FastMetal-14B-QAD`](https://huggingface.co/FastVideo/FastMetal-14B-QAD) | `mlx_wan_prompt_to_video.py` | 36 GB+ |

```bash
hf download FastVideo/FastMetal-1.3B-QAD --local-dir ./FastMetal-1.3B-QAD

python examples/inference/basic/mlx_wan_prompt_to_video.py \
  --model-root ./FastMetal-1.3B-QAD \
  --mlx-checkpoint ./FastMetal-1.3B-QAD \
  --height 480 --width 832 --num-frames 81 \
  --prompt "A bird's-eye view of a misty forest valley at dawn."
```

14B uses the same script. Point both flags at `./FastMetal-14B-QAD`. That repo also ships an EMA variant: keep `--model-root` at the repo root and set `--mlx-checkpoint ./FastMetal-14B-QAD/ema`.

Wan2.2 5B uses a different latent layout, so it has its own entrypoint:

```bash
hf download FastVideo/FastMetal-5B-QAD --local-dir ./FastMetal-5B-QAD

python examples/inference/basic/mlx_wan22_generate.py \
  --mlx-checkpoint ./FastMetal-5B-QAD \
  --text-encoder-root ./FastMetal-5B-QAD \
  --vae-root ./FastMetal-5B-QAD/vae \
  --height 704 --width 1280 --num-frames 81 \
  --prompt "A cinematic portrait with soft neon lighting and smooth camera motion."
```

CUDA FastWan-QAD (`FastVideo/FastWan-QAD-1.3B`, `FastVideo/FastWan-QAD-FP8-1.3B`) is a separate NVIDIA release. The MLX examples look for FastMetal packed weights (`mlx_dit.json`).

`basic_mps.py` is a generic PyTorch MPS demo. For local video on Mac, use the FastMetal commands above.

## Development Environment Setup

If you're planning to contribute to FastVideo please see the following page:
[Contributor Guide](../../contributing/overview.md)

## Hardware Requirements

- **1.3B / 5B:** 16 GB unified memory and up (M1 and later)
- **14B:** 36 GB unified memory and up
- Fanless 13-inch MacBook Air can run 1.3B and 5B at the same resolutions

## Troubleshooting

If you encounter any issues during installation, please open an issue on our [GitHub repository](https://github.com/hao-ai-lab/FastVideo).

You can also join our [Slack community](https://join.slack.com/t/fastvideo/shared_invite/zt-3f4lao1uq-u~Ipx6Lt4J27AlD2y~IdLQ) for additional support.
