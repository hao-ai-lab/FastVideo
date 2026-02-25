# NVIDIA GPU

Instructions to install FastVideo for NVIDIA CUDA GPUs.

## Requirements

- **OS: Linux or Windows WSL**
- **Python: 3.10-3.12**
- **CUDA 12.8**
- **At least 1 NVIDIA GPU**

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
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh
source ~/.bashrc
```

##### 2. Create and activate a Conda environment for FastVideo

```bash
# Create and activate a Conda environment
conda create -n fastvideo python=3.12 -y
conda activate fastvideo
```

### Installation

#### With uv (recommended)

```bash
uv pip install fastvideo
```

Also optionally install FlashAttention:

```bash
uv pip install flash-attn --no-build-isolation -v
```

#### With Conda environment (alternative)

```bash
pip install fastvideo
```

Also optionally install FlashAttention:

```bash
pip install flash-attn --no-build-isolation -v
```

### Installation from Source

#### 1. Clone the FastVideo repository

```bash
git clone https://github.com/hao-ai-lab/FastVideo.git && cd FastVideo
```

#### 2. Install FastVideo

Basic installation:

```bash
uv pip install -e .
```

Alternative with Conda environment:

```bash
pip install -e .
```

### Optional Dependencies

#### Flash Attention

```bash
uv pip install flash-attn --no-build-isolation -v
```

Alternative with Conda environment:

```bash
pip install flash-attn --no-build-isolation -v
```

## Set up using Docker
We also have prebuilt docker images with FastVideo dependencies pre-installed:
[Docker Images](../../contributing/developer_env/docker.md)

## Development Environment Setup

If you're planning to contribute to FastVideo please see the following page:
[Contributor Guide](../../contributing/overview.md)

## Hardware Requirements

### For Basic Inference
- NVIDIA GPU with CUDA 12.8 support

### For Lora Finetuning
- 40GB GPU memory each for 2 GPUs with lora
- 30GB GPU memory each for 2 GPUs with CPU offload and lora

### For Full Finetuning/Distillation
- Multiple high-memory GPUs recommended (e.g., H100)

## Troubleshooting

If you encounter any issues during installation, please open an issue on our [GitHub repository](https://github.com/hao-ai-lab/FastVideo).

You can also join our [Slack community](https://join.slack.com/t/fastvideo/shared_invite/zt-38u6p1jqe-yDI1QJOCEnbtkLoaI5bjZQ) for additional support.
