# MPS (Apple Silicon)

Instructions to install FastVideo for Apple Silicon.

## Requirements

- **OS: MacOS**
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

#### With uv (recommended)

```bash
uv pip install fastvideo
```

#### With Conda environment (alternative)

```bash
pip install fastvideo
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

## Development Environment Setup

If you're planning to contribute to FastVideo please see the following page:
[Contributor Guide](../../contributing/overview.md)

## Hardware Requirements

### For Basic Inference

- Mac M1, M2, M3, or M4 (at least 32 GB RAM is preferable for high quality video generation)

## Troubleshooting

If you encounter any issues during installation, please open an issue on our [GitHub repository](https://github.com/hao-ai-lab/FastVideo).

You can also join our [Slack community](https://join.slack.com/t/fastvideo/shared_invite/zt-38u6p1jqe-yDI1QJOCEnbtkLoaI5bjZQ) for additional support.
