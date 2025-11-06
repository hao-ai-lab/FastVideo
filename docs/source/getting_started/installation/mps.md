# MPS (Apple Silicon)

Instructions to install FastVideo for Apple Silicon.

## Requirements

- **OS: MacOS**
- **Python: 3.12.4**

## Set up using Python

### Create a new Python environment

#### Conda

You can create a new python environment using [Conda](https://docs.conda.io/projects/conda/en/stable/user-guide/getting-started.html)

##### 1. Install Miniconda (if not already installed)

```bash
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-MacOSX-arm64.sh
bash Miniconda3-latest-MacOSX-arm64.sh
source ~/.zshrc
```

##### 2. Create and activate a Conda environment for FastVideo

```bash
# (Recommended) Create a new conda environment.
conda create -n fastvideo python=3.12.4 -y
conda activate fastvideo
```

:::{note}
[PyTorch has deprecated the conda release channel](https://github.com/pytorch/pytorch/issues/138506). If you use `conda`, please only use it to create Python environment rather than installing packages.
:::

#### uv

:::{tip}
We highly recommend using `uv` to install FastVideo. In our experience, `uv` speeds up installation by at least 3x.
Note that you can also use `uv` to install FastVideo in a Conda environment.
:::

Or you can create a new Python environment using [uv](https://docs.astral.sh/uv/), a very fast Python environment manager. Please follow the [documentation](https://docs.astral.sh/uv/#getting-started) to install `uv`. After installing `uv`, you can create a new Python environment using the following command:

```console
# (Recommended) Create a new uv environment. Use `--seed` to install `pip` and `setuptools` in the environment.
uv venv --python 3.12 --seed
source .venv/bin/activate
```

### Dependencies

```
brew install ffmpeg
```

### Installation

```bash
pip install fastvideo

# or if you are using uv
uv pip install fastvideo
```

### Installation from Source

#### 1. Clone the FastVideo repository

```bash
git clone https://github.com/hao-ai-lab/FastVideo.git && cd FastVideo
```

#### 2. Install FastVideo

Basic installation:

```bash
pip install -e .

# or if you are using uv
uv pip install -e .
```

## Development Environment Setup

If you're planning to contribute to FastVideo please see the following page:
[Contributor Guide](#developer-overview)

## Hardware Requirements

### For Basic Inference

- Mac M1, M2, M3, or M4 (at least 32 GB RAM is preferable for high quality video generation)

## Troubleshooting

If you encounter any issues during installation, please open an issue on our [GitHub repository](https://github.com/hao-ai-lab/FastVideo).

You can also join our [Slack community](https://join.slack.com/t/fastvideo/shared_invite/zt-38u6p1jqe-yDI1QJOCEnbtkLoaI5bjZQ) for additional support.
