(developer-overview)=

# 🛠️ Contributing to FastVideo

Thank you for your interest in contributing to FastVideo. We want to make the process as smooth for you as possible and this is a guide to help get you started!

Our community is open to everyone and welcomes any contributions no matter how large or small.

# Developer Environment:
Do make sure you have CUDA 12.4 installed and supported. FastVideo currently only support Linux and CUDA GPUs, but we hope to support other platforms in the future.

We recommend using a fresh Python 3.10 Conda environment to develop FastVideo:

Install Miniconda:

```
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh
source ~/.bashrc
```

Create and activate a Conda environment for FastVideo:

```
conda create -n fastvideo python=3.12 -y
conda activate fastvideo
```

Install `uv` (optional, but recommended):

From instructions on [uv](https://astral.sh/uv/):

```
curl -LsSf https://astral.sh/uv/install.sh | sh
# or 
wget -qO- https://astral.sh/uv/install.sh | sh
```

Clone the FastVideo repository and go to the FastVideo directory:

```
git clone https://github.com/hao-ai-lab/FastVideo.git && cd FastVideo

```

Now you can install FastVideo and setup git hooks for running linting. By using `pre-commit`, the linters will run and have to pass before you'll be able to make a commit.

```bash
uv pip install -e .[dev]

# Can also install flash-attn (optional)
uv pip install flash-attn --no-build-isolation 

# Linting, formatting and static type checking
pre-commit install --hook-type pre-commit --hook-type commit-msg

# You can manually run pre-commit with
pre-commit run --all-files

# Unit tests
pytest tests/
```

If you are on a Hopper GPU, you should also install [FA3](https://github.com/Dao-AILab/flash-attention) for much better performance:

```
git clone https://github.com/Dao-AILab/flash-attention.git && cd flash-attention/hopper

# make sure you have ninja installed
uv pip install ninja

python setup.py install
```
