FROM rocm/pytorch:rocm7.1_ubuntu22.04_py3.10_pytorch_release_2.9.1

ENV DEBIAN_FRONTEND=noninteractive

SHELL ["/bin/bash", "-c"]

WORKDIR /FastVideo

RUN apt-get update && apt-get install -y --no-install-recommends \
    wget \
    git \
    ca-certificates \
    openssh-server \
    zsh \
    vim \
    curl \
    gcc-11 \
    g++-11 \
    clang-11 \
    && rm -rf /var/lib/apt/lists/*

# Set up C++20 compilers for ThunderKittens
RUN update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-11 100 --slave /usr/bin/g++ g++ /usr/bin/g++-11

# Install uv and source its environment
RUN curl -LsSf https://astral.sh/uv/install.sh | sh && \
    echo 'source $HOME/.local/bin/env' >> /root/.bashrc

# Copy just the pyproject.toml first to leverage Docker cache
COPY pyproject_other.toml ./pyproject.toml

# Create a dummy README to satisfy the installation
RUN echo "# Placeholder" > README.md

# Create and activate virtual environment with specific Python version and seed
RUN source $HOME/.local/bin/env && \
    uv venv --python 3.10 --seed /opt/venv && \
    source /opt/venv/bin/activate && \
    uv pip install --no-cache-dir --upgrade pip 

COPY . .

# Install dependencies using uv and set up shell configuration
RUN source $HOME/.local/bin/env && \
    source /opt/venv/bin/activate && \
    uv pip install --no-cache-dir -e .[rocm] && \
    git config --unset-all http.https://github.com/.extraheader || true && \
    echo 'source /opt/venv/bin/activate' >> /root/.bashrc && \
    echo 'if [ -n "$ZSH_VERSION" ] && [ -f ~/.zshrc ]; then . ~/.zshrc; elif [ -f ~/.bashrc ]; then . ~/.bashrc; fi' > /root/.profile

# Install FastVideo Unified Kernel
RUN source $HOME/.local/bin/env && \
    source /opt/venv/bin/activate && \
    cd csrc/fastvideo_kernel && \
    git submodule update --init --recursive && \
    SKIP_SM90_EXT=1 pip install --no-build-isolation -e .

EXPOSE 22
