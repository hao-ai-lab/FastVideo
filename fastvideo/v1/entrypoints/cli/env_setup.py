# SPDX-License-Identifier: Apache-2.0

import subprocess
import sys

from fastvideo.v1.logger import init_logger

logger = init_logger(__name__)


def install_pytorch():
    """
    Install PyTorch dependencies with CUDA support

    Installs torch 2.5.0 and torchvision with CUDA 12.4 support
    """
    logger.info("Installing PyTorch dependencies...")

    # Install PyTorch with CUDA
    torch_cmd = [
        sys.executable, "-m", "pip", "install",
        "torch==2.5.0", "torchvision",
        "--index-url", "https://download.pytorch.org/whl/cu124"
    ]

    logger.info("Installing PyTorch: %s", " ".join(torch_cmd))
    result = subprocess.run(torch_cmd,
                           stdout=subprocess.PIPE,
                           stderr=subprocess.STDOUT,
                           universal_newlines=True)

    if result.returncode == 0:
        logger.info("PyTorch installation complete")
        return True
    else:
        logger.error("PyTorch installation failed")
        logger.error(result.stdout)
        return False


def install_flash_attn():
    """
    Install Flash Attention dependencies

    Installs PyTorch with CUDA support, packaging, ninja and wheel,
    followed by flash-attn with the no-build-isolation flag
    """
    # First install PyTorch
    if not install_pytorch():
        return 1

    logger.info("Installing Flash Attention dependencies...")

    # Install prerequisites
    prereq_cmd = [
        sys.executable, "-m", "pip", "install", "packaging", "ninja", "wheel"
    ]

    logger.info("Installing prerequisites: %s", " ".join(prereq_cmd))
    subprocess.run(prereq_cmd, check=True)

    # Install flash-attn
    install_cmd = [
        sys.executable, "-m", "pip", "install",
        "flash-attn==2.7.0.post2", "--no-build-isolation"
    ]

    logger.info("Installing flash-attn: %s", " ".join(install_cmd))
    result = subprocess.run(install_cmd,
                           stdout=subprocess.PIPE,
                           stderr=subprocess.STDOUT,
                           universal_newlines=True)

    if result.returncode == 0:
        logger.info("Flash Attention installation complete")
        return 0
    else:
        logger.error("Flash Attention installation failed")
        logger.error(result.stdout)
        return 1


def install_st_attn():
    """
    Install STA (Structured Attention) dependencies

    Installs PyTorch with CUDA support followed by required
    packages for STA functionality
    """
    # First install PyTorch
    if not install_pytorch():
        return 1

    install_cmd = [
        sys.executable, "-m", "pip", "install",
        "st-attn==0.0.2", "--no-build-isolation"
    ]

    logger.info("Installing STA packages: %s", " ".join(install_cmd))
    result = subprocess.run(install_cmd,
                           stdout=subprocess.PIPE,
                           stderr=subprocess.STDOUT,
                           universal_newlines=True)

    if result.returncode == 0:
        logger.info("STA installation complete")
        return 0
    else:
        logger.error("STA installation failed")
        logger.error(result.stdout)
        return 1


if __name__ == "__main__":
    # Allow direct execution for testing
    install_flash_attn()
