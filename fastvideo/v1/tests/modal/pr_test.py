import modal

app = modal.App()

import os

image_version = os.getenv("IMAGE_VERSION", "latest")
image_tag = f"ghcr.io/hao-ai-lab/fastvideo/fastvideo-dev:{image_version}"
print(f"Using image: {image_tag}")

image = (
    modal.Image.from_registry(image_tag, add_python="3.12")
    .apt_install("cmake", "pkg-config", "build-essential", "curl", "libssl-dev")
    .run_commands("curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --default-toolchain stable")
    .run_commands("echo 'source ~/.cargo/env' >> ~/.bashrc")
    .env({"PATH": "/root/.cargo/bin:$PATH"})
    .run_commands("/bin/bash -c 'source $HOME/.local/bin/env && source /opt/venv/bin/activate && cd /FastVideo && uv pip install -e .[test]'")
)

@app.function(gpu="L40S:1", image=image, timeout=1800)
def run_encoder_tests():
    """Run encoder tests on L40S GPU"""
    import subprocess
    import sys
    import os
    
    os.chdir("/FastVideo")
    
    command = """
    source /opt/venv/bin/activate && 
    pytest ./fastvideo/v1/tests/encoders -s
    """
    
    result = subprocess.run([
        "/bin/bash", "-c", command
    ], stdout=sys.stdout, stderr=sys.stderr, check=False)
    
    sys.exit(result.returncode)

@app.function(gpu="L40S:1", image=image, timeout=1800)
def run_vae_tests():
    """Run VAE tests on L40S GPU"""
    import subprocess
    import sys
    import os
    
    os.chdir("/FastVideo")
    
    command = """
    source /opt/venv/bin/activate && 
    pytest ./fastvideo/v1/tests/vaes -s
    """
    
    result = subprocess.run([
        "/bin/bash", "-c", command
    ], stdout=sys.stdout, stderr=sys.stderr, check=False)
    
    sys.exit(result.returncode)

@app.function(gpu="L40S:1", image=image, timeout=1800)
def run_transformer_tests():
    """Run transformer tests on L40S GPU"""
    import subprocess
    import sys
    import os
    
    os.chdir("/FastVideo")
    
    command = """
    source /opt/venv/bin/activate && 
    pytest ./fastvideo/v1/tests/transformers -s
    """
    
    result = subprocess.run([
        "/bin/bash", "-c", command
    ], stdout=sys.stdout, stderr=sys.stderr, check=False)
    
    sys.exit(result.returncode)

@app.function(gpu="L40S:2", image=image, timeout=3600)
def run_ssim_tests():
    """Run SSIM tests on 2x L40S GPUs"""
    import subprocess
    import sys
    import os
    
    os.chdir("/FastVideo")
    
    command = """
    source /opt/venv/bin/activate && 
    pytest ./fastvideo/v1/tests/ssim -vs
    """
    
    result = subprocess.run([
        "/bin/bash", "-c", command
    ], stdout=sys.stdout, stderr=sys.stderr, check=False)
    
    sys.exit(result.returncode)
