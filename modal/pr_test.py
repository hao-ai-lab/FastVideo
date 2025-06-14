import modal

app = modal.App()

image = (
    modal.Image.from_registry("ghcr.io/hao-ai-lab/fastvideo/fastvideo-dev:py3.12-latest", add_python="3.12")
    .apt_install("cmake", "pkg-config", "build-essential", "curl", "libssl-dev")
    .run_commands("curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --default-toolchain stable")
    .run_commands("echo 'source ~/.cargo/env' >> ~/.bashrc")
    .env({"PATH": "/root/.cargo/bin:$PATH"})
    .pip_install("torch==2.6.0", "pytest")
    .run_commands("/bin/bash -c 'source /opt/conda/etc/profile.d/conda.sh && conda activate fastvideo-dev && cd /FastVideo && pip install -e .[test]'")
)

@app.function(gpu="L40S:1", image=image, timeout=1800)
def run_encoder_tests():
    """Run encoder tests on L40S GPU"""
    import subprocess
    import sys
    import os
    
    os.chdir("/FastVideo")
    
    command = """
    source /opt/conda/etc/profile.d/conda.sh && 
    conda activate fastvideo-dev && 
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
    source /opt/conda/etc/profile.d/conda.sh && 
    conda activate fastvideo-dev && 
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
    source /opt/conda/etc/profile.d/conda.sh && 
    conda activate fastvideo-dev && 
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
    source /opt/conda/etc/profile.d/conda.sh && 
    conda activate fastvideo-dev && 
    pytest ./fastvideo/v1/tests/ssim -vs
    """
    
    result = subprocess.run([
        "/bin/bash", "-c", command
    ], stdout=sys.stdout, stderr=sys.stderr, check=False)
    
    sys.exit(result.returncode)
