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

def run_test(pytest_command: str):
    """Helper function to run a test suite with custom pytest command"""
    import subprocess
    import sys
    import os
    
    os.chdir("/FastVideo")
    
    command = f"""
    source /opt/venv/bin/activate && 
    {pytest_command}
    """
    
    result = subprocess.run([
        "/bin/bash", "-c", command
    ], stdout=sys.stdout, stderr=sys.stderr, check=False)
    
    sys.exit(result.returncode)

def create_test_function(test_name: str, pytest_command: str, gpu_config: str, timeout: int, secrets: list = None):
    """Factory function to create modal test functions"""
    decorator_kwargs = {
        "gpu": gpu_config,
        "image": image,
        "timeout": timeout
    }

    if secrets:
        decorator_kwargs["secrets"] = secrets
    
    @app.function(**decorator_kwargs)
    def test_function():
        f"""Run {test_name} tests"""
        run_test(pytest_command)
    
    test_function.__name__ = f"run_{test_name}_tests"
    return test_function

run_encoder_tests = create_test_function(
    test_name="encoder",
    pytest_command="pytest ./fastvideo/v1/tests/encoders -vs",
    gpu_config="L40S:1",
    timeout=1800
)

run_vae_tests = create_test_function(
    test_name="vae",
    pytest_command="pytest ./fastvideo/v1/tests/vaes -vs",
    gpu_config="L40S:1",
    timeout=1800
)

run_transformer_tests = create_test_function(
    test_name="transformer",
    pytest_command="pytest ./fastvideo/v1/tests/transformers -vs",
    gpu_config="L40S:1",
    timeout=1800
)

run_ssim_tests = create_test_function(
    test_name="ssim",
    pytest_command="pytest ./fastvideo/v1/tests/ssim -vs",
    gpu_config="L40S:2",
    timeout=3600
)

run_training_tests = create_test_function(
    test_name="training",
    pytest_command="wandb login $WANDB_API_KEY && pytest ./fastvideo/v1/tests/training/Vanilla -srP",
    gpu_config="L40S:4",
    timeout=3600,
    secrets=[modal.Secret.from_dict({"WANDB_API_KEY": os.environ.get("WANDB_API_KEY", "")})]
)

run_training_tests_VSA = create_test_function(
    test_name="training_vsa",
    pytest_command="wandb login $WANDB_API_KEY && pytest ./fastvideo/v1/tests/training/VSA -srP",
    gpu_config="H100:1",
    timeout=3600,
    secrets=[modal.Secret.from_dict({"WANDB_API_KEY": os.environ.get("WANDB_API_KEY", "")})]
)

run_inference_tests_STA = create_test_function(
    test_name="inference_sta",
    pytest_command="pytest ./fastvideo/v1/tests/inference/STA -srP",
    gpu_config="H100:1",
    timeout=3600
)
