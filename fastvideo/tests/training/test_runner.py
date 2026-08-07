import subprocess
import sys

def test_runner_help():
    # Run the runner with --help to ensure it parses arguments correctly
    # and doesn't have any syntax errors or missing imports.
    result = subprocess.run(
        [sys.executable, "-m", "fastvideo.training.runner", "--help"],
        capture_output=True,
        text=True
    )
    
    # Check if the process completed successfully
    assert result.returncode == 0, f"Runner failed with exit code {result.returncode}.\nStderr: {result.stderr}"
    
    # Check if help output is printed
    assert "--pipeline_class" in result.stdout
    assert "--pipeline_module" in result.stdout
