import os
import sys
import subprocess
import torch

def launch_distributed(num_gpus=None, args=None, master_port=None):
    """
    Launch a distributed job with the given arguments
    
    Args:
        num_gpus: Number of GPUs to use
        args: Arguments to pass to v1_fastvideo_inference.py (defaults to sys.argv[1:])
        master_port: Port for the master process (default: random)
    """
    if args is None:
        # Use all command-line arguments by default, but skip the command name
        args = sys.argv[2:]  # Skip 'run.py' and the command name ('serve')
    
    current_env = os.environ.copy()
    
    # Get the path to the Python executable and main script
    python_executable = sys.executable
    
    # Fix the path to the inference script
    # Instead of using a relative path from the current file, use an absolute path
    # based on the project structure
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../.."))
    main_script = os.path.join(project_root, "fastvideo/v1/sample/v1_fastvideo_inference.py")
    
    # If num_gpus is None, use all available GPUs
    if num_gpus is None:
        num_gpus = torch.cuda.device_count()
    
    # Construct the torchrun command
    cmd = [
        python_executable, 
        "-m", "torch.distributed.run",
        f"--nproc_per_node={num_gpus}"
    ]
    
    # Add master port if specified
    if master_port is not None:
        cmd.append(f"--master_port={master_port}")
    
    # Add the main script and arguments
    cmd.append(main_script)
    cmd.extend(args)
    
    print(f"Launching command: {' '.join(cmd)}")
    
    # Use subprocess.run instead of Popen for simpler handling
    # Set bufsize=1 to enable line buffering
    process = subprocess.Popen(
        cmd,
        env=current_env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        universal_newlines=True,
        bufsize=1  # Line buffering
    )
    
    # Print output in real-time
    for line in iter(process.stdout.readline, ''):
        print(line.strip())
    
    return process.wait()  # Wait for process to complete and return exit code