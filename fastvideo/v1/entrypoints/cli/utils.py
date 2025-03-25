# SPDX-License-Identifier: Apache-2.0

import os
import sys
import subprocess
from fastvideo.v1.logger import init_logger

logger = init_logger(__name__)

def launch_distributed(num_gpus=None, args=None, master_port=None):
    """
    Launch a distributed job with the given arguments
    
    Args:
        num_gpus: Number of GPUs to use
        args: Arguments to pass to v1_fastvideo_inference.py (defaults to sys.argv[1:])
        master_port: Port for the master process (default: random)
    """

    current_env = os.environ.copy()
    python_executable = sys.executable
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../.."))
    main_script = os.path.join(project_root, "fastvideo/v1/sample/v1_fastvideo_inference.py")

    cmd = [
        python_executable, 
        "-m", "torch.distributed.run",
        f"--nproc_per_node={num_gpus}"
    ]

    if master_port is not None:
        cmd.append(f"--master_port={master_port}")

    cmd.append(main_script)
    cmd.extend(args)

    logger.info(f"Running inference with {num_gpus} GPU(s)")
    logger.info(f"Launching command: {' '.join(cmd)}")

    process = subprocess.Popen(
        cmd,
        env=current_env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        universal_newlines=True,
        bufsize=1
    )

    for line in iter(process.stdout.readline, ''):
        print(line.strip())

    return process.wait()