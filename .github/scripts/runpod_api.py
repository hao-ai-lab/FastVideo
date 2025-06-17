import argparse
import json
import os
import subprocess
import sys
import time

import requests


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Run tests on RunPod GPU')
    parser.add_argument('--gpu-type', type=str, help='GPU type to use')
    parser.add_argument('--gpu-count',
                        type=int,
                        help='Number of GPUs to use',
                        default=1)
    parser.add_argument('--test-command', type=str, help='Test command to run')
    parser.add_argument('--disk-size',
                        type=int,
                        default=20,
                        help='Container disk size in GB (default: 20)')
    parser.add_argument('--volume-size',
                        type=int,
                        default=20,
                        help='Persistent volume size in GB (default: 20)')
    parser.add_argument(
        '--image',
        type=str,
        required=True,
        help='Docker image to use')
    return parser.parse_args()


args = parse_arguments()
API_KEY = os.environ['RUNPOD_API_KEY']
RUN_ID = os.environ['GITHUB_RUN_ID']
JOB_ID = os.environ['JOB_ID']
PODS_API = "https://rest.runpod.io/v1/pods"
HEADERS = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {API_KEY}"
}


def create_pod():
    """Create a RunPod instance"""
    # Ensure image name is lowercase (Docker requirement)
    image_name = args.image.lower()
    print(f"Using specified image: {image_name}")

    docker_start_cmd = [
        "bash", 
        "-c", 
        "apt update;DEBIAN_FRONTEND=noninteractive apt-get install openssh-server -y;mkdir -p ~/.ssh;cd $_;chmod 700 ~/.ssh;echo \"$PUBLIC_KEY\" >> authorized_keys;chmod 700 authorized_keys;service ssh start;sleep infinity"
    ]
    
    print(f"Creating RunPod instance with GPU: {args.gpu_type}...")
    payload = {
        "name": f"fastvideo-{JOB_ID}-{RUN_ID}",
        "containerDiskInGb": args.disk_size,
        "volumeInGb": args.volume_size,
        "gpuTypeIds": [args.gpu_type],
        "gpuCount": args.gpu_count,
        "imageName": image_name,
        "allowedCudaVersions": ["12.4"],
        "dockerStartCmd": docker_start_cmd
    }

    response = requests.post(PODS_API, headers=HEADERS, json=payload)
    response_data = response.json()
    print(f"Response: {json.dumps(response_data, indent=2)}")

    return response_data["id"]


def wait_for_pod(pod_id):
    """Wait for pod to be in RUNNING state and fully ready with SSH access"""
    print("Waiting for RunPod to be ready...")

    # First wait for RUNNING status
    max_attempts = 10
    attempts = 0
    while attempts < max_attempts:
        response = requests.get(f"{PODS_API}/{pod_id}", headers=HEADERS)
        pod_data = response.json()
        status = pod_data["desiredStatus"]

        if status == "RUNNING":
            print("RunPod is running! Now waiting for ports to be assigned...")
            break

        print(
            f"Current status: {status}, waiting... (attempt {attempts+1}/{max_attempts})"
        )
        time.sleep(2)
        attempts += 1

    if attempts >= max_attempts:
        raise TimeoutError(
            "Timed out waiting for RunPod to reach RUNNING state")

    # Wait for ports to be assigned
    max_attempts = 50
    attempts = 0
    while attempts < max_attempts:
        response = requests.get(f"{PODS_API}/{pod_id}", headers=HEADERS)
        pod_data = response.json()
        port_mappings = pod_data.get("portMappings")

        if (port_mappings is not None and "22" in port_mappings
                and pod_data.get("publicIp", "") != ""):
            print("RunPod is ready with SSH access!")
            print(f"SSH IP: {pod_data['publicIp']}")
            print(f"SSH Port: {port_mappings['22']}")
            break

        print(
            f"Waiting for SSH port and public IP to be available... (attempt {attempts+1}/{max_attempts})"
        )
        time.sleep(20)
        attempts += 1

    if attempts >= max_attempts:
        raise TimeoutError("Timed out waiting for RunPod SSH access")


def execute_command(pod_id):
    """Execute command on the pod via SSH using system SSH client"""
    print(f"Running command: {args.test_command}")

    response = requests.get(f"{PODS_API}/{pod_id}", headers=HEADERS)
    pod_data = response.json()
    ssh_ip = pod_data["publicIp"]
    ssh_port = pod_data["portMappings"]["22"]

    # Copy the repository to the pod using scp
    repo_dir = os.path.abspath(os.getcwd())
    repo_name = os.path.basename(repo_dir)

    print(f"Copying repository from {repo_dir} to RunPod...")

    tar_command = [
        "tar", "-czf", "/tmp/repo.tar.gz", "-C",
        os.path.dirname(repo_dir), repo_name
    ]
    subprocess.run(tar_command, check=True)

    # Copy the tarball to the pod
    scp_command = [
        "scp", "-o", "StrictHostKeyChecking=no", "-o",
        "UserKnownHostsFile=/dev/null", "-o", "ServerAliveInterval=60", "-o",
        "ServerAliveCountMax=10", "-P",
        str(ssh_port), "/tmp/repo.tar.gz", f"root@{ssh_ip}:/tmp/"
    ]
    subprocess.run(scp_command, check=True)

    # For custom image, we can use the pre-configured environment
    setup_steps = [
        "tar -xzf /tmp/repo.tar.gz --no-same-owner -C /workspace/",
        f"cd /workspace/{repo_name}",
        "source $HOME/.local/bin/env && source /opt/venv/bin/activate",
        args.test_command
    ]
    
    remote_command = " && ".join(setup_steps)

    ssh_command = [
        "ssh", "-o", "StrictHostKeyChecking=no", "-o",
        "UserKnownHostsFile=/dev/null", "-o", "ServerAliveInterval=60", "-o",
        "ServerAliveCountMax=10", "-p",
        str(ssh_port), f"root@{ssh_ip}", remote_command
    ]

    print(f"Connecting to {ssh_ip}:{ssh_port}...")

    try:
        process = subprocess.Popen(ssh_command,
                                   stdout=subprocess.PIPE,
                                   stderr=subprocess.STDOUT,
                                   universal_newlines=True,
                                   bufsize=0)

        stdout_lines = []

        print("Command output:")

        for line in iter(process.stdout.readline, ''):
            print(line.strip())
            stdout_lines.append(line)

        process.wait()

        return_code = process.returncode
        success = return_code == 0

        stdout_str = "".join(stdout_lines)

        if success:
            print("Command executed successfully")
        else:
            print(f"Command failed with exit code {return_code}")

        result = {
            "success": success,
            "return_code": return_code,
            "stdout": stdout_str,
            "stderr": ""
        }
        return result

    except Exception as e:
        print(f"Error executing SSH command: {str(e)}")
        result = {"success": False, "error": str(e), "stdout": "", "stderr": ""}
        return result


def terminate_pod(pod_id):
    """Terminate the pod"""
    print("Terminating RunPod...")
    requests.delete(f"{PODS_API}/{pod_id}", headers=HEADERS)
    print(f"Terminated pod {pod_id}")


def main():
    pod_id = None
    try:
        pod_id = create_pod()
        wait_for_pod(pod_id)
        result = execute_command(pod_id)

        if result.get("error") is not None:
            print(f"Error executing command: {result['error']}")
            sys.exit(1)

        if not result.get("success", False):
            print(
                "Tests failed - check the output above for details on which tests failed"
            )
            sys.exit(1)

    finally:
        if pod_id:
            terminate_pod(pod_id)


if __name__ == "__main__":
    main()
