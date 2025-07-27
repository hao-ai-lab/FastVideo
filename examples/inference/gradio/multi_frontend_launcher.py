import argparse
import os
import subprocess
import sys
import time
import threading
import signal
import requests
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))


def check_frontend_health(frontend_url: str, max_retries: int = 30) -> bool:
    """Check if the frontend is healthy"""
    for i in range(max_retries):
        try:
            response = requests.get(frontend_url, timeout=5)
            if response.status_code == 200:
                print(f"✅ Frontend is healthy at {frontend_url}")
                return True
        except requests.exceptions.RequestException:
            pass
        
        if i < max_retries - 1:
            print(f"⏳ Waiting for frontend to start... ({i+1}/{max_retries})")
            time.sleep(2)
    
    print(f"❌ Frontend failed to start within {max_retries * 2} seconds")
    return False


def start_frontend_instance(args, instance_id: int, backend_url: str):
    """Start a single frontend instance"""
    frontend_script = Path(__file__).parent / "gradio_frontend.py"
    frontend_port = args.frontend_base_port + instance_id
    
    cmd = [
        sys.executable, str(frontend_script),
        "--backend_url", backend_url,
        "--t2v_model_path", args.t2v_model_path,
        "--i2v_model_path", args.i2v_model_path,
        "--host", args.frontend_host,
        "--port", str(frontend_port)
    ]
    
    print(f"🎨 Starting Frontend {instance_id + 1} on port {frontend_port}...")
    print(f"Command: {' '.join(cmd)}")
    
    # Start the frontend process
    frontend_process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        universal_newlines=True,
        bufsize=1
    )
    
    # Monitor frontend output
    def monitor_frontend():
        for line in frontend_process.stdout:
            print(f"[FRONTEND-{instance_id + 1}] {line.rstrip()}")
    
    monitor_thread = threading.Thread(target=monitor_frontend, daemon=True)
    monitor_thread.start()
    
    return frontend_process, frontend_port


def main():
    parser = argparse.ArgumentParser(description="FastVideo Multi-Frontend Launcher")
    
    # Model and output settings
    parser.add_argument("--t2v_model_path",
                        type=str,
                        default="FastVideo/FastWan2.1-T2V-1.3B-Diffusers",
                        help="Path to the T2V model")
    parser.add_argument("--i2v_model_path",
                        type=str,
                        default="Wan-AI/Wan2.1-I2V-14B-480P-Diffusers",
                        help="Path to the I2V model")
    
    # Frontend settings
    parser.add_argument("--frontend_host",
                        type=str,
                        default="0.0.0.0",
                        help="Frontend host to bind to")
    parser.add_argument("--frontend_base_port",
                        type=int,
                        default=7860,
                        help="Base port for frontend instances")
    parser.add_argument("--num_frontends",
                        type=int,
                        default=2,
                        help="Number of frontend instances to start")
    
    # Backend settings
    parser.add_argument("--backend_url",
                        type=str,
                        default="http://localhost:8000",
                        help="Backend URL for frontends to connect to")
    
    # Other settings
    parser.add_argument("--skip_health_check",
                        action="store_true",
                        help="Skip frontend health check")
    
    args = parser.parse_args()
    
    print("🎬 FastVideo Multi-Frontend Launcher")
    print("=" * 50)
    print(f"T2V Model: {args.t2v_model_path}")
    print(f"I2V Model: {args.i2v_model_path}")
    print(f"Backend URL: {args.backend_url}")
    print(f"Number of Frontends: {args.num_frontends}")
    print(f"Frontend Base Port: {args.frontend_base_port}")
    print("=" * 50)
    
    # Start multiple frontend instances
    frontend_processes = []
    frontend_urls = []
    
    for i in range(args.num_frontends):
        process, port = start_frontend_instance(args, i, args.backend_url)
        frontend_processes.append(process)
        frontend_urls.append(f"http://{args.frontend_host}:{port}")
    
    # Wait for frontends to be ready
    if not args.skip_health_check:
        print("\n⏳ Waiting for frontends to start...")
        for i, url in enumerate(frontend_urls):
            if not check_frontend_health(url):
                print(f"❌ Frontend {i + 1} failed to start. Terminating...")
                for process in frontend_processes:
                    process.terminate()
                sys.exit(1)
    
    print("\n🎉 All frontend instances are starting up!")
    for i, url in enumerate(frontend_urls):
        print(f"📺 Frontend {i + 1}: {url}")
    print("\nPress Ctrl+C to stop all frontend instances...")
    
    # Signal handler for graceful shutdown
    def signal_handler(signum, frame):
        print("\n🛑 Shutting down frontend instances...")
        for process in frontend_processes:
            process.terminate()
        
        # Wait for processes to terminate
        try:
            for process in frontend_processes:
                process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            print("⚠️  Force killing processes...")
            for process in frontend_processes:
                process.kill()
        
        print("✅ Frontend instances stopped")
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Monitor processes
    try:
        while True:
            # Check if processes are still running
            for i, process in enumerate(frontend_processes):
                if process.poll() is not None:
                    print(f"❌ Frontend {i + 1} process died unexpectedly")
                    break
            
            time.sleep(1)
    
    except KeyboardInterrupt:
        signal_handler(signal.SIGINT, None)


if __name__ == "__main__":
    main()