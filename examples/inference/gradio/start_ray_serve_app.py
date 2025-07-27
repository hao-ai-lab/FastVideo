"""
Startup script for FastVideo with Ray Serve backend and Gradio frontend.
This script starts both the backend and frontend services.
"""

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


def check_backend_health(backend_url: str, max_retries: int = 100) -> bool:
    """Check if the backend is healthy"""
    health_url = f"{backend_url}/health"
    
    for i in range(max_retries):
        try:
            response = requests.get(health_url, timeout=5)
            if response.status_code == 200:
                print(f"✅ Backend is healthy at {backend_url}")
                return True
        except requests.exceptions.RequestException:
            pass
        
        if i < max_retries - 1:
            print(f"⏳ Waiting for backend to start... ({i+1}/{max_retries})")
            time.sleep(2)
    
    print(f"❌ Backend failed to start within {max_retries * 2} seconds")
    return False


def start_backend(args):
    """Start the Ray Serve backend"""
    backend_script = Path(__file__).parent / "ray_serve_backend.py"
    
    cmd = [
        sys.executable, str(backend_script),
        "--t2v_model_paths", args.t2v_model_paths,
        "--t2v_model_replicas", args.t2v_model_replicas,
        # "--i2v_model_path", args.i2v_model_path,  # I2V functionality commented out
        "--output_path", args.output_path,
        "--host", args.backend_host,
        "--port", str(args.backend_port)
    ]
    
    print(f"🚀 Starting Ray Serve backend...")
    print(f"Command: {' '.join(cmd)}")
    
    # Start the backend process
    backend_process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        universal_newlines=True,
        bufsize=1
    )
    
    # Monitor backend output
    def monitor_backend():
        for line in backend_process.stdout:
            print(f"[BACKEND] {line.rstrip()}")
    
    monitor_thread = threading.Thread(target=monitor_backend, daemon=True)
    monitor_thread.start()
    
    return backend_process


def start_frontend(args):
    """Start the Gradio frontend"""
    frontend_script = Path(__file__).parent / "gradio_frontend.py"
    backend_url = f"http://{args.backend_host}:{args.backend_port}"
    
    cmd = [
        sys.executable, str(frontend_script),
        "--backend_url", backend_url,
        "--t2v_model_paths", args.t2v_model_paths,
        # "--t2v_model_replicas", args.t2v_model_replicas,
        # "--i2v_model_path", args.i2v_model_path,  # I2V functionality commented out
        "--host", args.frontend_host,
        "--port", str(args.frontend_port)
    ]
    
    print(f"🎨 Starting Gradio frontend...")
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
            print(f"[FRONTEND] {line.rstrip()}")
    
    monitor_thread = threading.Thread(target=monitor_frontend, daemon=True)
    monitor_thread.start()
    
    return frontend_process


def main():
    parser = argparse.ArgumentParser(description="FastVideo Ray Serve App")
    
    # Model and output settings
    parser.add_argument("--t2v_model_paths",
                        type=str,
                        default="FastVideo/FastWan2.1-T2V-1.3B-Diffusers,FastVideo/FastWan2.1-T2V-14B-Diffusers",
                        help="Comma separated list of paths to the T2V model(s)")
    parser.add_argument("--t2v_model_replicas",
                        type=str,
                        default="4,4",
                        help="Comma separated list of number of replicas for the T2V model(s)")
    # parser.add_argument("--t2v_14b_model_path",
    #                     type=str,
    #                     default="FastVideo/FastWan2.1-T2V-14B-Diffusers",
    #                     help="Path to the T2V 14B model")
    # parser.add_argument("--i2v_model_path",  # I2V functionality commented out
    #                     type=str,
    #                     default="Wan-AI/Wan2.1-I2V-14B-480P-Diffusers",
    #                     help="Path to the I2V model")
    parser.add_argument("--output_path",
                        type=str,
                        default="outputs",
                        help="Path to save generated videos")
    
    # Backend settings
    parser.add_argument("--backend_host",
                        type=str,
                        default="0.0.0.0",
                        help="Backend host to bind to")
    parser.add_argument("--backend_port",
                        type=int,
                        default=8000,
                        help="Backend port to bind to")
    
    # Frontend settings
    parser.add_argument("--frontend_host",
                        type=str,
                        default="0.0.0.0",
                        help="Frontend host to bind to")
    parser.add_argument("--frontend_port",
                        type=int,
                        default=7861,
                        help="Frontend port to bind to")
    
    # Other settings
    parser.add_argument("--skip_backend_check",
                        action="store_true",
                        help="Skip backend health check")
    
    args = parser.parse_args()
    
    # Ensure output directory exists
    os.makedirs(args.output_path, exist_ok=True)
    
    print("🎬 FastVideo Ray Serve App")
    print("=" * 50)
    print(f"T2V Models: {args.t2v_model_paths}")
    print(f"T2V Model Replicas: {args.t2v_model_replicas}")
    # print(f"I2V Model: {args.i2v_model_path}")  # I2V functionality commented out
    print(f"Output: {args.output_path}")
    print(f"Backend: http://{args.backend_host}:{args.backend_port}")
    print(f"Frontend: http://{args.frontend_host}:{args.frontend_port}")
    print("=" * 50)
    
    # Start backend
    backend_process = start_backend(args)
    
    # Wait for backend to be ready
    backend_url = f"http://{args.backend_host}:{args.backend_port}"
    
    if not args.skip_backend_check:
        if not check_backend_health(backend_url):
            print("❌ Backend failed to start. Terminating...")
            backend_process.terminate()
            sys.exit(1)
    
    # Start frontend
    frontend_process = start_frontend(args)
    
    print("\n🎉 Both services are starting up!")
    print(f"📺 Frontend will be available at: http://{args.frontend_host}:{args.frontend_port}")
    print(f"🔧 Backend API will be available at: {backend_url}")
    print("\nPress Ctrl+C to stop both services...")
    # return
    
    # Signal handler for graceful shutdown
    def signal_handler(signum, frame):
        print("\n🛑 Shutting down services...")
        frontend_process.terminate()
        backend_process.terminate()
        
        # Wait for processes to terminate
        try:
            frontend_process.wait(timeout=5)
            backend_process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            print("⚠️  Force killing processes...")
            frontend_process.kill()
            backend_process.kill()
        
        print("✅ Services stopped")
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Monitor processes
    try:
        while True:
            # Check if processes are still running
            if frontend_process.poll() is not None:
                print("❌ Frontend process died unexpectedly")
                break
            
            if backend_process.poll() is not None:
                print("❌ Backend process died unexpectedly")
                break
            
            time.sleep(1)
    
    except KeyboardInterrupt:
        signal_handler(signal.SIGINT, None)


if __name__ == "__main__":
    main()