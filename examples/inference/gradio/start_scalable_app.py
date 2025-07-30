"""
Startup script for FastVideo scalable architecture:
ngrok -> nginx reverse proxy -> frontend1/frontend2 -> backend1×8/backend2×8

This script starts:
1. Multiple backend instances (8 GPU replicas each)
2. Multiple frontend instances (2 instances)
3. Nginx reverse proxy
4. Optional ngrok tunnel
"""

import argparse
import os
import subprocess
import sys
import time
import threading
import signal
import requests
import json
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))


def check_service_health(url: str, max_retries: int = 50) -> bool:
    """Check if a service is healthy"""
    for i in range(max_retries):
        try:
            response = requests.get(url, timeout=5)
            if response.status_code == 200:
                print(f"✅ Service is healthy at {url}")
                return True
        except requests.exceptions.RequestException:
            pass
        
        if i < max_retries - 1:
            print(f"⏳ Waiting for service to start... ({i+1}/{max_retries})")
            time.sleep(2)
    
    print(f"❌ Service failed to start within {max_retries * 2} seconds")
    return False


def start_backend_cluster(args, cluster_id: int):
    """Start one backend cluster (Ray-Serve application)."""
    backend_script = Path(__file__).parent / "ray_serve_backend_scalable.py"

    # All Ray Serve apps share the same HTTP server (default 8000).
    # We still forward the port flag for completeness, but keep it
    # identical for every cluster.
    base_port = args.backend_base_port
    
    cmd = [
        sys.executable, str(backend_script),
        "--t2v_model_path", args.t2v_model_path,
        "--i2v_model_path", args.i2v_model_path,
        "--output_path", args.output_path,
        "--host", args.backend_host,
        "--port", str(base_port),
        "--num_gpus", str(args.num_gpus_per_cluster),
        "--cluster_id", str(cluster_id),
    ]
    
    print(f"🚀 Starting Backend Cluster {cluster_id + 1} (HTTP port {base_port})...")
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
            print(f"[BACKEND-{cluster_id + 1}] {line.rstrip()}")
    
    monitor_thread = threading.Thread(target=monitor_backend, daemon=True)
    monitor_thread.start()
    
    return backend_process, base_port


def start_frontend_instance(args, instance_id: int, backend_url: str):
    """Start a single frontend instance"""
    frontend_script = Path(__file__).parent / "gradio_frontend.py"
    frontend_port = args.frontend_base_port + instance_id
    
    # Update backend URL to include cluster-specific path
    cluster_id = instance_id % args.num_backend_clusters
    backend_url_with_cluster = f"{backend_url}/cluster_{cluster_id}"
    
    cmd = [
        sys.executable, str(frontend_script),
        "--backend_url", backend_url_with_cluster,
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


def start_nginx(args):
    """Start nginx reverse proxy"""
    nginx_conf = Path(__file__).parent / "nginx.conf"
    
    # Update nginx configuration with actual ports
    update_nginx_config(args)
    
    cmd = [
        "nginx",
        "-c", str(nginx_conf),
        "-g", "daemon off;"
    ]
    
    print(f"🌐 Starting Nginx reverse proxy...")
    print(f"Command: {' '.join(cmd)}")
    
    # Start nginx process
    nginx_process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        universal_newlines=True,
        bufsize=1
    )
    
    # Monitor nginx output
    def monitor_nginx():
        for line in nginx_process.stdout:
            print(f"[NGINX] {line.rstrip()}")
    
    monitor_thread = threading.Thread(target=monitor_nginx, daemon=True)
    monitor_thread.start()
    
    return nginx_process


def update_nginx_config(args):
    """Rewrite nginx.conf with the correct ports – NO “/cluster_X” in upstreams."""
    nginx_conf        = Path(__file__).parent / "nginx.conf"
    nginx_conf_backup = Path(__file__).parent / "nginx.conf.backup"

    if not nginx_conf_backup.exists():
        nginx_conf_backup.write_text(nginx_conf.read_text())

    config_content = nginx_conf_backup.read_text()

    # ── 1. front-end pool ───────────────────────────────────────────────
    frontend_servers = "\n        ".join(
        f"server 127.0.0.1:{args.frontend_base_port + i} "
        f"weight=1 max_fails=3 fail_timeout=30s;"
        for i in range(args.num_frontends)
    )
    config_content = config_content.replace(
        "# Upstream for frontend load balancing",
        f"# Upstream for frontend load balancing\n    upstream frontend_servers {{\n"
        f"        # Round-robin load balancing between frontends\n        {frontend_servers}"
    )

    # Shared Ray-Serve HTTP port
    backend_port = args.backend_base_port           # default 8000
    backend_line = (f"server 127.0.0.1:{backend_port} "
                    f"weight=1 max_fails=3 fail_timeout=30s;")

    # ── 2. backend-1 pool ───────────────────────────────────────────────
    config_content = config_content.replace(
        "# Upstream for backend1 load balancing",
        f"# Upstream for backend1 load balancing\n    upstream backend1_servers {{\n"
        f"        {backend_line}"
    )

    # ── 3. backend-2 pool ───────────────────────────────────────────────
    config_content = config_content.replace(
        "# Upstream for backend2 load balancing",
        f"# Upstream for backend2 load balancing\n    upstream backend2_servers {{\n"
        f"        {backend_line}"
    )

    # ── 4. strip any stray “/cluster_X” fragments ───────────────────────
    config_content = config_content.replace("/cluster_0", "").replace("/cluster_1", "")

    # ── 5. use user-writable log directory ---------------------------------
    log_dir = Path(args.output_path).resolve()
    config_content = config_content.replace(
        "access_log /var/log/nginx/access.log;",
        f"access_log {log_dir}/nginx_access.log;")
    config_content = config_content.replace(
        "error_log /var/log/nginx/error.log;",
        f"error_log  {log_dir}/nginx_error.log;")

    nginx_conf.write_text(config_content)
    print("✅ nginx.conf updated (no path suffixes & custom log paths)")


def start_ngrok(args):
    """Start ngrok tunnel"""
    if not args.use_ngrok:
        return None
    
    cmd = [
        "ngrok",
        "http",
        str(args.nginx_port),
        "--log=stdout"
    ]
    
    print(f"🌍 Starting ngrok tunnel to port {args.nginx_port}...")
    print(f"Command: {' '.join(cmd)}")
    
    # Start ngrok process
    ngrok_process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        universal_newlines=True,
        bufsize=1
    )
    
    # Monitor ngrok output
    def monitor_ngrok():
        for line in ngrok_process.stdout:
            print(f"[NGROK] {line.rstrip()}")
    
    monitor_thread = threading.Thread(target=monitor_ngrok, daemon=True)
    monitor_thread.start()
    
    return ngrok_process


def main():
    parser = argparse.ArgumentParser(description="FastVideo Scalable Architecture Launcher")
    
    # Model and output settings
    parser.add_argument("--t2v_model_path",
                        type=str,
                        default="FastVideo/FastWan2.1-T2V-1.3B-Diffusers",
                        help="Path to the T2V model")
    parser.add_argument("--i2v_model_path",
                        type=str,
                        default="Wan-AI/Wan2.1-I2V-14B-480P-Diffusers",
                        help="Path to the I2V model")
    parser.add_argument("--output_path",
                        type=str,
                        default="outputs",
                        help="Path to save generated videos")
    
    # Backend settings
    parser.add_argument("--backend_host",
                        type=str,
                        default="0.0.0.0",
                        help="Backend host to bind to")
    parser.add_argument("--backend_base_port",
                        type=int,
                        default=8000,
                        help="Base port for backend clusters")
    parser.add_argument("--num_backend_clusters",
                        type=int,
                        default=2,
                        help="Number of backend clusters")
    parser.add_argument("--num_gpus_per_cluster",
                        type=int,
                        default=3,  # Changed from 8 to 3 (3+3=6 GPUs total, leaving 1 GPU buffer)
                        help="Number of GPUs per backend cluster")
    
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
                        help="Number of frontend instances")
    
    # Nginx settings
    parser.add_argument("--nginx_port",
                        type=int,
                        default=80,
                        help="Port for nginx reverse proxy")
    
    # Ngrok settings
    parser.add_argument("--use_ngrok",
                        action="store_true",
                        help="Start ngrok tunnel")
    
    # Other settings
    parser.add_argument("--skip_health_check",
                        action="store_true",
                        help="Skip health checks")
    
    args = parser.parse_args()
    
    # Ensure output directory exists
    os.makedirs(args.output_path, exist_ok=True)
    
    print(" FastVideo Scalable Architecture")
    print("=" * 60)
    print(f"Architecture: ngrok -> nginx -> frontend1/frontend2 -> backend1×{args.num_gpus_per_cluster}/backend2×{args.num_gpus_per_cluster}")
    print(f"T2V Model: {args.t2v_model_path}")
    print(f"I2V Model: {args.i2v_model_path}")
    print(f"Output: {args.output_path}")
    print(f"Backend Clusters: {args.num_backend_clusters}")
    print(f"GPUs per Cluster: {args.num_gpus_per_cluster}")
    print(f"Total GPUs needed: {args.num_backend_clusters * args.num_gpus_per_cluster}")
    print(f"Frontend Instances: {args.num_frontends}")
    print(f"Nginx Port: {args.nginx_port}")
    print(f"Use Ngrok: {args.use_ngrok}")
    print("=" * 60)
    
    # Start backend clusters
    backend_processes = []
    backend_urls = []
    
    for i in range(args.num_backend_clusters):
        process, _ = start_backend_cluster(args, i)
        backend_processes.append(process)
        backend_urls.append(f"http://{args.backend_host}:{args.backend_base_port}")
    
    # Wait for backends to be ready
    if not args.skip_health_check:
        print("\n⏳ Waiting for backend clusters to start...")
        for i, url in enumerate(backend_urls):
            if not check_service_health(f"{url}/cluster_{i}/health"):
                print(f"❌ Backend cluster {i + 1} failed to start. Terminating...")
                for process in backend_processes:
                    process.terminate()
                sys.exit(1)
    
    # Start frontend instances
    frontend_processes = []
    frontend_urls = []
    
    for i in range(args.num_frontends):
        # Each frontend connects to a different backend cluster
        backend_url = backend_urls[i % len(backend_urls)]
        process, port = start_frontend_instance(args, i, backend_url)
        frontend_processes.append(process)
        frontend_urls.append(f"http://{args.frontend_host}:{port}")
    
    # Wait for frontends to be ready
    if not args.skip_health_check:
        print("\n⏳ Waiting for frontend instances to start...")
        for i, url in enumerate(frontend_urls):
            if not check_service_health(url):
                print(f"❌ Frontend {i + 1} failed to start. Terminating...")
                for process in backend_processes + frontend_processes:
                    process.terminate()
                sys.exit(1)
    
    # Start nginx reverse proxy
    nginx_process = start_nginx(args)
    
    # Wait for nginx to be ready
    if not args.skip_health_check:
        print("\n⏳ Waiting for nginx to start...")
        if not check_service_health(f"http://localhost:{args.nginx_port}/health"):
            print("❌ Nginx failed to start. Terminating...")
            for process in backend_processes + frontend_processes + [nginx_process]:
                process.terminate()
            sys.exit(1)
    
    # Start ngrok tunnel (optional)
    ngrok_process = start_ngrok(args)
    
    print("\n🎉 All services are starting up!")
    print(f"🌐 Nginx reverse proxy: http://localhost:{args.nginx_port}")
    for i, url in enumerate(frontend_urls):
        print(f"📺 Frontend {i + 1}: {url}")
    for i, url in enumerate(backend_urls):
        print(f" Backend Cluster {i + 1}: {url}")
    if args.use_ngrok:
        print("🌍 Ngrok tunnel is starting...")
    print("\nPress Ctrl+C to stop all services...")
    
    # Signal handler for graceful shutdown
    def signal_handler(signum, frame):
        print("\n🛑 Shutting down all services...")
        all_processes = backend_processes + frontend_processes + [nginx_process]
        if ngrok_process:
            all_processes.append(ngrok_process)
        
        for process in all_processes:
            if process:
                process.terminate()
        
        # Wait for processes to terminate
        try:
            for process in all_processes:
                if process:
                    process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            print("⚠️  Force killing processes...")
            for process in all_processes:
                if process:
                    process.kill()
        
        print("✅ All services stopped")
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Monitor processes
    try:
        while True:
            # Check if processes are still running
            for i, process in enumerate(backend_processes):
                if process.poll() is not None:
                    print(f"❌ Backend cluster {i + 1} process died unexpectedly")
                    break
            
            for i, process in enumerate(frontend_processes):
                if process.poll() is not None:
                    print(f"❌ Frontend {i + 1} process died unexpectedly")
                    break
            
            if nginx_process and nginx_process.poll() is not None:
                print("❌ Nginx process died unexpectedly")
                break
            
            if ngrok_process and ngrok_process.poll() is not None:
                print("❌ Ngrok process died unexpectedly")
                break
            
            time.sleep(1)
    
    except KeyboardInterrupt:
        signal_handler(signal.SIGINT, None)


if __name__ == "__main__":
    main()