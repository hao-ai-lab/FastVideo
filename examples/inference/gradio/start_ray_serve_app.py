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
from typing import Dict, Any, Optional


DEFAULT_BACKEND_HOST = "0.0.0.0"
DEFAULT_BACKEND_PORT = 8000
DEFAULT_FRONTEND_HOST = "0.0.0.0"
DEFAULT_FRONTEND_PORT = 7860
DEFAULT_OUTPUT_PATH = "outputs"
DEFAULT_T2V_MODELS = "FastVideo/FastWan2.1-T2V-1.3B-Diffusers,FastVideo/FastWan2.2-TI2V-5B-FullAttn-Diffusers"
DEFAULT_T2V_REPLICAS = "4,4"

HEALTH_CHECK_TIMEOUT = 5
HEALTH_CHECK_MAX_RETRIES = 100
HEALTH_CHECK_INTERVAL = 2
PROCESS_SHUTDOWN_TIMEOUT = 5
PROCESS_MONITOR_INTERVAL = 1

PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


class ServiceManager:
    
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.backend_process: Optional[subprocess.Popen] = None
        self.frontend_process: Optional[subprocess.Popen] = None
        self.backend_url = f"http://{args.backend_host}:{args.backend_port}"
    
    def check_backend_health(self, max_retries: int = HEALTH_CHECK_MAX_RETRIES) -> bool:
        health_url = f"{self.backend_url}/health"
        
        for attempt in range(max_retries):
            try:
                response = requests.get(health_url, timeout=HEALTH_CHECK_TIMEOUT)
                if response.status_code == 200:
                    print(f"✅ Backend is healthy at {self.backend_url}")
                    return True
            except requests.exceptions.RequestException:
                pass
            
            if attempt < max_retries - 1:
                print(f"⏳ Waiting for backend to start... ({attempt + 1}/{max_retries})")
                time.sleep(HEALTH_CHECK_INTERVAL)
        
        print(f"❌ Backend failed to start within {max_retries * HEALTH_CHECK_INTERVAL} seconds")
        return False
    
    def _create_monitor_thread(self, process: subprocess.Popen, service_name: str) -> threading.Thread:
        def monitor():
            if process.stdout:
                for line in process.stdout:
                    print(f"[{service_name}] {line.rstrip()}")
        
        thread = threading.Thread(target=monitor, daemon=True)
        thread.start()
        return thread
    
    def _start_service(self, script_name: str, args_dict: Dict[str, Any], service_name: str) -> subprocess.Popen:
        script_path = Path(__file__).parent / script_name
        
        cmd = [sys.executable, str(script_path)]
        for key, value in args_dict.items():
            cmd.extend([f"--{key}", str(value)])
        
        print(f"🚀 Starting {service_name}...")
        print(f"Command: {' '.join(cmd)}")
        
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True,
            bufsize=1
        )
        
        self._create_monitor_thread(process, service_name.upper())
        return process
    
    def start_backend(self) -> subprocess.Popen:
        backend_args = {
            "t2v_model_paths": self.args.t2v_model_paths,
            "t2v_model_replicas": self.args.t2v_model_replicas,
            "output_path": self.args.output_path,
            "host": self.args.backend_host,
            "port": self.args.backend_port
        }
        
        self.backend_process = self._start_service("ray_serve_backend.py", backend_args, "backend")
        return self.backend_process
    
    def start_frontend(self) -> subprocess.Popen:
        frontend_args = {
            "backend_url": self.backend_url,
            "t2v_model_paths": self.args.t2v_model_paths,
            "host": self.args.frontend_host,
            "port": self.args.frontend_port
        }
        
        self.frontend_process = self._start_service("gradio_frontend.py", frontend_args, "frontend")
        return self.frontend_process
    
    def shutdown_services(self) -> None:
        print("\n🛑 Shutting down services...")
        
        processes = []
        if self.frontend_process:
            self.frontend_process.terminate()
            processes.append(("frontend", self.frontend_process))
        
        if self.backend_process:
            self.backend_process.terminate()
            processes.append(("backend", self.backend_process))
        
        for name, process in processes:
            try:
                process.wait(timeout=PROCESS_SHUTDOWN_TIMEOUT)
                print(f"✅ {name.capitalize()} stopped gracefully")
            except subprocess.TimeoutExpired:
                print(f"⚠️  Force killing {name} process...")
                process.kill()
        
        print("✅ All services stopped")
    
    def monitor_processes(self) -> None:
        if not self.backend_process or not self.frontend_process:
            print("❌ Processes not properly initialized")
            return
        
        try:
            while True:
                if self.frontend_process.poll() is not None:
                    print("❌ Frontend process died unexpectedly")
                    break
                
                if self.backend_process.poll() is not None:
                    print("❌ Backend process died unexpectedly")
                    break
                
                time.sleep(PROCESS_MONITOR_INTERVAL)
        
        except KeyboardInterrupt:
            pass
        
        self.shutdown_services()


def setup_signal_handlers(service_manager: ServiceManager) -> None:
    def signal_handler(signum: int, frame: Any) -> None:
        service_manager.shutdown_services()
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)


def print_startup_info(args: argparse.Namespace) -> None:
    print("🎬 FastVideo Ray Serve App")
    print("=" * 50)
    print(f"T2V Models: {args.t2v_model_paths}")
    print(f"T2V Model Replicas: {args.t2v_model_replicas}")
    print(f"Output: {args.output_path}")
    print(f"Backend: http://{args.backend_host}:{args.backend_port}")
    print(f"Frontend: http://{args.frontend_host}:{args.frontend_port}")
    print("=" * 50)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="FastVideo Ray Serve App")
    
    parser.add_argument("--t2v_model_paths", 
                        type=str,
                        default=DEFAULT_T2V_MODELS,
                        help="Comma separated list of paths to the T2V model(s)")
    parser.add_argument("--t2v_model_replicas", 
                        type=str, 
                        default=DEFAULT_T2V_REPLICAS,
                        help="Comma separated list of number of replicas for the T2V model(s)")
    parser.add_argument("--output_path", 
                        type=str, 
                        default=DEFAULT_OUTPUT_PATH,
                        help="Path to save generated videos")
    
    parser.add_argument("--backend_host", 
                        type=str, 
                        default=DEFAULT_BACKEND_HOST,
                        help="Backend host to bind to")
    parser.add_argument("--backend_port", 
                        type=int, 
                        default=DEFAULT_BACKEND_PORT,
                        help="Backend port to bind to")
    
    parser.add_argument("--frontend_host", 
                        type=str, 
                        default=DEFAULT_FRONTEND_HOST,
                        help="Frontend host to bind to")
    parser.add_argument("--frontend_port", 
                        type=int, 
                        default=DEFAULT_FRONTEND_PORT,
                        help="Frontend port to bind to")
    
    parser.add_argument("--skip_backend_check", 
                        action="store_true",
                        help="Skip backend health check")
    
    return parser.parse_args()


def main() -> None:
    args = parse_arguments()
    
    os.makedirs(args.output_path, exist_ok=True)
    print_startup_info(args)
    
    service_manager = ServiceManager(args)
    setup_signal_handlers(service_manager)
    
    try:
        service_manager.start_backend()
        
        if not args.skip_backend_check:
            if not service_manager.check_backend_health():
                print("❌ Backend failed to start. Terminating...")
                service_manager.shutdown_services()
                sys.exit(1)
        
        service_manager.start_frontend()
        
        print("\n🎉 Both services are starting up!")
        print(f"📺 Frontend will be available at: http://{args.frontend_host}:{args.frontend_port}")
        print(f"🔧 Backend API will be available at: http://{args.backend_host}:{args.backend_port}")
        print("\nPress Ctrl+C to stop both services...")
        
        service_manager.monitor_processes()
    
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        service_manager.shutdown_services()
        sys.exit(1)


if __name__ == "__main__":
    main()