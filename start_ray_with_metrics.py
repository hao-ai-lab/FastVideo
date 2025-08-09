#!/usr/bin/env python3
"""
Start Ray with metrics collection enabled for Prometheus/Grafana monitoring
"""
import os

import ray


def start_ray_with_metrics():
    """Initialize Ray with metrics collection enabled"""

    # Ray configuration for metrics
    ray_config = {
        "dashboard_host": "0.0.0.0",  # Allow external access to dashboard
        "dashboard_port": 8265,  # Default Ray dashboard port
        "metrics_export_port": 8080,  # Port for Prometheus metrics
        "include_dashboard": True,  # Enable Ray dashboard
        "log_to_driver": True,
    }

    # Set environment variables for better metrics collection
    os.environ["RAY_ENABLE_RECORD_TASK_SCHEDULING"] = "1"
    os.environ["RAY_record_ref_creation_sites"] = "1"

    # Initialize Ray
    if ray.is_initialized():
        print("Ray is already initialized. Shutting down...")
        ray.shutdown()

    print("Starting Ray with metrics collection enabled...")
    ray.init(**ray_config)

    print("✅ Ray started successfully!")
    print("📊 Ray Dashboard: http://localhost:8265")
    print("📈 Prometheus metrics: http://localhost:8080/metrics")
    print(f"🔧 Ray status: {ray.cluster_resources()}")

    return True


if __name__ == "__main__":
    start_ray_with_metrics()
