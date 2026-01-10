# SPDX-License-Identifier: Apache-2.0
"""
Pytest configuration for SSIM tests.
Includes GPU and CPU memory logging to debug OOM issues.
"""
import gc
import os
import pytest
import torch

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False


def get_cpu_memory_info():
    """Get CPU/RAM memory info for the current process and system."""
    if not PSUTIL_AVAILABLE:
        return "psutil not available (pip install psutil)"
    
    process = psutil.Process(os.getpid())
    proc_mem = process.memory_info()
    proc_rss = proc_mem.rss / 1024**3  # GB - actual physical memory used
    proc_vms = proc_mem.vms / 1024**3  # GB - virtual memory size
    
    sys_mem = psutil.virtual_memory()
    sys_used = sys_mem.used / 1024**3  # GB
    sys_total = sys_mem.total / 1024**3  # GB
    sys_percent = sys_mem.percent
    
    return (
        f"Process: {proc_rss:.2f}GB RSS, {proc_vms:.2f}GB VMS | "
        f"System: {sys_used:.2f}GB/{sys_total:.2f}GB ({sys_percent}% used)"
    )


def get_gpu_memory_info():
    """Get memory info for all available GPUs."""
    if not torch.cuda.is_available():
        return "CUDA not available"
    
    info_lines = []
    for i in range(torch.cuda.device_count()):
        allocated = torch.cuda.memory_allocated(i) / 1024**3  # GB
        reserved = torch.cuda.memory_reserved(i) / 1024**3    # GB
        max_allocated = torch.cuda.max_memory_allocated(i) / 1024**3  # GB
        total = torch.cuda.get_device_properties(i).total_memory / 1024**3  # GB
        info_lines.append(
            f"GPU {i}: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved, "
            f"{max_allocated:.2f}GB peak, {total:.2f}GB total"
        )
    return " | ".join(info_lines)


@pytest.fixture(autouse=True)
def log_memory(request):
    """Log GPU and CPU memory before and after each test."""
    test_name = request.node.name
    
    # Before test
    print(f"\n{'='*60}")
    print(f"[MEMORY] BEFORE {test_name}")
    print(f"[GPU] {get_gpu_memory_info()}")
    print(f"[CPU] {get_cpu_memory_info()}")
    print(f"{'='*60}")
    
    yield
    
    # After test (before cleanup)
    print(f"\n{'='*60}")
    print(f"[MEMORY] AFTER {test_name} (before cleanup)")
    print(f"[GPU] {get_gpu_memory_info()}")
    print(f"[CPU] {get_cpu_memory_info()}")
    
    # Force cleanup
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    
    # After cleanup
    print(f"[MEMORY] AFTER {test_name} (after cleanup)")
    print(f"[GPU] {get_gpu_memory_info()}")
    print(f"[CPU] {get_cpu_memory_info()}")
    print(f"{'='*60}\n")


@pytest.fixture(autouse=True, scope="module")
def log_module_memory(request):
    """Log GPU and CPU memory at module start/end."""
    module_name = request.node.name
    
    print(f"\n{'#'*60}")
    print(f"[MODULE START] {module_name}")
    print(f"[GPU] {get_gpu_memory_info()}")
    print(f"[CPU] {get_cpu_memory_info()}")
    print(f"{'#'*60}\n")
    
    yield
    
    # Force cleanup at module end
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    
    print(f"\n{'#'*60}")
    print(f"[MODULE END] {module_name}")
    print(f"[GPU] {get_gpu_memory_info()}")
    print(f"[CPU] {get_cpu_memory_info()}")
    print(f"{'#'*60}\n")
