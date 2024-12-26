import platform

import accelerate
import peft
import torch
import transformers
from transformers.utils import is_torch_cuda_available, is_torch_npu_available


VERSION = "1.2.0"


if __name__ == "__main__":
    info = {
        "FastVideo version": VERSION,
        "Platform": platform.platform(),
        "Python version": platform.python_version(),
        "PyTorch version": torch.__version__,
        "Transformers version": transformers.__version__,
        "Accelerate version": accelerate.__version__,
        "PEFT version": peft.__version__,
    }

    if is_torch_cuda_available():
        info["PyTorch version"] += " (GPU)"
        info["GPU type"] = torch.cuda.get_device_name()

    if is_torch_npu_available():
        info["PyTorch version"] += " (NPU)"
        info["NPU type"] = torch.npu.get_device_name()
        info["CANN version"] = torch.version.cann

    try:
        import bitsandbytes

        info["Bitsandbytes version"] = bitsandbytes.__version__
    except Exception:
        pass

    print("\n" + "\n".join([f"- {key}: {value}" for key, value in info.items()]) + "\n")