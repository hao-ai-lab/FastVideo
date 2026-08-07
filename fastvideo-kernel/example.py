import torch
from kernels import get_kernel


fastvideo_kernel = get_kernel("hao-ai-lab/fastvideo-kernel", version=1)

x = torch.randn((16, 1024), device="cuda", dtype=torch.float16)
weight = torch.ones((1024,), device="cuda", dtype=torch.float16)
y = fastvideo_kernel.rms_norm(x, 1e-6, weight)

print(y.shape)
