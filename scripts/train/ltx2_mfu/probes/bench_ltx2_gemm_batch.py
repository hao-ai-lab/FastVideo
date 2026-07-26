import statistics
import time

import torch
import torch.nn.functional as F


DEVICE = torch.device("cuda:0")
DTYPE = torch.bfloat16
TOKENS = 4290
HIDDEN = 4096
FFN = 16384


class FFNBlock(torch.nn.Module):

    def __init__(self) -> None:
        super().__init__()
        self.fc1 = torch.nn.Linear(HIDDEN, FFN, bias=True, device=DEVICE, dtype=DTYPE)
        self.fc2 = torch.nn.Linear(FFN, HIDDEN, bias=True, device=DEVICE, dtype=DTYPE)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(F.gelu(self.fc1(x), approximate="tanh"))


class Projections(torch.nn.Module):

    def __init__(self) -> None:
        super().__init__()
        self.layers = torch.nn.ModuleList([
            torch.nn.Linear(HIDDEN, HIDDEN, bias=True, device=DEVICE, dtype=DTYPE)
            for _ in range(4)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return sum(layer(x) for layer in self.layers)


def measure(module: torch.nn.Module, batch: int, matmul_flops: float) -> tuple[float, float]:
    x = torch.randn(batch, TOKENS, HIDDEN, device=DEVICE, dtype=DTYPE, requires_grad=True)

    def step() -> None:
        out = module(x)
        out.backward(torch.ones_like(out))
        x.grad = None
        for parameter in module.parameters():
            parameter.grad = None

    for _ in range(5):
        step()
    torch.cuda.synchronize()

    samples = []
    for _ in range(12):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        step()
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end))
    median_ms = statistics.median(samples)
    return median_ms, matmul_flops * batch / (median_ms * 1e9)


def main() -> None:
    torch.manual_seed(0)
    torch.cuda.set_device(DEVICE)
    cases = (
        ("ffn", FFNBlock, 12.0 * TOKENS * HIDDEN * FFN),
        ("four_4096_projections", Projections, 24.0 * TOKENS * HIDDEN * HIDDEN),
    )
    for name, factory, flops in cases:
        for compiled in (False, True):
            results = []
            for batch in (1, 2):
                module = factory()
                if compiled:
                    module = torch.compile(module, fullgraph=True)
                median_ms, tflops = measure(module, batch, flops)
                results.append((batch, median_ms, tflops))
                del module
                torch.cuda.empty_cache()
            ratio = results[1][1] / results[0][1]
            print(name, "compiled=" + str(compiled), results, "b2_over_b1=" + f"{ratio:.4f}")


if __name__ == "__main__":
    started = time.time()
    main()
    print("wall_sec", time.time() - started)
