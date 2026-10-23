"""All-to-all bandwidth on 8x RTX PRO 6000 at FastH3 Ulysses SP8 payload sizes (10 s, 1344x768: 73.6k tokens)."""
import modal

image = (modal.Image.debian_slim(python_version="3.12")
         .pip_install("torch==2.8.0", index_url="https://download.pytorch.org/whl/cu128"))
app = modal.App("h3-a2a8", image=image)

TOKENS, WIDTH = 73642, 7168
WORLD = 8


def _worker(rank: int, env: dict, out_q) -> None:
    import os
    import time
    import torch
    import torch.distributed as dist
    os.environ.update(env)
    os.environ.update({"MASTER_ADDR": "127.0.0.1", "MASTER_PORT": "29511"})
    torch.cuda.set_device(rank)
    dist.init_process_group("nccl", rank=rank, world_size=WORLD)
    rows = -(-TOKENS // WORLD)
    payloads = {
        "qkvg_bf16": 4 * rows * WIDTH * 2,          # q, k, v, gate shards before attention
        "qkv_fp4_plus_gate_bf16": 3 * rows * WIDTH * 9 // 16 + rows * WIDTH * 2,
        "qkvg_fp4": 4 * rows * WIDTH * 9 // 16,      # 4-bit values + one e4m3 scale per 16
        "out_bf16": rows * WIDTH * 2,                # attention output back to sequence shards
    }
    res = {}
    for name, nbytes in payloads.items():
        n = (nbytes // 2 // WORLD) * WORLD
        send = torch.empty(n, dtype=torch.bfloat16, device="cuda")
        recv = torch.empty_like(send)
        for _ in range(3):
            dist.all_to_all_single(recv, send)
        torch.cuda.synchronize()
        dist.barrier()
        iters = 10
        t = time.perf_counter()
        for _ in range(iters):
            dist.all_to_all_single(recv, send)
        torch.cuda.synchronize()
        ms = (time.perf_counter() - t) / iters * 1e3
        res[name] = {"mb_per_rank": round(n * 2 / 1e6, 1), "ms": round(ms, 2),
                     "algbw_gbps": round(n * 2 / ms / 1e6, 1)}
    if rank == 0:
        out_q.put(res)
    dist.destroy_process_group()


@app.function(gpu="RTX-PRO-6000:8", cpu=16, memory=65536, timeout=1800)
def bench() -> dict:
    import subprocess
    import torch.multiprocessing as mp
    report = {"topo": subprocess.run(["nvidia-smi", "topo", "-m"], capture_output=True, text=True).stdout[-3000:]}
    for label, env in (("p2p_default", {}), ("p2p_disabled", {"NCCL_P2P_DISABLE": "1"})):
        ctx = mp.get_context("spawn")
        q = ctx.Queue()
        procs = [ctx.Process(target=_worker, args=(r, env, q)) for r in range(WORLD)]
        for p in procs:
            p.start()
        report[label] = q.get(timeout=900)
        for p in procs:
            p.join()
    return report


@app.local_entrypoint()
def main():
    import json
    r = bench.remote()
    print(r.pop("topo"))
    print("A2A", json.dumps(r, indent=1))
