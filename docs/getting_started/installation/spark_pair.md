# Pair two NVIDIA DGX Sparks

One GB10 is 128 GB of unified LPDDR5X. FastH3 still fits on a single Spark with
[`lazy_module_load`](../../inference/offloading.md) (auto on GB10; sequential
load stands down when lazy owns deferral). Two boxes connected
by the QSFP ConnectX-7 cables can run **one clip faster** and can hold a
**longer clip** (up to the FastH3 15 s cap).

This is FastVideo sequence parallel (`sp_size=2`) over Ray, not a third-party
xDiT vendor. Do not install xDiT for this path.

## What two Sparks buy you

| Goal | How | Use two Sparks? |
|---|---|---|
| Two independent videos at once | One process per box, `num_gpus=1` | Throughput only. Each clip still takes the 1-GPU time for that size. |
| One clip, faster | Ray + `sp_size=2` + parallel VAE | **Yes.** One 768×1344×124 recipe was 292 s vs 374 s on one GB10. |
| One clip, longer | Same, more frames | **Yes.** 345 frames (~14.4 s at 24 fps) finished in 587 s at 768×1344. |

Sequence parallel **replicates** the DiT (~66 GiB per node). Lazy module load
is still required on each box. FSDP would shard weights;
it is untested on this fabric and is likely slower because every layer gathers
over ~21 GB/s RoCE.

## Requirements

- Two DGX Sparks with FastVideo [installed](spark.md) (CUDA 13, `aarch64`).
- The QSFP cables that ship with a dual-Spark kit, **ACTIVE** at 200 Gb/s:
  `ibstat` should show the ConnectX-7 ports `LinkUp`.
- The same FastH3 snapshot on **both** NVMes. Copy the Hugging Face cache over
  QSFP; do not download 100+ GB twice over Wi-Fi.
- Ray in the FastVideo venv (`uv pip install ray` if it is not already there).

Each Spark has **one** GPU. `num_gpus=2` therefore means two nodes, which is
why the executor must be Ray (`mp` only works inside one process tree).

## 1. Put IPv4 on the QSFP NICs

The RoCE links often come up with no IPv4. Wi-Fi (`192.168.1.x`) is fine for
SSH and must stay the default route. NCCL and Ray must **not** use it.

Pick a /24 that does not collide with your LAN. Example:

| Node | QSFP IPv4 | Interface (typical) |
|---|---|---|
| Spark A | `192.168.23.1/24` | `enp1s0f1np1` |
| Spark B (Ray head) | `192.168.23.2/24` | `enp1s0f1np1` |

Confirm names with `ibdev2netdev` and `ip -br link`. Then, as root, on each
box (NetworkManager likes to steal the NIC; unmanaged is enough for a session):

```bash
sudo nmcli device set enp1s0f1np1 managed no
sudo ip addr replace 192.168.23.1/24 dev enp1s0f1np1   # .2 on the other box
sudo ip link set enp1s0f1np1 mtu 9000 up
```

These addresses do **not** survive reboot. Ping across the cable before
continuing: `ping -c 3 -I enp1s0f1np1 192.168.23.2`.

A healthy fabric on this hardware looks like:

- TCP iperf (jumbo 9000): ~40 Gb/s
- NCCL allreduce 1 GiB × 10: ~21 GB/s busbw (NVIDIA's dual-Spark figure is ~21.7)

## 2. Start a two-node Ray cluster on the cable

On **both** nodes, from the FastVideo repo, with the venv active:

```bash
source examples/inference/optimizations/spark_pair_env.sh
```

That script pins NCCL and Gloo to the QSFP NIC/HCA, disables NVLink-style P2P
(there is none between boxes), and turns off Ray's memory monitor. The monitor
treats GB10 unified RSS during a 14-shard DiT load as a runaway and SIGTERMs
the worker around shard 11/14. Override `NCCL_SOCKET_IFNAME` /
`GLOO_SOCKET_IFNAME` if `ibdev2netdev` shows a different name.

Cap Ray's object store. The default (~30% of 128 GB) leaves too little room
for the DiT:

```bash
# Spark B — head
export FASTVIDEO_HOST_IP=192.168.23.2
ray start --head --node-ip-address=192.168.23.2 --port=6379 --num-gpus=1 \
  --disable-usage-stats --object-store-memory=2147483648 --memory=4294967296

# Spark A — worker
export FASTVIDEO_HOST_IP=192.168.23.1
ray start --address=192.168.23.2:6379 --node-ip-address=192.168.23.1 --num-gpus=1 \
  --disable-usage-stats --object-store-memory=2147483648 --memory=4294967296
```

`FASTVIDEO_HOST_IP` **must** match `--node-ip-address`. If you omit it, Ray
advertises the Wi-Fi address, FastVideo builds a placement group for
`node:192.168.1.x`, and the QSFP workers never match.

Check `ray status` on the head: `0.0/2.0 GPU` idle.

## 3. Generate one FastH3 clip on both GPUs

Run the driver on the **head**, same venv, same QSFP IP.

`basic_fasth3.py` defaults target a four-GPU GB200 profile: 768×1344, `sm100a`
VSA, FA4, four GPUs. On Sparks you must override the kernel flags. Height,
width, frames, steps, seed, and prompt are yours. Change them. Legal
`num_frames` values are `17n+5`, capped at 345.

GB10 has no FA4 / sm_100a VSA kernel, so `--vsa-kernel triton --no-fa4` stays
required on this box. `--execution-backend ray` is optional when `RAY_ADDRESS`
is already set.

`--warmup --repeats 3` prints a median of three `generate()` calls after an
excluded warmup. Sequential load reloads Qwen for each later request, so that
protocol works. For a single cold process, pass `--no-warmup --repeats 1`.

The command below is one example, not a required recipe:

```bash
source examples/inference/optimizations/spark_pair_env.sh
export RAY_ADDRESS=192.168.23.2:6379
export FASTVIDEO_HOST_IP=192.168.23.2

python examples/inference/basic/basic_fasth3.py \
  --model-path FastVideo/FastVideo-FastH3-4-step-Preview-v1-VSA-DataFree \
  --num-gpus 2 --execution-backend ray \
  --vsa-kernel triton --no-fa4 \
  --warmup --repeats 3 --parallel-vae \
  --height 768 --width 1344 --num-frames 124 --steps 5 \
  --seed 2026 \
  --prompt "A wide cinematic shot of an alpine meadow at sunrise, pale pink mountain peaks above a blue valley filled with thin morning mist." \
  --output outputs/fasth3_spark_pair
```

Config-first equivalent. Edit the YAML the same way, `request.sampling` is not
locked:

```bash
FASTVIDEO_VSA_SM100A=0 FASTVIDEO_FA4=0 \
FASTVIDEO_ATTENTION_BACKEND=VIDEO_SPARSE_ATTN_H3 \
FASTVIDEO_VAE_PARALLEL_DECODE=1 FASTVIDEO_STAGE_LOGGING=1 \
fastvideo generate --config examples/inference/basic/basic_fasth3_spark_pair.yaml
```

Stop the cluster when you are done: `ray stop` on both nodes.

## FastH3 frame counts

H3 is 24 fps. Legal `num_frames` values are `17n+5`. The pipeline rejects
clips longer than **15 s**. The longest legal length is **345 frames**
(14.375 s). 360 frames aligns to 362 and fails the duration check.

## Measured on two GB10s (2026-08-31)

These rows are full H3 VAE decode, Triton VSA, sequential + lazy load (auto on
GB10), parallel VAE. They are not a required size. Denoise times include
deferred DiT load (~35 s on the first generate).

Cold process, `--no-warmup --repeats 1`, alpine prompt, 768×1344, 5 sigma
points (4 DiT forwards):

| Run | GPUs | Frames | E2E | Denoise | VAE decode |
|---|---:|---:|---:|---:|---:|
| One Spark | 1 | 124 | 374–393 s | 180–188 s | 151–156 s |
| Two Sparks, SP=2 | 2 | 124 | **292 s** | **122 s** | **102 s** |
| Two Sparks, SP=2 | 2 | 345 | **587 s** | **351 s** | **173 s** |

Warmup excluded, `--warmup --repeats 3` median, 512×896, 5 sigma points, full
VAE, same 4-step schedule:

| Run | GPUs | Frames | Median E2E | Median denoise |
|---|---:|---:|---:|---:|
| One Spark | 1 | 124 | **251.4 s** | 94.2 s |
| Two Sparks, SP=2 | 2 | 124 | **215.2 s** | 72.4 s |

Those medians used `--height` / `--width` / `--num-frames` as CLI flags. Swap
them. Native 480p on this model is 480×832, 124 frames. The 15 s cap is 345
frames.

The first VAE decode still pays `torch.compile`. Later `generate()` calls in
the same workers are cheaper. GB10 regional DiT compile stays off because the
sm_100a VSA kernel is not on this chip, so denoise is slower than a GB200
`sm100a` run at the same geometry.

## Troubleshooting

| Symptom | Fix |
|---|---|
| Placement group waits forever / `node:192.168.1.x` | Set `FASTVIDEO_HOST_IP` to the QSFP address on **every** `ray start` **and** on the driver. |
| `RayDistributedExecutor` TypeError / abstract `set_log_queue` | Use a FastVideo build that implements those methods on the Ray executor (this page). |
| Worker SIGTERM during DiT shard 11/14 | `RAY_memory_monitor_refresh_ms=0` **before** `ray start`. Do not leave Ray's default 30% object store. |
| NCCL hangs or uses Wi-Fi | `source spark_pair_env.sh`. Confirm `NCCL_SOCKET_IFNAME` is the QSFP NIC. |
| Gloo `connectFullMesh` / `remote=[127.0.0.1]` | Two 1-GPU nodes must not use loopback as the Gloo store. Source `spark_pair_env.sh` so `GLOO_SOCKET_IFNAME` is the QSFP NIC on **each** box. FastVideo no longer copies that NIC name from the driver onto workers. |
| Second `generate()` crashes `NoneType.parameters` | Sequential load used to drop the text encoder without reloading it. This branch reloads Qwen for later requests so `--warmup --repeats N` works. |
| OOM / `earlyoom` prefers Python | Lazy module load must stay on (do not pass `--no-lazy-module-load`). Peak GPU during 345-frame denoise is ~90 GiB/node. |
| `num_gpus=2` on one Spark | Each Spark has one GPU. Use Ray across two nodes, or `num_gpus=1` on one box. |

## What we are not claiming

- **Throughput of many clips.** Two independent 1-GPU jobs still win if you
  want two videos, not one faster video.
- **xDiT PipeFusion / CFG-parallel.** FastH3 is 4-step and has no CFG.
- **FSDP or tensor parallel as a speedup** on this 21 GB/s link.
- **Persistent networking.** The example IPs are session `ip addr replace`.

More GPUs are legal while `num_attention_heads` (56 on FastH3) is divisible by
`sp_size`. Four Sparks would need a four-node fabric that this bring-up did
not exercise.
