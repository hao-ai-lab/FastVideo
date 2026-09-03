# MiniMax H3 component disaggregation on two DGX Sparks

This runbook tests the component-disaggregated MiniMax H3 inference path:

- Spark A: text/multimodal encoder, video VAE, audio VAE, and the driver.
- Spark B: DiT denoising only.
- Ray transports CPU-contiguous conditioning and latent tensors between the two roles.
- Each role uses one GPU with tensor parallelism and sequence parallelism disabled.
- All role components remain resident until the generator shuts down.

The first test is a synchronous correctness smoke test. A second test exercises
the bounded asynchronous request pipeline.

## Important: public SSH ports are not Ray node identities

It is fine for both Sparks to share one public or NAT IP while SSH exposes them
through different ports:

| Purpose | Spark A | Spark B |
|---|---|---|
| SSH from another machine | PUBLIC_IP:SSH_PORT_A | PUBLIC_IP:SSH_PORT_B |
| Ray and FastVideo | unique internal A_IP | unique internal B_IP |
| Component role | encoder/decoder | DiT |

Do not pass PUBLIC_IP, SSH_PORT_A, or SSH_PORT_B to
h3_encoder_node_ip or h3_dit_node_ip. The current runtime pins actors using
Ray resources named node:A_IP and node:B_IP and rejects equal node addresses.

Use one of the following for A_IP and B_IP:

- The two QSFP/RoCE addresses.
- Two distinct LAN addresses that are mutually reachable.
- Two distinct Tailscale or WireGuard addresses.

If Ray itself reports the same NodeManagerAddress for both physical machines,
stop. Port differences alone cannot select the two component roles. Configure
unique internal or overlay addresses before continuing.

## 1. Connect and identify the internal interfaces

From the machine used for SSH:

~~~bash
ssh -p SSH_PORT_A USER@PUBLIC_IP
ssh -p SSH_PORT_B USER@PUBLIC_IP
~~~

On each Spark:

~~~bash
hostname
ip -br addr
ibdev2netdev
~~~

The examples below use:

~~~text
Spark A: 192.168.23.1
Spark B: 192.168.23.2
Ray head: Spark A, port 6379
~~~

Replace these values with the real addresses.

If the direct QSFP interface does not already have IPv4 addresses, first find
the interface name with ibdev2netdev. The commonly used interface is
enp1s0f1np1.

On Spark A:

~~~bash
export H3_QSFP_IFACE=enp1s0f1np1
sudo nmcli device set "$H3_QSFP_IFACE" managed no
sudo ip addr replace 192.168.23.1/24 dev "$H3_QSFP_IFACE"
sudo ip link set "$H3_QSFP_IFACE" mtu 9000 up
~~~

On Spark B:

~~~bash
export H3_QSFP_IFACE=enp1s0f1np1
sudo nmcli device set "$H3_QSFP_IFACE" managed no
sudo ip addr replace 192.168.23.2/24 dev "$H3_QSFP_IFACE"
sudo ip link set "$H3_QSFP_IFACE" mtu 9000 up
~~~

Verify both directions:

~~~bash
# Spark A
ping -c 3 192.168.23.2

# Spark B
ping -c 3 192.168.23.1
~~~

These manually assigned addresses do not survive a reboot unless they are
added to the persistent network configuration.

## 2. Put the same FastVideo code and environment on both Sparks

The component-disaggregation implementation must exist on both machines. A
stock checkout that predates this change will not recognize the new flags.

On both Sparks:

~~~bash
cd /path/to/FastVideo
source .venv/bin/activate

git rev-parse HEAD
UV_TORCH_BACKEND=cu130 uv pip install -e .
~~~

If the model requires Hugging Face authentication, export the token before
starting Ray so the Ray worker processes inherit it:

~~~bash
export HF_TOKEN=YOUR_TOKEN
~~~

Verify the software and GPU on both machines:

~~~bash
python - <<'PY'
import ray
import torch
from fastvideo.worker.minimax_h3_disaggregated import (
    MiniMaxH3DisaggregatedExecutor,
)

print("Ray:", ray.__version__)
print("Torch:", torch.__version__)
print("CUDA:", torch.version.cuda)
print("CUDA available:", torch.cuda.is_available())
print("GPU:", torch.cuda.get_device_name(0))
print("Disaggregated executor import: OK")
PY
~~~

Both machines must use the same Ray version and should report NVIDIA GB10.
FastVideo currently requires Ray 2.49.1 or newer.

For local model directories, use the same absolute path on both machines. A
Hugging Face model ID can be used instead; each role downloads only its owned
component directories plus small required metadata.

## 3. Start the Ray cluster

Stop an old cluster first only if it is safe to terminate all Ray work on that
machine:

~~~bash
ray stop
~~~

### Spark A: Ray head and encoder/decoder node

~~~bash
cd /path/to/FastVideo
source .venv/bin/activate

export A_IP=192.168.23.1
export B_IP=192.168.23.2
export FASTVIDEO_HOST_IP="$A_IP"
export RAY_memory_monitor_refresh_ms=0
export RAY_memory_usage_threshold=1.0
export FASTVIDEO_STAGE_LOGGING=1

ray start \
  --head \
  --node-ip-address="$A_IP" \
  --port=6379 \
  --num-gpus=1 \
  --disable-usage-stats \
  --object-store-memory=2147483648 \
  --memory=4294967296
~~~

### Spark B: DiT node

~~~bash
cd /path/to/FastVideo
source .venv/bin/activate

export A_IP=192.168.23.1
export B_IP=192.168.23.2
export FASTVIDEO_HOST_IP="$B_IP"
export RAY_memory_monitor_refresh_ms=0
export RAY_memory_usage_threshold=1.0
export FASTVIDEO_STAGE_LOGGING=1

ray start \
  --address="$A_IP:6379" \
  --node-ip-address="$B_IP" \
  --num-gpus=1 \
  --disable-usage-stats \
  --object-store-memory=2147483648 \
  --memory=4294967296
~~~

The Ray port 6379 is the head/GCS port. It is unrelated to either SSH port.
Ray uses additional internal service ports, so forwarding only SSH and 6379
through NAT is not a substitute for a mutually reachable internal network.

The Ray memory monitor is disabled because GB10 unified-memory usage during
large checkpoint loading can otherwise be mistaken for host-memory pressure.

## 4. Verify Ray sees two distinct one-GPU nodes

Run on Spark A:

~~~bash
export A_IP=192.168.23.1
export RAY_ADDRESS="$A_IP:6379"

ray status

python - <<'PY'
import os
import ray

ray.init(address=os.environ["RAY_ADDRESS"])

print("Live nodes:")
for node in ray.nodes():
    if node["Alive"]:
        print({
            "node_id": node["NodeID"],
            "address": node["NodeManagerAddress"],
            "gpu": node["Resources"].get("GPU", 0),
        })

print("Placement resources:")
print(sorted(
    key for key in ray.cluster_resources()
    if key.startswith("node:")
))
PY
~~~

Expected placement resources:

~~~text
node:192.168.23.1
node:192.168.23.2
~~~

Expected total capacity from ray status:

~~~text
0.0/2.0 GPU
~~~

Do not continue if only one node appears, either GPU is unavailable, or both
NodeManagerAddress values are identical.

## 5. Create the synchronous smoke-test configuration

Run on Spark A:

~~~bash
export A_IP=192.168.23.1
export B_IP=192.168.23.2
export H3_MODEL_PATH=MiniMaxAI/MiniMax-H3

tee /tmp/h3-disaggregated-smoke.yaml >/dev/null <<YAML
generator:
  model_path: "$H3_MODEL_PATH"

  engine:
    # num_gpus is one per independent component role. The specialized
    # runtime still reserves two physical GPUs, one on each named node.
    num_gpus: 1
    execution_backend: ray
    use_fsdp_inference: false

    parallelism:
      tp_size: 1
      sp_size: 1

    offload:
      dit: false
      dit_layerwise: false
      text_encoder: false
      image_encoder: false
      vae: false
      pin_cpu_memory: false
      lazy_module_load: false

    compile:
      enabled: false
      text_encoder_enabled: false
      vae_enabled: false
      audio_vae_enabled: false

  pipeline:
    experimental:
      h3_disaggregated: true
      h3_encoder_node_ip: "$A_IP"
      h3_dit_node_ip: "$B_IP"
      h3_ray_address: "$A_IP:6379"

      h3_sequential_load: false
      vae_parallel_encode: false
      vae_parallel_decode: false
      video_decode_backend: h3-vae

      # Conservative correctness-first backend for the base H3 checkpoint.
      attention_backend: TORCH_SDPA

request:
  prompt: >-
    A cinematic wide shot of an alpine lake at sunrise, with gentle mist
    drifting above the water.
  negative_prompt: ""

  sampling:
    seed: 2026
    height: 256
    width: 256
    # H3 requires at least five seconds. 124 is the smallest normal legal
    # frame count satisfying its 17n+5 causal-VAE geometry.
    num_frames: 124
    fps: 24
    # Two sigma points produce one DiT forward, suitable for plumbing checks.
    num_inference_steps: 2
    guidance_scale: 1.0
    batch_cfg: false

  output:
    output_path: outputs/h3-disaggregated-smoke/
    save_video: true
    return_frames: false
YAML
~~~

Inspect the resolved file before launching:

~~~bash
sed -n '1,220p' /tmp/h3-disaggregated-smoke.yaml
~~~

Critical settings:

- num_gpus must be 1, not 2.
- tp_size and sp_size must both be 1.
- execution_backend must be ray.
- h3_encoder_node_ip and h3_dit_node_ip must be distinct Ray node IPs.
- FSDP, lazy loading, sequential loading, VAE parallelism, and TAEH3 must be off.
- The full H3 video VAE is selected with video_decode_backend: h3-vae.

## 6. Run the synchronous correctness smoke test

On Spark A:

~~~bash
cd /path/to/FastVideo
source .venv/bin/activate

export A_IP=192.168.23.1
export RAY_ADDRESS="$A_IP:6379"
export FASTVIDEO_HOST_IP="$A_IP"

fastvideo generate --config /tmp/h3-disaggregated-smoke.yaml
~~~

Successful actor startup validates both placement and component isolation.
The expected resident module sets are:

~~~text
Spark A / encoder_decoder:
audio_vae, processor, scheduler, text_encoder, tokenizer, vae

Spark B / dit:
audio_scheduler, scheduler, transformer
~~~

The output should be written under:

~~~text
outputs/h3-disaggregated-smoke/
~~~

The first startup can be slow because the two large component sets are loaded
and, when using a Hugging Face model ID, downloaded into each node's cache.

## 7. Exercise three-request pipeline concurrency

This test directly uses the disaggregated executor's bounded lookahead API. It
keeps both actors alive, reports their health, and submits three requests.

Run on Spark A:

~~~bash
cd /path/to/FastVideo
source .venv/bin/activate

export A_IP=192.168.23.1
export RAY_ADDRESS="$A_IP:6379"
export FASTVIDEO_HOST_IP="$A_IP"

python - <<'PY'
import time

from fastvideo import VideoGenerator
from fastvideo.pipelines import ForwardBatch

generator = VideoGenerator.from_file("/tmp/h3-disaggregated-smoke.yaml")

try:
    print("Worker health:")
    for receipt in generator.executor.runtime.health():
        print(receipt)

    prompts = [
        "An alpine lake at sunrise with thin mist.",
        "Ocean waves beneath a cloudy blue sky.",
        "A quiet forest path illuminated by morning light.",
    ]

    batches = [
        ForwardBatch(
            data_type="video",
            prompt=prompt,
            negative_prompt="",
            height=256,
            width=256,
            num_frames=124,
            fps=24,
            num_inference_steps=2,
            guidance_scale=1.0,
            batch_cfg=False,
            num_videos_per_prompt=1,
            seed=2026 + index,
            save_video=False,
            return_frames=False,
        )
        for index, prompt in enumerate(prompts)
    ]

    started = time.perf_counter()
    for index, output in enumerate(
        generator.executor.iter_forward(batches),
        start=1,
    ):
        audio = output.extra["audio"]
        print(
            f"request={index}",
            f"video={tuple(output.output.shape)}",
            f"audio={tuple(audio.shape)}",
            f"sample_rate={output.extra['audio_sample_rate']}",
            f"elapsed={time.perf_counter() - started:.1f}s",
        )
finally:
    generator.shutdown()
PY
~~~

Expected health properties:

~~~text
role: encoder_decoder
node_ip: 192.168.23.1
all_resident: True

role: dit
node_ip: 192.168.23.2
all_resident: True
~~~

The scheduling order is:

~~~text
Spark A encodes request N+1
Spark B denoises request N
Spark A decodes request N-1
~~~

Spark A serializes its own encode and decode operations, while those operations
overlap with Spark B's independent denoising work. Spark B schedules the next
denoise before the driver waits for the previous decode result.

## 8. Observe utilization and confirm components are not reloaded

In a separate SSH session on each Spark:

~~~bash
nvidia-smi dmon -s pucvmet
~~~

Ray worker logs are normally under:

~~~text
/tmp/ray/session_latest/logs/
~~~

Inspect relevant lifecycle messages:

~~~bash
grep -R -E \
  "Loading pipeline modules|Released MiniMax|Reloading MiniMax|component-disaggregated" \
  /tmp/ray/session_latest/logs/ | tail -n 100
~~~

For one generator lifetime:

- Each actor should initialize its pipeline once.
- No request should release or reload the text encoder, DiT, or VAEs.
- Spark B should continue denoising without waiting for Spark A's VAE decode.

## 9. Optional FastH3/VSA test after correctness passes

The first smoke test intentionally uses the base MiniMax H3 checkpoint with
TORCH_SDPA. For the FastH3 VSA preview checkpoint, change:

~~~yaml
generator:
  model_path: FastVideo/FastVideo-FastH3-4-step-Preview-v1-VSA-DataFree

  pipeline:
    experimental:
      attention_backend: VIDEO_SPARSE_ATTN_H3
      VSA_sparsity: 0.9
      VSA_tile_size: 64
~~~

Use five sigma points for the four-forward preview schedule:

~~~yaml
request:
  sampling:
    num_inference_steps: 5
~~~

On GB10, do not enable the sm_100a FA4 path. Keep compilation disabled until
the basic distributed path is known to work.

## 10. Troubleshooting

### Both Sparks share one public IP

Different SSH ports are only a login mechanism. Use internal LAN, QSFP, or
overlay-network addresses for Ray. Do not write PUBLIC_IP:SSH_PORT into either
H3 node-IP field.

### Ray reports the same node IP twice

The current runtime cannot distinguish those nodes because it pins actors with
node:IP resources and validates actual placement by IP. Configure distinct
reachable addresses. If that is impossible, the runtime must be extended to
accept Ray node IDs or explicit named per-role resources.

### Ray reports no live node resource

Confirm that FASTVIDEO_HOST_IP matched --node-ip-address before ray start on
each machine:

~~~bash
python - <<'PY'
import ray
ray.init(address="auto")
print(sorted(k for k in ray.cluster_resources() if k.startswith("node:")))
PY
~~~

Restart Ray after correcting the environment.

### An actor remains pending

Check:

~~~bash
ray status
~~~

Each requested node must have one free GPU. Stop unrelated GPU jobs or stale
Ray actors before retrying.

### Worker dies while loading checkpoint shards

Verify that these were exported before ray start on both nodes:

~~~bash
export RAY_memory_monitor_refresh_ms=0
export RAY_memory_usage_threshold=1.0
~~~

Also confirm that no other large process is consuming the GB10's unified
memory.

### Import or unknown-argument error

The new implementation is missing from one node or that node is using a
different Python environment. On both machines:

~~~bash
which python
which ray
python -c "from fastvideo.worker.minimax_h3_disaggregated import MiniMaxH3DisaggregatedExecutor; print('OK')"
~~~

### Hugging Face download or authentication error

Ensure the model is accessible from both nodes. Export HF_TOKEN before starting
Ray, or pre-populate each node's Hugging Face cache.

### TAEH3 validation error

TAEH3 intentionally omits the full video VAE and is therefore incompatible
with this topology's resident video-VAE requirement. Use h3-vae.

### num_gpus, TP, or SP validation error

For component disaggregation, use:

~~~yaml
num_gpus: 1
tp_size: 1
sp_size: 1
~~~

Although two physical GPUs are reserved, each component role is an independent
one-GPU process rather than one two-rank model-parallel world.

## 11. Stop the cluster

The generator shuts down its two actors, but it does not stop the shared Ray
cluster. When testing is complete, run this on both machines:

~~~bash
ray stop
~~~

## Relevant implementation files

- fastvideo/pipelines/basic/minimax_h3/disaggregated.py
- fastvideo/worker/minimax_h3_disaggregated.py
- fastvideo/fastvideo_args.py
- fastvideo/worker/executor.py
- fastvideo/tests/worker/test_minimax_h3_disaggregated.py

The existing docs/getting_started/installation/spark_pair.md describes the
separate sequence-parallel two-Spark mode. Do not copy its num_gpus=2 and
sp_size=2 settings into this component-disaggregated mode.
