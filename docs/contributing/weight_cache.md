# Retaining native weights across process restarts

A weight-cache donor owns loaded GPU tensors. Consumers create fresh Python
modules on `meta`, validate compatibility, and map those tensors through CUDA
IPC. Exiting a consumer does not unload the donor. This foundation is independent
of an edit runner and does not pickle or transfer a model object.

The initial contract is Linux, one GPU, native strided tensors, inference only,
and one active consumer at a time. CPU registered tensors, DTensor/FSDP,
quantized representations and training are not supported. Multiple components
and parallel-rank routing need additional adapters and identity fields.

## Native Wan VAE example

Use the same GPU, user and IPC namespace for all processes. The model directory
is a local Wan 2.1 1.3B Diffusers VAE component. Supply its immutable revision.
Create a private socket directory:

```bash
install -d -m 700 /tmp/fv-weights
python examples/benchmark/vae_weight_cache.py donor \
  --model-path /path/to/checkpoint/vae --checkpoint-id IMMUTABLE_REVISION \
  --socket /tmp/fv-weights/vae.sock --output /tmp/donor.json
```

After the socket exists, run consumers from another terminal. Each invocation is
an independent process:

```bash
python examples/benchmark/vae_weight_cache.py consumer \
  --model-path /path/to/checkpoint/vae --checkpoint-id IMMUTABLE_REVISION \
  --socket /tmp/fv-weights/vae.sock --output /tmp/consumer-1.json \
  --tensor-output /tmp/consumer-1.pt
```

Repeat with new output filenames. The same donor retains the tensors. The example
reads configuration in each consumer, constructs fresh VAE caches and module
objects, and attaches registered weights without checkpoint hydration. Its FP32
codec check decodes identical input twice, checks cache reset and weight
fingerprints, and can export output for an independent `disk`-role comparison.
Use Ctrl-C on the donor after consumers finish; graceful shutdown waits for the
active connection to close and writes a final weight fingerprint.

## Python boundary

The donor receives an already loaded, idle inference model:

```python
serve_weight_cache(model.eval(), key, socket_path, stop=stop_event)
```

Construction and attachment happen in the consumer:

```python
client = WeightCacheClient(socket_path, key, device)
with client.load_model(factory) as attached:
    output = attached.model(inputs)
```

`factory` constructs the component without reading checkpoint weights. It runs
inside a meta-device context. Initialize required non-registered configuration
constants outside that context. Model/request state belongs to the consumer;
only registered parameters and buffers are shared. Non-persistent buffers and
tied-parameter identities are included. Consumer bindings are restored to meta
on context exit, after GPU synchronization. Do not retain mapped tensors outside
the context.

The donor must outlive every mapped tensor. Shared tensors have a read-only
inference contract; CUDA does not enforce read-only mappings. In-place weight
changes require private copies or a new donor. Donor crashes invalidate consumers
and require recovery. Imports, CUDA contexts, compilation, graphs and request
caches are not transferred or retained by this API.

## Compatibility and transport

`WeightCacheKey` requires checkpoint, component, configuration, dtype and
weight-processing identities. The caller must provide the correct immutable
checkpoint identity and increment the processing version when weight preparation
changes. A method-body edit can reuse the same weights. Both peers also check the
Torch version and physical GPU UUID. The complete registered layout checks class,
names, shapes, dtypes, strides, offsets, aliases and buffer persistence before
any bindings change. A mismatch or missing donor raises; no disk fallback exists.

The Unix socket and its directory must be private to the current user. Each peer
checks Linux peer credentials before decoding trusted same-user descriptors.
Descriptors contain CUDA IPC handles, not GPU weight bytes. New PyTorch reduction
arguments are created for every consumer to preserve per-mapping refcounts. A
per-socket lock prevents a second donor from replacing an active donor's socket.

The design follows the separation used by vLLM's
[weight-cache donor](https://github.com/vllm-project/vllm/blob/b965ee564d1897fdaeddaeceecc316aa15a45312/vllm/model_executor/model_loader/weight_cache/daemon.py)
and [IPC loader](https://github.com/vllm-project/vllm/blob/b965ee564d1897fdaeddaeceecc316aa15a45312/vllm/model_executor/model_loader/weight_cache/ipc_loader.py).
vLLM additionally handles post-processing, quantization and parallel shards.
Those capabilities are not implied by this initial native-tensor foundation.
