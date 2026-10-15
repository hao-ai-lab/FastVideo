# Retaining native weights across process restarts

A weight-cache donor owns loaded GPU tensors. Consumers create fresh Python
modules on `meta`, validate compatibility, and map those tensors through CUDA
IPC. Exiting a consumer does not unload the donor. This foundation is independent
of an edit runner and does not pickle or transfer a model object.

The initial contract is Linux, one GPU, native strided tensors, inference only,
and one active consumer pipeline at a time. The normal inference loader supports
Wan 2.1 T2V 1.3B DiT, UMT5 text encoder and VAE together. CPU registered tensors,
DTensor/FSDP, offload, compilation, quantization, LoRA and training are unsupported.
These combinations raise rather than falling back to disk.

## Complete Wan inference through the normal loader

The donor example loads DiT, T5 and VAE once through `VideoGenerator`. One process
owns all three components and serves one socket per component. Tokenizers and
schedulers remain local; each consumer builds new model objects and request state.

Start the donor on the same GPU, user and IPC namespace as the consumers:

```bash
python examples/benchmark/wan_weight_cache.py donor \
  --model-path /path/to/Wan2.1-T2V-1.3B-Diffusers \
  --checkpoint-id IMMUTABLE_REVISION --cache-dir /tmp/fv-wan-weights \
  --output /tmp/wan-donor.json
```

Wait for `DiT, T5 and VAE donor sockets ready`, then run independent consumers:

```bash
python examples/benchmark/wan_weight_cache.py consumer \
  --model-path /path/to/Wan2.1-T2V-1.3B-Diffusers \
  --checkpoint-id IMMUTABLE_REVISION --cache-dir /tmp/fv-wan-weights \
  --output /tmp/wan-consumer-1.json --tensor-output /tmp/wan-consumer-1.pt
```

Repeat with new output filenames. Use role `disk` for fresh-loading parity checks.
Each worker runs two identical seeded 9-frame, 128x128, four-step requests. The
in-process executor records full component fingerprints and IPC-handle witnesses.
`--executor mp` also exercises the default executor selection; FastVideo resolves
single-GPU `mp` to `UniprocExecutor`, which the report records explicitly.
Stop the donor with Ctrl-C after consumers finish; it waits for active leases.

For application code, use the ordinary public inference entry point:

```python
generator = VideoGenerator.from_pretrained(
    local_checkpoint, num_gpus=1, tp_size=1, sp_size=1,
    weight_cache_dir="/tmp/fv-wan-weights",
    weight_cache_checkpoint="IMMUTABLE_REVISION",
    dit_cpu_offload=False, dit_layerwise_offload=False,
    text_encoder_cpu_offload=False, vae_cpu_offload=False,
    lazy_module_load=False, inference_torch_compile=False,
)
try:
    result = generator.generate_video(prompt, save_video=False)
finally:
    generator.shutdown()
```

The same fields have CLI options `--weight-cache-dir` and
`--weight-cache-checkpoint`. A checkpoint identity without a directory keeps disk
loading and records the donor keys. Supplying the directory makes attachment
mandatory for all three components. There is no silent hydration fallback.

`PipelineComponentLoader` retains normal config resolution and class dispatch.
Its native loader adapters attach after configuration but before checkpoint
hydration. Identity includes the resolved architecture, config constants, dtype,
attention backend request and loader processing version. The caller must provide
an immutable checkpoint revision or content identity; the cache does not hash
all checkpoint files during attachment. Recreate the donor when the checkpoint,
layout, or weight-processing implementation changes.

The pipeline owns an `ExitStack` of attachment leases. Pipeline close and worker
shutdown synchronize and detach all components; partial load failure releases
already attached components. Direct `PipelineComponentLoader` users must supply
`weight_cache_leases=stack` and keep that stack open throughout inference. Runtime
LoRA merge/change APIs reject shared weights. Arbitrary user tensor mutations
remain outside the read-only contract and cannot be prevented by CUDA IPC.

The donor reserves the combined weight memory for its lifetime. Attached workers
still need their own CUDA contexts, communication state and activation memory.
Keep the donor and sequential workers inside the same GPU allocation; this does
not retain GPU memory across allocation teardown or machine restart.

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
