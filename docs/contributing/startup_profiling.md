# Profiling model startup

Measure startup separately from inference latency before choosing what to retain
between development runs. The initial instrumentation covers component totals
and the native VAE safetensors loader, including Wan. It does not implement a
weight donor or CUDA IPC.

## Run the Wan VAE profile

Use a prepared CUDA environment and a local Wan 2.1 1.3B Diffusers VAE component directory:

```bash
python examples/benchmark/vae_startup.py \
  --model-path /path/to/checkpoint/vae \
  --output /tmp/vae-startup.json \
  --tensor-output /tmp/vae-output.pt \
  --synchronize
```

The script measures torch and FastVideo/helper imports separately, device and
process-group setup, VAE loading, and the first decode of a fixed seeded latent
input. It preserves launcher-provided rendezvous settings; otherwise it chooses a
local rendezvous port. It uses one GPU, FP32, and the default cached Wan VAE path.
Imports and configuration can themselves initialize CUDA. The
`cuda_initialized_before_device_setup` field records whether the explicit
device-setup interval begins with an existing context; import time includes
any initialization side effects.
It verifies output shape and finiteness; those checks alone do not establish
parity. Compare the saved tensors against an independent unchanged-source run.

Run the script in independent processes to measure repeated startup. Record the
source revision, checkpoint revision, hardware, runtime versions, cache state,
and process wall time alongside the JSON. The script excludes checkpoint
downloads. Cached local files do not establish a cold filesystem-cache run.

## Collect component timings in a worker

```python
from fastvideo.models.loader.startup import collect_startup_timings

with collect_startup_timings() as timings:
    pipeline = build_pipeline(...)  # Your existing construction call.
report = timings.to_dict()
```

Collection is opt-in and local to the current context/process. Install a
collector inside each spawned worker to measure its component loads; wrapping
only a multiprocess driver does not collect worker timings. Lazy components are
recorded when they actually load, if collection is still active.

Each phase has an ID, parent ID, start offset, duration, and success/error status.
Parent durations include child durations: do not sum both. Recording duration
also includes gaps between phases. A failed phase propagates its exception and
records an error; the collector restores the previous context on exit.

Default durations measure host wall time, including asynchronous submission.
For GPU completion timing, supply a synchronization callback. The example uses a
callback that synchronizes only after CUDA is initialized, so the first callback
does not move context initialization out of the device-setup interval.
Synchronization changes overlap and adds overhead. Compare end-to-end latency
with an uninstrumented run; do not interpret instrumented timings as free.

## Native VAE boundaries

`VAELoader.load()` retains the existing disk-loading behavior. Internally it now
separates two operations:

- `build_model()` constructs the native module and moves its initial tensors to
  the target device. The caller supplies the resolved class, architecture config,
  and dtype context.
- `load_weights()` reads safetensors, preserves checkpoint remapping and loading
  strictness, assigns weights, applies existing overlays/memory-format changes,
  and switches the module to evaluation mode.

Phases are `vae.construct`, `vae.device_transfer`, `vae.checkpoint_read`,
`vae.weight_assignment`, and `vae.finalize`, nested inside `component.vae` when
called through `PipelineComponentLoader`. Checkpoint reads can map files lazily;
page faults and transfers can occur during assignment. These labels describe
loader operations, not independent physical disk-I/O and PCIe measurements.

Checkpoint-backed special constructors retain their existing paths and only
receive the component-total interval. Transformer/text-encoder internals,
compilation, and kernel warm-up need further instrumentation. Reusing GPU tensor
storage across processes will require a separate compatibility manifest,
ownership/lifetime protocol, and loader implementation. The extracted `load_weights()` method still copies from disk; a future IPC
loader must bind storage instead of calling `load_state_dict()` and claiming
zero-copy attachment.
