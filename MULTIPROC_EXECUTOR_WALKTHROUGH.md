# FastVideo Multiprocessing Architecture: Complete Walkthrough

This document traces the execution flow of `test_longcat_set_lora_adapter.py`, explaining how FastVideo's multiprocessing architecture works from the outermost test script down to the GPU worker processes.

---

## Table of Contents
1. [Architecture Overview](#architecture-overview)
2. [Layer 1: Test Script](#layer-1-test-script-test_longcat_set_lora_adapterpy)
3. [Layer 2: VideoGenerator](#layer-2-videogenerator)
4. [Layer 3: MultiprocExecutor](#layer-3-multiprocexecutor)
5. [Layer 4: Worker Process](#layer-4-worker-process)
6. [Layer 5: GPU Worker](#layer-5-gpu-worker)
7. [The Bug and Fix](#the-bug-and-fix)
8. [Complete Execution Trace](#complete-execution-trace)

---

## Architecture Overview

FastVideo uses a **multiprocessing architecture** where:
- **Main Process**: Runs your script, coordinates work
- **Worker Processes**: Run on each GPU, do the actual computation
- **RPC (Remote Procedure Call)**: Main process sends commands to workers via pipes

```
┌─────────────────────────────────────────────────────────────┐
│ Main Process (your script)                                  │
│  test_longcat_set_lora_adapter.py                          │
│         │                                                    │
│         ▼                                                    │
│  VideoGenerator                                             │
│         │                                                    │
│         ▼                                                    │
│  MultiprocExecutor ──────RPC over pipes──────►              │
└─────────────────────────────────────────────────────────────┘
                                                ▼
                         ┌──────────────────────────────────────┐
                         │ Worker Process (separate Python proc)│
                         │  WorkerMultiprocProc.worker_main()  │
                         │         │                            │
                         │         ▼                            │
                         │  worker_busy_loop()                 │
                         │  (receives RPC calls)               │
                         │         │                            │
                         │         ▼                            │
                         │  Worker.execute_method()            │
                         │         │                            │
                         │         ▼                            │
                         │  GPU computation on CUDA:0          │
                         └──────────────────────────────────────┘
```

---

## Layer 1: Test Script (`test_longcat_set_lora_adapter.py`)

### What It Does
The test script is the outermost layer - it's what YOU run. It:
1. Creates a `VideoGenerator` (without LoRA initially)
2. Calls `set_lora_adapter()` to load LoRA weights
3. Calls `generate_video()` to create a video

### Code Flow

#### Step 1: Create VideoGenerator (lines 45-51)

```python
generator = VideoGenerator.from_pretrained(
    model_path,                      # "weights/longcat-native"
    num_gpus=num_gpus,              # 1
    use_fsdp_inference=(num_gpus > 1),  # False (only 1 GPU)
    dit_cpu_offload=False,          # Keep model on GPU
)
```

**What happens here:**
- `VideoGenerator.from_pretrained()` is a factory method
- It reads the model config and creates the appropriate pipeline
- Initializes worker processes (one per GPU)
- Returns a ready-to-use generator object

**Logs from Step 1:**
```
Step 1: Creating VideoGenerator without LoRA...
INFO [utils.py:600] Diffusers version: 0.32.0
...
INFO [multiproc_executor.py:41] Use master port: 59377
...
INFO [multiproc_executor.py:446] 1 workers ready
✓ Generator initialized in 25.88s
```

The key log is `"1 workers ready"` - this means the worker process is running and waiting for commands.

---

#### Step 2: Load LoRA via `set_lora_adapter()` (lines 63-68)

```python
generator.set_lora_adapter(
    lora_nickname="distilled",               # Internal name for this LoRA
    lora_path=lora_path                      # "weights/longcat-native/lora/distilled"
)
```

**What happens here:**
- This method sends a command to the worker process
- Worker converts model layers to LoRA layers (if not already)
- Worker loads LoRA weights from disk
- Worker merges LoRA weights into the model

**Logs from Step 2:**
```
Step 2: Loading LoRA using set_lora_adapter()...
INFO [multiproc_executor.py:123] 🔍 Main process: collective_rpc called with method=set_lora_adapter, type=str
...
✓ LoRA loaded in 12.73s
```

We'll trace this in detail in Layer 3.

---

#### Step 3: Generate Video (lines 89-104)

```python
video = generator.generate_video(
    prompt=prompt,
    height=480,
    width=832,
    num_frames=93,
    num_inference_steps=16,     # Fast! (distilled LoRA allows this)
    guidance_scale=1.0,         # Distilled uses guidance_scale=1.0
    seed=seed,
    output_path=str(output_path),
    save_video=True,
    return_frames=True,
)
```

**What happens here:**
- Main process sends another RPC: `execute_forward`
- Worker runs the diffusion model 16 times (16 steps)
- Returns the generated latents
- Main process decodes to video and saves to disk

**Logs from Step 3:**
```
Step 3: Generating video...
...
[Worker] LongCat Denoising: 100%|██████████| 16/16 [01:33<00:00, 5.86s/it]
...
✓ Generated in 98.71s → outputs/output_t2v_set_lora_adapter.mp4
```

---

## Layer 2: VideoGenerator

### Location: `fastvideo/video_generator.py`

The `VideoGenerator` is a high-level API that hides the complexity of multiprocessing.

### Key Methods

#### `from_pretrained()` (factory method)

```python
@classmethod
def from_pretrained(cls, model_path, num_gpus=1, ...):
    # 1. Create FastVideoArgs (configuration object)
    args = FastVideoArgs.from_kwargs(model_path=model_path, num_gpus=num_gpus, ...)
    
    # 2. Create executor (multiprocessing manager)
    executor = MultiprocExecutor(fastvideo_args=args)
    
    # 3. Return generator that wraps the executor
    return cls(executor=executor, ...)
```

This is why you see logs about creating workers - `MultiprocExecutor.__init__()` spawns the worker processes.

---

#### `set_lora_adapter()` (delegates to executor)

```python
def set_lora_adapter(self, lora_nickname: str, lora_path: str | None = None):
    # Simply forward the call to the executor
    self.executor.set_lora_adapter(lora_nickname, lora_path)
```

**Key insight:** `VideoGenerator` is just a wrapper! The real work happens in the executor.

---

## Layer 3: MultiprocExecutor

### Location: `fastvideo/worker/multiproc_executor.py`

This is the **central coordination layer** that manages worker processes.

### Initialization: `_init_executor()` (lines 31-68)

Called when you create `VideoGenerator.from_pretrained()`.

```python
def _init_executor(self) -> None:
    self.world_size = self.fastvideo_args.num_gpus  # 1 in our case
    
    # Get a free port for inter-process communication
    master_port = get_open_port(...)
    
    # Create worker processes (one per GPU)
    for rank in range(self.world_size):  # rank=0 (only one worker)
        unready_workers.append(
            WorkerMultiprocProc.make_worker_process(
                fastvideo_args=self.fastvideo_args,
                local_rank=rank,     # 0
                rank=rank,           # 0
                distributed_init_method=distributed_init_method,
            ))
    
    # Wait for workers to initialize
    self.workers = WorkerMultiprocProc.wait_for_ready(unready_workers)
```

**What `make_worker_process()` does (lines 312-338):**

```python
@staticmethod
def make_worker_process(...) -> UnreadyWorkerProcHandle:
    context = get_mp_context()  # Get multiprocessing context
    
    # Create a two-way pipe for communication
    executor_pipe, worker_pipe = context.Pipe(duplex=True)
    
    # Create a separate ready signal pipe
    reader, writer = context.Pipe(duplex=False)
    
    # Spawn a new Python process
    proc = context.Process(
        target=WorkerMultiprocProc.worker_main,  # Entry point in worker
        kwargs={...},
        name=f"FVWorkerProc-{rank}",
        daemon=True
    )
    
    proc.start()  # 🚀 Worker process starts running!
    
    return UnreadyWorkerProcHandle(proc, rank, executor_pipe, reader)
```

**Logs:**
```
INFO [multiproc_executor.py:41] Use master port: 59377
[Worker pid=1487771] INFO [parallel_state.py:976] Initializing distributed environment...
...
INFO [multiproc_executor.py:446] 1 workers ready
```

The `[Worker pid=1487771]` prefix indicates this log comes from the worker process, not the main process!

---

### RPC Communication: `collective_rpc()` (lines 116-148)

This is the **message passing system** that sends commands to workers.

```python
def collective_rpc(self,
                   method: str | Callable,  # e.g., "set_lora_adapter"
                   timeout: float | None = None,
                   args: tuple = (),
                   kwargs: dict | None = None) -> list[Any]:
    
    kwargs = kwargs or {}
    
    logger.info("🔍 Main process: collective_rpc called with method=%s, type=%s", 
               method, type(method).__name__)
    
    # Step 1: Send command to ALL workers
    for worker in self.workers:
        logger.info("🔍 Main process: sending RPC to worker %d", worker.rank)
        worker.pipe.send({              # 📤 Send over pipe
            "method": method,            # "set_lora_adapter"
            "args": args,                # ()
            "kwargs": kwargs             # {"lora_nickname": "distilled", "lora_path": "..."}
        })
    
    logger.info("🔍 Main process: sent RPC to all %d workers, now waiting for responses...", 
               len(self.workers))
    
    # Step 2: Wait for response from ALL workers
    responses = []
    for worker in self.workers:
        logger.info("🔍 Main process: waiting for response from worker %d...", worker.rank)
        response = worker.pipe.recv()   # 📥 BLOCKS until worker responds!
        logger.info("🔍 Main process: received response from worker %d: %s", 
                   worker.rank, type(response).__name__)
        responses.append(response)
    
    logger.info("🔍 Main process: received all responses")
    return responses
```

**Key Concept: Pipes**

`worker.pipe` is a `multiprocessing.Pipe` - a two-way communication channel between processes.

```
Main Process                Worker Process
    |                             |
    | pipe.send({"method":        |
    |   "set_lora_adapter"})      |
    |─────────────────────────────>│
    |                             | pipe.recv()  ← Gets the message
    |                             | (processes it)
    |                             |
    |         pipe.recv()  ←──────| pipe.send({"status": "lora_adapter_set"})
    |  ← BLOCKS here until        |
    |     worker responds          |
```

**Logs for set_lora_adapter:**
```
INFO [multiproc_executor.py:123] 🔍 Main process: collective_rpc called with method=set_lora_adapter, type=str
INFO [multiproc_executor.py:128] 🔍 Main process: sending RPC to worker 0
INFO [multiproc_executor.py:134] 🔍 Main process: sent RPC to all 1 workers, now waiting for responses...
INFO [multiproc_executor.py:139] 🔍 Main process: waiting for response from worker 0...
```

At this point, the main process is **BLOCKED** waiting for the worker to respond!

---

### Specific RPC Methods

#### `set_lora_adapter()` (lines 91-102)

```python
def set_lora_adapter(self,
                     lora_nickname: str,
                     lora_path: str | None = None) -> None:
    # Call the generic RPC mechanism
    responses = self.collective_rpc("set_lora_adapter",
                                    kwargs={
                                        "lora_nickname": lora_nickname,
                                        "lora_path": lora_path
                                    })
    
    # Verify all workers succeeded
    for i, response in enumerate(responses):
        if response["status"] != "lora_adapter_set":
            raise RuntimeError(
                f"Worker {i} failed to set LoRA adapter to {lora_path}")
```

The method name `"set_lora_adapter"` is sent as a **string**, not the actual method. The worker will look up the method by name.

---

## Layer 4: Worker Process

### Entry Point: `worker_main()` (lines 341-399)

This is a **static method** that runs in the **separate worker process**.

```python
@staticmethod
def worker_main(*args, **kwargs):
    """Worker initialization and execution loops.
    This runs in a background process"""
    
    # Setup signal handlers for graceful shutdown
    shutdown_requested = False
    
    def signal_handler(signum, frame):
        nonlocal shutdown_requested
        if not shutdown_requested:
            shutdown_requested = True
            raise SystemExit()
    
    signal.signal(signal.SIGTERM, signal_handler)
    signal.signal(signal.SIGINT, signal_handler)
    
    ready_pipe = kwargs.pop("ready_pipe")
    rank = kwargs.get("rank")
    
    try:
        # Step 1: Initialize the worker
        worker = WorkerMultiprocProc(*args, **kwargs)
        
        # Step 2: Send READY signal to main process
        ready_pipe.send({
            "status": WorkerMultiprocProc.READY_STR,  # "READY"
        })
        ready_pipe.close()
        
        # Step 3: Enter busy loop (never returns until shutdown)
        worker.worker_busy_loop()  # 🔁 Infinite loop!
        
    except Exception:
        logger.exception("WorkerMultiprocProc failed.")
        # Signal parent that we crashed
        parent_process.send_signal(signal.SIGQUIT)
```

**Logs:**
```
[Worker pid=1487771] INFO [parallel_state.py:976] Initializing distributed environment with world_size=1, device=cuda:0
...
[Worker pid=1487771] INFO [multiproc_executor.py:455] Worker 0 starting event loop...
```

The `[Worker pid=1487771]` prefix is added by the logging system to distinguish worker logs from main process logs.

---

### The Busy Loop: `worker_busy_loop()` (lines 442-511)

This is where the **worker waits for and handles RPC calls**.

```python
def worker_busy_loop(self) -> None:
    """Main busy loop for Multiprocessing Workers"""
    while True:  # ♾️ Infinite loop
        logger.info("Worker %d starting event loop...", self.rank)
        
        try:
            # Step 1: Wait for RPC call from main process
            rpc_call = self.pipe.recv()  # 📥 BLOCKS until message arrives
            method = rpc_call.get("method")
            args = rpc_call.get("args", ())
            kwargs = rpc_call.get("kwargs", {})
            
            logger.info("🔍 Worker %d received RPC call: method=%s, isinstance(method, str)=%s", 
                       self.rank, method, isinstance(method, str))
            
            # Step 2: Route based on method type
            if isinstance(method, str):
                logger.info("🔍 Worker %d: method is string, checking which method...", self.rank)
                
                # Special case: shutdown
                if method == "shutdown":
                    logger.info("🔍 Worker %d: handling shutdown", self.rank)
                    response = self.shutdown()
                    with contextlib.suppress(Exception):
                        self.pipe.send(response)
                    break  # Exit loop
                
                # Special case: execute_forward (optimized path)
                if method == 'execute_forward':
                    logger.info("🔍 Worker %d: handling execute_forward", self.rank)
                    forward_batch = kwargs['forward_batch']
                    fastvideo_args = kwargs['fastvideo_args']
                    output_batch = self.worker.execute_forward(
                        forward_batch, fastvideo_args)
                    self.pipe.send({
                        "output_batch": output_batch.output.cpu(),
                        "logging_info": ...
                    })
                    logger.info("🔍 Worker %d: execute_forward response sent", self.rank)
                
                else:
                    # ✅ THE FIX: Handle other string methods
                    logger.info("🔍 Worker %d: method='%s' not shutdown/execute_forward, calling execute_method", 
                               self.rank, method)
                    result = self.worker.execute_method(method, *args, **kwargs)
                    self.pipe.send(result)  # 📤 Send response back
                    logger.info("🔍 Worker %d: execute_method response sent for '%s'", self.rank, method)
            
            else:
                # Method is a callable (function/method object)
                logger.info("🔍 Worker %d: method is callable, calling execute_method", self.rank)
                result = self.worker.execute_method(method, *args, **kwargs)
                self.pipe.send(result)
                logger.info("🔍 Worker %d: execute_method response sent", self.rank)
                
        except KeyboardInterrupt:
            logger.error("Worker %d in loop received KeyboardInterrupt", self.rank)
            self.pipe.send({"error": "Operation aborted by KeyboardInterrupt"})
            continue
```

**Logs when `set_lora_adapter` is received:**
```
[Worker pid=1487771] INFO [multiproc_executor.py:462] 🔍 Worker 0 received RPC call: method=set_lora_adapter, isinstance(method, str)=True
[Worker pid=1487771] INFO [multiproc_executor.py:466] 🔍 Worker 0: method is string, checking which method...
[Worker pid=1487771] INFO [multiproc_executor.py:489] 🔍 Worker 0: method='set_lora_adapter' not shutdown/execute_forward, calling execute_method
```

Now it calls `self.worker.execute_method("set_lora_adapter", ...)`.

---

## Layer 5: GPU Worker

### Location: `fastvideo/worker/gpu_worker.py`

The `Worker` class handles the actual GPU computation.

### `execute_method()` (generic method dispatcher)

```python
def execute_method(self, method: str | Callable, *args, **kwargs) -> dict:
    """Execute a method by name or callable on the worker"""
    
    if isinstance(method, str):
        # Look up method by name
        if hasattr(self.pipeline, method):
            fn = getattr(self.pipeline, method)
        elif hasattr(self, method):
            fn = getattr(self, method)
        else:
            return {"status": f"failed: method {method} not found"}
    else:
        fn = method
    
    # Call the method
    result = fn(*args, **kwargs)
    return result
```

For `method="set_lora_adapter"`, it does:
```python
fn = getattr(self.pipeline, "set_lora_adapter")
result = fn(lora_nickname="distilled", lora_path="weights/longcat-native/lora/distilled")
```

This calls into the **pipeline's** `set_lora_adapter` method.

---

### Pipeline: `LoRAPipeline.set_lora_adapter()`

### Location: `fastvideo/pipelines/lora_pipeline.py`

```python
def set_lora_adapter(self,
                     lora_nickname: str,
                     lora_path: str | None = None):
    """
    Load a LoRA adapter into the pipeline and merge it into the transformer.
    """
    
    # Step 1: Convert layers to LoRA if not already done
    if not self.lora_initialized:
        self.convert_to_lora_layers()  # Wraps Linear layers with LoRA capability
    
    # Step 2: Load LoRA weights from disk
    if lora_path is not None:
        lora_local_path = maybe_download_lora(lora_path)
        lora_state_dict = load_file(lora_local_path)  # Load .safetensors
        
        # Map weight names from file to model layer names
        for name, weight in lora_state_dict.items():
            # Extract lora_A and lora_B weights
            if ".lora_A" in name:
                self.lora_adapters[lora_nickname][name] = weight
            elif ".lora_B" in name:
                self.lora_adapters[lora_nickname][name] = weight
    
    # Step 3: Merge LoRA weights into model
    for name, layer in self.lora_layers.items():
        lora_A_name = name + ".lora_A"
        lora_B_name = name + ".lora_B"
        if lora_A_name in self.lora_adapters[lora_nickname]:
            layer.set_lora_weights(
                self.lora_adapters[lora_nickname][lora_A_name],
                self.lora_adapters[lora_nickname][lora_B_name],
                training_mode=False,  # Inference mode
                lora_path=lora_path
            )
    
    self.cur_adapter_path = lora_path
    self.cur_adapter_nickname = lora_nickname
    
    return {"status": "lora_adapter_set"}  # 📤 This gets sent back to main process
```

**Logs from this layer:**
```
[Worker pid=1487771] INFO [lora_pipeline.py:138] Converted 582 layers to LoRA layers
[Worker pid=1487771] INFO [lora_pipeline.py:232] Rank 0: loaded LoRA adapter weights/longcat-native/lora/distilled
[Worker pid=1487771] WARNING [lora_pipeline.py:262] LoRA adapter does not contain weights for layer time_embedder.linear_1
...
[Worker pid=1487771] INFO [lora_pipeline.py:266] Rank 0: LoRA adapter applied to 480 layers
```

The warnings are normal - not all layers have LoRA weights, only the ones that were trained.

---

### Return Path

```
Pipeline.set_lora_adapter()
    returns: {"status": "lora_adapter_set"}
        ↓
Worker.execute_method()
    returns: {"status": "lora_adapter_set"}
        ↓
worker_busy_loop()
    pipe.send({"status": "lora_adapter_set"})  # 📤
        ↓
    (over the pipe to main process)
        ↓
MultiprocExecutor.collective_rpc()
    response = worker.pipe.recv()  # 📥 Unblocks!
    return [{"status": "lora_adapter_set"}]
        ↓
MultiprocExecutor.set_lora_adapter()
    checks: response["status"] == "lora_adapter_set" ✅
    returns: None
        ↓
VideoGenerator.set_lora_adapter()
    returns: None
        ↓
test script
    print("✓ LoRA loaded")
```

---

## The Bug and Fix

### The Original Bug (BEFORE the fix)

The `worker_busy_loop()` had this structure:

```python
if isinstance(method, str):
    if method == "shutdown":
        # handle shutdown
        break
    if method == 'execute_forward':
        # handle execute_forward
        self.pipe.send(...)
    # ❌ NO ELSE CLAUSE!
    # If method is "set_lora_adapter", nothing happens here!
    # Loop continues to next iteration without sending response
else:
    # This only runs if method is NOT a string
    result = self.worker.execute_method(method, *args, **kwargs)
    self.pipe.send(result)
```

**What went wrong:**
1. Main process sends: `{"method": "set_lora_adapter"}` as a **string**
2. Worker receives it
3. `isinstance(method, str)` → **True**
4. `method == "shutdown"` → **False**
5. `method == "execute_forward"` → **False**
6. **No else clause!** → Falls through to bottom of `if` block
7. Loop continues to next `rpc_call = self.pipe.recv()` without sending response
8. Main process still waiting at `response = worker.pipe.recv()` → **DEADLOCK!** 🔒

### The Fix

Add an `else` clause to handle other string method names:

```python
if isinstance(method, str):
    if method == "shutdown":
        # handle shutdown
        break
    if method == 'execute_forward':
        # handle execute_forward
        self.pipe.send(...)
    else:  # ✅ THE FIX!
        # Handle other string methods like 'set_lora_adapter'
        result = self.worker.execute_method(method, *args, **kwargs)
        self.pipe.send(result)
else:
    # method is a callable
    result = self.worker.execute_method(method, *args, **kwargs)
    self.pipe.send(result)
```

Now when `method="set_lora_adapter"`:
1. `isinstance(method, str)` → **True**
2. `method == "shutdown"` → **False**
3. `method == "execute_forward"` → **False**
4. **else clause executes!** ✅
5. Calls `execute_method("set_lora_adapter", ...)`
6. Sends response back to main process
7. Main process unblocks and continues → **SUCCESS!** 🎉

---

## Complete Execution Trace

Here's the full trace with log line numbers and timing:

### Phase 1: Initialization (0s - 25.88s)

```
[Main] test_longcat_set_lora_adapter.py line 45
       generator = VideoGenerator.from_pretrained(...)
           ↓
[Main] VideoGenerator.__init__()
           ↓
[Main] MultiprocExecutor._init_executor()
       → spawns worker process
           ↓
[Worker PID=1487771] WorkerMultiprocProc.worker_main() starts
       → initializes GPU, loads model
       → sends "READY" signal
           ↓
[Main] wait_for_ready() receives "READY"
       LOG: INFO [multiproc_executor.py:446] 1 workers ready
       ✓ Generator initialized in 25.88s
```

### Phase 2: Load LoRA (25.88s - 38.61s)

```
[Main] test_longcat_set_lora_adapter.py line 63
       generator.set_lora_adapter("distilled", "weights/longcat-native/lora/distilled")
           ↓
[Main] VideoGenerator.set_lora_adapter()
           ↓
[Main] MultiprocExecutor.set_lora_adapter()
           ↓
[Main] collective_rpc("set_lora_adapter", kwargs={...})
       LOG: INFO [multiproc_executor.py:123] 🔍 Main process: collective_rpc called with method=set_lora_adapter
           ↓
[Main] worker.pipe.send({"method": "set_lora_adapter", ...})
       LOG: INFO [multiproc_executor.py:128] 🔍 Main process: sending RPC to worker 0
           ↓
[Main] worker.pipe.recv()  ← BLOCKS HERE
       LOG: INFO [multiproc_executor.py:139] 🔍 Main process: waiting for response from worker 0...
```

Meanwhile, in the worker process:

```
[Worker] worker_busy_loop() waiting at pipe.recv()
           ↓
[Worker] receives {"method": "set_lora_adapter", ...}
         LOG: INFO [multiproc_executor.py:462] 🔍 Worker 0 received RPC call: method=set_lora_adapter
           ↓
[Worker] isinstance(method, str) → True
         LOG: INFO [multiproc_executor.py:466] 🔍 Worker 0: method is string, checking which method...
           ↓
[Worker] method not shutdown or execute_forward → else clause
         LOG: INFO [multiproc_executor.py:489] 🔍 Worker 0: method='set_lora_adapter' not shutdown/execute_forward
           ↓
[Worker] execute_method("set_lora_adapter", ...)
           ↓
[Worker] Worker.execute_method() → finds self.pipeline.set_lora_adapter
           ↓
[Worker] LoRAPipeline.set_lora_adapter()
         LOG: INFO [lora_pipeline.py:138] Converted 582 layers to LoRA layers
         LOG: INFO [lora_pipeline.py:232] Rank 0: loaded LoRA adapter
         LOG: INFO [lora_pipeline.py:266] Rank 0: LoRA adapter applied to 480 layers
           ↓
[Worker] returns {"status": "lora_adapter_set"}
           ↓
[Worker] pipe.send({"status": "lora_adapter_set"})
         LOG: INFO [multiproc_executor.py:493] 🔍 Worker 0: execute_method response sent for 'set_lora_adapter'
           ↓
[Worker] back to top of while loop, waits at pipe.recv() again
         LOG: INFO [multiproc_executor.py:455] Worker 0 starting event loop...
```

Back in main process:

```
[Main] worker.pipe.recv() ← UNBLOCKS!
       LOG: INFO [multiproc_executor.py:141] 🔍 Main process: received response from worker 0: dict
           ↓
[Main] collective_rpc() returns [{"status": "lora_adapter_set"}]
       LOG: INFO [multiproc_executor.py:144] 🔍 Main process: received all responses
           ↓
[Main] set_lora_adapter() checks status ✅
           ↓
[Main] test_longcat_set_lora_adapter.py
       ✓ LoRA loaded in 12.73s
```

### Phase 3: Generate Video (38.61s - 137.32s)

```
[Main] test_longcat_set_lora_adapter.py line 89
       generator.generate_video(...)
           ↓
[Main] collective_rpc("execute_forward", ...)
       → Similar RPC dance, but with execute_forward method
           ↓
[Worker] Runs diffusion model 16 times
         LOG: [Worker] LongCat Denoising: 100%|██████████| 16/16 [01:33<00:00, 5.86s/it]
           ↓
[Worker] Returns generated latents
           ↓
[Main] Decodes to video, saves to disk
       ✓ Generated in 98.71s → outputs/output_t2v_set_lora_adapter.mp4
```

---

## Why String Methods vs Callable Methods?

You might wonder: why does the RPC system support both `method: str` and `method: Callable`?

### Design Rationale

The `collective_rpc` signature is:
```python
def collective_rpc(self, method: str | Callable, ...) -> list[Any]:
```

This supports **two patterns** for different use cases:

---

### Pattern 1: String Methods (Common Case)

**Used for**: Predefined worker methods

**Examples**: `"set_lora_adapter"`, `"execute_forward"`, `"shutdown"`, `"merge_lora_weights"`

**How it works**:
```python
# Main process
executor.collective_rpc("set_lora_adapter", kwargs={"lora_nickname": "distilled", ...})

# Sends over pipe: {"method": "set_lora_adapter", "kwargs": {...}}

# Worker receives string "set_lora_adapter"
# Looks up method by name: getattr(self.pipeline, "set_lora_adapter")
# Calls: self.pipeline.set_lora_adapter(lora_nickname="distilled", ...)
```

**Advantages**:
- ✅ Simple and fast
- ✅ Easy to debug (method name visible in logs)
- ✅ Works across all executor backends (multiprocessing, Ray)
- ✅ No serialization overhead (just a string)

**When to use**: For all standard worker operations

---

### Pattern 2: Callable Methods (Advanced Case)

**Used for**: Custom logic not defined on the worker

**Examples**: One-off operations, testing, dynamic behavior

**How it works**:

```python
# Define a custom function
def custom_diagnostic(worker):
    """Get memory usage from worker"""
    import torch
    return {
        "gpu_memory": torch.cuda.memory_allocated(),
        "model_params": sum(p.numel() for p in worker.pipeline.parameters())
    }

# Main process
results = executor.collective_rpc(custom_diagnostic)
# results[0] = {"gpu_memory": 12884901888, "model_params": 13580000000}
```

**What happens under the hood**:

**For MultiprocExecutor** (multiprocessing):
```python
# Main process sends callable directly
worker.pipe.send({"method": custom_diagnostic, "args": (), "kwargs": {}})
# Python's Pipe automatically pickles the function

# Worker receives the callable object
# Since isinstance(method, str) == False, goes to else branch
result = self.worker.execute_method(custom_diagnostic, *args, **kwargs)
# execute_method calls: custom_diagnostic(self.worker)
```

**For RayDistributedExecutor** (Ray):
```python
# Main process serializes with cloudpickle
import cloudpickle
sent_method = cloudpickle.dumps(custom_diagnostic)  # → bytes
worker.execute_method.remote(sent_method, ...)

# Worker receives bytes
# execute_method checks: isinstance(method, bytes)
func = cloudpickle.loads(method)  # Deserialize back to function
result = func(self.worker)
```

**Advantages**:
- ✅ Ultimate flexibility - send ANY code to workers
- ✅ No need to modify worker class
- ✅ Great for debugging, experimentation, one-off tasks

**Disadvantages**:
- ❌ Harder to debug (function is serialized bytes)
- ❌ Serialization overhead (using pickle/cloudpickle)
- ❌ Can break if function uses non-picklable objects

**When to use**: Testing, debugging, custom operations not worth adding to the worker permanently

---

### Why Have Both in `worker_busy_loop()`?

Looking back at the code:

```python
if isinstance(method, str):
    if method == "shutdown":
        # Special optimized path for shutdown
    if method == 'execute_forward':
        # Special optimized path for forward pass (hot path!)
    else:
        # Generic path for other string methods
        result = self.worker.execute_method(method, *args, **kwargs)
        self.pipe.send(result)
else:
    # Callable path - for advanced users
    result = self.worker.execute_method(method, *args, **kwargs)
    self.pipe.send(result)
```

**Why special cases for `shutdown` and `execute_forward`?**

1. **`shutdown`**: Needs to break out of the loop (not just return a response)
2. **`execute_forward`**: This is the **hot path** - called hundreds of times during video generation
   - Optimized to avoid extra function lookups
   - Directly accesses `forward_batch` from kwargs
   - Custom response format for performance

**Why the else branch for other strings?**

- Handles all other worker methods generically: `set_lora_adapter`, `merge_lora_weights`, etc.
- **This was the missing piece that caused the bug!**

---

### Complete Method Resolution Flow

Here's how `execute_method` resolves a method:

**Location**: `fastvideo/worker/worker_base.py`

```python
def execute_method(self, method: str | bytes, *args, **kwargs):
    return run_method(self, method, args, kwargs)
```

**Location**: `fastvideo/utils.py`

```python
def run_method(obj: Any, method: str | bytes | Callable, args, kwargs):
    if isinstance(method, bytes):
        # Callable serialized by cloudpickle (Ray executor)
        func = partial(cloudpickle.loads(method), obj)
    elif isinstance(method, str):
        # String method name - look it up
        func = getattr(obj, method)  # e.g., getattr(worker, "set_lora_adapter")
    else:
        # Direct callable object (multiprocessing executor)
        func = partial(method, obj)  # Makes callable(obj, *args, **kwargs)
    
    return func(*args, **kwargs)
```

**Resolution order** for string methods:
1. Check if `WorkerWrapperBase` has the method (e.g., `init_worker`)
2. If not, check if `Worker` has it (via `__getattr__` delegation)
3. If not, check if `Pipeline` has it (Worker delegates to pipeline)
4. If not found anywhere: raise `NotImplementedError`

---

### Real-World Example: Why This Matters

**Scenario**: You want to inspect worker state without modifying the codebase

**Without callables** (would need to modify `Worker` class):
```python
# 1. Edit fastvideo/worker/gpu_worker.py
class Worker:
    def get_memory_stats(self):  # Add new method
        return {"gpu_mem": torch.cuda.memory_allocated()}

# 2. Restart everything
# 3. Call it
results = executor.collective_rpc("get_memory_stats")
```

**With callables** (no code changes needed):
```python
# Just define a function and send it!
def get_memory_stats(worker):
    import torch
    return {"gpu_mem": torch.cuda.memory_allocated()}

results = executor.collective_rpc(get_memory_stats)
```

This is why FastVideo supports both patterns - **strings for efficiency, callables for flexibility**.

---

## Why Ray Backend Didn't Have This Bug

**Important Discovery**: You were right that `set_lora_adapter()` worked on Ray but not multiprocessing!

### Ray Architecture (No Bug)

Ray doesn't use `worker_busy_loop()` at all! Instead:

```python
# RayDistributedExecutor
def _run_ray_workers(self, method: str | Callable, *args, **kwargs):
    # Send method to all workers via Ray's RPC system
    ray_worker_outputs = [
        worker.execute_method.remote(method, *args, **kwargs)  # Ray magic!
        for worker in ray_workers
    ]
    return ray.get(ray_worker_outputs)
```

**Key difference**: `.remote()` is Ray's RPC decorator. It **always** routes to `WorkerWrapperBase.execute_method()`, which calls `run_method()`, which correctly handles string method names via `getattr()`.

**No custom busy loop = No bug!**

### Why Multiprocessing Needed a Busy Loop

Multiprocessing uses Python's `multiprocessing.Pipe` for communication, which is just a dumb pipe. The worker needs a custom loop to:
1. Wait for messages (`pipe.recv()`)
2. Parse the method name
3. Call the appropriate handler
4. Send response back (`pipe.send()`)

This custom loop is where the bug was introduced - the missing `else` clause.

### Backend Comparison

| Feature | Multiprocessing | Ray |
|---------|----------------|-----|
| RPC System | Custom (pipes + busy loop) | Ray's built-in `.remote()` |
| Bug location | `worker_busy_loop()` | N/A (no busy loop) |
| Affected by this bug? | ✅ YES | ❌ NO |
| Multi-node support | ❌ | ✅ |
| Dependencies | Python stdlib | Ray |

See `RAY_VS_MULTIPROC_ANALYSIS.md` for detailed comparison.

---

## Summary

The FastVideo multiprocessing architecture:

1. **Main Process**: Your script runs here, coordinates work
2. **MultiprocExecutor**: Manages worker processes, sends RPC calls over pipes
3. **Worker Process**: Separate Python process per GPU
4. **worker_busy_loop**: Infinite loop waiting for RPC calls
5. **execute_method**: Dispatches method calls by name to the pipeline
6. **Pipeline**: Does actual GPU work (loading LoRA, running diffusion)

### Key Design Choices:

- **String methods**: Fast, debuggable, used for 99% of operations
- **Callable methods**: Flexible, used for custom/debugging operations
- **Special cases**: `shutdown` breaks loop, `execute_forward` optimized for hot path
- **Generic else**: Handles all other string methods uniformly

### The Bug and Fix:

The bug was a missing `else` clause in `worker_busy_loop()` that caused the worker to ignore `set_lora_adapter` calls (and any other string method except `shutdown`/`execute_forward`), resulting in a deadlock where the main process waited forever for a response.

The fix ensures all string method names are properly dispatched to `execute_method()` and send a response back to the main process.

