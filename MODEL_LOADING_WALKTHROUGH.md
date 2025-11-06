# Model Loading and Initialization Process in FastVideo

A detailed walkthrough of where and how models are loaded in FastVideo, tracing from user input to fully initialized pipeline.

---

## Table of Contents

1. [High-Level Overview](#high-level-overview)
2. [Complete Initialization Flow](#complete-initialization-flow)
3. [Detailed Component Loading](#detailed-component-loading)
4. [Code Trace with Line Numbers](#code-trace-with-line-numbers)
5. [Memory and Device Management](#memory-and-device-management)
6. [Advanced Loading Patterns](#advanced-loading-patterns)

---

## High-Level Overview

**When are models loaded?**

Models are loaded during **worker initialization**, NOT during `VideoGenerator.from_pretrained()`. The VideoGenerator creates executor/workers, and each worker loads its own copy of the model components.

**Where are models loaded?**

```
User Code
    ↓
VideoGenerator.from_pretrained()  ← Creates executor
    ↓
MultiprocExecutor._init_executor()  ← Spawns worker processes
    ↓
Worker.__init__() → Worker.init_device()  ← MODELS LOADED HERE
    ↓
build_pipeline() → WanPipeline.__init__()
    ↓
ComposedPipelineBase.load_modules()  ← ACTUAL LOADING HAPPENS
    ↓
PipelineComponentLoader.load_module()  ← Per-component loading
    ↓
[TextEncoderLoader, TransformerLoader, VAELoader, SchedulerLoader]
```

---

## Complete Initialization Flow

### Phase 1: VideoGenerator Creation (Main Process)

```python
# User code
generator = VideoGenerator.from_pretrained(
    "FastVideo/FastWan2.1-T2V-1.3B-Diffusers",
    num_gpus=2
)
```

**Step 1.1: Parse Arguments**

File: `fastvideo/entrypoints/video_generator.py`

```python
@classmethod
def from_pretrained(cls, model_path: str, **kwargs) -> "VideoGenerator":
    # Line 52-76
    kwargs['model_path'] = model_path
    fastvideo_args = FastVideoArgs.from_kwargs(**kwargs)
    # Creates FastVideoArgs with:
    # - model_path: "FastVideo/FastWan2.1-T2V-1.3B-Diffusers"
    # - num_gpus: 2
    # - pipeline_config: loaded from model's config.json
    
    return cls.from_fastvideo_args(fastvideo_args)
```

**Step 1.2: Get Executor Class**

```python
@classmethod
def from_fastvideo_args(cls, fastvideo_args: FastVideoArgs) -> "VideoGenerator":
    # Line 78-98
    executor_class = Executor.get_class(fastvideo_args)
    # Returns: MultiprocExecutor (for distributed_executor_backend="mp")
    
    return cls(
        fastvideo_args=fastvideo_args,
        executor_class=executor_class,
        log_stats=False,
    )
```

**Step 1.3: Create Executor**

```python
def __init__(self, fastvideo_args: FastVideoArgs, executor_class: type[Executor], ...):
    # Line 39-49
    self.fastvideo_args = fastvideo_args
    self.executor = executor_class(fastvideo_args)  # Spawns workers here!
```

**At this point:**
- ✅ VideoGenerator created
- ✅ Executor initialized
- ✅ Worker processes spawned
- ❌ Models NOT loaded yet (happens in workers)

---

### Phase 2: Executor Initialization (Spawns Workers)

File: `fastvideo/worker/multiproc_executor.py`

```python
class MultiprocExecutor(Executor):
    def _init_executor(self) -> None:
        # Line 31-69
        self.world_size = self.fastvideo_args.num_gpus  # 2
        
        # Get master port for distributed communication
        master_port = get_open_port(self.fastvideo_args.master_port)
        distributed_init_method = get_distributed_init_method(
            get_loopback_ip(), master_port
        )
        # e.g., "tcp://127.0.0.1:29500"
        
        # Spawn one worker process per GPU
        unready_workers: list[UnreadyWorkerProcHandle] = []
        for rank in range(self.world_size):  # rank = 0, 1
            unready_workers.append(
                WorkerMultiprocProc.make_worker_process(
                    fastvideo_args=self.fastvideo_args,
                    local_rank=rank,
                    rank=rank,
                    distributed_init_method=distributed_init_method,
                )
            )
        
        # Wait for all workers to be ready
        self.workers = WorkerMultiprocProc.wait_for_ready(unready_workers)
```

**What `make_worker_process` does:**

```python
@staticmethod
def make_worker_process(...) -> UnreadyWorkerProcHandle:
    # Line 312-338
    context = get_mp_context()  # Usually "spawn"
    executor_pipe, worker_pipe = context.Pipe(duplex=True)
    reader, writer = context.Pipe(duplex=False)
    
    process_kwargs = {
        "fastvideo_args": fastvideo_args,
        "local_rank": local_rank,
        "rank": rank,
        "distributed_init_method": distributed_init_method,
        "pipe": worker_pipe,
        "ready_pipe": writer,
    }
    
    # Start worker process (runs worker_main in new process)
    proc = context.Process(
        target=WorkerMultiprocProc.worker_main,
        kwargs=process_kwargs,
        name=f"FVWorkerProc-{rank}",
        daemon=True
    )
    proc.start()
    
    return UnreadyWorkerProcHandle(proc, rank, executor_pipe, reader)
```

**At this point:**
- ✅ 2 worker processes spawned
- ✅ Communication pipes set up
- ⏳ Workers are initializing (in parallel)
- ❌ Models still not loaded

---

### Phase 3: Worker Initialization (IN WORKER PROCESS)

Each worker process runs `worker_main`:

File: `fastvideo/worker/multiproc_executor.py`

```python
@staticmethod
def worker_main(*args, **kwargs):
    # Line 341-399
    
    # Set up signal handlers for graceful shutdown
    signal.signal(signal.SIGTERM, signal_handler)
    signal.signal(signal.SIGINT, signal_handler)
    
    ready_pipe = kwargs.pop("ready_pipe")
    rank = kwargs.get("rank")
    
    # Create the worker instance
    worker = WorkerMultiprocProc(*args, **kwargs)
    # This calls __init__, which creates Worker and loads models!
    
    # Signal to executor that we're ready
    ready_pipe.send({"status": "READY"})
    ready_pipe.close()
    
    # Enter busy loop waiting for RPC calls
    worker.worker_busy_loop()
```

**Worker initialization:**

```python
def __init__(self, fastvideo_args, local_rank, rank, distributed_init_method, pipe):
    # Line 282-309
    self.rank = rank
    self.pipe = pipe
    
    # Create wrapper
    wrapper = WorkerWrapperBase(
        fastvideo_args=fastvideo_args,
        rpc_rank=rank
    )
    
    # Initialize the worker (THIS CREATES THE ACTUAL WORKER)
    all_kwargs = [{} for _ in range(fastvideo_args.num_gpus)]
    all_kwargs[rank] = {
        "fastvideo_args": fastvideo_args,
        "local_rank": local_rank,
        "rank": rank,
        "distributed_init_method": distributed_init_method,
    }
    wrapper.init_worker(all_kwargs)  # Creates Worker instance
    self.worker = wrapper
    
    # Initialize device (sets up GPU, torch.distributed)
    self.worker.init_device()  # MODELS LOADED HERE!
```

---

### Phase 4: Worker Device and Model Initialization

File: `fastvideo/worker/gpu_worker.py`

```python
class Worker:
    def __init__(self, fastvideo_args, local_rank, rank, distributed_init_method):
        # Line 20-36
        self.fastvideo_args = fastvideo_args
        self.local_rank = local_rank
        self.rank = rank
        self.distributed_init_method = distributed_init_method
    
    def init_device(self) -> None:
        # Line 38-73
        
        # Set environment variables
        os.environ["TORCH_NCCL_AVOID_RECORD_STREAMS"] = "1"
        os.environ["LOCAL_RANK"] = str(self.local_rank)
        os.environ["RANK"] = str(self.rank)
        os.environ["WORLD_SIZE"] = str(self.fastvideo_args.num_gpus)
        
        # Get device
        self.device = get_local_torch_device()
        # e.g., cuda:0 for rank 0, cuda:1 for rank 1
        
        # Initialize torch.distributed
        maybe_init_distributed_environment_and_model_parallel(
            self.fastvideo_args.tp_size,  # Tensor parallelism size
            self.fastvideo_args.sp_size,  # Sequence parallelism size
            self.distributed_init_method  # "tcp://127.0.0.1:29500"
        )
        
        # *** BUILD AND LOAD THE PIPELINE ***
        self.pipeline = build_pipeline(self.fastvideo_args)
        # THIS IS WHERE MODELS ARE ACTUALLY LOADED!
```

**At this point (after `build_pipeline`):**
- ✅ Worker process initialized
- ✅ GPU device set up (cuda:0 or cuda:1)
- ✅ torch.distributed initialized
- ✅ **MODELS LOADED** (see next section)

---

### Phase 5: Pipeline Building and Model Loading

File: `fastvideo/pipelines/__init__.py`

```python
def build_pipeline(fastvideo_args: FastVideoArgs) -> PipelineWithLoRA:
    # Line 28-69
    
    # Download model if needed (from HuggingFace)
    model_path = maybe_download_model(fastvideo_args.model_path)
    # e.g., ~/.cache/huggingface/hub/models--FastVideo--FastWan2.1-T2V-1.3B-Diffusers
    
    # Read model_index.json to determine pipeline class
    config = verify_model_config_and_directory(model_path)
    pipeline_name = config.get("_class_name")
    # e.g., "WanVideoPipeline"
    
    # Get pipeline registry and resolve pipeline class
    pipeline_registry = get_pipeline_registry(PipelineType.BASIC)
    pipeline_cls = pipeline_registry.resolve_pipeline_cls(
        pipeline_name,
        PipelineType.BASIC,
        fastvideo_args.workload_type  # T2V
    )
    # Returns: WanPipeline class
    
    # Instantiate the pipeline (THIS LOADS MODELS!)
    pipeline = pipeline_cls(model_path, fastvideo_args)
    
    return pipeline
```

**Pipeline instantiation:**

File: `fastvideo/pipelines/composed_pipeline_base.py`

```python
class ComposedPipelineBase(ABC):
    def __init__(self, model_path, fastvideo_args, ...):
        # Line 49-86
        
        self.fastvideo_args = fastvideo_args
        self.model_path = model_path
        self._stages = []
        self._stage_name_mapping = {}
        
        # Initialize distributed environment (if not already done)
        maybe_init_distributed_environment_and_model_parallel(
            fastvideo_args.tp_size,
            fastvideo_args.sp_size
        )
        
        # *** LOAD ALL MODULES ***
        logger.info("Loading pipeline modules...")
        self.modules = self.load_modules(fastvideo_args, loaded_modules)
        # THIS IS WHERE THE ACTUAL MODEL LOADING HAPPENS!
```

---

## Detailed Component Loading

### The `load_modules` Method

File: `fastvideo/pipelines/composed_pipeline_base.py`

```python
def load_modules(self, fastvideo_args, loaded_modules=None) -> dict[str, Any]:
    # Line 255-358
    
    # Load model_index.json
    model_index = self._load_config(self.model_path)
    # Contents:
    # {
    #   "_class_name": "WanVideoPipeline",
    #   "_diffusers_version": "0.31.0",
    #   "scheduler": ["diffusers", "FlowMatchEulerDiscreteScheduler"],
    #   "text_encoder": ["transformers", "T5EncoderModel"],
    #   "tokenizer": ["transformers", "T5Tokenizer"],
    #   "transformer": ["diffusers", "WanTransformer3DModel"],
    #   "vae": ["diffusers", "AutoencoderKLWan"]
    # }
    
    # Get required modules from pipeline class
    required_modules = self.required_config_modules
    # For WanPipeline: ["text_encoder", "tokenizer", "vae", "transformer", "scheduler"]
    
    modules = {}
    for module_name, (library, architecture) in model_index.items():
        if module_name not in required_modules:
            continue
        
        # Path to this specific module
        component_model_path = os.path.join(self.model_path, module_name)
        # e.g., "~/.cache/.../FastWan2.1-T2V-1.3B-Diffusers/text_encoder"
        
        # Load the module
        module = PipelineComponentLoader.load_module(
            module_name=module_name,
            component_model_path=component_model_path,
            transformers_or_diffusers=library,
            fastvideo_args=fastvideo_args,
        )
        
        modules[module_name] = module
    
    return modules
```

### Component Loading: PipelineComponentLoader

File: `fastvideo/models/loader/component_loader.py`

```python
class PipelineComponentLoader:
    @staticmethod
    def load_module(module_name, component_model_path, 
                   transformers_or_diffusers, fastvideo_args):
        # Line 579-608
        
        # Get the appropriate loader for this module type
        loader = ComponentLoader.for_module_type(
            module_name,
            transformers_or_diffusers
        )
        # Returns:
        # - TextEncoderLoader for "text_encoder"
        # - TransformerLoader for "transformer"
        # - VAELoader for "vae"
        # - SchedulerLoader for "scheduler"
        # - TokenizerLoader for "tokenizer"
        
        # Load the module
        module = loader.load(component_model_path, fastvideo_args)
        
        return module
```

---

## Detailed Loading: Each Component Type

### 1. Text Encoder Loading (T5)

File: `fastvideo/models/loader/component_loader.py`

```python
class TextEncoderLoader(ComponentLoader):
    def load(self, model_path: str, fastvideo_args: FastVideoArgs):
        # Line 165-271
        
        # Get configuration from model config
        config = get_diffusers_config(model=model_path)
        # config.json contains T5 architecture details
        
        # Determine precision
        precision = fastvideo_args.pipeline_config.text_encoder_precisions[0]
        target_dtype = PRECISION_TO_TYPE[precision]  # Usually torch.float32
        
        # Get encoder class from config
        encoder_config = EncoderConfig(...)
        encoder_cls, _ = ModelRegistry.resolve_model_cls(encoder_config.prefix)
        # Returns: T5EncoderModel class
        
        # Determine device
        if fastvideo_args.text_encoder_cpu_offload:
            device = torch.device("cpu")
        else:
            device = get_local_torch_device()  # cuda:0 or cuda:1
        
        # Initialize model with config
        model = encoder_cls(encoder_config)
        
        # Load weights from safetensors
        hf_weights_files, use_safetensors = self._prepare_weights(
            model_path, fall_back_to_pt=True
        )
        # Finds: ["model.safetensors"] or ["model-00001-of-00002.safetensors", ...]
        
        if use_safetensors:
            weights_iterator = safetensors_weights_iterator(hf_weights_files)
        else:
            weights_iterator = pt_weights_iterator(hf_weights_files)
        
        # Load weights shard by shard
        for name, loaded_weight in weights_iterator:
            # Map weight names if needed
            if name in model.state_dict():
                param = model.state_dict()[name]
                param.copy_(loaded_weight)
        
        # Move to device and set precision
        model = model.to(device).to(target_dtype).eval()
        
        return model
```

**Result:**
- T5-XXL encoder loaded
- 4.6B parameters
- Device: CPU or GPU (based on cpu_offload setting)
- Precision: FP32 (default for T5)
- Memory: ~18 GB (FP32) or ~9 GB (FP16)

### 2. Transformer (DiT) Loading

File: `fastvideo/models/loader/component_loader.py`

```python
class TransformerLoader(ComponentLoader):
    def load(self, model_path: str, fastvideo_args: FastVideoArgs):
        # Line 429-513
        
        # Get config
        config = get_diffusers_config(model=model_path)
        cls_name = config.pop("_class_name")
        # e.g., "WanTransformer3DModel"
        
        # Get model class
        model_cls, _ = ModelRegistry.resolve_model_cls(cls_name)
        
        # Find safetensors files
        safetensors_list = glob.glob(
            os.path.join(model_path, "*.safetensors")
        )
        # e.g., ["diffusion_pytorch_model.safetensors"] or
        #       ["diffusion_pytorch_model-00001-of-00003.safetensors", ...]
        
        # Precision
        default_dtype = PRECISION_TO_TYPE[
            fastvideo_args.pipeline_config.dit_precision
        ]  # Usually torch.bfloat16
        
        # Load with FSDP wrapper (for large models)
        model = maybe_load_fsdp_model(
            model_cls=model_cls,
            init_params={
                "config": dit_config,
                "hf_config": hf_config
            },
            weight_dir_list=safetensors_list,
            device=get_local_torch_device(),
            hsdp_replicate_dim=fastvideo_args.hsdp_replicate_dim,
            hsdp_shard_dim=fastvideo_args.hsdp_shard_dim,
            cpu_offload=fastvideo_args.dit_cpu_offload,
            pin_cpu_memory=fastvideo_args.pin_cpu_memory,
            fsdp_inference=fastvideo_args.use_fsdp_inference,
            default_dtype=default_dtype,
            param_dtype=torch.bfloat16,
            reduce_dtype=torch.float32,
            training_mode=fastvideo_args.training_mode,
            enable_torch_compile=fastvideo_args.enable_torch_compile,
        )
        
        total_params = sum(p.numel() for p in model.parameters())
        logger.info("Loaded model with %.2fB parameters", total_params / 1e9)
        # Output: "Loaded model with 1.30B parameters"
        
        return model.eval()
```

**What is FSDP?**

FSDP (Fully Sharded Data Parallel) allows loading models larger than GPU memory by:
1. Sharding model weights across GPUs
2. Only loading needed weights for each layer
3. Prefetching next layer while computing current layer

**For Wan 1.3B model:**
- Parameters: 1.3B
- Precision: BF16
- Memory per GPU: ~2.6 GB (weights only)
- With FSDP: can run on 1× 4GB GPU
- Without FSDP: needs ~4-6 GB GPU

### 3. VAE Loading

File: `fastvideo/models/loader/component_loader.py`

```python
class VAELoader(ComponentLoader):
    def load(self, model_path: str, fastvideo_args: FastVideoArgs):
        # Line 378-423
        
        # Get config
        config = get_diffusers_config(model=model_path)
        vae_config = fastvideo_args.pipeline_config.vae_config
        vae_config.update_arch_config(config)
        
        class_name = config.pop("_class_name")
        # e.g., "AutoencoderKLWan"
        
        # Determine device
        if fastvideo_args.vae_cpu_offload:
            target_device = torch.device("cpu")
        else:
            target_device = get_local_torch_device()
        
        # Get precision
        vae_precision = fastvideo_args.pipeline_config.vae_precision
        # Usually "fp32" for better quality
        
        # Load VAE model
        with set_default_torch_dtype(PRECISION_TO_TYPE[vae_precision]):
            vae_cls, _ = ModelRegistry.resolve_model_cls(class_name)
            vae = vae_cls(vae_config).to(target_device)
        
        # Load weights
        safetensors_list = glob.glob(
            os.path.join(model_path, "*.safetensors")
        )
        loaded = safetensors_load_file(safetensors_list[0])
        vae.load_state_dict(loaded, strict=False)
        
        return vae.eval()
```

**Result:**
- VAE decoder loaded (encoder usually skipped for inference)
- Precision: FP32 (for quality)
- Device: CPU or GPU
- Memory: ~1-2 GB

### 4. Scheduler Loading

File: `fastvideo/models/loader/component_loader.py`

```python
class SchedulerLoader(ComponentLoader):
    def load(self, model_path: str, fastvideo_args: FastVideoArgs):
        # Line 516-534
        
        # Get config
        config = get_diffusers_config(model=model_path)
        class_name = config.pop("_class_name")
        # e.g., "FlowMatchEulerDiscreteScheduler"
        
        # Get scheduler class
        scheduler_cls, _ = ModelRegistry.resolve_model_cls(class_name)
        
        # Create scheduler (no weights to load)
        scheduler = scheduler_cls(**config)
        
        # Apply flow shift if specified
        if fastvideo_args.pipeline_config.flow_shift is not None:
            scheduler.set_shift(fastvideo_args.pipeline_config.flow_shift)
        
        return scheduler
```

**Result:**
- Scheduler object created
- No parameters (pure algorithm)
- Configured with flow_shift=3.0 for Wan2.1

### 5. Tokenizer Loading

File: `fastvideo/models/loader/component_loader.py`

```python
class TokenizerLoader(ComponentLoader):
    def load(self, model_path: str, fastvideo_args: FastVideoArgs):
        # Line 345-375
        
        # Use HuggingFace's AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=fastvideo_args.trust_remote_code,
            revision=fastvideo_args.revision,
        )
        
        return tokenizer
```

**Result:**
- T5Tokenizer loaded
- Vocabulary: ~32k tokens
- No GPU memory needed

---

## Timeline Summary: Wan 1.3B T2V on 2 GPUs

```
Time    Event                                           GPU 0 Memory    GPU 1 Memory
──────────────────────────────────────────────────────────────────────────────────────
0.0s    User calls VideoGenerator.from_pretrained()    0 GB            0 GB
0.1s    Executor spawns worker processes               0 GB            0 GB
        
        [Worker 0 initialization starts]
0.2s    Worker 0: init_device()                        0 GB            [starting]
0.3s    Worker 0: torch.distributed.init_process_group 0.5 GB          [starting]
0.5s    Worker 0: build_pipeline()                     0.5 GB          0.5 GB
        
        [Model loading starts - both workers load in parallel]
1.0s    Worker 0 & 1: Load tokenizer                   0.5 GB          0.5 GB
3.0s    Worker 0 & 1: Load T5 encoder                  18.5 GB         18.5 GB
        (if CPU offload: stays on CPU)
        
8.0s    Worker 0 & 1: Load Transformer (FSDP)          21.0 GB         21.0 GB
        (1.3B params, BF16, sharded)
        
10.0s   Worker 0 & 1: Load VAE decoder                 23.0 GB         23.0 GB
        (if CPU offload: stays on CPU)
        
10.5s   Worker 0 & 1: Create scheduler                 23.0 GB         23.0 GB
        (no memory)
        
11.0s   Workers send "READY" signal                    23.0 GB         23.0 GB
11.1s   Executor receives "READY" from all workers
11.2s   VideoGenerator.from_pretrained() returns       23.0 GB         23.0 GB
        
        [User now has VideoGenerator instance]
        [Models are loaded and ready in workers]
```

**Total initialization time: ~11 seconds**
**Peak memory per GPU: ~23 GB**

With CPU offloading enabled:
- T5 on CPU: saves ~18 GB
- VAE on CPU: saves ~2 GB
- Peak GPU memory: ~3 GB per GPU
- Slightly slower inference due to CPU↔GPU transfers

---

## Code Trace with Line Numbers

### Complete Call Stack

```
user_code.py
  └─ VideoGenerator.from_pretrained()
       📄 fastvideo/entrypoints/video_generator.py:52-76

     └─ FastVideoArgs.from_kwargs()
          📄 fastvideo/fastvideo_args.py:503-516

     └─ VideoGenerator.from_fastvideo_args()
          📄 fastvideo/entrypoints/video_generator.py:78-98

        └─ Executor.get_class()
             📄 fastvideo/worker/executor.py:26-37
             Returns: MultiprocExecutor

        └─ MultiprocExecutor()  [calls __init__ → _init_executor]
             📄 fastvideo/worker/multiproc_executor.py:31-69

           └─ WorkerMultiprocProc.make_worker_process()  [for each GPU]
                📄 fastvideo/worker/multiproc_executor.py:312-338
                
              └─ Process.start() → worker_main()  [IN NEW PROCESS]
                   📄 fastvideo/worker/multiproc_executor.py:341-399

                 └─ WorkerMultiprocProc.__init__()
                      📄 fastvideo/worker/multiproc_executor.py:282-309

                    └─ WorkerWrapperBase.init_worker()
                         📄 fastvideo/worker/worker_base.py:61-72

                       └─ Worker()
                            📄 fastvideo/worker/gpu_worker.py:20-36

                       └─ Worker.init_device()  ⚡ MODELS LOADED HERE
                            📄 fastvideo/worker/gpu_worker.py:38-73

                          └─ build_pipeline()
                               📄 fastvideo/pipelines/__init__.py:28-69

                             └─ WanPipeline()  [calls parent __init__]
                                  📄 fastvideo/pipelines/basic/wan/wan_pipeline.py:23

                                └─ ComposedPipelineBase.__init__()
                                     📄 fastvideo/pipelines/composed_pipeline_base.py:49-86

                                   └─ load_modules()  ⚡ ACTUAL LOADING
                                        📄 fastvideo/pipelines/composed_pipeline_base.py:255-358

                                      └─ PipelineComponentLoader.load_module()  [for each module]
                                           📄 fastvideo/models/loader/component_loader.py:579-608

                                         └─ ComponentLoader.for_module_type()
                                              📄 fastvideo/models/loader/component_loader.py:58-94
                                              Returns appropriate loader

                                            └─ [Specific Loader].load()
                                                 📄 fastvideo/models/loader/component_loader.py
                                                 
                                               ├─ TextEncoderLoader.load()  :165-271
                                               ├─ TransformerLoader.load()  :429-513
                                               ├─ VAELoader.load()          :378-423
                                               ├─ SchedulerLoader.load()    :516-534
                                               └─ TokenizerLoader.load()    :345-375
```

---

## Memory and Device Management

### Device Placement Strategy

```python
# Determined by fastvideo_args flags

if fastvideo_args.text_encoder_cpu_offload:
    text_encoder_device = torch.device("cpu")
else:
    text_encoder_device = get_local_torch_device()  # cuda:0 or cuda:1

if fastvideo_args.dit_cpu_offload:
    transformer_device = torch.device("cpu")
    # But with FSDP: weights on CPU, computation on GPU
else:
    transformer_device = get_local_torch_device()

if fastvideo_args.vae_cpu_offload:
    vae_device = torch.device("cpu")
else:
    vae_device = get_local_torch_device()
```

### Memory Optimization: FSDP

**Without FSDP:**
```
GPU 0: [Full Transformer Model]  ← 2.6 GB
GPU 1: [Full Transformer Model]  ← 2.6 GB
Total: 5.2 GB wasted (duplicated)
```

**With FSDP (use_fsdp_inference=True):**
```
GPU 0: [Transformer Shard 0]  ← 1.3 GB
GPU 1: [Transformer Shard 1]  ← 1.3 GB
Total: 2.6 GB (no duplication!)

During forward pass:
- Each GPU loads needed weights dynamically
- Prefetches next layer while computing current
- Synchronizes via all-gather
```

**With FSDP + CPU Offload:**
```
CPU:   [Transformer Weights]  ← 2.6 GB
GPU 0: [Active Layer]          ← 0.2 GB
GPU 1: [Active Layer]          ← 0.2 GB
Total GPU: 0.4 GB only!

Trade-off: ~15-20% slower inference
```

---

## Advanced Loading Patterns

### 1. Lazy Loading (Deferred Loading)

Some models can be loaded lazily:

```python
# In load_modules()
if fastvideo_args.dit_cpu_offload:
    # Don't load transformer immediately
    fastvideo_args.model_loaded["transformer"] = False
    fastvideo_args.model_paths["transformer"] = transformer_path
    modules["transformer"] = None

# Later, in DenoisingStage.forward()
if not fastvideo_args.model_loaded["transformer"]:
    loader = TransformerLoader()
    self.transformer = loader.load(
        fastvideo_args.model_paths["transformer"],
        fastvideo_args
    )
    fastvideo_args.model_loaded["transformer"] = True
```

### 2. Custom Weight Initialization

For training/distillation:

```python
if fastvideo_args.init_weights_from_safetensors:
    # Use custom weights instead of model's default weights
    custom_weights_path = fastvideo_args.init_weights_from_safetensors
    safetensors_list = glob.glob(
        os.path.join(custom_weights_path, "*.safetensors")
    )
    # Load from custom path
```

### 3. LoRA Adapter Loading

Loaded separately after base model:

```python
# After model is loaded
if fastvideo_args.lora_path:
    from fastvideo.layers.lora import inject_lora
    
    inject_lora(
        model=transformer,
        lora_path=fastvideo_args.lora_path,
        lora_rank=fastvideo_args.lora_rank,
        target_modules=fastvideo_args.lora_target_modules,
    )
```

### 4. Torch Compile (JIT Optimization)

Applied after loading:

```python
if fastvideo_args.enable_torch_compile:
    transformer = torch.compile(
        transformer,
        backend="inductor",
        mode="reduce-overhead",
        **fastvideo_args.torch_compile_kwargs
    )
```

---

## Common Questions

**Q: Why load models in workers instead of main process?**

A: Because:
1. Each worker runs on a different GPU
2. Models need to be on the correct GPU
3. torch.distributed requires per-process initialization
4. Avoids large memory transfers between processes

**Q: Can I pre-load models and pass them in?**

A: Yes! Use `loaded_modules` parameter:

```python
# Pre-load models
my_transformer = load_my_transformer()
my_vae = load_my_vae()

pipeline = WanPipeline(
    model_path=model_path,
    fastvideo_args=fastvideo_args,
    loaded_modules={
        "transformer": my_transformer,
        "vae": my_vae,
    }
)
```

**Q: How do I reduce memory usage?**

A: Multiple options:

```python
fastvideo_args = FastVideoArgs(
    # Option 1: CPU offloading
    text_encoder_cpu_offload=True,  # Saves ~18 GB
    vae_cpu_offload=True,            # Saves ~2 GB
    dit_cpu_offload=True,            # Saves ~2.6 GB
    
    # Option 2: FSDP (share weights across GPUs)
    use_fsdp_inference=True,
    hsdp_shard_dim=num_gpus,
    
    # Option 3: Lower precision (if quality ok)
    pipeline_config=PipelineConfig(
        dit_precision="fp16",      # Instead of bf16
        text_encoder_precisions=["fp16"],  # Instead of fp32
    ),
)
```

**Q: What files are read during loading?**

A: For each module, these files are read:

```
model_path/
├── model_index.json          ← Pipeline structure
├── text_encoder/
│   ├── config.json           ← T5 architecture config
│   ├── model.safetensors     ← T5 weights (~18 GB)
│   └── tokenizer_config.json
├── tokenizer/
│   ├── special_tokens_map.json
│   ├── spiece.model          ← Vocabulary
│   └── tokenizer_config.json
├── transformer/
│   ├── config.json           ← Transformer architecture
│   └── diffusion_pytorch_model.safetensors  ← Weights (~2.6 GB)
├── vae/
│   ├── config.json           ← VAE architecture
│   └── diffusion_pytorch_model.safetensors  ← Weights (~2 GB)
└── scheduler/
    └── scheduler_config.json ← Scheduler settings (no weights)
```

---

## Summary

**Model loading happens in 3 key places:**

1. **`Worker.init_device()`** - Entry point for loading
   - File: `fastvideo/worker/gpu_worker.py:38-73`

2. **`build_pipeline()`** - Creates pipeline instance
   - File: `fastvideo/pipelines/__init__.py:28-69`

3. **`ComposedPipelineBase.load_modules()`** - Loads all components
   - File: `fastvideo/pipelines/composed_pipeline_base.py:255-358`

**Each component loaded by specialized loader:**
- `TextEncoderLoader` - T5 encoder
- `TransformerLoader` - DiT model with FSDP
- `VAELoader` - VAE decoder
- `SchedulerLoader` - Noise scheduler
- `TokenizerLoader` - T5 tokenizer

**Memory can be optimized with:**
- CPU offloading (save GPU memory)
- FSDP (share weights across GPUs)
- Lower precision (fp16 instead of fp32)

**Total loading time:** 10-15 seconds for Wan 1.3B on 2× A100s

The key insight is that **models are loaded per-worker**, not in the main process, because each worker needs its own copy on its GPU for distributed execution.

