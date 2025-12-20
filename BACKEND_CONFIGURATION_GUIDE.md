# FastVideo Backend Configuration Guide

## Where the Backend is Specified

The distributed executor backend (`"mp"` for multiprocessing or `"ray"` for Ray) is controlled by the `distributed_executor_backend` parameter in `FastVideoArgs`.

---

## Method 1: Pass as Keyword Argument (EASIEST)

When creating a `VideoGenerator`, pass `distributed_executor_backend` as a keyword argument:

```python
from fastvideo import VideoGenerator

# Use multiprocessing backend (DEFAULT)
generator = VideoGenerator.from_pretrained(
    "Wan-AI/Wan2.1-T2V-1.3B-Diffusers",
    num_gpus=1,
    distributed_executor_backend="mp"  # Multiprocessing (default)
)

# Use Ray backend
generator = VideoGenerator.from_pretrained(
    "Wan-AI/Wan2.1-T2V-1.3B-Diffusers",
    num_gpus=2,
    distributed_executor_backend="ray"  # Ray
)
```

**Location**: Any keyword argument you pass to `from_pretrained()` gets forwarded to `FastVideoArgs.from_kwargs()` which creates a `FastVideoArgs` object with your settings.

---

## Method 2: Command-Line Arguments

If you're using FastVideo from the command line (e.g., in training scripts), you can specify it via CLI:

```bash
# Multiprocessing (default)
python my_script.py --distributed-executor-backend mp

# Ray
python my_script.py --distributed-executor-backend ray
```

**Location**: `fastvideo/fastvideo_args.py` lines 234-240:

```python
parser.add_argument(
    "--distributed-executor-backend",
    type=str,
    choices=["mp"],  # NOTE: Only "mp" is listed, but "ray" also works
    default=FastVideoArgs.distributed_executor_backend,
    help="The distributed executor backend to use",
)
```

**Note**: The `choices=["mp"]` is outdated - it should include `"ray"`, but Ray works anyway since the actual validation happens in `Executor.get_class()`.

---

## Method 3: Directly Create FastVideoArgs

For advanced use cases, create `FastVideoArgs` directly:

```python
from fastvideo.fastvideo_args import FastVideoArgs
from fastvideo import VideoGenerator

# Create args with Ray backend
args = FastVideoArgs(
    model_path="Wan-AI/Wan2.1-T2V-1.3B-Diffusers",
    num_gpus=2,
    distributed_executor_backend="ray"
)

# Create generator from args
generator = VideoGenerator.from_fastvideo_args(args)
```

---

## Default Value

**Location**: `fastvideo/fastvideo_args.py` line 98:

```python
@dataclasses.dataclass
class FastVideoArgs:
    # ... other fields ...
    
    # Distributed executor backend
    distributed_executor_backend: str = "mp"  # DEFAULT: multiprocessing
```

**Default**: `"mp"` (multiprocessing)

If you don't specify `distributed_executor_backend`, it will use multiprocessing.

---

## How the Backend is Selected

**Location**: `fastvideo/worker/executor.py` lines 27-37:

```python
class Executor(ABC):
    @staticmethod
    def get_class(fastvideo_args: FastVideoArgs) -> type["Executor"]:
        if fastvideo_args.distributed_executor_backend == "mp":
            from fastvideo.worker.multiproc_executor import MultiprocExecutor
            return cast(type["Executor"], MultiprocExecutor)
        
        elif fastvideo_args.distributed_executor_backend == "ray":
            from fastvideo.worker.ray_distributed_executor import RayDistributedExecutor
            return cast(type["Executor"], RayDistributedExecutor)
        
        else:
            raise ValueError(
                f"Unsupported distributed executor backend: {fastvideo_args.distributed_executor_backend}"
            )
```

**Flow**:
1. You create `VideoGenerator.from_pretrained(..., distributed_executor_backend="ray")`
2. Internally calls `FastVideoArgs.from_kwargs(**kwargs)` → creates `FastVideoArgs` object
3. Calls `Executor.get_class(fastvideo_args)` → returns `MultiprocExecutor` or `RayDistributedExecutor` class
4. Creates executor: `executor = executor_class(fastvideo_args)`

---

## Supported Values

| Value | Backend | Import Path |
|-------|---------|-------------|
| `"mp"` | Multiprocessing | `fastvideo.worker.multiproc_executor.MultiprocExecutor` |
| `"ray"` | Ray | `fastvideo.worker.ray_distributed_executor.RayDistributedExecutor` |

Any other value will raise: `ValueError: Unsupported distributed executor backend: <value>`

---

## Real-World Example: Ray Backend

**File**: `examples/inference/basic/basic_ray.py`

```python
from fastvideo import VideoGenerator

generator = VideoGenerator.from_pretrained(
    "Wan-AI/Wan2.1-T2V-1.3B-Diffusers",
    num_gpus=2,                              # Multiple GPUs
    use_fsdp_inference=True,
    dit_cpu_offload=False,
    distributed_executor_backend="ray",      # ← HERE: Use Ray
)

video = generator.generate_video(
    "A curious raccoon peers through sunflowers...",
    output_path="video_samples",
    save_video=True
)
```

---

## When to Use Each Backend

### Use Multiprocessing (`"mp"`) when:
- ✅ Single node (one machine)
- ✅ 1-8 GPUs
- ✅ Simple setup (no dependencies)
- ✅ Quick prototyping
- ✅ You don't need fault tolerance

### Use Ray (`"ray"`) when:
- ✅ Multi-node clusters
- ✅ Need fault tolerance
- ✅ Advanced scheduling
- ✅ Large-scale production deployments
- ✅ Integration with Ray ecosystem

---

## Checking What Backend You're Using

You can check the backend in a running generator:

```python
generator = VideoGenerator.from_pretrained(...)

# Check the executor type
print(type(generator.executor))
# Output: <class 'fastvideo.worker.multiproc_executor.MultiprocExecutor'>
# OR
# Output: <class 'fastvideo.worker.ray_distributed_executor.RayDistributedExecutor'>

# Check the args
print(generator.executor.fastvideo_args.distributed_executor_backend)
# Output: "mp" or "ray"
```

---

## Summary

**Quick Reference**:

```python
# Multiprocessing (default)
VideoGenerator.from_pretrained(model_path, distributed_executor_backend="mp")

# Ray
VideoGenerator.from_pretrained(model_path, distributed_executor_backend="ray")

# If omitted, defaults to "mp"
VideoGenerator.from_pretrained(model_path)  # Uses multiprocessing
```

**The parameter flows through**:
1. `VideoGenerator.from_pretrained()` kwargs
2. → `FastVideoArgs.from_kwargs()` 
3. → `FastVideoArgs.distributed_executor_backend` (stored)
4. → `Executor.get_class()` (selects backend)
5. → Creates `MultiprocExecutor` or `RayDistributedExecutor`















