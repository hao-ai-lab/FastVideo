# SPDX-License-Identifier: Apache-2.0
# Adapted from vllm: https://github.com/vllm-project/vllm/blob/v0.7.3/vllm/envs.py
"""Registry of the environment variables that FastVideo reads.

Every FastVideo-owned environment variable is declared here once, as a typed
field with a default, a category, and a description. Code reads a variable
with ``envs.NAME.get()`` inside a function, writes it with ``envs.NAME.set()``,
and tests change it temporarily with ``envs.NAME.override()``. Each field type
has one parsing rule, and a value that the rule rejects raises ``EnvVarError``.

The policy for environment variables is in ``docs/contributing/env_vars.md``,
and ``fastvideo/tests/contract/test_env_policy.py`` enforces it.
"""

import os
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from typing import Generic, TypeVar

T = TypeVar("T")
# String fields accept a string default or None (unset).
S = TypeVar("S", str, str | None)

POLICY_DOC = "docs/contributing/env_vars.md"

# Allowed values of EnvField.category.
CATEGORIES = ("build", "path", "distributed", "external", "logging", "attention", "performance", "profiling", "debug",
              "sampling")

_TRUE_VALUES = frozenset({"1", "true", "yes", "on"})
_FALSE_VALUES = frozenset({"0", "false", "no", "off", ""})


class EnvVarError(ValueError):
    """An environment variable holds a value that its registered type rejects."""


class EnvField(Generic[T]):
    """One registered environment variable: its type, default, category, and description.

    ``default`` is either the value itself or a zero-argument function that
    computes it on each read while the variable is unset.
    """

    type_name = ""

    def __init__(self, default: T | Callable[[], T], *, category: str, doc: str) -> None:
        self.name = ""  # Set by _register_fields() from the module attribute name.
        self.default = default
        self.category = category
        self.doc = doc

    def parse(self, raw: str) -> T:
        raise NotImplementedError

    def format(self, value: T) -> str:
        return str(value)

    def get(self) -> T:
        """Return the parsed value, or the default when the variable is unset."""
        raw = os.environ.get(self.name)
        if raw is None:
            return self.default() if callable(self.default) else self.default
        try:
            return self.parse(raw)
        except ValueError as exc:
            raise EnvVarError(f"Invalid value {raw!r} for {self.name}: {exc}. See {POLICY_DOC}.") from None

    def is_set(self) -> bool:
        return self.name in os.environ

    def set(self, value: T) -> None:
        os.environ[self.name] = self.format(value)

    def clear(self) -> None:
        os.environ.pop(self.name, None)

    @contextmanager
    def override(self, value: T | None) -> Iterator[None]:
        """Set the variable, or unset it when ``value`` is None, and restore the previous value on exit."""
        previous = os.environ.get(self.name)
        if value is None:
            self.clear()
        else:
            self.set(value)
        try:
            yield
        finally:
            if previous is None:
                self.clear()
            else:
                os.environ[self.name] = previous

    def __bool__(self) -> bool:
        raise TypeError(f"Use envs.{self.name}.get() to read {self.name}.")


class EnvBool(EnvField[bool]):
    """True for 1, true, yes, on; false for 0, false, no, off, and the empty string; case-insensitive."""

    type_name = "bool"

    def parse(self, raw: str) -> bool:
        value = raw.strip().lower()
        if value in _TRUE_VALUES:
            return True
        if value in _FALSE_VALUES:
            return False
        raise ValueError("expected 1, true, yes, on, 0, false, no, off, or an empty string")

    def format(self, value: bool) -> str:
        return "1" if value else "0"


class EnvInt(EnvField[int]):
    type_name = "int"

    def parse(self, raw: str) -> int:
        return int(raw)


class EnvFloat(EnvField[float]):
    type_name = "float"

    def parse(self, raw: str) -> float:
        return float(raw)


class EnvStr(EnvField[S]):
    type_name = "str"

    def parse(self, raw: str) -> S:
        return raw


class EnvPath(EnvField[S]):
    """A filesystem path; a leading ``~`` is expanded."""

    type_name = "path"

    def parse(self, raw: str) -> S:
        return os.path.expanduser(raw)


class EnvChoice(EnvField[str]):
    """One of ``choices``; the value is stripped and lower-cased before the check."""

    def __init__(self, default: str, *, choices: tuple[str, ...], category: str, doc: str) -> None:
        super().__init__(default, category=category, doc=doc)
        self.choices = choices
        self.type_name = "one of " + ", ".join(choices)

    def parse(self, raw: str) -> str:
        value = raw.strip().lower()
        if value not in self.choices:
            raise ValueError(f"expected one of {', '.join(self.choices)}")
        return value


def get_default_cache_root() -> str:
    return os.getenv(
        "XDG_CACHE_HOME",
        os.path.join(os.path.expanduser("~"), ".cache"),
    )


def get_default_config_root() -> str:
    return os.getenv(
        "XDG_CONFIG_HOME",
        os.path.join(os.path.expanduser("~"), ".config"),
    )


# ================== Installation ==================

FASTVIDEO_TARGET_DEVICE = EnvStr("cuda",
                                 category="build",
                                 doc="Target device of FastVideo: cuda, rocm, neuron, cpu, or openvino.")
MAX_JOBS = EnvStr(None,
                  category="build",
                  doc="Maximum number of parallel compilation jobs. Defaults to the number of CPUs.")
NVCC_THREADS = EnvStr(None,
                      category="build",
                      doc="Number of nvcc threads. When set, MAX_JOBS is reduced to avoid oversubscribing the CPU.")
FASTVIDEO_USE_PRECOMPILED = EnvBool(False, category="build", doc="Use precompiled binaries (*.so).")
CMAKE_BUILD_TYPE = EnvStr(None, category="build", doc="CMake build type: Debug, Release, or RelWithDebInfo.")
VERBOSE = EnvBool(False, category="build", doc="Print verbose logs during installation.")

# ================== Paths ==================

FASTVIDEO_CONFIG_ROOT = EnvPath(
    lambda: os.path.expanduser(os.path.join(get_default_config_root(), "fastvideo")),
    category="path",
    doc="Root directory for FastVideo configuration files, at runtime and at installation. "
    "Defaults to ~/.config/fastvideo, or $XDG_CONFIG_HOME/fastvideo when XDG_CONFIG_HOME is set.")
FASTVIDEO_CACHE_ROOT = EnvPath(
    lambda: os.path.expanduser(os.path.join(get_default_cache_root(), "fastvideo")),
    category="path",
    doc="Root directory for FastVideo cache files. "
    "Defaults to ~/.cache/fastvideo, or $XDG_CACHE_HOME/fastvideo when XDG_CACHE_HOME is set.")

# ================== Distributed ==================

FASTVIDEO_HOST_IP = EnvStr(
    "",
    category="distributed",
    doc="IP address of this node when the node has several network interfaces. Set it on each node for multi-node "
    "inference.")
FASTVIDEO_LOOPBACK_IP = EnvStr("",
                               category="distributed",
                               doc="Loopback IP address to use instead of the detected one.")
FASTVIDEO_RAY_PER_WORKER_GPUS = EnvFloat(
    1.0,
    category="distributed",
    doc="GPUs per Ray worker. A fraction lets Ray schedule several actors on one GPU, so other actors can share "
    "the GPUs with FastVideo.")
FASTVIDEO_RINGBUFFER_WARNING_INTERVAL = EnvInt(60,
                                               category="distributed",
                                               doc="Seconds between warnings while the ring buffer is full.")
FASTVIDEO_NCCL_SO_PATH = EnvStr(
    None,
    category="distributed",
    doc="Path to the NCCL library file. Needed because the nccl>=2.19 that PyTorch ships has a bug "
    "(https://github.com/NVIDIA/nccl/issues/1234).")
HCCL_SO_PATH = EnvStr(None, category="distributed", doc="Path to the HCCL library file on Ascend NPUs.")
FASTVIDEO_ENGINE_ITERATION_TIMEOUT_S = EnvInt(60,
                                              category="distributed",
                                              doc="Timeout in seconds for each engine iteration.")
FASTVIDEO_WORKER_MULTIPROC_METHOD = EnvChoice("spawn",
                                              choices=("spawn", "fork", "forkserver"),
                                              category="distributed",
                                              doc="Multiprocessing start method for worker processes.")
FASTVIDEO_ULYSSES_A2A = EnvChoice(
    "off",
    choices=("off", "auto"),
    category="distributed",
    doc="Sequence-parallel all-to-all backend. off uses the NCCL path in DistributedAutograd.AllToAll4D. auto uses "
    "the fused NVLink kernel when the group is a load-store accessible mesh of 2, 4, 6, or 8 ranks in eager "
    "execution, and the NCCL path otherwise.")

# ================== External variables ==================
# Variables that other tools set. They stay registered so that Ray copies them
# to its workers until the external-variable allowlist replaces them.

LD_LIBRARY_PATH = EnvStr(None,
                         category="external",
                         doc="Searched for the NCCL library when FASTVIDEO_NCCL_SO_PATH is unset.")
LOCAL_RANK = EnvInt(0,
                    category="external",
                    doc="Local rank of the process in a distributed run; selects the GPU device id.")
CUDA_VISIBLE_DEVICES = EnvStr(None, category="external", doc="Visible devices in a distributed run.")

# ================== Logging ==================

FASTVIDEO_CONFIGURE_LOGGING = EnvBool(
    True,
    category="logging",
    doc="Configure logging at import. When true, FastVideo uses its default logging configuration or the file in "
    "FASTVIDEO_LOGGING_CONFIG_PATH.")
FASTVIDEO_LOGGING_CONFIG_PATH = EnvStr(None, category="logging", doc="Path to a JSON logging configuration file.")
FASTVIDEO_LOGGING_LEVEL = EnvStr("INFO", category="logging", doc="Default logging level.")
FASTVIDEO_LOGGING_PREFIX = EnvStr("", category="logging", doc="Prefix prepended to every log message.")
FASTVIDEO_STAGE_LOGGING = EnvBool(False, category="logging", doc="Log the time that each pipeline stage takes.")

# ================== Attention ==================

FASTVIDEO_ATTENTION_BACKEND = EnvStr(
    None,
    category="attention",
    doc="Attention backend, as an AttentionBackendEnum name such as TORCH_SDPA, FLASH_ATTN, VIDEO_SPARSE_ATTN, "
    "SAGE_ATTN, or SAGE_ATTN_THREE. FastVideoArgs uses it when FastVideoArgs.attention_backend is unset.")
# FA4 is opt-in and never auto-selected just because it is installed. Below
# sm90, grad-enabled and GQA calls are routed to FA2 (FA4's backward asserts
# sm90+ and its pack_gqa fails to JIT there).
FASTVIDEO_FA4 = EnvBool(False,
                        category="attention",
                        doc="The FLASH_ATTN backend uses FlashAttention-4 (flash_attn.cute) instead of FA3 or FA2.")
FASTVIDEO_MINIMAX_H3_FA4_PACKED_VARLEN = EnvBool(
    False,
    category="attention",
    doc="MiniMax-H3 dense DiT self-attention uses the FlashAttention-4 packed-varlen entry point. This changes the "
    "floating-point reduction order, so it is an inference-only opt-in.")
FASTVIDEO_VSA_SM100A = EnvBool(
    False,
    category="attention",
    doc="VIDEO_SPARSE_ATTN_H3 sends no-grad tile-64 forwards to the data-center Blackwell (sm_100a) kernel. "
    "fastvideo-kernel reads the same variable with the same rule.")

# ================== Performance ==================

# Non-fullgraph-traceable attention backends such as VSA degrade to eager with
# one warning; see _regional_compile_unsupported_reason in
# fastvideo/models/loader/fsdp_load.py.
FASTVIDEO_INFERENCE_TORCH_COMPILE = EnvBool(
    False,
    category="performance",
    doc="Compile each DiT transformer block with fullgraph torch.compile at inference. Same as "
    "FastVideoArgs.inference_torch_compile=True.")
FASTVIDEO_VAE_PARALLEL_DECODE = EnvBool(
    False,
    category="performance",
    doc="MiniMax-H3 VAE decode splits its temporal chunks across the sequence-parallel ranks instead of running "
    "serially on the output rank. Same as FastVideoArgs.vae_parallel_decode=True.")
FASTVIDEO_VAE_PARALLEL_ENCODE = EnvBool(
    False,
    category="performance",
    doc="MiniMax-H3 reference-video VAE encode splits its temporal chunks across the sequence-parallel ranks. "
    "Same as FastVideoArgs.vae_parallel_encode=True.")
FASTVIDEO_VAE_PARALLEL_DECODE_STRATEGY = EnvStr(
    None,
    category="performance",
    doc="Collective that moves chunks in parallel VAE decode: gather (used when unset) or all_gather.")
# Adapted from the NVlabs/Sana Sol-Engine implementation.
FASTVIDEO_MINIMAX_H3_FUSIONS = EnvStr(
    "",
    category="performance",
    doc="MiniMax-H3 inference-only Triton fusions: all, 1, or a comma-separated subset of "
    "modulate,qknorm_rope,swiglu. Empty, 0, or none keeps the eager implementation.")
FASTVIDEO_TEST_DYNAMO_FULLGRAPH_CAPTURE = EnvBool(True, category="debug", doc="Enable Dynamo fullgraph capture.")

# ================== Profiling ==================

FASTVIDEO_NVTX_PROFILE = EnvBool(False,
                                 category="profiling",
                                 doc="Emit NVTX ranges for external profilers such as Nsight Systems.")
FASTVIDEO_TORCH_PROFILER_DIR = EnvPath(
    None,
    category="profiling",
    doc="Enables the torch profiler and sets the directory for its traces. Must be an absolute path.")
FASTVIDEO_TORCH_PROFILER_RECORD_SHAPES = EnvBool(False, category="profiling", doc="Torch profiler records shapes.")
FASTVIDEO_TORCH_PROFILER_WITH_PROFILE_MEMORY = EnvBool(False,
                                                       category="profiling",
                                                       doc="Torch profiler profiles memory.")
FASTVIDEO_TORCH_PROFILER_WITH_STACK = EnvBool(
    False, category="profiling", doc="Torch profiler captures stacks. Costs about 1.5x runtime and 1.4x trace size.")
FASTVIDEO_TORCH_PROFILER_WITH_FLOPS = EnvBool(False, category="profiling", doc="Torch profiler profiles FLOPs.")
FASTVIDEO_TORCH_PROFILE_REGIONS = EnvStr(
    "",
    category="profiling",
    doc="Comma-separated profiler regions to record. The torch profiler requires at least one region.")

# ================== Debug ==================

FASTVIDEO_SERVER_DEV_MODE = EnvBool(False,
                                    category="debug",
                                    doc="Run the server in development mode with extra debugging endpoints.")
FASTVIDEO_TRACE_FUNCTION = EnvBool(False, category="debug", doc="Trace function calls.")
FASTVIDEO_TRACE_ACTIVATIONS = EnvBool(False, category="debug", doc="Enable activation trace hooks.")
FASTVIDEO_TRACE_LAYERS = EnvStr("", category="debug", doc="Regex filter for traced module names. Empty means all.")
FASTVIDEO_TRACE_STATS = EnvStr("abs_mean,sum",
                               category="debug",
                               doc="Comma-separated activation statistics dumped for each output tensor.")
FASTVIDEO_TRACE_OUTPUT = EnvStr("/tmp/fv_trace_<pid>.jsonl",
                                category="debug",
                                doc="JSONL path for activation traces. The literal <pid> is replaced at runtime.")
FASTVIDEO_TRACE_STEPS = EnvStr("", category="debug", doc="Comma-separated denoising step indices. Empty means all.")

# ================== Sampling ==================

# CFG gating fraction for stale-uncond reuse (Adaptive Guidance / LinearAG
# variant — Castillo et al. 2023, arXiv:2312.12487).  Float in [0, 1].
# Interpretation: for step index `i < len(timesteps) * X`, run both
# cond and uncond forwards and refresh delta_cached = cond - uncond.
# Once `i >= len(timesteps) * X`, skip the uncond forward and reuse
# the cached delta:  noise_pred = cond + (guidance_scale - 1) * delta.
#
# Edge cases:
#   1.0 (default) : disables gating; identical to baseline two-pass CFG.
#   0.5           : run uncond for the first half of steps, reuse delta
#                    for the second half (~25% inference time saved on
#                    bandwidth-bound SP setups).
#   0.0           : step 0 still computes uncond fresh (cache is empty
#                    at start) — all subsequent steps reuse the step-0
#                    delta.  This is the most aggressive setting; does
#                    NOT mean "no uncond forward ever."
#
# Caveats:
#   - Algorithmically approximate; not bit-exact vs baseline CFG.
#     Validate per-pipeline with SSIM / VBench before lowering below 1.0.
#   - Interaction with `guidance_rescale > 0` is unvalidated; the
#     denoising stage logs a warning when both are active.
#   - Wan2.2 high/low-noise expert switch invalidates the cache.
FASTVIDEO_CFG_GATE_STEP = EnvFloat(
    1.0,
    category="sampling",
    doc="CFG gating fraction in [0, 1]. Steps before len(timesteps) * X run the conditional and unconditional "
    "forwards; later steps reuse the cached difference. 1.0 disables gating.")


def _register_fields() -> dict[str, EnvField]:
    """Name each module-level EnvField after its attribute and return the fields by name."""
    fields = {name: value for name, value in globals().items() if isinstance(value, EnvField)}
    for name, field in fields.items():
        field.name = name
    return fields


environment_variables: dict[str, EnvField] = _register_fields()
