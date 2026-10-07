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

import logging
import os
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from typing import Generic, TypeVar

T = TypeVar("T")
# String fields accept a string default or None (unset).
S = TypeVar("S", str, str | None)

POLICY_DOC = "docs/contributing/env_vars.md"

# Allowed values of EnvField.category.
CATEGORIES = ("path", "distributed", "logging", "attention", "performance", "output", "profiling", "debug", "sampling",
              "eval", "test")

_TRUE_VALUES = frozenset({"1", "true", "yes", "on"})
_FALSE_VALUES = frozenset({"0", "false", "no", "off", ""})

# fastvideo.logger imports this module, so warnings go through the standard
# logging module; the "fastvideo" logger configuration still applies.
_logger = logging.getLogger(__name__)
_warned_messages: set[str] = set()


def _warn_once(message: str) -> None:
    if message not in _warned_messages:
        _warned_messages.add(message)
        _logger.warning(message)


class EnvVarError(ValueError):
    """An environment variable holds a value that its registered type rejects."""


class EnvField(Generic[T]):
    """One registered environment variable: its type, default, category, and description.

    ``default`` is either the value itself or a zero-argument function that
    computes it on each read while the variable is unset. ``deprecated_names``
    lists earlier names of a renamed variable; they are read, with a warning,
    only when the variable itself is unset.
    """

    type_name = ""

    def __init__(self, default: T | Callable[[], T], *, category: str, doc: str,
                 deprecated_names: tuple[str, ...] = ()) -> None:
        self.name = ""  # Set by _register_fields() from the module attribute name.
        self.default = default
        self.category = category
        self.doc = doc
        self.deprecated_names = deprecated_names

    def parse(self, raw: str) -> T:
        raise NotImplementedError

    def format(self, value: T) -> str:
        return str(value)

    def _read_raw(self) -> tuple[str, str] | None:
        """Return (name, raw value) from the variable or its first set deprecated name."""
        raw = os.environ.get(self.name)
        if raw is not None:
            return self.name, raw
        for old_name in self.deprecated_names:
            raw = os.environ.get(old_name)
            if raw is not None:
                _warn_once(f"{old_name} is deprecated and will be removed in the next minor release; "
                           f"set {self.name} instead.")
                return old_name, raw
        return None

    def get(self) -> T:
        """Return the parsed value, or the default when the variable is unset."""
        found = self._read_raw()
        if found is None:
            return self.default() if callable(self.default) else self.default
        name, raw = found
        try:
            return self.parse(raw)
        except ValueError as exc:
            raise EnvVarError(f"Invalid value {raw!r} for {name}: {exc}. See {POLICY_DOC}.") from None

    def is_set(self) -> bool:
        return any(name in os.environ for name in (self.name, *self.deprecated_names))

    def set(self, value: T) -> None:
        os.environ[self.name] = self.format(value)

    def clear(self) -> None:
        for name in (self.name, *self.deprecated_names):
            os.environ.pop(name, None)

    @contextmanager
    def override(self, value: T | None) -> Iterator[None]:
        """Set the variable, or unset it when ``value`` is None, and restore the previous values on exit."""
        previous = {name: os.environ.get(name) for name in (self.name, *self.deprecated_names)}
        self.clear()
        if value is not None:
            self.set(value)
        try:
            yield
        finally:
            for name, old_value in previous.items():
                if old_value is None:
                    os.environ.pop(name, None)
                else:
                    os.environ[name] = old_value

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


# Variables that other tools read (CUDA, NCCL, PyTorch, launchers) are written
# only through these helpers. The contract test checks each name against its
# write allowlist.


def set_external(name: str, value: str) -> None:
    """Set an environment variable that another tool reads."""
    os.environ[name] = value


def setdefault_external(name: str, value: str) -> None:
    """Set an environment variable that another tool reads, unless it is already set."""
    os.environ.setdefault(name, value)


def unset_external(name: str) -> None:
    """Remove an environment variable that another tool reads."""
    os.environ.pop(name, None)


@contextmanager
def override_external(name: str, value: str | None) -> Iterator[None]:
    """Set, or unset when ``value`` is None, a variable outside the registry, and restore it on exit.

    Tests use it instead of ``monkeypatch.setenv`` for variables that other
    tools or the CI read.
    """
    previous = os.environ.get(name)
    if value is None:
        os.environ.pop(name, None)
    else:
        os.environ[name] = value
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop(name, None)
        else:
            os.environ[name] = previous


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
FASTVIDEO_REASON1_WEIGHTS_PATH = EnvStr(None,
                                        category="path",
                                        doc="Local path or Hugging Face id of Reason1 weights to load instead of the "
                                        "checkpoint's own.")

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
FASTVIDEO_NCCL_SO_PATH = EnvStr(
    None,
    category="distributed",
    doc="Path to the NCCL library file. Needed because the nccl>=2.19 that PyTorch ships has a bug "
    "(https://github.com/NVIDIA/nccl/issues/1234).")
FASTVIDEO_HCCL_SO_PATH = EnvStr(None,
                                category="distributed",
                                doc="Path to the HCCL library file on Ascend NPUs.",
                                deprecated_names=("HCCL_SO_PATH", ))
FASTVIDEO_WORKER_MULTIPROC_METHOD = EnvChoice("spawn",
                                              choices=("spawn", "fork", "forkserver"),
                                              category="distributed",
                                              doc="Multiprocessing start method for worker processes.")
FASTVIDEO_EXTERNAL_LAUNCHER = EnvBool(
    False,
    category="distributed",
    doc="With the default mp backend, offline inference runs SPMD under torchrun or srun: each launched process is "
    "one worker joining the env:// rendezvous. All ranks must call generate() together; world rank 0 owns outputs.")
FASTVIDEO_ULYSSES_A2A = EnvChoice(
    "off",
    choices=("off", "auto"),
    category="distributed",
    doc="Sequence-parallel all-to-all backend. off uses the NCCL path in DistributedAutograd.AllToAll4D. auto uses "
    "the fused NVLink kernel when the group is a load-store accessible mesh of 2, 4, 6, or 8 ranks in eager "
    "execution, and the NCCL path otherwise.")
# Keep the original long-training transport by default. Opt into bounded
# chunks after validating the training recipe's activation memory budget.
FASTVIDEO_ULYSSES_A2A_LONG_TRAINING = EnvChoice(
    "auto",
    choices=("auto", "chunked"),
    category="distributed",
    doc="Fused Ulysses all-to-all policy for grad-enabled calls whose window exceeds the long-training plane limit. "
    "auto keeps the original unchunked transport. chunked uses bounded chunks; validate the recipe's activation "
    "memory first.")

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
FASTVIDEO_LOG_ALL_PROCESSES = EnvBool(
    False,
    category="logging",
    doc="logger.info logs from every process, ignoring the default local-main-process filter and the "
    "main_process_only and local_main_process_only arguments. Read at each call, so it can be set after "
    "importing fastvideo. Useful for debugging distributed runs.")

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
FASTVIDEO_VSA_TRITON = EnvBool(
    False,
    category="attention",
    doc="Force the Triton MiniMax-H3 sparse attention kernel. fastvideo-kernel reads the same variable.")
FASTVIDEO_NVFP4_FA4 = EnvBool(
    False,
    category="attention",
    doc="FlashAttention-4 quantizes Q and K to NVFP4. An explicit nvfp4_fa4 attention implementation argument "
    "takes precedence.")
FASTVIDEO_FA4_PV_MODE = EnvChoice(
    "bf16",
    choices=("bf16", "fp8"),
    category="attention",
    doc="V dtype on the NVFP4 FlashAttention-4 path (FLASH_ATTN with nvfp4_fa4, ATTN_QAT_INFER on sm_100a/sm_103a): "
    "bf16 keeps V in BF16; fp8 casts V to float8 e4m3 without scaling. An explicit fa4_pv_mode attention "
    "implementation argument takes precedence.")
FASTVIDEO_DISABLE_ATTENTION_COMPILE = EnvBool(
    True,
    category="attention",
    doc="Keep attention forward out of torch.compile graphs (torch.compiler.disable). Set it to 0 to let attention "
    "constructed under that setting be traced. Setting it explicitly to true also blocks regional compile.")
FASTVIDEO_MLX_WINDOW = EnvInt(0,
                              category="attention",
                              doc="MLX FastWan windowed attention size in tokens. 0 uses full attention.")
FASTVIDEO_MLX_WINDOW_SINK = EnvInt(0,
                                   category="attention",
                                   doc="Number of sink tokens that MLX windowed attention always attends to.")
FASTVIDEO_DISABLE_VSA64_FUSED_LAYOUT = EnvBool(
    False,
    category="attention",
    doc="VIDEO_SPARSE_ATTN keeps the original tile scatter and BSHD->BHSD transposes instead of the fused Triton "
    "layout kernel that no-grad SM100 BF16 head_dim-128 forwards with 64-token tiles use by default.")

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
# MiniMax-H3 component-level pipeline parallel: dedicate the first
# FASTVIDEO_H3_ENCODER_NODES nodes to the Qwen3-VL text encoder so the
# denoising ranks never load it (720p needs the ~48 GiB headroom on GB10).
FASTVIDEO_H3_ENCODER_SPLIT = EnvBool(
    False,
    category="distributed",
    doc="MiniMax-H3 runs the Qwen3-VL text encoder on a dedicated group of nodes, so the denoising ranks never "
    "load it. Same as FastVideoArgs.h3_encoder_split=True.")
FASTVIDEO_H3_ENCODER_NODES = EnvInt(1,
                                    category="distributed",
                                    doc="Number of leading nodes that FASTVIDEO_H3_ENCODER_SPLIT dedicates to the "
                                    "MiniMax-H3 text encoder. Values below 1 count as 1.")
# Adapted from the NVlabs/Sana Sol-Engine implementation.
FASTVIDEO_MINIMAX_H3_FUSIONS = EnvStr(
    "",
    category="performance",
    doc="MiniMax-H3 inference-only Triton fusions: all, 1, or a comma-separated subset of "
    "modulate,qknorm_rope,swiglu. Empty, 0, or none keeps the eager implementation.")
FASTVIDEO_MINIMAX_H3_EXACT_KERNELS = EnvStr(
    "",
    category="performance",
    doc="MiniMax-H3 inference-only Triton kernels that reproduce the eager output bit for bit: all, 1, or a "
    "comma-separated subset of rope,modulate,swiglu. An op whose FASTVIDEO_MINIMAX_H3_FUSIONS fusion is on keeps "
    "the fusion. Empty, 0, or none keeps the eager implementation.")
FASTVIDEO_FSDP2_AUTOWRAP = EnvBool(False,
                                   category="performance",
                                   doc="FSDP2 shards modules by parameter count instead of the model's shard "
                                   "conditions. Not supported by self-forcing distillation.")
FASTVIDEO_FSDP2_MIN_PARAMS = EnvInt(10000000,
                                    category="performance",
                                    doc="Minimum parameter count of a module that FASTVIDEO_FSDP2_AUTOWRAP shards.")
FASTVIDEO_MLX_COMPILE = EnvBool(False, category="performance", doc="Compile the MLX DiT forward with mx.compile.")
FASTVIDEO_MLX_FAST_NORM = EnvBool(False, category="performance", doc="Use MLX fast normalization kernels.")
FASTVIDEO_MLX_DQ_GEMM = EnvStr(
    "1",
    category="performance",
    doc="MLX dequantized GEMM for affine-quantized weights: 0 turns it off, 1 uses the measured minimum row count, "
    "and an integer sets the minimum row count.")
FASTVIDEO_LTX2_VAE_CHANNELS_LAST_3D = EnvBool(True,
                                              category="performance",
                                              doc="LTX-2 VAE uses the channels_last_3d memory format.")
FASTVIDEO_LTX2_DISABLE_AUDIO_AUTOCAST = EnvBool(True,
                                                category="performance",
                                                doc="LTX-2 audio decoding runs without CUDA autocast.",
                                                deprecated_names=("LTX2_DISABLE_AUDIO_AUTOCAST", ))
FASTVIDEO_FLUX2_DISABLE_BF16_REDUCED_PRECISION_REDUCTION = EnvBool(
    False,
    category="performance",
    doc="Flux denoising disables reduced-precision reductions in bf16 matmuls, which tightens accumulation for "
    "the 4-step Klein model.")
# Global kill switch for the opt-in, inference-only Triton fusion in
# fastvideo/layers/triton_fused_norm.py, for debugging numerics or Triton issues.
FASTVIDEO_DISABLE_FUSED_NORM = EnvBool(False,
                                       category="performance",
                                       doc="Turn off the Triton-fused residual + LayerNorm + modulate inference "
                                       "path that Wan blocks opt into, and use the eager path.")

# ================== Output encoding ==================

FASTVIDEO_FFMPEG_BIN = EnvStr("ffmpeg", category="output", doc="ffmpeg executable used to save video with audio.")
FASTVIDEO_VIDEO_CODEC = EnvStr("libx264", category="output", doc="ffmpeg video codec for saved videos.")
FASTVIDEO_NVENC_PRESET = EnvStr("p1", category="output", doc="NVENC preset when the codec is an *_nvenc codec.")
FASTVIDEO_NVENC_TUNE = EnvStr("ull", category="output", doc="NVENC tune option.")
FASTVIDEO_NVENC_RC = EnvStr("constqp", category="output", doc="NVENC rate-control mode.")
FASTVIDEO_NVENC_QP = EnvStr("28", category="output", doc="NVENC quantization parameter.")
FASTVIDEO_NVENC_BF = EnvStr("0", category="output", doc="NVENC number of B-frames.")
FASTVIDEO_X264_PRESET = EnvStr("ultrafast", category="output", doc="x264 preset for non-NVENC codecs.")
FASTVIDEO_OUTPUT_PIX_FMT = EnvStr("yuv420p", category="output", doc="ffmpeg pixel format for saved videos.")

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

FASTVIDEO_TRACE_ACTIVATIONS = EnvBool(False, category="debug", doc="Enable activation trace hooks.")
FASTVIDEO_TRACE_LAYERS = EnvStr("", category="debug", doc="Regex filter for traced module names. Empty means all.")
FASTVIDEO_TRACE_STATS = EnvStr("abs_mean,sum",
                               category="debug",
                               doc="Comma-separated activation statistics dumped for each output tensor.")
FASTVIDEO_TRACE_OUTPUT = EnvStr("/tmp/fv_trace_<pid>.jsonl",
                                category="debug",
                                doc="JSONL path for activation traces. The literal <pid> is replaced at runtime.")
FASTVIDEO_TRACE_STEPS = EnvStr("", category="debug", doc="Comma-separated denoising step indices. Empty means all.")
FASTVIDEO_H3_VSA_PROBE = EnvStr(
    None,
    category="debug",
    doc="Output directory for the VSA-H3 attention-mass probe, which writes one .pt file per step, layer, and rank. "
    "Keeps the model out of regional compile.")
FASTVIDEO_LTX2_GEMMA_LOG = EnvStr("",
                                  category="debug",
                                  doc="Log file for LTX-2 Gemma text-encoder hidden states, used by parity tests.",
                                  deprecated_names=("LTX2_FASTVIDEO_GEMMA_LOG", ))
FASTVIDEO_COSMOS25_LOG_KNOBS = EnvBool(False,
                                       category="debug",
                                       doc="Log the Cosmos 2.5 latent-preparation conditioning inputs.")

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
FASTVIDEO_LTX2_USE_DISTILLED_SIGMAS = EnvBool(
    True,
    category="sampling",
    doc="LTX-2 uses the distilled sigma schedule when FastVideoArgs.ltx2_use_distilled_sigmas is also true.",
    deprecated_names=("LTX2_USE_DISTILLED_SIGMAS", ))

# ================== Evaluation ==================

FASTVIDEO_EVAL_CACHE = EnvPath(lambda: os.path.join(FASTVIDEO_CACHE_ROOT.get(), "eval"),
                               category="eval",
                               doc="Cache directory for evaluation models and datasets. Defaults to "
                               "$FASTVIDEO_CACHE_ROOT/eval.")
FASTVIDEO_PHYSICS_IQ_BUCKET_URL = EnvStr("https://storage.googleapis.com/physics-iq-benchmark",
                                         category="eval",
                                         doc="Base URL of the Physics-IQ benchmark bucket.")
FASTVIDEO_VBENCH_FULL_INFO_JSON = EnvStr(None,
                                         category="eval",
                                         doc="Path to VBench_full_info.json, used instead of the vendored copy.",
                                         deprecated_names=("VBENCH_FULL_INFO_JSON", ))
FASTVIDEO_FVD_REF_FEATURES = EnvStr(None, category="eval", doc="Cached reference-feature file for the FVD metric.")
FASTVIDEO_FAD_REF_FEATURES = EnvStr(None,
                                    category="eval",
                                    doc="Cached reference-feature file for the audio Frechet distance metric.")

# ================== MiniMax-H3 single-GPU switches ==================

FASTVIDEO_H3_VSA_FP4 = EnvBool(False,
                               category="attention",
                               doc="Run MiniMax-H3 VSA attention on the block-sparse SageAttention3 FP4 kernel "
                               "(sm_120, no-grad, single sequence-parallel rank).")
FASTVIDEO_H3_VSA_HEADS_FIRST_TILE = EnvBool(False,
                                            category="performance",
                                            doc="VSA-H3 64/128-token tiles: scatter rows straight into the heads-first "
                                            "layout the sparse kernels read, removing their per-call transpose copy "
                                            "of query, key and value. Bit-identical; no-grad, uncompiled CUDA only.")
FASTVIDEO_H3_VSA_TILE_FIRST = EnvBool(False,
                                      category="attention",
                                      doc="Single-rank MiniMax-H3 VSA with one tile gather of the block input "
                                      "instead of separate Q/K/V/gate scatters.")
FASTVIDEO_H3_VSA_SM89_KERNEL = EnvChoice("original",
                                         choices=("original", "bf16", "int8"),
                                         category="attention",
                                         doc="Fine-attention kernel for MiniMax-H3 VSA on sm_89: original, bf16, "
                                         "or int8 (INT8 QK, BF16 PV).")
FASTVIDEO_H3_SIM_SP_FP8 = EnvBool(False,
                                  category="debug",
                                  doc="Simulate the FP8 sequence-parallel exchange of the MiniMax-H3 FP4 VSA "
                                  "path on one rank.")
FASTVIDEO_H3_FFN_CHUNK_TOKENS = EnvInt(0,
                                       category="performance",
                                       doc="Inference-only MiniMax-H3 FFN token chunk size; 0 runs the FFN "
                                       "unchunked.")
FASTVIDEO_H3_FP8_ATTENTION = EnvBool(False,
                                     category="performance",
                                     doc="With NVFP4 layer_profile h3_dit_ffn, run MiniMax-H3 attention "
                                     "projections in FP8.")
FASTVIDEO_H3_FP8_GRANULARITY = EnvChoice("tensor",
                                         choices=("tensor", "channel"),
                                         category="performance",
                                         doc="FP8 scaling granularity for FASTVIDEO_H3_FP8_ATTENTION.")
FASTVIDEO_NVFP4_MM_BACKEND = EnvStr("auto",
                                    category="performance",
                                    doc="FlashInfer mm_fp4 backend for NVFP4 linears, e.g. auto or cutlass.")
FASTVIDEO_NVFP4_ACT_AMAX = EnvPath(None,
                                   category="performance",
                                   doc="JSON of calibrated NVFP4 input amax per linear, keyed b<block>.<sub> or "
                                   "full prefix; sets a static activation scale.")
FASTVIDEO_NVFP4_DYNAMIC_ACT = EnvStr("",
                                     category="performance",
                                     doc="NVFP4 linears that derive the activation scale per call: all, or "
                                     "comma-separated layer-name suffixes such as ff.fc_out.")
FASTVIDEO_H3_ADALN_CACHE = EnvBool(False,
                                   category="performance",
                                   doc="Cache MiniMax-H3 AdaLN modulation per timestep instead of keeping the "
                                   "projection weights resident.")
FASTVIDEO_H3_ADALN_TABLE = EnvPath(None,
                                   category="performance",
                                   doc="Precomputed MiniMax-H3 AdaLN modulation table; enables the cache and "
                                   "skips loading the AdaLN projection weights.")
FASTVIDEO_H3_ADALN_DUMP = EnvPath(None,
                                  category="debug",
                                  doc="Write the MiniMax-H3 AdaLN modulation table to this path while sampling.")
FASTVIDEO_H3_SPLICE_TRANSFORMER = EnvPath(None,
                                          category="eval",
                                          doc="Second MiniMax-H3 transformer that runs the late DMD steps "
                                          "(checkpoint step-splice evaluation).")
FASTVIDEO_H3_SPLICE_FROM_STEP = EnvInt(4,
                                       category="eval",
                                       doc="First denoising step run by FASTVIDEO_H3_SPLICE_TRANSFORMER.")
FASTVIDEO_H3_ENCODER_LAYERWISE = EnvBool(False,
                                         category="performance",
                                         doc="Stream MiniMax-H3 text-encoder language layers through exact-size "
                                         "pinned host memory (text-only prompts).")
FASTVIDEO_H3_ENCODER_FUSED_DEQUANT = EnvBool(False,
                                             category="performance",
                                             doc="Expand the serialized NVFP4 MiniMax-H3 text encoder with one "
                                             "fused Triton pass on GPUs without FP4 GEMM.")
FASTVIDEO_H3_VAE_TILE_BATCH = EnvInt(1,
                                     category="performance",
                                     doc="Spatial tiles per MiniMax-H3 video VAE decoder call; 1 decodes per "
                                     "tile.")
FASTVIDEO_H3_REF2VA_MEMO_ENTRIES = EnvInt(0,
                                          category="performance",
                                          doc="Entries in each content-keyed MiniMax-H3 memo of reference encodes "
                                          "(Qwen3-VL Ref2VA presentation, VAE keyframe latents); repeat "
                                          "references are reused exactly. 0 disables.")
FASTVIDEO_H3_VAE_TILE_PARALLEL = EnvBool(False,
                                         category="performance",
                                         doc="Split MiniMax-H3 VAE spatial tiles (not only temporal chunks) across "
                                         "the sequence-parallel ranks in the parallel decode and keyframe encode; "
                                         "bitwise equal to the serial path.")
FASTVIDEO_H3_VAE_INT8_SHARED_QKV = EnvBool(False,
                                           category="performance",
                                           doc="Share the INT8 activation rotation and quantization across the "
                                           "MiniMax-H3 VAE Q/K/V projections.")
FASTVIDEO_H3_VAE_INT8_TRANSPOSE_VIEW = EnvBool(False,
                                               category="performance",
                                               doc="Use transposed weight views in the MiniMax-H3 VAE INT8 "
                                               "projections.")
FASTVIDEO_H3_VAE_INT8_FUSED_DEQUANT = EnvBool(False,
                                              category="performance",
                                              doc="Fused dequantization epilogue for the MiniMax-H3 VAE INT8 "
                                              "projections.")
FASTVIDEO_H3_PINNED_SWAP = EnvBool(True,
                                   category="performance",
                                   doc="Swap offloaded MiniMax-H3 modules through exact-size pinned host "
                                   "arenas.")
FASTVIDEO_H3_PARK_MODULES = EnvStr(None,
                                   category="performance",
                                   doc="Comma-separated MiniMax-H3 denoise modules parked on the host while the "
                                   "text encoder runs, e.g. vae,audio_vae.")
FASTVIDEO_LAYERWISE_OFFLOAD_BUFFERS = EnvBool(False,
                                              category="performance",
                                              doc="Layerwise offload also streams large buffers such as packed "
                                              "FP4/FP8 weights.")
FASTVIDEO_LAYERWISE_RESIDENT_BLOCKS = EnvInt(0,
                                             category="performance",
                                             doc="Keep the first N layerwise-offloaded blocks resident on the "
                                             "GPU.")
FASTVIDEO_H3_SP_PROFILE = EnvBool(False,
                                  category="profiling",
                                  doc="CUDA-event spans per stage over one MiniMax-H3 FP4 VSA DiT forward.")
FASTVIDEO_H3_CAPTURE_QKV = EnvPath(None,
                                   category="debug",
                                   doc="Directory for captured real MiniMax-H3 Q/K/V attention inputs.")
FASTVIDEO_CUDA_MEMORY_CAP_GIB = EnvFloat(0.0,
                                         category="debug",
                                         doc="Cap this process's CUDA allocator at this many GiB to emulate a "
                                         "smaller GPU; 0 leaves it uncapped.")
FASTVIDEO_MEMORY_REPORT = EnvBool(False,
                                  category="debug",
                                  doc="Log bytes held per pipeline component by device and dtype after "
                                  "loading.")

# ================== Tests ==================

FASTVIDEO_TEST_WAN_S2V_MODEL_PATH = EnvPath(None,
                                            category="test",
                                            doc="Wan2.2-S2V-14B weights for test_wan_s2v.py; unset means "
                                            "official_weights/Wan2.2-S2V-14B in the repository.",
                                            deprecated_names=("WAN_S2V_MODEL_PATH", ))
FASTVIDEO_TEST_LTX2_OVERFIT_DATA_DIR = EnvStr("data/cats",
                                              category="test",
                                              doc="Raw data directory for preprocess_ltx2_overfit.py.",
                                              deprecated_names=("LTX2_OVERFIT_DATA_DIR", ))
FASTVIDEO_TEST_LTX2_OVERFIT_CAPTION_JSON = EnvStr("videos2caption_1_sample.json",
                                                  category="test",
                                                  doc="Caption file, relative to the raw data directory.",
                                                  deprecated_names=("LTX2_OVERFIT_CAPTION_JSON", ))
FASTVIDEO_TEST_LTX2_OVERFIT_VIDEO_SUBDIR = EnvStr("video",
                                                  category="test",
                                                  doc="Video subdirectory, relative to the raw data directory.",
                                                  deprecated_names=("LTX2_OVERFIT_VIDEO_SUBDIR", ))
FASTVIDEO_TEST_LTX2_OVERFIT_OUTPUT_DIR = EnvStr("data/ltx2_overfit_preprocessed",
                                                category="test",
                                                doc="Output directory for preprocess_ltx2_overfit.py.",
                                                deprecated_names=("LTX2_OVERFIT_OUTPUT_DIR", ))
FASTVIDEO_TEST_LTX2_OVERFIT_MODEL = EnvStr("FastVideo/LTX2-Distilled-Diffusers",
                                           category="test",
                                           doc="Model repository whose encoders preprocess_ltx2_overfit.py uses.",
                                           deprecated_names=("LTX2_OVERFIT_MODEL", ))
FASTVIDEO_TEST_LTX2_OVERFIT_NUM_COPIES = EnvInt(4,
                                                category="test",
                                                doc="Number of copies of the overfit sample in the parquet file.",
                                                deprecated_names=("LTX2_OVERFIT_NUM_COPIES", ))
FASTVIDEO_TEST_KANDINSKY5_OVERFIT_DATA_DIR = EnvStr("data/kandinsky5_overfit",
                                                    category="test",
                                                    doc="Raw data directory for preprocess_kandinsky5_overfit.py.",
                                                    deprecated_names=("KANDINSKY5_OVERFIT_DATA_DIR", ))
FASTVIDEO_TEST_KANDINSKY5_OVERFIT_OUTPUT_DIR = EnvStr("data/kandinsky5_overfit_preprocessed",
                                                      category="test",
                                                      doc="Output directory for preprocess_kandinsky5_overfit.py.",
                                                      deprecated_names=("KANDINSKY5_OVERFIT_OUTPUT_DIR", ))
FASTVIDEO_TEST_HUNYUAN15_OVERFIT_DATA_DIR = EnvStr("data/hunyuan15_overfit",
                                                   category="test",
                                                   doc="Raw data directory for preprocess_hunyuan15_overfit.py.")
FASTVIDEO_TEST_HUNYUAN15_OVERFIT_OUTPUT_DIR = EnvStr("data/hunyuan15_overfit_preprocessed",
                                                     category="test",
                                                     doc="Output directory for preprocess_hunyuan15_overfit.py.")

# Switches and paths that only tests read. Each old name stays readable, with a
# warning, until the next minor release.
FASTVIDEO_TEST_SSIM_REFERENCE_HF_REPO = EnvStr("FastVideo/ssim-reference-videos",
                                               category="test",
                                               doc="Hugging Face repository that holds the SSIM reference videos.",
                                               deprecated_names=("FASTVIDEO_SSIM_REFERENCE_HF_REPO", ))
FASTVIDEO_TEST_SSIM_REFERENCE_HF_REPO_TYPE = EnvStr("dataset",
                                                    category="test",
                                                    doc="Repository type of FASTVIDEO_TEST_SSIM_REFERENCE_HF_REPO.",
                                                    deprecated_names=("FASTVIDEO_SSIM_REFERENCE_HF_REPO_TYPE", ))
FASTVIDEO_TEST_SSIM_SKIP_REFERENCE_DOWNLOAD = EnvBool(False,
                                                      category="test",
                                                      doc="SSIM tests use local reference videos without downloading.",
                                                      deprecated_names=("FASTVIDEO_SSIM_SKIP_REFERENCE_DOWNLOAD", ))
FASTVIDEO_TEST_SSIM_FULL_QUALITY = EnvBool(False,
                                           category="test",
                                           doc="SSIM tests use the full-quality sampling configurations.",
                                           deprecated_names=("FASTVIDEO_SSIM_FULL_QUALITY", ))
FASTVIDEO_TEST_NIGHTLY = EnvBool(False,
                                 category="test",
                                 doc="Run the nightly end-to-end overfit tests.",
                                 deprecated_names=("FASTVIDEO_NIGHTLY", ))
FASTVIDEO_TEST_ULYSSES_FAULT_RANK = EnvStr(
    None,
    category="test",
    doc="Rank that fails in the Ulysses fault-injection test. The test sets it for its worker processes.",
    deprecated_names=("FASTVIDEO_ULYSSES_FAULT_RANK", ))
FASTVIDEO_TEST_ULYSSES_FAULT_STAGE = EnvStr(
    None,
    category="test",
    doc="Stage that fails in the Ulysses fault-injection test. The test sets it for its worker processes.",
    deprecated_names=("FASTVIDEO_ULYSSES_FAULT_STAGE", ))
FASTVIDEO_TEST_GOLDEN_GATE_DIR = EnvStr(None,
                                        category="test",
                                        doc="Local directory of golden-gate reference tensors.",
                                        deprecated_names=("FASTVIDEO_GOLDEN_GATE_DIR", ))
FASTVIDEO_TEST_WAN22_5B_ALLOW_LOW_MEMORY = EnvBool(
    False,
    category="test",
    doc="Run the MLX Wan2.2 5B real-weights parity test on hosts with little memory.",
    deprecated_names=("FASTVIDEO_WAN22_5B_ALLOW_LOW_MEMORY", ))
FASTVIDEO_TEST_WAN22_5B_ROOT = EnvStr(None,
                                      category="test",
                                      doc="Local Wan2.2 5B checkpoint for the MLX real-weights parity test.",
                                      deprecated_names=("FASTVIDEO_WAN22_5B_ROOT", ))
FASTVIDEO_TEST_GRADNORM_UPDATE = EnvBool(False,
                                         category="test",
                                         doc="Gradient-norm regression tests update their references.",
                                         deprecated_names=("FASTVIDEO_GRADNORM_UPDATE", ))
FASTVIDEO_TEST_DREAMX_WORLD_SSIM_MODEL_PATH = EnvStr("FastVideo/DreamX-World-5B-Cam-Diffusers",
                                                     category="test",
                                                     doc="Model for the DreamX-World camera SSIM test.",
                                                     deprecated_names=("DREAMX_WORLD_SSIM_MODEL_PATH", ))
FASTVIDEO_TEST_DREAMX_WORLD_AR_SSIM_MODEL_PATH = EnvStr("FastVideo/DreamX-World-5B-Diffusers",
                                                        category="test",
                                                        doc="Model for the DreamX-World autoregressive SSIM test.",
                                                        deprecated_names=("DREAMX_WORLD_AR_SSIM_MODEL_PATH", ))
FASTVIDEO_TEST_FLUX_T2I_MODEL_DIR = EnvStr("black-forest-labs/FLUX.1-dev",
                                           category="test",
                                           doc="Model for the Flux text-to-image SSIM test.",
                                           deprecated_names=("FLUX_T2I_MODEL_DIR", ))
FASTVIDEO_TEST_FLUX_TRANSFORMER_PATH = EnvStr(None,
                                              category="test",
                                              doc="Local Flux transformer for the Flux transformer test.",
                                              deprecated_names=("FLUX_TRANSFORMER_PATH", ))
FASTVIDEO_TEST_GAMECRAFT_MODEL_PATH = EnvStr("FastVideo/HunyuanGameCraft-Diffusers",
                                             category="test",
                                             doc="Model for the HunyuanGameCraft SSIM test.",
                                             deprecated_names=("GAMECRAFT_MODEL_PATH", ))
FASTVIDEO_TEST_GEN3C_MODEL_PATH = EnvStr("FastVideo/GEN3C-Cosmos-7B-Diffusers",
                                         category="test",
                                         doc="Model for the GEN3C SSIM test.",
                                         deprecated_names=("GEN3C_MODEL_PATH", ))
FASTVIDEO_TEST_GEN3C_IMAGE_PATH = EnvStr(None,
                                         category="test",
                                         doc="Input image for the GEN3C SSIM test.",
                                         deprecated_names=("GEN3C_TEST_IMAGE_PATH", ))
FASTVIDEO_TEST_GLM_IMAGE_LOCAL_WEIGHTS_DIR = EnvStr(None,
                                                    category="test",
                                                    doc="Local official GLM-Image weights for the GLM-Image SSIM test.",
                                                    deprecated_names=("GLM_IMAGE_LOCAL_WEIGHTS_DIR", ))
FASTVIDEO_TEST_GLM_IMAGE_MODEL_DIR = EnvStr(None,
                                            category="test",
                                            doc="Model for the GLM-Image SSIM test.",
                                            deprecated_names=("GLM_IMAGE_MODEL_DIR", ))
FASTVIDEO_TEST_KANDINSKY5_E2E_NUM_GPUS = EnvInt(1,
                                                category="test",
                                                doc="GPUs for the Kandinsky5 nightly end-to-end overfit test.",
                                                deprecated_names=("KANDINSKY5_E2E_NUM_GPUS", ))
FASTVIDEO_TEST_KANDINSKY5_E2E_WRITE_REFERENCE = EnvBool(
    False,
    category="test",
    doc="The Kandinsky5 nightly end-to-end test writes a missing reference video.",
    deprecated_names=("KANDINSKY5_E2E_WRITE_REFERENCE", ))
FASTVIDEO_TEST_LONGCAT_MODEL_ROOT = EnvStr(None,
                                           category="test",
                                           doc="Local LongCat-Video checkpoint for the golden-gate test.",
                                           deprecated_names=("LONGCAT_MODEL_ROOT", ))
FASTVIDEO_TEST_MINIMAX_H3_GATE_GOLDEN_DIR = EnvStr(None,
                                                   category="test",
                                                   doc="Local directory of MiniMax-H3 golden-gate tensors.",
                                                   deprecated_names=("MINIMAX_H3_GATE_GOLDEN_DIR", ))
FASTVIDEO_TEST_MINIMAX_H3_GATE_LAYER = EnvInt(0,
                                              category="test",
                                              doc="Transformer layer that the MiniMax-H3 golden-gate test checks.",
                                              deprecated_names=("MINIMAX_H3_GATE_LAYER", ))
FASTVIDEO_TEST_MINIMAX_H3_MODEL_ROOT = EnvStr(None,
                                              category="test",
                                              doc="Local MiniMax-H3 checkpoint for the golden-gate test.",
                                              deprecated_names=("MINIMAX_H3_MODEL_ROOT", ))
FASTVIDEO_TEST_SD35_MODEL_DIR = EnvStr("stabilityai/stable-diffusion-3.5-medium",
                                       category="test",
                                       doc="Model for the Stable Diffusion 3.5 SSIM test.",
                                       deprecated_names=("SD35_MODEL_DIR", ))
FASTVIDEO_TEST_TAEH3_REFERENCE_DIR = EnvStr(None,
                                            category="test",
                                            doc="Upstream taehv checkout for the MLX TAEH3 parity test.",
                                            deprecated_names=("TAEH3_REFERENCE_DIR", ))
FASTVIDEO_TEST_WAN_ANIMATE_MODEL_DIR = EnvStr(
    None,
    category="test",
    doc="Local Wan2.2-Animate-14B checkpoint for the Wan-Animate weight tests.",
    deprecated_names=("WAN_ANIMATE_MODEL_PATH", ))
FASTVIDEO_TEST_ZIMAGE_MODEL_DIR = EnvStr("Tongyi-MAI/Z-Image-Turbo",
                                         category="test",
                                         doc="Model for the Z-Image SSIM test.",
                                         deprecated_names=("ZIMAGE_MODEL_DIR", ))
FASTVIDEO_TEST_ZIMAGE_MODEL_REVISION = EnvStr("f332072aa78be7aecdf3ee76d5c247082da564a6",
                                              category="test",
                                              doc="Hugging Face revision of the Z-Image model for its SSIM test.",
                                              deprecated_names=("ZIMAGE_MODEL_REVISION", ))

# Variables that FastVideo no longer reads. Setting one logs a warning; delete
# the entries in the next minor release.
DEPRECATED_VARIABLES = {
    "FASTVIDEO_TARGET_DEVICE": "no code reads it",
    "FASTVIDEO_USE_PRECOMPILED": "no code reads it",
    "FASTVIDEO_RINGBUFFER_WARNING_INTERVAL": "no code reads it",
    "FASTVIDEO_ENGINE_ITERATION_TIMEOUT_S": "no code reads it",
    "FASTVIDEO_SERVER_DEV_MODE": "no code reads it",
    "FASTVIDEO_TEST_DYNAMO_FULLGRAPH_CAPTURE": "no code reads it",
    "FASTVIDEO_TRACE_FUNCTION": "no code reads it",
}


def warn_deprecated_variables() -> None:
    """Log a warning for each variable in DEPRECATED_VARIABLES that is set."""
    for name, reason in DEPRECATED_VARIABLES.items():
        if name in os.environ:
            _warn_once(f"{name} is deprecated and has no effect ({reason}); it will be removed in the next minor "
                       "release.")


def _register_fields() -> dict[str, EnvField]:
    """Name each module-level EnvField after its attribute and return the fields by name."""
    fields = {name: value for name, value in globals().items() if isinstance(value, EnvField)}
    for name, field in fields.items():
        field.name = name
    return fields


environment_variables: dict[str, EnvField] = _register_fields()
