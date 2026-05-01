from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor

import torch

from fastvideo.eval.memory import clear_cache, is_batch_too_large, slice_sample
from fastvideo.eval.registry import get_metric, list_metrics, resolve_group
from fastvideo.eval.types import MetricResult


class Evaluator:
    """Pre-initialized evaluator for repeated, batched evaluation.

    Created via :func:`create_evaluator`.  Models are loaded once during
    construction; subsequent :meth:`evaluate` calls run pure inference.

    Parameters
    ----------
    metrics : list[str] | str
        Metric names, or ``"all"``.
    device : str
        PyTorch device string (e.g. ``"cuda"``).  Ignored when
        *num_gpus* > 1 — each GPU gets its own device automatically.
    num_gpus : int
        Number of GPUs for data-parallel evaluation.  When > 1, models
        are replicated across GPUs and the batch is sharded automatically.
        Uses threads (GIL released during CUDA ops) — no subprocess overhead.
    compile : bool
        Apply ``torch.compile`` to models.
    """

    def __init__(
        self,
        metrics: list[str] | str = "all",
        device: str = "cuda",
        num_gpus: int = 1,
        compile: bool = False,
    ) -> None:
        # Stash construction config so unload() / reload() can rebuild
        # without making the caller remember the original args.
        self._names = _resolve_metric_names(metrics)
        self._num_gpus = num_gpus
        self._device = "cuda:0" if num_gpus > 1 else device
        self._compile = compile
        self._unloaded = False

        self._gpu_metrics: list[dict] | None = None
        self._metrics: dict = {}
        self._load()

    def _load(self) -> None:
        """Instantiate every configured metric on its target device.

        Used by ``__init__`` and :meth:`reload`. Idempotent only in the
        sense that calling it on an already-loaded evaluator simply
        re-creates the metric instances (re-paying the model load cost);
        no leak, but no caching.
        """
        if self._num_gpus > 1:
            self._gpu_metrics = []
            for gpu_id in range(self._num_gpus):
                dev = f"cuda:{gpu_id}"
                gpu_m: dict = {}
                for name in self._names:
                    gpu_m[name] = self._build_metric(name, dev)
                self._gpu_metrics.append(gpu_m)
            self._metrics = self._gpu_metrics[0]
        else:
            self._gpu_metrics = None
            self._metrics = {
                name: self._build_metric(name, self._device)
                for name in self._names
            }
        self._unloaded = False

    def _build_metric(self, name: str, device: str):
        m = get_metric(name)
        m.to(device)
        m.setup()
        if self._compile and hasattr(m, "_model") and m._model is not None:
            m._model = torch.compile(m._model)
        return m

    def release_cuda_memory(self) -> None:
        """Free CUDA caches between calls. Eval models stay loaded.

        Call in a ``finally`` after each ``evaluate(...)`` in a training
        loop so transient activation buffers don't pin GPU memory the
        trainer needs::

            try:
                scores = ev.evaluate(video=...)
            finally:
                ev.release_cuda_memory()
        """
        clear_cache()                              # gc + torch.cuda.empty_cache
        if torch.cuda.is_available():
            try:
                torch.cuda.ipc_collect()           # release cross-process IPC handles
            except Exception:
                pass

    def unload(self) -> None:
        """Drop every metric reference so the underlying models become
        GC-able, then free CUDA caches. Use when training-time memory
        pressure spikes and you need the eval model GPU memory back.

        Reversible — call :meth:`reload` to rebuild the same metrics.
        Subsequent ``evaluate`` calls before ``reload`` raise.
        """
        self._metrics = {}
        if self._gpu_metrics is not None:
            self._gpu_metrics = []
        self._unloaded = True
        self.release_cuda_memory()

    def reload(self) -> None:
        """Rebuild metrics dropped by :meth:`unload` using the same
        configuration. No-op if the evaluator is already loaded.

        This re-pays the model-load cost (`from_pretrained`, weight
        copy to GPU). Equivalent to constructing a new ``Evaluator``
        with the original args, but you keep the same handle.
        """
        if not self._unloaded:
            return
        self._load()

    def calibrate(
        self,
        *,
        height: int,
        width: int,
        num_frames: int,
        safety_margin: float = 0.85,
    ) -> dict[str, int]:
        """Auto-detect max batch size for each metric's batch unit.

        Calibrates on the first GPU; applies the same chunk_size to all
        GPU replicas when using multi-GPU.
        """
        device = torch.device(self._device)
        result = {}

        if device.type != "cuda":
            for name, m in self._metrics.items():
                m._chunk_size = 2**30
                result[name] = m._chunk_size
            return result

        trial_kw = dict(height=height, width=width, num_frames=num_frames)

        for name, m in self._metrics.items():
            clear_cache()

            torch.cuda.reset_peak_memory_stats(device)
            baseline = torch.cuda.memory_allocated(device)

            try:
                m.trial_forward(1, **trial_kw)
            except Exception:
                m._chunk_size = 1
                result[name] = 1
                clear_cache()
                continue

            peak = torch.cuda.max_memory_allocated(device)
            per_unit = peak - baseline
            clear_cache()

            total = torch.cuda.get_device_properties(device).total_memory
            free = (total - baseline) * safety_margin
            if per_unit > 0:
                raw = max(1, int(free / per_unit))
                mem_limit = 1 << (raw.bit_length() - 1)
            else:
                mem_limit = 2**30

            validated = mem_limit
            while validated > 1:
                clear_cache()
                try:
                    m.trial_forward(validated, **trial_kw)
                    break
                except RuntimeError as e:
                    if is_batch_too_large(e):
                        validated //= 2
                    else:
                        raise
            clear_cache()

            probe = min(validated, 4)
            if probe > 1 and m.needs_gpu:
                def _time_trial(bs):
                    m.trial_forward(bs, **trial_kw)
                    torch.cuda.synchronize(device)
                    torch.cuda.synchronize(device)
                    t0 = time.perf_counter()
                    for _ in range(3):
                        m.trial_forward(bs, **trial_kw)
                    torch.cuda.synchronize(device)
                    return (time.perf_counter() - t0) / 3

                t1 = _time_trial(1)
                clear_cache()
                tp = _time_trial(probe)
                clear_cache()

                speedup = (t1) / (tp / probe) if tp > 0 else 1.0

                if speedup < 1.2:
                    m._chunk_size = 1
                else:
                    marginal = (tp - t1) / (probe - 1)
                    fixed = t1 - marginal
                    if fixed > 0 and marginal > 0:
                        tp_limit = max(1, int(9 * fixed / marginal))
                        tp_limit = 1 << (tp_limit.bit_length() - 1)
                        m._chunk_size = min(validated, tp_limit)
                    else:
                        m._chunk_size = validated
            else:
                m._chunk_size = validated

            result[name] = m._chunk_size

        # Propagate calibrated chunk_sizes to all GPU replicas
        if self._gpu_metrics is not None:
            for gpu_id in range(1, self._num_gpus):
                for name in self._metrics:
                    self._gpu_metrics[gpu_id][name]._chunk_size = (
                        self._metrics[name]._chunk_size
                    )

        return result

    def evaluate(self, **kwargs) -> dict[str, MetricResult] | list[dict[str, MetricResult]]:
        """Evaluate all metrics on the given inputs.

        Pass tensors as keyword arguments.  ``video`` should be
        ``(T, C, H, W)`` for a single sample or ``(B, T, C, H, W)``
        for a batch.  Returns a single result dict for one sample,
        or a list of result dicts for a batch.

        When ``num_gpus > 1``, the batch is automatically sharded
        across GPUs and results are gathered in order.
        """
        if getattr(self, "_unloaded", False):
            raise RuntimeError(
                "Evaluator was unloaded; build a new one with "
                "create_evaluator(...) before scoring again."
            )

        sample, batch_size = _add_batch_dim(kwargs)

        if self._num_gpus > 1 and batch_size > 1:
            return self._evaluate_multi_gpu(sample, batch_size)

        return self._evaluate_single(sample, batch_size)

    def _evaluate_single(
        self, sample: dict, batch_size: int,
        metrics: dict | None = None,
    ) -> dict[str, MetricResult] | list[dict[str, MetricResult]]:
        """Single-device evaluation."""
        if metrics is None:
            metrics = self._metrics

        all_results: list[dict[str, MetricResult]] = [{} for _ in range(batch_size)]
        for name, m in metrics.items():
            if m.batch_unit == "video" and m._chunk_size is not None and batch_size > m._chunk_size:
                batch_results = self._compute_chunked(m, sample, batch_size, m._chunk_size)
            else:
                batch_results = m.compute(sample)
            for i, res in enumerate(batch_results):
                all_results[i][name] = res

        if batch_size == 1:
            return all_results[0]
        return all_results

    def _evaluate_multi_gpu(
        self, sample: dict, batch_size: int,
    ) -> list[dict[str, MetricResult]]:
        """Shard batch across GPUs, evaluate in parallel threads."""
        n = self._num_gpus
        # Compute shard boundaries
        shard_sizes = []
        for i in range(n):
            start = i * batch_size // n
            end = (i + 1) * batch_size // n
            shard_sizes.append((start, end))

        def _run_on_gpu(gpu_id: int) -> list[dict[str, MetricResult]]:
            start, end = shard_sizes[gpu_id]
            if start >= end:
                return []
            shard = slice_sample(sample, start, end)
            # Move tensors to this GPU
            dev = f"cuda:{gpu_id}"
            gpu_shard = {}
            for k, v in shard.items():
                if isinstance(v, torch.Tensor):
                    gpu_shard[k] = v.to(dev)
                else:
                    gpu_shard[k] = v
            shard_bs = end - start
            result = self._evaluate_single(gpu_shard, shard_bs,
                                           metrics=self._gpu_metrics[gpu_id])
            # Ensure result is a list
            if isinstance(result, dict):
                result = [result]
            # Move results back to CPU
            return result

        # Run all GPUs in parallel — GIL is released during CUDA ops
        with ThreadPoolExecutor(max_workers=n) as pool:
            futures = [pool.submit(_run_on_gpu, i) for i in range(n)]
            shard_results = [f.result() for f in futures]

        # Merge in original order
        all_results = []
        for shard in shard_results:
            all_results.extend(shard)

        return all_results

    @staticmethod
    def _compute_chunked(metric, sample: dict, batch_size: int, chunk_size: int) -> list[MetricResult]:
        all_results = []
        for start in range(0, batch_size, chunk_size):
            end = min(start + chunk_size, batch_size)
            chunk = slice_sample(sample, start, end)
            all_results.extend(metric.compute(chunk))
        return all_results


def create_evaluator(
    metrics: list[str] | str = "all",
    device: str = "cuda",
    num_gpus: int = 1,
    compile: bool = False,
) -> Evaluator:
    """Create a reusable evaluator.

    Parameters
    ----------
    metrics : list[str] | str
        Metric names, or ``"all"``.
    device : str
        PyTorch device string.  Ignored when *num_gpus* > 1.
    num_gpus : int
        Number of GPUs.  When > 1, the batch is automatically sharded
        across GPUs.  Models are loaded once per GPU.
    compile : bool
        Apply ``torch.compile`` to models.
    """
    return Evaluator(metrics=metrics, device=device, num_gpus=num_gpus,
                     compile=compile)


# --- helpers ---

def _resolve_metric_names(metrics: list[str] | str) -> list[str]:
    """Resolve metric names, supporting groups like ``"vbench"`` or ``"common"``.

    Examples::

        "all"                      → every registered metric
        "vbench"                   → all vbench.* metrics
        "vbench.aesthetic_quality"  → just that one
        ["vbench", "common.ssim"]  → all vbench.* + common.ssim
    """
    if metrics == "all":
        return list_metrics()
    if isinstance(metrics, str):
        metrics = [metrics]

    names = []
    for m in metrics:
        group = resolve_group(m)
        if group is not None:
            names.extend(group)
        else:
            names.append(m)
    return names


def _add_batch_dim(kwargs: dict) -> tuple[dict, int]:
    """Ensure tensor args have a leading batch dim. Returns (sample, B)."""
    sample = dict(kwargs)
    video = sample.get("video")
    if video is None:
        return sample, 1
    if video.dim() == 4:  # (T,C,H,W) → (1,T,C,H,W)
        sample["video"] = video.unsqueeze(0)
        ref = sample.get("reference")
        if ref is not None and isinstance(ref, torch.Tensor) and ref.dim() == 4:
            sample["reference"] = ref.unsqueeze(0)
        return sample, 1
    return sample, video.shape[0]
