"""Extract a FastVideo LoRA adapter from the difference between two checkpoints.

The extractor supports ordinary low-rank matrix deltas as well as parameters
that cannot be represented by a LoRA product:

* ``.diff`` / ``.diff_b`` / ``.diff_param`` store exact additive deltas.
* ``.set_weight`` / ``.set_param`` store parameters absent from the base checkpoint.

Indexed safetensors are streamed one tensor at a time, so extracting from large
transformers does not require both state dictionaries in host memory. Exact CPU
SVD remains the default for compatibility; GPU and randomized SVD are opt-in.
"""

from __future__ import annotations

import argparse
from collections.abc import Iterable, Iterator, Sequence
from contextlib import ExitStack, contextmanager, suppress
from dataclasses import asdict, dataclass
import hashlib
import json
import logging
import os
from pathlib import Path
import re
import shutil
from typing import Any, Protocol

# Pipeline loading imports distributed code. Keep the single-process defaults in
# place for the legacy --load-mode pipeline fallback.
os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
os.environ.setdefault("MASTER_PORT", "29500")
os.environ.setdefault("WORLD_SIZE", "1")
os.environ.setdefault("RANK", "0")
os.environ.setdefault("LOCAL_RANK", "0")

import torch
from huggingface_hub import snapshot_download
from safetensors import safe_open
from safetensors.torch import load_file, save_file
from tqdm import tqdm

LOG = logging.getLogger("extract_lora")
INDEX_FILENAME = "diffusion_pytorch_model.safetensors.index.json"
FORMAT_VERSION = "fastvideo-lora-v2"
# Scratch lives in a subdirectory we create so cleanup never reaches a caller's files
# when --work-dir points at a directory that already holds something else.
WORK_SUBDIR = "fastvideo-lora-extract"

DIFF_SUFFIX = ".diff"
DIFF_BIAS_SUFFIX = ".diff_b"
DIFF_PARAM_SUFFIX = ".diff_param"
SET_WEIGHT_SUFFIX = ".set_weight"
SET_PARAM_SUFFIX = ".set_param"

_DTYPE_MAP = {
    "float32": torch.float32,
    "float": torch.float32,
    "fp32": torch.float32,
    "float16": torch.float16,
    "half": torch.float16,
    "fp16": torch.float16,
    "bfloat16": torch.bfloat16,
    "bf16": torch.bfloat16,
}


class TensorReader(Protocol):
    """Random access to one checkpoint's transformer tensors."""

    source: str
    fingerprint: str

    @property
    def keys(self) -> set[str]: ...

    def get_tensor(self, key: str) -> torch.Tensor: ...

    def get_shape(self, key: str) -> tuple[int, ...]: ...

    def assert_unchanged(self) -> None: ...

    def __enter__(self) -> "TensorReader": ...

    def __exit__(self, *args: object) -> None: ...


class DictTensorReader:
    """Reader wrapper for the legacy pipeline-loading path."""

    def __init__(self, state_dict: dict[str, torch.Tensor], source: str) -> None:
        self.state_dict = state_dict
        self.source = source
        self.fingerprint = f"pipeline:{source}"

    @property
    def keys(self) -> set[str]:
        return set(self.state_dict)

    def get_tensor(self, key: str) -> torch.Tensor:
        return self.state_dict[key]

    def get_shape(self, key: str) -> tuple[int, ...]:
        return tuple(self.state_dict[key].shape)

    def assert_unchanged(self) -> None:
        return None

    def __enter__(self) -> "DictTensorReader":
        return self

    def __exit__(self, *args: object) -> None:
        return None


class IndexedSafetensorsReader:
    """Stream tensors from an indexed or unsharded transformer component."""

    def __init__(self, transformer_dir: Path) -> None:
        self.transformer_dir = transformer_dir
        self.source = str(transformer_dir.resolve())
        index_path = transformer_dir / INDEX_FILENAME
        if index_path.is_file():
            index = json.loads(index_path.read_text(encoding="utf-8"))
            self.weight_map: dict[str, str] = index["weight_map"]
        else:
            self.weight_map = {}
            for path in sorted(transformer_dir.glob("*.safetensors")):
                with safe_open(path, framework="pt") as handle:
                    for key in handle.keys():
                        if key in self.weight_map:
                            raise ValueError(f"Tensor {key} occurs in multiple shards under {transformer_dir}")
                        self.weight_map[key] = path.name
        if not self.weight_map:
            raise ValueError(f"No transformer safetensors found under {transformer_dir}")

        shard_names = sorted(set(self.weight_map.values()))
        identity_files = ([index_path] if index_path.is_file() else []) + [
            transformer_dir / shard for shard in shard_names
        ]
        before = _file_stats(identity_files)
        self.fingerprint = _fingerprint_files(identity_files, transformer_dir)
        after = _file_stats(identity_files)
        if before != after:
            raise RuntimeError(f"Checkpoint files changed while fingerprinting {transformer_dir}")
        self._identity_files = identity_files
        self._file_stats = after

        stack = ExitStack()
        try:
            shards = {
                shard: stack.enter_context(safe_open(transformer_dir / shard, framework="pt", device="cpu"))
                for shard in shard_names
            }
        except Exception:
            stack.close()
            raise
        self._stack = stack
        self._shards = shards

    @property
    def keys(self) -> set[str]:
        return set(self.weight_map)

    def get_tensor(self, key: str) -> torch.Tensor:
        return self._shards[self.weight_map[key]].get_tensor(key)

    def get_shape(self, key: str) -> tuple[int, ...]:
        return tuple(self._shards[self.weight_map[key]].get_slice(key).get_shape())

    def assert_unchanged(self) -> None:
        if _file_stats(self._identity_files) != self._file_stats:
            raise RuntimeError(f"Checkpoint files changed during extraction: {self.transformer_dir}")

    def __enter__(self) -> "IndexedSafetensorsReader":
        return self

    def __exit__(self, *args: object) -> None:
        self._stack.close()


@dataclass(frozen=True)
class ExtractionConfig:
    base_source: str
    finetuned_source: str
    base_fingerprint: str
    finetuned_fingerprint: str
    rank: int
    full_rank: bool
    min_delta: float
    device: str
    svd_method: str
    randomized_q: int | None
    oversample: int
    niter: int
    seed: int
    factor_dtype: str
    dense_dtype: str
    replacement_dtype: str
    dense_payload: bool
    exact_tensor_patterns: tuple[str, ...]


def configure_logging(level: str = "INFO") -> None:
    if LOG.handlers:
        LOG.setLevel(level)
        return
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s", datefmt="%Y-%m-%d %H:%M:%S"))
    LOG.addHandler(handler)
    LOG.setLevel(level)


def _atomic_json_dump(data: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def _file_stats(paths: Sequence[Path]) -> dict[str, tuple[int, int, int]]:
    return {
        str(path.resolve()): (path.stat().st_size, path.stat().st_mtime_ns, path.stat().st_ino)
        for path in paths
    }


def _fingerprint_files(paths: Sequence[Path], root: Path) -> str:
    """Strong checkpoint identity used to reject stale resume shards."""
    digest = hashlib.sha256()
    for path in sorted(paths, key=lambda item: str(item)):
        digest.update(str(path.relative_to(root)).encode("utf-8"))
        with path.open("rb") as handle:
            while chunk := handle.read(8 * 1024 * 1024):
                digest.update(chunk)
    return digest.hexdigest()


def _torch_dtype(name: str) -> torch.dtype:
    try:
        return _DTYPE_MAP[name.lower()]
    except KeyError as error:
        raise ValueError(f"Unsupported dtype {name!r}; choose from {sorted(_DTYPE_MAP)}") from error


def _resolve_output_dtype(name: str, source_dtype: torch.dtype) -> torch.dtype:
    return source_dtype if name.lower() == "source" else _torch_dtype(name)


def _resolve_transformer_dir(model: str, revision: str | None = None) -> Path:
    """Resolve a local/HF model to its transformer directory.

    For a Hub model, download only the transformer component. This is material
    for compound models: MiniMax-H3 is roughly 464 GiB in total while one
    transformer is about 65 GiB.
    """
    path = Path(model).expanduser()
    if path.exists():
        if revision is not None:
            raise ValueError(f"revision={revision!r} cannot be used with local model path {path}")
        if (path / "transformer").is_dir():
            return path / "transformer"
        if (path / INDEX_FILENAME).is_file() or any(path.glob("*.safetensors")):
            return path
        raise FileNotFoundError(f"No transformer safetensors found under {path}")

    snapshot = Path(
        snapshot_download(
            repo_id=model,
            revision=revision,
            allow_patterns=["transformer/*"],
        ))
    transformer_dir = snapshot / "transformer"
    if not transformer_dir.is_dir():
        raise FileNotFoundError(f"Downloaded repository {model} has no transformer directory")
    return transformer_dir


def get_pipeline_class_for_model(model_path: str):
    """Return the FastVideo pipeline class for the legacy loading mode."""
    from fastvideo.fastvideo_args import WorkloadType
    from fastvideo.pipelines.pipeline_registry import PipelineType, get_pipeline_registry
    from fastvideo.utils import maybe_download_model_index

    config = maybe_download_model_index(model_path)
    pipeline_name = config.get("_class_name")
    if pipeline_name is None:
        raise ValueError(f"Model config for {model_path} is missing _class_name")
    registry = get_pipeline_registry(PipelineType.BASIC)
    return registry.resolve_pipeline_cls(pipeline_name, PipelineType.BASIC, WorkloadType.T2V)


def load_transformer_state_dict_from_model(
    model_path: str,
    num_gpus: int = 1,
    dit_cpu_offload: bool = True,
    vae_cpu_offload: bool = True,
    text_encoder_cpu_offload: bool = True,
    pin_cpu_memory: bool = True,
) -> dict[str, torch.Tensor]:
    """Load a transformer through FastVideo. Prefer indexed loading for extraction."""
    pipeline_cls = get_pipeline_class_for_model(model_path)
    pipeline = pipeline_cls.from_pretrained(
        model_path,
        num_gpus=num_gpus,
        inference_mode=True,
        dit_cpu_offload=dit_cpu_offload,
        # This helper must materialize real parameters. Layerwise offload may expose
        # placeholder tensors through state_dict() before a layer is activated.
        dit_layerwise_offload=False,
        vae_cpu_offload=vae_cpu_offload,
        text_encoder_cpu_offload=text_encoder_cpu_offload,
        pin_cpu_memory=pin_cpu_memory,
    )
    transformer = getattr(pipeline, "transformer", None)
    if transformer is None:
        modules = getattr(pipeline, "modules", None)
        transformer = modules.get("transformer") if isinstance(modules, dict) else None
    if transformer is None:
        nested_pipeline = getattr(pipeline, "pipeline", None)
        transformer = getattr(nested_pipeline, "transformer", None)
    if transformer is None:
        raise RuntimeError("Transformer not found in pipeline")

    try:
        from torch.distributed.tensor import DTensor
    except ImportError:
        DTensor = None  # type: ignore[assignment,misc]

    result: dict[str, torch.Tensor] = {}
    for key, value in transformer.state_dict().items():
        if DTensor is not None and isinstance(value, DTensor):
            value = value.to_local()
        result[key] = value.detach().cpu().contiguous()
    del pipeline, transformer
    torch.cuda.empty_cache()
    return result


def load_transformer_state_dict_from_safetensors(model_path: str) -> dict[str, torch.Tensor]:
    """Compatibility helper that materializes a directly loaded state dictionary."""
    with IndexedSafetensorsReader(_resolve_transformer_dir(model_path)) as reader:
        return {key: reader.get_tensor(key) for key in sorted(reader.keys)}


@contextmanager
def _open_readers(
    base: str,
    finetuned: str,
    base_revision: str | None,
    finetuned_revision: str | None,
    load_mode: str,
) -> Iterator[tuple[TensorReader, TensorReader]]:
    revisions_requested = base_revision is not None or finetuned_revision is not None
    if load_mode == "pipeline" and revisions_requested:
        raise ValueError("--base-revision/--ft-revision require indexed loading; pipeline loading cannot honor them")
    if load_mode in {"auto", "indexed"}:
        stack = ExitStack()
        try:
            base_reader = stack.enter_context(IndexedSafetensorsReader(_resolve_transformer_dir(base, base_revision)))
            finetuned_reader = stack.enter_context(
                IndexedSafetensorsReader(_resolve_transformer_dir(finetuned, finetuned_revision)))
        except Exception:
            stack.close()
            if load_mode == "indexed" or revisions_requested:
                raise
            LOG.warning("Indexed loading failed; falling back to pipeline loading", exc_info=True)
        else:
            LOG.info("Streaming indexed transformers: base=%s finetuned=%s", base_reader.source,
                     finetuned_reader.source)
            try:
                yield base_reader, finetuned_reader
            finally:
                stack.close()
            return

    base_state = load_transformer_state_dict_from_model(base)
    finetuned_state = load_transformer_state_dict_from_model(finetuned)
    yield DictTensorReader(base_state, base), DictTensorReader(finetuned_state, finetuned)


def is_extractable_weight(key: str) -> bool:
    """Backward-compatible name filter for matrices suitable for LoRA."""
    if not key.endswith("weight"):
        return False
    lowered = key.lower()
    return not any(fragment in lowered for fragment in ("norm", "bias", "embedding"))


def dense_payload_key(param_name: str) -> str:
    if param_name.endswith(".weight"):
        return param_name.removesuffix(".weight") + DIFF_SUFFIX
    if param_name.endswith(".bias"):
        return param_name.removesuffix(".bias") + DIFF_BIAS_SUFFIX
    return param_name + DIFF_PARAM_SUFFIX


def build_dense_payload(
    base_sd: dict[str, torch.Tensor],
    ft_sd: dict[str, torch.Tensor],
    low_rank_keys: set[str],
    min_delta: float,
) -> dict[str, torch.Tensor]:
    """Compatibility helper for callers using in-memory state dictionaries."""
    payload: dict[str, torch.Tensor] = {}
    for key in sorted(ft_sd):
        if key in low_rank_keys:
            continue
        finetuned = ft_sd[key].detach().cpu()
        base = base_sd.get(key)
        if base is None:
            output_key = (key.removesuffix(".weight") + SET_WEIGHT_SUFFIX
                          if key.endswith(".weight") else key + SET_PARAM_SUFFIX)
            payload[output_key] = finetuned.contiguous()
            continue
        if base.shape != finetuned.shape or torch.equal(base.cpu(), finetuned):
            continue
        delta = finetuned.float() - base.cpu().float()
        if float(delta.abs().max()) <= min_delta:
            continue
        output_key = dense_payload_key(key)
        if output_key is not None:
            payload[output_key] = delta.to(finetuned.dtype).contiguous()
    return payload


def _seed_for_key(seed: int, key: str) -> int:
    digest = hashlib.sha256(f"{seed}:{key}".encode()).digest()
    return int.from_bytes(digest[:8], "little") % (2**31)


def _compile_patterns(patterns: Sequence[str]) -> tuple[re.Pattern[str], ...]:
    return tuple(re.compile(pattern) for pattern in patterns)


def _validate_exact_patterns(patterns: Sequence[str], keys: Iterable[str]) -> None:
    """Reject a pattern that matches no tensor rather than silently factorizing it anyway.

    A pattern is only ever used to *exclude* tensors from factorization, so one that
    matches nothing is indistinguishable from not passing it at all -- the usual cause
    is shell escaping, where a doubled backslash makes ``\\.`` mean "backslash, any
    character" instead of a literal dot.
    """
    keys = sorted(keys)
    unmatched = [pattern for pattern in patterns if not any(re.search(pattern, key) for key in keys)]
    if unmatched:
        rendered = ", ".join(repr(pattern) for pattern in unmatched)
        raise ValueError(
            "--exact-tensor-pattern matched no tensor in the fine-tuned checkpoint: " + rendered +
            ". Those tensors would be rank-truncated instead of kept exact; check the escaping "
            r"(inside shell single quotes write '\.', not '\\.').")


def _should_factor(
    key: str,
    shape: tuple[int, ...],
    exact_patterns: Sequence[re.Pattern[str]],
) -> bool:
    if not is_extractable_weight(key) or len(shape) != 2:
        return False
    return not any(pattern.search(key) for pattern in exact_patterns)


def _factorize_delta(
    delta: torch.Tensor,
    rank: int,
    full_rank: bool,
    method: str,
    randomized_q: int | None,
    oversample: int,
    niter: int,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, str]:
    available_rank = min(delta.shape)
    chosen_rank = available_rank if full_rank or rank <= 0 else min(rank, available_rank)
    if chosen_rank == 0:
        raise ValueError(f"Cannot factorize empty matrix with shape {tuple(delta.shape)}")

    if method == "exact":
        u, singular_values, vh = torch.linalg.svd(delta, full_matrices=False)
        v = vh.mT
        method_description = "exact"
    else:
        q = randomized_q if randomized_q is not None else chosen_rank + oversample
        q = min(available_rank, max(chosen_rank, q))
        if q == available_rank:
            u, singular_values, vh = torch.linalg.svd(delta, full_matrices=False)
            v = vh.mT
            method_description = "exact-full-basis"
        else:
            devices = [delta.device] if delta.device.type == "cuda" else []
            with torch.random.fork_rng(devices=devices):
                torch.manual_seed(seed)
                if delta.device.type == "cuda":
                    torch.cuda.manual_seed(seed)
                u, singular_values, v = torch.svd_lowrank(delta, q=q, niter=niter)
            method_description = f"randomized-q{q}-niter{niter}"

    singular_values = singular_values[:chosen_rank].float()
    sqrt_s = singular_values.sqrt()
    lora_b = (u[:, :chosen_rank].float() * sqrt_s.unsqueeze(0)).contiguous()
    lora_a = (v[:, :chosen_rank].float() * sqrt_s.unsqueeze(0)).mT.contiguous()
    return lora_a, lora_b, singular_values, method_description


def _work_namespace(out_path: Path) -> str:
    resolved = str(out_path.expanduser().resolve(strict=False))
    digest = hashlib.sha256(resolved.encode("utf-8")).hexdigest()[:12]
    stem = re.sub(r"[^A-Za-z0-9_.-]+", "-", out_path.stem).strip("-.") or "adapter"
    return f"{stem}-{digest}"


def _clear_work_dir(work_dir: Path) -> None:
    """Remove the scratch artifacts this script writes, never the directory itself.

    Assembly is driven by the manifest rather than by a directory listing, so dropping
    the manifest is what makes a rerun start clean; the tensor shards only cost disk.
    """
    shutil.rmtree(work_dir / "tensors", ignore_errors=True)
    (work_dir / "manifest.json").unlink(missing_ok=True)


def _prepare_work_dir(work_dir: Path, config: ExtractionConfig, resume: bool) -> tuple[Path, dict[str, Any]]:
    manifest_path = work_dir / "manifest.json"
    expected = asdict(config)
    expected["exact_tensor_patterns"] = list(config.exact_tensor_patterns)
    if resume and manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("config") != expected:
            raise ValueError(f"Resume configuration does not match {manifest_path}")
        return manifest_path, manifest
    if not resume:
        _clear_work_dir(work_dir)

    (work_dir / "tensors").mkdir(parents=True, exist_ok=True)
    manifest = {"format": FORMAT_VERSION, "config": expected, "layers": {}}
    _atomic_json_dump(manifest, manifest_path)
    return manifest_path, manifest


def _validate_key_sets(base: TensorReader, finetuned: TensorReader) -> None:
    missing = sorted(base.keys - finetuned.keys)
    if missing:
        raise ValueError(f"Fine-tuned checkpoint is missing {len(missing)} base tensors; first keys: {missing[:5]}")
    for key in sorted(base.keys & finetuned.keys):
        if base.get_shape(key) != finetuned.get_shape(key):
            raise ValueError(
                f"Shape mismatch for {key}: base={base.get_shape(key)}, finetuned={finetuned.get_shape(key)}")


def _save_layer_payload(path: Path, payload: dict[str, torch.Tensor], source_key: str) -> None:
    save_file(
        {key: tensor.detach().cpu().contiguous() for key, tensor in payload.items()},
        str(path),
        metadata={"source_key": source_key, "format": FORMAT_VERSION},
    )


def _extract_layers(
    base: TensorReader,
    finetuned: TensorReader,
    work_dir: Path,
    manifest_path: Path,
    manifest: dict[str, Any],
    config: ExtractionConfig,
) -> None:
    device = torch.device(config.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA device requested but CUDA is unavailable: {device}")
    factor_dtype = _torch_dtype(config.factor_dtype)
    exact_patterns = _compile_patterns(config.exact_tensor_patterns)

    for index, key in enumerate(tqdm(sorted(finetuned.keys), desc="extracting LoRA", unit="tensor")):
        tensor_file = work_dir / "tensors" / f"{index:05d}.safetensors"
        existing = manifest["layers"].get(key)
        if existing is not None and (existing.get("tensor_file") is None or tensor_file.is_file()):
            continue

        finetuned_tensor = finetuned.get_tensor(key)
        if key not in base.keys:
            if not config.dense_payload:
                manifest["layers"][key] = {
                    "kind": "skipped",
                    "shape": list(finetuned_tensor.shape),
                    "tensor_file": None,
                    "reason": "dense payload disabled",
                }
                _atomic_json_dump(manifest, manifest_path)
                continue
            is_weight = key.endswith(".weight")
            output_key = (key.removesuffix(".weight") + SET_WEIGHT_SUFFIX if is_weight else key + SET_PARAM_SUFFIX)
            output_dtype = _resolve_output_dtype(config.replacement_dtype, finetuned_tensor.dtype)
            _save_layer_payload(tensor_file, {output_key: finetuned_tensor.to(output_dtype)}, key)
            manifest["layers"][key] = {
                "kind": "set_weight" if is_weight else "set_param",
                "shape": list(finetuned_tensor.shape),
                "tensor_file": tensor_file.name,
                "output_keys": [output_key],
            }
            _atomic_json_dump(manifest, manifest_path)
            continue

        base_tensor = base.get_tensor(key)
        delta = finetuned_tensor.to(device=device, dtype=torch.float32, copy=True)
        delta.sub_(base_tensor.to(device=device, dtype=torch.float32))
        max_abs_delta = float(delta.abs().max().item()) if delta.numel() else 0.0
        if max_abs_delta <= config.min_delta:
            manifest["layers"][key] = {
                "kind": "unchanged",
                "shape": list(delta.shape),
                "tensor_file": None,
                "max_abs_delta": max_abs_delta,
            }
            _atomic_json_dump(manifest, manifest_path)
            del delta, base_tensor, finetuned_tensor
            continue

        shape = tuple(delta.shape)
        if _should_factor(key, shape, exact_patterns):
            delta_fro_sq = float(delta.double().square().sum().item())
            lora_a, lora_b, singular_values, method = _factorize_delta(
                delta,
                rank=config.rank,
                full_rank=config.full_rank,
                method=config.svd_method,
                randomized_q=config.randomized_q,
                oversample=config.oversample,
                niter=config.niter,
                seed=_seed_for_key(config.seed, key),
            )
            module_name = key.removesuffix(".weight")
            actual_rank = lora_a.shape[0]
            output_keys = [
                f"{module_name}.lora_A.weight",
                f"{module_name}.lora_B.weight",
            ]
            # The extracted factors use alpha == rank, which is the loader's
            # default when no scalar is present. Emitting per-layer rank/alpha
            # bookkeeping adds hundreds of keys, and several adapter naming
            # schemes map those scalars differently from their A/B factors.
            payload = {
                output_keys[0]: lora_a.to(factor_dtype),
                output_keys[1]: lora_b.to(factor_dtype),
            }
            _save_layer_payload(tensor_file, payload, key)
            captured = float(singular_values.double().square().sum().item())
            residual = (max(0.0, 1.0 - captured / delta_fro_sq)**0.5) if delta_fro_sq else 0.0
            manifest["layers"][key] = {
                "kind": "lora",
                "shape": list(shape),
                "tensor_file": tensor_file.name,
                "output_keys": output_keys,
                "rank": actual_rank,
                "method": method,
                "delta_frobenius_norm": delta_fro_sq**0.5,
                "relative_residual": residual,
                "max_abs_delta": max_abs_delta,
            }
            del lora_a, lora_b, singular_values, payload
        else:
            output_key = dense_payload_key(key)
            if config.dense_payload:
                output_dtype = _resolve_output_dtype(config.dense_dtype, finetuned_tensor.dtype)
                _save_layer_payload(tensor_file, {output_key: delta.to(output_dtype)}, key)
                manifest["layers"][key] = {
                    "kind": "diff",
                    "shape": list(shape),
                    "tensor_file": tensor_file.name,
                    "output_keys": [output_key],
                    "max_abs_delta": max_abs_delta,
                }
            else:
                manifest["layers"][key] = {
                    "kind": "skipped",
                    "shape": list(shape),
                    "tensor_file": None,
                    "reason": "dense payload disabled",
                    "max_abs_delta": max_abs_delta,
                }

        _atomic_json_dump(manifest, manifest_path)
        del delta, base_tensor, finetuned_tensor
        if device.type == "cuda":
            torch.cuda.empty_cache()


def _assemble_adapter(
    out_path: Path,
    work_dir: Path,
    manifest: dict[str, Any],
    metadata: dict[str, str],
) -> None:
    adapter: dict[str, torch.Tensor] = {}
    for report in tqdm(manifest["layers"].values(), desc="assembling adapter", unit="tensor"):
        tensor_file = report.get("tensor_file")
        if tensor_file is None:
            continue
        payload = load_file(str(work_dir / "tensors" / tensor_file), device="cpu")
        overlap = set(adapter) & set(payload)
        if overlap:
            raise ValueError(f"Duplicate adapter keys while assembling {out_path}: {sorted(overlap)[:5]}")
        adapter.update(payload)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = out_path.with_suffix(out_path.suffix + ".tmp")
    save_file(adapter, str(temporary), metadata=metadata)
    temporary.replace(out_path)
    out_path.chmod(0o644)


def _verify_adapter(out_path: Path, manifest: dict[str, Any]) -> None:
    expected_keys = {
        key
        for layer in manifest["layers"].values()
        for key in layer.get("output_keys", [])
    }
    with safe_open(out_path, framework="pt") as adapter:
        actual_keys = set(adapter.keys())
        if actual_keys != expected_keys:
            missing = sorted(expected_keys - actual_keys)
            extra = sorted(actual_keys - expected_keys)
            raise ValueError(f"Output key mismatch: missing={missing[:5]}, extra={extra[:5]}")
        for source_key, layer in manifest["layers"].items():
            kind = layer["kind"]
            shape = tuple(layer["shape"])
            if kind == "lora":
                module_name = source_key.removesuffix(".weight")
                rank = int(layer["rank"])
                a_shape = tuple(adapter.get_slice(f"{module_name}.lora_A.weight").get_shape())
                b_shape = tuple(adapter.get_slice(f"{module_name}.lora_B.weight").get_shape())
                if a_shape != (rank, shape[1]) or b_shape != (shape[0], rank):
                    raise ValueError(f"Invalid factor shapes for {source_key}: A={a_shape}, B={b_shape}")
            elif kind in {"diff", "set_weight", "set_param"}:
                output_key = layer["output_keys"][0]
                output_shape = tuple(adapter.get_slice(output_key).get_shape())
                if output_shape != shape:
                    raise ValueError(f"Invalid dense payload shape for {source_key}: {output_shape} != {shape}")


def _build_report(manifest: dict[str, Any], out_path: Path) -> dict[str, Any]:
    counts: dict[str, int] = {}
    delta_energy = 0.0
    residual_energy = 0.0
    for layer in manifest["layers"].values():
        kind = layer["kind"]
        counts[kind] = counts.get(kind, 0) + 1
        if kind == "lora":
            norm = float(layer["delta_frobenius_norm"])
            residual = float(layer["relative_residual"])
            delta_energy += norm * norm
            residual_energy += (norm * residual)**2
    weighted_residual = (residual_energy / delta_energy)**0.5 if delta_energy else 0.0
    return {
        "format": FORMAT_VERSION,
        "adapter": str(out_path.resolve()),
        "adapter_size_bytes": out_path.stat().st_size,
        "counts": counts,
        "factorized_weighted_relative_residual": weighted_residual,
        "config": manifest["config"],
        "layers": manifest["layers"],
    }


def extract_lora_adapter(
    base: str,
    ft: str,
    out: str,
    rank: int = 32,
    full_rank: bool = False,
    min_delta: float = 1e-6,
    checkpoint: str | None = None,
    resume: bool = False,
    log_level: str = "INFO",
    dense_payload: bool = True,
    *,
    base_revision: str | None = None,
    ft_revision: str | None = None,
    load_mode: str = "auto",
    device: str = "cpu",
    svd_method: str = "exact",
    randomized_q: int | None = None,
    oversample: int = 64,
    niter: int = 4,
    seed: int = 42,
    factor_dtype: str = "float32",
    dense_dtype: str = "float32",
    replacement_dtype: str = "source",
    exact_tensor_patterns: Sequence[str] = (),
    work_dir: str | None = None,
    keep_work_dir: bool = False,
) -> Path:
    """Extract one adapter while streaming transformer tensors."""
    configure_logging(log_level)
    if load_mode not in {"auto", "indexed", "pipeline"}:
        raise ValueError(f"Unsupported load mode: {load_mode}")
    if svd_method not in {"exact", "randomized"}:
        raise ValueError(f"Unsupported SVD method: {svd_method}")
    if randomized_q is not None and randomized_q < 1:
        raise ValueError("randomized_q must be positive")
    if resume and load_mode == "pipeline":
        raise ValueError("--resume requires indexed safetensors; pipeline loading cannot validate checkpoint identity")
    reader_load_mode = "indexed" if resume else load_mode

    out_path = Path(out).expanduser()
    if out_path.suffix != ".safetensors":
        raise ValueError(f"--out must end in .safetensors; adapters are always written as safetensors: {out_path}")
    work_root: Path | None = None
    if work_dir is not None:
        # --work-dir is a shared root; each output gets an independent namespace.
        work_root = Path(work_dir).expanduser() / WORK_SUBDIR
        effective_work_dir = work_root / _work_namespace(out_path)
    elif checkpoint is not None:
        effective_work_dir = Path(checkpoint).expanduser().with_suffix(".work")
    else:
        effective_work_dir = out_path.parent / f".{out_path.name}.work"

    with _open_readers(base, ft, base_revision, ft_revision, reader_load_mode) as (base_reader, finetuned_reader):
        if resume and (not isinstance(base_reader, IndexedSafetensorsReader)
                       or not isinstance(finetuned_reader, IndexedSafetensorsReader)):
            raise ValueError("--resume requires indexed safetensors so checkpoint identity can be validated")
        _validate_key_sets(base_reader, finetuned_reader)
        _validate_exact_patterns(exact_tensor_patterns, finetuned_reader.keys)
        config = ExtractionConfig(
            base_source=base_reader.source,
            finetuned_source=finetuned_reader.source,
            base_fingerprint=base_reader.fingerprint,
            finetuned_fingerprint=finetuned_reader.fingerprint,
            rank=rank,
            full_rank=full_rank,
            min_delta=min_delta,
            device=str(torch.device(device)),
            svd_method=svd_method,
            randomized_q=randomized_q,
            oversample=oversample,
            niter=niter,
            seed=seed,
            factor_dtype=str(_torch_dtype(factor_dtype)).removeprefix("torch."),
            dense_dtype=dense_dtype,
            replacement_dtype=replacement_dtype,
            dense_payload=dense_payload,
            exact_tensor_patterns=tuple(exact_tensor_patterns),
        )
        manifest_path, manifest = _prepare_work_dir(effective_work_dir, config, resume)
        _extract_layers(base_reader, finetuned_reader, effective_work_dir, manifest_path, manifest, config)
        base_reader.assert_unchanged()
        finetuned_reader.assert_unchanged()

    counts: dict[str, int] = {}
    for layer in manifest["layers"].values():
        counts[layer["kind"]] = counts.get(layer["kind"], 0) + 1
    metadata = {
        "format": FORMAT_VERSION,
        "base_model": base,
        "base_revision": base_revision or "unspecified",
        "finetuned_model": ft,
        "finetuned_revision": ft_revision or "unspecified",
        "requested_rank": str(rank),
        "full_rank": str(bool(full_rank)),
        "svd_method": svd_method,
        "randomized_q": str(randomized_q) if randomized_q is not None else "automatic",
        "niter": str(niter),
        "seed": str(seed),
        "factor_dtype": config.factor_dtype,
        "dense_diff_dtype": dense_dtype,
        "replacement_dtype": replacement_dtype,
        "lora_layers": str(counts.get("lora", 0)),
        "diff_tensors": str(counts.get("diff", 0)),
        "set_weight_tensors": str(counts.get("set_weight", 0)),
        "set_param_tensors": str(counts.get("set_param", 0)),
        "dropped_unchanged": str(counts.get("unchanged", 0)),
        "application": "W = W_base + lora_B @ lora_A; then dense diffs added and replacements assigned",
    }
    _assemble_adapter(out_path, effective_work_dir, manifest, metadata)
    _verify_adapter(out_path, manifest)
    report = _build_report(manifest, out_path)
    report_path = out_path.with_suffix(out_path.suffix + ".report.json")
    _atomic_json_dump(report, report_path)
    LOG.info("Saved adapter to %s (%.2f GiB); report=%s", out_path, out_path.stat().st_size / 2**30, report_path)

    if not keep_work_dir:
        _clear_work_dir(effective_work_dir)
        with suppress(OSError):
            effective_work_dir.rmdir()
        if work_root is not None:
            with suppress(OSError):
                work_root.rmdir()
    return out_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", required=True, help="Base model ID or local path")
    parser.add_argument("--ft", required=True, help="Fine-tuned model ID or local path")
    parser.add_argument("--out", default="fastvideo_adapter.safetensors")
    parser.add_argument("--base-revision", default=None)
    parser.add_argument("--ft-revision", default=None)
    parser.add_argument("--rank", type=int, default=16)
    parser.add_argument("--full-rank", action="store_true")
    parser.add_argument("--min-delta", type=float, default=1e-8,
                        help="Drop tensors whose maximum absolute FP32 delta does not exceed this value")
    parser.add_argument("--load-mode", choices=("auto", "indexed", "pipeline"), default="auto")
    parser.add_argument("--device", default="cpu", help="Factorization device, for example cpu or cuda:0")
    parser.add_argument("--svd-method", choices=("exact", "randomized"), default="exact")
    parser.add_argument("--randomized-q", type=int, default=None,
                        help="Randomized basis width; q=320 is validated for rank-64 MiniMax-H3")
    parser.add_argument("--oversample", type=int, default=64,
                        help="Used when --randomized-q is omitted: q = rank + oversample")
    parser.add_argument("--niter", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--factor-dtype", choices=("float32", "float16", "bfloat16"), default="float32")
    parser.add_argument("--dense-dtype", choices=("source", "float32", "float16", "bfloat16"), default="float32")
    parser.add_argument("--replacement-dtype", choices=("source", "float32", "float16", "bfloat16"),
                        default="source")
    parser.add_argument("--exact-tensor-pattern", action="append", default=[],
                        help="Regex for a tensor to retain as an exact dense delta instead of factorizing; repeatable")
    parser.add_argument("--work-dir", default=None)
    parser.add_argument("--checkpoint", default=None,
                        help="Deprecated work-directory alias retained for CLI compatibility")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--keep-work-dir", action="store_true")
    parser.add_argument("--no-dense-payload", dest="dense_payload", action="store_false",
                        help="Emit low-rank factors only; changed norms/biases and fine-tuned-only weights are omitted")
    parser.add_argument("--log-level", default="INFO", choices=("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    extract_lora_adapter(
        base=args.base,
        ft=args.ft,
        out=args.out,
        rank=args.rank,
        full_rank=args.full_rank,
        min_delta=args.min_delta,
        checkpoint=args.checkpoint,
        resume=args.resume,
        log_level=args.log_level,
        dense_payload=args.dense_payload,
        base_revision=args.base_revision,
        ft_revision=args.ft_revision,
        load_mode=args.load_mode,
        device=args.device,
        svd_method=args.svd_method,
        randomized_q=args.randomized_q,
        oversample=args.oversample,
        niter=args.niter,
        seed=args.seed,
        factor_dtype=args.factor_dtype,
        dense_dtype=args.dense_dtype,
        replacement_dtype=args.replacement_dtype,
        exact_tensor_patterns=args.exact_tensor_pattern,
        work_dir=args.work_dir,
        keep_work_dir=args.keep_work_dir,
    )


if __name__ == "__main__":
    main()
