"""Extract publishable multi-rank MiniMax-H3 LoRAs while streaming weights.

This is intentionally separate from :mod:`extract_lora`: MiniMax-H3 is a 33B
transformer, and a VSA-trained student has 50 dense ``to_gate_compress``
matrices that do not exist in the base checkpoint. The generic extractor loads
both complete models, repeats the work for every requested rank, and silently
drops fine-tuned-only tensors.

Large block-matrix deltas are encoded as ``lora_A``/``lora_B`` factors. Other
changed base parameters use exact ``.diff``/``.diff_b`` payloads, while each
VSA gate absent from the base uses ``.set_weight``. Base-identical tensors are
omitted. These suffixes match FastVideo's dense LoRA patch loader and ComfyUI's
adapter convention.

Example::

    python scripts/lora_extraction/extract_minimax_h3_lora.py \
        --base /path/to/MiniMax-H3 \
        --finetuned /path/to/training-output/checkpoint-1300 \
        --finetuned-role student \
        --output-dir /path/to/fasth3-loras \
        --ranks 64 128 256 \
        --device cuda:0
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import logging
import math
import os
import re
from contextlib import ExitStack
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Protocol

import torch
import torch.distributed.checkpoint as dcp
from huggingface_hub import snapshot_download
from safetensors import safe_open
from safetensors.torch import load_file, save_file
from tqdm import tqdm

LOG = logging.getLogger("extract_minimax_h3_lora")
_INDEX_FILENAME = "diffusion_pytorch_model.safetensors.index.json"
_GATE_PATTERN = re.compile(r"^transformer_blocks\.\d+\.attn\.to_gate_compress\.weight$")
_FORMAT_VERSION = "fastvideo-lora-v2"
_DIFF_SUFFIX = ".diff"
_DIFF_BIAS_SUFFIX = ".diff_b"
_SET_WEIGHT_SUFFIX = ".set_weight"


@dataclass(frozen=True)
class FactorizationConfig:
    """Settings that must remain stable when resuming an extraction."""

    base_transformer: str
    finetuned_source: str
    ranks: tuple[int, ...]
    oversample: int
    niter: int
    seed: int
    factor_dtype: str


class TensorReader(Protocol):
    """Minimal random-access interface used by the factorizer."""

    source: str

    @property
    def keys(self) -> set[str]: ...

    def get_tensor(self, key: str) -> torch.Tensor: ...

    def get_shape(self, key: str) -> tuple[int, ...]: ...

    def __enter__(self) -> TensorReader: ...

    def __exit__(self, *args: object) -> None: ...


class IndexedSafetensors:
    """Read individual tensors from an indexed safetensors component."""

    def __init__(self, transformer_dir: Path) -> None:
        self.transformer_dir = transformer_dir
        self.source = str(transformer_dir)
        index_path = transformer_dir / _INDEX_FILENAME
        if not index_path.is_file():
            raise FileNotFoundError(f"Safetensors index not found: {index_path}")

        index = json.loads(index_path.read_text(encoding="utf-8"))
        self.weight_map: dict[str, str] = index["weight_map"]
        self.metadata: dict[str, Any] = index.get("metadata", {})
        self._stack = ExitStack()
        self._shards = {
            shard: self._stack.enter_context(safe_open(transformer_dir / shard, framework="pt", device="cpu"))
            for shard in sorted(set(self.weight_map.values()))
        }

    def close(self) -> None:
        self._stack.close()

    def __enter__(self) -> IndexedSafetensors:
        return self

    def __exit__(self, *args: object) -> None:
        self.close()

    @property
    def keys(self) -> set[str]:
        return set(self.weight_map)

    def get_tensor(self, key: str) -> torch.Tensor:
        return self._shards[self.weight_map[key]].get_tensor(key)

    def get_shape(self, key: str) -> tuple[int, ...]:
        return tuple(self._shards[self.weight_map[key]].get_slice(key).get_shape())


def _fastvideo_to_diffusers_key(key: str) -> str:
    """Map a FastVideo-native H3 state key back to its Diffusers name."""
    key = re.sub(r"^time_embedder\.fc_in\.", "time_embedder.linear_1.", key)
    key = re.sub(r"^time_embedder\.fc_out\.", "time_embedder.linear_2.", key)
    key = re.sub(r"^(.*)\.attn\.to_out\.(.*)$", r"\1.attn.to_out.0.\2", key)
    key = re.sub(r"^(.*)\.ff\.fc_in\.(.*)$", r"\1.ff.net.0.proj.\2", key)
    key = re.sub(r"^(.*)\.ff\.fc_out\.(.*)$", r"\1.ff.net.2.\2", key)
    return key


def _nested_dict(parts: list[str], value: torch.Tensor) -> dict[str, Any]:
    nested: Any = value
    for part in reversed(parts):
        nested = {part: nested}
    return nested


class DCPRoleReader:
    """Load one model role from a torch.distributed.checkpoint on demand."""

    def __init__(self, checkpoint: Path, role: str) -> None:
        dcp_dir = checkpoint / "dcp" if (checkpoint / "dcp" / ".metadata").is_file() else checkpoint
        if not (dcp_dir / ".metadata").is_file():
            raise FileNotFoundError(f"DCP metadata not found under {checkpoint}")
        self.source = str(dcp_dir.resolve())
        self.role = role
        self._reader = dcp.FileSystemReader(self.source)
        self._metadata = self._reader.read_metadata()
        self._prefix = f"roles.{role}.transformer."
        self._weight_map: dict[str, str] = {}
        for flat_key in self._metadata.state_dict_metadata:
            if not flat_key.startswith(self._prefix):
                continue
            fastvideo_key = flat_key.removeprefix(self._prefix)
            diffusers_key = _fastvideo_to_diffusers_key(fastvideo_key)
            if diffusers_key in self._weight_map:
                raise ValueError(f"DCP key mapping collision for {diffusers_key}")
            self._weight_map[diffusers_key] = flat_key
        if not self._weight_map:
            raise ValueError(f"No tensors found for DCP role {role!r} in {self.source}")

    def __enter__(self) -> DCPRoleReader:
        return self

    def __exit__(self, *args: object) -> None:
        return None

    @property
    def keys(self) -> set[str]:
        return set(self._weight_map)

    def get_shape(self, key: str) -> tuple[int, ...]:
        metadata = self._metadata.state_dict_metadata[self._weight_map[key]]
        return tuple(metadata.size)

    def get_tensor(self, key: str) -> torch.Tensor:
        flat_key = self._weight_map[key]
        metadata = self._metadata.state_dict_metadata[flat_key]
        tensor = torch.empty(tuple(metadata.size), dtype=metadata.properties.dtype, device="cpu")
        fastvideo_key = flat_key.removeprefix(self._prefix)
        state = {
            "roles": {
                self.role: {
                    "transformer": _nested_dict(fastvideo_key.split("."), tensor)
                }
            }
        }
        dcp.load(state, storage_reader=self._reader)
        return tensor


def _configure_logging(level: str) -> None:
    if LOG.handlers:
        LOG.setLevel(level)
        return
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s", datefmt="%Y-%m-%d %H:%M:%S"))
    LOG.addHandler(handler)
    LOG.setLevel(level)


def _resolve_transformer_dir(model: str, revision: str | None) -> Path:
    path = Path(model).expanduser()
    candidates = (path / "transformer", path)
    for candidate in candidates:
        if (candidate / _INDEX_FILENAME).is_file():
            return candidate.resolve()

    if path.exists():
        raise FileNotFoundError(f"Could not find transformer/{_INDEX_FILENAME} under {path}")

    snapshot = Path(
        snapshot_download(
            repo_id=model,
            revision=revision,
            allow_patterns=["transformer/*", "provenance.json", "checkpoint_content.json"],
            token=os.environ.get("HF_TOKEN"),
        ))
    transformer_dir = snapshot / "transformer"
    if not (transformer_dir / _INDEX_FILENAME).is_file():
        raise FileNotFoundError(f"Downloaded snapshot has no transformer index: {transformer_dir}")
    return transformer_dir.resolve()


def _resolve_finetuned_reader(model: str, revision: str | None, role: str) -> TensorReader:
    path = Path(model).expanduser()
    if path.exists() and ((path / ".metadata").is_file() or (path / "dcp" / ".metadata").is_file()):
        return DCPRoleReader(path.resolve(), role)
    return IndexedSafetensors(_resolve_transformer_dir(model, revision))


def _torch_dtype(name: str) -> torch.dtype:
    normalized = name.lower()
    choices = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }
    if normalized not in choices:
        raise ValueError(f"Unsupported factor dtype {name!r}; choose from {sorted(choices)}")
    return choices[normalized]


def _atomic_json_dump(data: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def _is_lora_matrix(key: str, shape: tuple[int, ...]) -> bool:
    # Shape is a safer discriminator than name fragments.  In particular,
    # norm_out.linear.weight is a real 2-D AdaLN projection and must not be
    # discarded merely because its path contains "norm".
    return key.endswith(".weight") and len(shape) == 2


def _is_boundary_tensor(key: str) -> bool:
    """Keep the small I/O, timestep, final, and normalization state exact."""
    return not key.startswith(("transformer_blocks.", "token_refiner.refiner_blocks."))


def _dense_passthrough_keys(
    base: TensorReader,
    matrix_keys: list[str],
    gate_keys: list[str],
) -> list[str]:
    """Return tensors that need an exact dense delta or replacement.

    The name is retained for compatibility with existing analysis utilities;
    returned base tensors are not copied through to the adapter. Their FP32
    differences are emitted only when nonzero.
    """
    matrix_key_set = set(matrix_keys)
    auxiliary = [
        key for key in sorted(base.keys)
        if key not in matrix_key_set or _is_boundary_tensor(key)
    ]
    return sorted(set(auxiliary) | set(gate_keys))


def _dense_payload(
    base: TensorReader,
    finetuned: TensorReader,
    candidate_keys: list[str],
    gate_keys: list[str],
    dense_dtype: torch.dtype,
    gate_dtype: torch.dtype | None,
) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    """Build exact dense deltas/replacements and omit unchanged parameters."""
    payload: dict[str, torch.Tensor] = {}
    diff_source_keys: list[str] = []
    replacement_source_keys: list[str] = []
    identical_source_keys: list[str] = []
    gate_key_set = set(gate_keys)

    for key in tqdm(candidate_keys, desc="building dense LoRA payload", unit="tensor"):
        finetuned_tensor = finetuned.get_tensor(key)
        if key in gate_key_set:
            if not key.endswith(".weight"):
                raise ValueError(f"VSA replacement is not a weight: {key}")
            output_key = key.removesuffix(".weight") + _SET_WEIGHT_SUFFIX
            tensor = finetuned_tensor if gate_dtype is None else finetuned_tensor.to(gate_dtype)
            payload[output_key] = tensor.cpu().contiguous()
            replacement_source_keys.append(key)
            continue

        base_tensor = base.get_tensor(key)
        if base_tensor.shape != finetuned_tensor.shape:
            raise ValueError(f"Dense payload shape mismatch for {key}: "
                             f"base={tuple(base_tensor.shape)}, finetuned={tuple(finetuned_tensor.shape)}")
        delta = finetuned_tensor.to(dtype=torch.float32, copy=True).sub_(base_tensor.to(torch.float32))
        if not bool(torch.count_nonzero(delta)):
            identical_source_keys.append(key)
            continue
        if key.endswith(".weight"):
            output_key = key.removesuffix(".weight") + _DIFF_SUFFIX
        elif key.endswith(".bias"):
            output_key = key.removesuffix(".bias") + _DIFF_BIAS_SUFFIX
        else:
            raise ValueError(f"No dense LoRA suffix for changed parameter: {key}")
        payload[output_key] = delta.to(dense_dtype).cpu().contiguous()
        diff_source_keys.append(key)

    report = {
        "diff_source_keys": diff_source_keys,
        "replacement_source_keys": replacement_source_keys,
        "dropped_identical_source_keys": identical_source_keys,
        "diff_tensor_count": len(diff_source_keys),
        "replacement_tensor_count": len(replacement_source_keys),
        "dropped_identical_tensor_count": len(identical_source_keys),
    }
    return payload, report


def _seed_for_key(seed: int, key: str) -> int:
    digest = hashlib.sha256(f"{seed}:{key}".encode()).digest()
    return int.from_bytes(digest[:8], "little") % (2**31)


def _factorize_delta(
    delta: torch.Tensor,
    max_rank: int,
    oversample: int,
    niter: int,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, str]:
    """Return symmetric ``A``, ``B`` factors and singular values."""
    if delta.ndim != 2:
        raise ValueError(f"Expected a matrix, got shape {tuple(delta.shape)}")

    available_rank = min(delta.shape)
    chosen_rank = min(max_rank, available_rank)
    q = min(available_rank, chosen_rank + oversample)

    # Exact SVD is cheap for the small boundary projections and avoids asking
    # the randomized routine for a full-width basis.
    if q == available_rank:
        u, singular_values, vh = torch.linalg.svd(delta, full_matrices=False)
        v = vh.mT
        method = "exact"
    else:
        devices = [delta.device] if delta.device.type == "cuda" else []
        with torch.random.fork_rng(devices=devices):
            torch.manual_seed(seed)
            if delta.device.type == "cuda":
                torch.cuda.manual_seed(seed)
            u, singular_values, v = torch.svd_lowrank(delta, q=q, niter=niter)
        method = f"randomized-q{q}-niter{niter}"

    singular_values = singular_values[:chosen_rank].to(torch.float32)
    sqrt_s = singular_values.sqrt()
    lora_b = (u[:, :chosen_rank].to(torch.float32) * sqrt_s.unsqueeze(0)).contiguous()
    lora_a = (v[:, :chosen_rank].to(torch.float32) * sqrt_s.unsqueeze(0)).mT.contiguous()
    return lora_a, lora_b, singular_values, method


def _relative_residual(delta_fro_sq: float, singular_values: torch.Tensor, rank: int) -> float:
    if delta_fro_sq == 0.0:
        return 0.0
    captured = float(singular_values[:rank].square().sum().item())
    return math.sqrt(max(0.0, 1.0 - captured / delta_fro_sq))


def _validate_key_sets(
    base: TensorReader,
    finetuned: TensorReader,
    expected_gate_count: int | None,
) -> tuple[list[str], list[str]]:
    missing_from_finetuned = sorted(base.keys - finetuned.keys)
    if missing_from_finetuned:
        raise ValueError(f"Fine-tuned checkpoint is missing {len(missing_from_finetuned)} base tensors; first keys: "
                         f"{missing_from_finetuned[:5]}")

    extra_keys = sorted(finetuned.keys - base.keys)
    gate_keys = [key for key in extra_keys if _GATE_PATTERN.fullmatch(key)]
    unexpected_extra = sorted(set(extra_keys) - set(gate_keys))
    if unexpected_extra:
        raise ValueError(f"Fine-tuned checkpoint has unexpected tensors besides VSA gates: {unexpected_extra[:10]}")
    if expected_gate_count is not None and len(gate_keys) != expected_gate_count:
        raise ValueError(f"Expected {expected_gate_count} full VSA gates, found {len(gate_keys)}")

    matrix_keys: list[str] = []
    for key in sorted(base.keys):
        base_shape = base.get_shape(key)
        finetuned_shape = finetuned.get_shape(key)
        if base_shape != finetuned_shape:
            raise ValueError(f"Shape mismatch for {key}: base={base_shape}, finetuned={finetuned_shape}")
        if _is_lora_matrix(key, base_shape):
            matrix_keys.append(key)
    return matrix_keys, gate_keys


def _load_or_initialize_manifest(
    work_dir: Path,
    config: FactorizationConfig,
    resume: bool,
) -> tuple[Path, dict[str, Any]]:
    manifest_path = work_dir / "factorization_manifest.json"
    expected_config = asdict(config)
    # JSON represents tuples as lists.
    expected_config["ranks"] = list(config.ranks)

    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("config") != expected_config:
            raise ValueError("Existing factorization manifest uses different inputs or settings; "
                             f"remove {work_dir} or select another --work-dir")
        if not resume:
            raise FileExistsError(f"Factorization work directory already exists: {work_dir}; pass --resume to reuse it")
        return manifest_path, manifest

    work_dir.mkdir(parents=True, exist_ok=True)
    manifest = {"format": _FORMAT_VERSION, "config": expected_config, "layers": {}}
    _atomic_json_dump(manifest, manifest_path)
    return manifest_path, manifest


def _factorize_matrices(
    base: TensorReader,
    finetuned: TensorReader,
    matrix_keys: list[str],
    ranks: tuple[int, ...],
    device: torch.device,
    oversample: int,
    niter: int,
    seed: int,
    factor_dtype: torch.dtype,
    factors_dir: Path,
    manifest_path: Path,
    manifest: dict[str, Any],
) -> None:
    factors_dir.mkdir(parents=True, exist_ok=True)
    max_rank = max(ranks)

    for index, key in enumerate(tqdm(matrix_keys, desc="factorizing H3 deltas", unit="matrix")):
        factor_path = factors_dir / f"{index:04d}.safetensors"
        existing = manifest["layers"].get(key)
        if existing and factor_path.is_file():
            continue

        # ``copy=True`` matters for CPU extraction: safetensors may otherwise
        # return the mapped source storage, which the in-place subtraction
        # below would corrupt for later dense passthrough.
        finetuned_weight = finetuned.get_tensor(key).to(device=device, dtype=torch.float32, copy=True)
        base_weight = base.get_tensor(key).to(device=device, dtype=torch.float32)
        delta = finetuned_weight.sub_(base_weight)
        del base_weight

        delta_fro_sq = float(delta.square().sum().item())
        mean_abs_delta = float(delta.abs().mean().item())
        if delta_fro_sq == 0.0:
            manifest["layers"][key] = {
                "factor_file": None,
                "shape": list(delta.shape),
                "delta_fro_norm": 0.0,
                "mean_abs_delta": 0.0,
                "relative_residual": {str(rank): 0.0 for rank in ranks},
                "method": "unchanged",
            }
            _atomic_json_dump(manifest, manifest_path)
            del delta, finetuned_weight
            continue

        lora_a, lora_b, singular_values, method = _factorize_delta(
            delta,
            max_rank=max_rank,
            oversample=oversample,
            niter=niter,
            seed=_seed_for_key(seed, key),
        )
        computed_rank = lora_a.shape[0]
        layer_report = {
            "factor_file": factor_path.name,
            "shape": list(delta.shape),
            "computed_rank": computed_rank,
            "delta_fro_norm": math.sqrt(delta_fro_sq),
            "mean_abs_delta": mean_abs_delta,
            "relative_residual": {
                str(rank): _relative_residual(delta_fro_sq, singular_values, min(rank, computed_rank))
                for rank in ranks
            },
            "method": method,
        }

        save_file(
            {
                "lora_A": lora_a.to(device="cpu", dtype=factor_dtype),
                "lora_B": lora_b.to(device="cpu", dtype=factor_dtype),
                "singular_values": singular_values.cpu(),
            },
            str(factor_path),
            metadata={"source_key": key, "format": _FORMAT_VERSION},
        )
        manifest["layers"][key] = layer_report
        _atomic_json_dump(manifest, manifest_path)

        del delta, finetuned_weight, lora_a, lora_b, singular_values
        if device.type == "cuda":
            torch.cuda.empty_cache()


def _save_rank_checkpoint(
    rank: int,
    output_dir: Path,
    factors_dir: Path,
    manifest: dict[str, Any],
    dense_payload: dict[str, torch.Tensor],
    dense_report: dict[str, Any],
    output_dtype: torch.dtype,
    dense_dtype: torch.dtype,
    gate_dtype_name: str,
    base_model_id: str,
    base_revision: str | None,
    finetuned_model_id: str,
    finetuned_revision: str | None,
) -> Path:
    rank_dir = output_dir / f"rank-{rank}"
    rank_dir.mkdir(parents=True, exist_ok=True)
    output_path = rank_dir / "adapter_model.safetensors"

    adapter_state: dict[str, torch.Tensor] = dict(dense_payload)
    actual_ranks: dict[str, int] = {}
    for key, report in tqdm(sorted(manifest["layers"].items()), desc=f"assembling rank {rank}", unit="matrix"):
        factor_file = report.get("factor_file")
        if factor_file is None:
            continue
        factors = load_file(str(factors_dir / factor_file), device="cpu")
        chosen_rank = min(rank, factors["lora_A"].shape[0])
        module_name = key.removesuffix(".weight")
        adapter_state[f"{module_name}.lora_A.weight"] = factors["lora_A"][:chosen_rank].to(
            output_dtype).contiguous()
        adapter_state[f"{module_name}.lora_B.weight"] = factors["lora_B"][:, :chosen_rank].to(
            output_dtype).contiguous()
        actual_ranks[module_name] = chosen_rank
        del factors

    metadata = {
        "format": _FORMAT_VERSION,
        "application": "W = W_base + lora_B @ lora_A; then .diff/.diff_b added and .set_weight assigned",
        "base_model": base_model_id,
        "base_revision": base_revision or "unspecified",
        "finetuned_model": finetuned_model_id,
        "finetuned_revision": finetuned_revision or "unspecified",
        "rank": str(rank),
        "requested_rank": str(rank),
        "factor_dtype": str(output_dtype).removeprefix("torch."),
        "dense_diff_dtype": str(dense_dtype).removeprefix("torch."),
        "gate_dtype": gate_dtype_name,
        "low_rank_tensors": str(2 * len(actual_ranks)),
        "diff_tensors": str(dense_report["diff_tensor_count"]),
        "set_weight_tensors": str(dense_report["replacement_tensor_count"]),
        "dropped_unchanged": str(dense_report["dropped_identical_tensor_count"]),
        "dense_tensor_policy": "nonzero FP32 auxiliary/boundary deltas plus full VSA gate replacements",
    }
    temporary_path = output_path.with_suffix(".safetensors.tmp")
    save_file(adapter_state, str(temporary_path), metadata=metadata)
    temporary_path.replace(output_path)

    config = {
        **metadata,
        "adapter_file": output_path.name,
        "lora_tensor_count": 2 * len(actual_ranks),
        "lora_layer_count": len(actual_ranks),
        "dense_tensor_count": len(dense_payload),
        "dense_tensor_keys": sorted(dense_payload),
        "dense_report": dense_report,
        "actual_rank_by_module": actual_ranks,
    }
    _atomic_json_dump(config, rank_dir / "adapter_config.json")
    LOG.info("Saved rank-%d checkpoint to %s (%.2f GiB)", rank, output_path, output_path.stat().st_size / 2**30)
    del adapter_state
    gc.collect()
    return output_path


def _verify_checkpoint(
    checkpoint: Path,
    manifest: dict[str, Any],
    dense_payload: dict[str, torch.Tensor],
    requested_rank: int,
    output_dtype: torch.dtype,
) -> None:
    with safe_open(checkpoint, framework="pt", device="cpu") as adapter:
        expected_keys = set(dense_payload)
        for key, report in manifest["layers"].items():
            if report.get("factor_file") is None:
                continue
            module_name = key.removesuffix(".weight")
            expected_keys.add(f"{module_name}.lora_A.weight")
            expected_keys.add(f"{module_name}.lora_B.weight")

        actual_keys = set(adapter.keys())
        if actual_keys != expected_keys:
            missing = sorted(expected_keys - actual_keys)
            extra = sorted(actual_keys - expected_keys)
            raise ValueError(f"Output key mismatch for {checkpoint}: missing={missing[:5]}, extra={extra[:5]}")

        for key, report in manifest["layers"].items():
            if report.get("factor_file") is None:
                continue
            module_name = key.removesuffix(".weight")
            shape = report["shape"]
            actual_rank = min(requested_rank, report["computed_rank"])
            a = adapter.get_slice(f"{module_name}.lora_A.weight")
            b = adapter.get_slice(f"{module_name}.lora_B.weight")
            if tuple(a.get_shape()) != (actual_rank, shape[1]):
                raise ValueError(f"Invalid lora_A shape for {module_name}: {a.get_shape()}")
            if tuple(b.get_shape()) != (shape[0], actual_rank):
                raise ValueError(f"Invalid lora_B shape for {module_name}: {b.get_shape()}")
            if a.get_dtype() != str(output_dtype).removeprefix("torch.").upper().replace("FLOAT", "F"):
                # Safetensors spells torch.bfloat16 as BF16 and float32 as F32.
                expected = {torch.bfloat16: "BF16", torch.float16: "F16", torch.float32: "F32"}[output_dtype]
                if a.get_dtype() != expected or b.get_dtype() != expected:
                    raise ValueError(f"Invalid factor dtype for {module_name}: A={a.get_dtype()}, B={b.get_dtype()}")

        # Dense additive/replacement payloads must remain exact while saving.
        for key, expected in tqdm(dense_payload.items(),
                                  desc=f"verifying rank {requested_rank} dense tensors",
                                  unit="tensor"):
            if not torch.equal(adapter.get_tensor(key), expected):
                raise ValueError(f"Dense tensor changed while saving {checkpoint}: {key}")


def extract_minimax_h3_loras(
    *,
    base: str,
    finetuned: str,
    output_dir: str,
    ranks: tuple[int, ...] = (64, 128, 256),
    base_revision: str | None = None,
    finetuned_revision: str | None = None,
    finetuned_role: str = "student",
    base_model_id: str | None = None,
    finetuned_model_id: str | None = None,
    device: str = "cuda:0",
    oversample: int = 64,
    niter: int = 4,
    seed: int = 42,
    factor_dtype: str = "float32",
    output_dtype: str = "float16",
    dense_dtype: str = "float32",
    gate_dtype: str = "source",
    work_dir: str | None = None,
    resume: bool = False,
    expected_gate_count: int | None = 50,
    verify: bool = True,
) -> list[Path]:
    """Extract all requested ranks from one maximum-rank factorization."""
    ranks = tuple(sorted(set(int(rank) for rank in ranks)))
    if not ranks or any(rank <= 0 for rank in ranks):
        raise ValueError(f"All ranks must be positive, got {ranks}")
    if oversample < 0 or niter < 0:
        raise ValueError("oversample and niter must be non-negative")

    output_path = Path(output_dir).expanduser().resolve()
    output_path.mkdir(parents=True, exist_ok=True)
    factor_work_dir = (Path(work_dir).expanduser().resolve()
                       if work_dir else output_path / f".factorization-r{max(ranks)}")
    factors_dir = factor_work_dir / "factors"

    base_transformer = _resolve_transformer_dir(base, base_revision)
    finetuned_reader = _resolve_finetuned_reader(finetuned, finetuned_revision, finetuned_role)
    torch_device = torch.device(device)
    if torch_device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA device requested but CUDA is unavailable: {device}")

    factor_torch_dtype = _torch_dtype(factor_dtype)
    output_torch_dtype = _torch_dtype(output_dtype)
    dense_torch_dtype = _torch_dtype(dense_dtype)
    gate_torch_dtype = None if gate_dtype == "source" else _torch_dtype(gate_dtype)
    factor_config = FactorizationConfig(
        base_transformer=str(base_transformer),
        finetuned_source=finetuned_reader.source,
        ranks=ranks,
        oversample=oversample,
        niter=niter,
        seed=seed,
        factor_dtype=factor_dtype,
    )
    manifest_path, manifest = _load_or_initialize_manifest(factor_work_dir, factor_config, resume)

    with IndexedSafetensors(base_transformer) as base_reader, finetuned_reader:
        matrix_keys, gate_keys = _validate_key_sets(base_reader, finetuned_reader, expected_gate_count)
        dense_keys = _dense_passthrough_keys(base_reader, matrix_keys, gate_keys)
        factorized_matrix_keys = sorted(set(matrix_keys) - set(dense_keys))
        LOG.info("Validated checkpoints: %d common matrices, %d LoRA-factorized matrices, "
                 "%d dense candidates (%d VSA gates)", len(matrix_keys),
                 len(factorized_matrix_keys), len(dense_keys), len(gate_keys))
        _factorize_matrices(
            base_reader,
            finetuned_reader,
            factorized_matrix_keys,
            ranks,
            torch_device,
            oversample,
            niter,
            seed,
            factor_torch_dtype,
            factors_dir,
            manifest_path,
            manifest,
        )

        # Reload because the in-memory object was updated after every layer.
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        dense_payload, dense_report = _dense_payload(
            base_reader,
            finetuned_reader,
            dense_keys,
            gate_keys,
            dense_torch_dtype,
            gate_torch_dtype,
        )
        LOG.info("Dense payload: %d changed deltas, %d replacements, %d base-identical tensors dropped",
                 dense_report["diff_tensor_count"], dense_report["replacement_tensor_count"],
                 dense_report["dropped_identical_tensor_count"])
        checkpoints = [
            _save_rank_checkpoint(
                rank,
                output_path,
                factors_dir,
                manifest,
                dense_payload,
                dense_report,
                output_torch_dtype,
                dense_torch_dtype,
                gate_dtype,
                base_model_id or base,
                base_revision,
                finetuned_model_id or finetuned,
                finetuned_revision,
            ) for rank in ranks
        ]
        if verify:
            for rank, checkpoint in zip(ranks, checkpoints, strict=True):
                _verify_checkpoint(
                    checkpoint,
                    manifest,
                    dense_payload,
                    rank,
                    output_torch_dtype,
                )

    metrics_path = output_path / "factorization_metrics.json"
    _atomic_json_dump(manifest, metrics_path)
    report = {
        "format": _FORMAT_VERSION,
        "base_model": base_model_id or base,
        "base_revision": base_revision,
        "finetuned_model": finetuned_model_id or finetuned,
        "finetuned_revision": finetuned_revision,
        "ranks": list(ranks),
        "gate_count": len(gate_keys),
        "dense_candidate_tensor_count": len(dense_keys),
        "dense_tensor_count": len(dense_payload),
        "dense_report": dense_report,
        "matrix_count": len(matrix_keys),
        "lora_matrix_count": len(factorized_matrix_keys),
        "factor_dtype": factor_dtype,
        "output_dtype": output_dtype,
        "dense_dtype": dense_dtype,
        "gate_dtype": gate_dtype,
        "factorization_metrics": metrics_path.name,
        "checkpoints": [str(path) for path in checkpoints],
    }
    _atomic_json_dump(report, output_path / "extraction_report.json")
    return checkpoints


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", required=True, help="Base model ID, model directory, or transformer directory")
    parser.add_argument("--finetuned",
                        required=True,
                        help="Fine-tuned model ID, model directory, or transformer directory")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--ranks", type=int, nargs="+", default=[64, 128, 256])
    parser.add_argument("--base-revision")
    parser.add_argument("--finetuned-revision")
    parser.add_argument("--finetuned-role", default="student", help="Model role when --finetuned is a DCP checkpoint")
    parser.add_argument("--base-model-id", help="Canonical base ID to record when --base is a local path")
    parser.add_argument("--finetuned-model-id", help="Canonical student ID to record when --finetuned is a local path")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--oversample", type=int, default=64)
    parser.add_argument("--niter", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--factor-dtype", choices=["float32", "float16", "bfloat16"], default="float32")
    parser.add_argument("--output-dtype",
                        choices=["float32", "float16", "bfloat16"],
                        default="float16",
                        help="Storage dtype for LoRA A/B factors")
    parser.add_argument("--dense-dtype",
                        choices=["float32", "float16", "bfloat16"],
                        default="float32",
                        help="Storage dtype for exact .diff/.diff_b payloads")
    parser.add_argument("--gate-dtype",
                        choices=["source", "float32", "float16", "bfloat16"],
                        default="source",
                        help="Storage dtype for full .set_weight VSA gates")
    parser.add_argument("--work-dir")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--expected-gate-count", type=int, default=50)
    parser.add_argument("--no-verify", action="store_true")
    parser.add_argument("--log-level", choices=["DEBUG", "INFO", "WARNING", "ERROR"], default="INFO")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    _configure_logging(args.log_level)
    extract_minimax_h3_loras(
        base=args.base,
        finetuned=args.finetuned,
        output_dir=args.output_dir,
        ranks=tuple(args.ranks),
        base_revision=args.base_revision,
        finetuned_revision=args.finetuned_revision,
        finetuned_role=args.finetuned_role,
        base_model_id=args.base_model_id,
        finetuned_model_id=args.finetuned_model_id,
        device=args.device,
        oversample=args.oversample,
        niter=args.niter,
        seed=args.seed,
        factor_dtype=args.factor_dtype,
        output_dtype=args.output_dtype,
        dense_dtype=args.dense_dtype,
        gate_dtype=args.gate_dtype,
        work_dir=args.work_dir,
        resume=args.resume,
        expected_gate_count=args.expected_gate_count,
        verify=not args.no_verify,
    )


if __name__ == "__main__":
    main()
