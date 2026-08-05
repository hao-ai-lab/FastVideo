# SPDX-License-Identifier: Apache-2.0
"""Shared machinery for golden-gate tests: single-layer bitwise DiT fingerprints.

A gate loads ONE transformer block of a DiT (selective safetensors read via the
checkpoint index when present), feeds a fixed seeded batch, and compares the
output bitwise against a device-keyed golden latent. Zero tolerance is the
point: on one device these blocks reproduce bit-identically, so any drift is a
real change to the compute path — and the golden's environment fingerprint
turns legitimate env changes into named failures instead of mystery drift.

Per-model tests provide a :class:`GateSpec`; everything else lives here.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import pytest
import torch
from torch.testing import assert_close

GOLDEN_REPO_ID = os.environ.get("FASTVIDEO_SSIM_REFERENCE_HF_REPO", "FastVideo/ssim-reference-videos")
GOLDEN_ROOT = Path(os.environ.get("FASTVIDEO_GOLDEN_GATE_DIR", Path(__file__).resolve().parent / "goldens"))
DEFAULT_SEED = 20260805

# Must be set before cuBLAS initializes for use_deterministic_algorithms.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
os.environ.setdefault("MASTER_PORT", "29661")
os.environ.setdefault("RANK", "0")
os.environ.setdefault("WORLD_SIZE", "1")
os.environ.setdefault("LOCAL_RANK", "0")


@dataclass
class GateSpec:
    """Everything model-specific about one golden gate."""
    name: str                                  # golden filename stem, e.g. "wan_t2v"
    repo_id: str                               # HF checkpoint repo
    build_block: Callable[[int], torch.nn.Module]
    make_inputs: Callable[[torch.device, int], dict[str, Any]]
    subfolder: str = "transformer"
    layer: int = 0
    prefix_template: str = "blocks.{N}."
    renames: tuple[tuple[str, str], ...] = ()  # (regex, replacement) on ".{stripped}" names
    attention_backend: str = "FLASH_ATTN"
    seed: int = DEFAULT_SEED
    # compute dtype for the block; matrixgame2's int64 null-modulation promotes
    # activations to fp32 mid-block, so that DiT runs fp32 in production
    dtype: torch.dtype = torch.bfloat16
    # some blocks return tuples (e.g. double-stream img/txt) — reduce to one tensor
    postprocess: Callable[[Any], torch.Tensor] = field(default=lambda out: out)
    model_root_env: str | None = None          # env var naming a local checkpoint root
    weight_file: str = "diffusion_pytorch_model.safetensors"  # single-file fallback


def env_fingerprint(spec: GateSpec) -> dict[str, str]:
    try:
        import flash_attn
        fa_version = str(getattr(flash_attn, "__version__", "?"))
    except ImportError:
        fa_version = "<not installed>"
    return {
        "torch": str(torch.__version__),
        "cuda": str(torch.version.cuda),
        "cudnn": str(torch.backends.cudnn.version()),
        "flash_attn": fa_version,
        "device_name": torch.cuda.get_device_name(0),
        "attention_backend": spec.attention_backend,
        "fastvideo_fa4": os.environ.get("FASTVIDEO_FA4", "<unset>"),
        "tf32_matmul": str(torch.backends.cuda.matmul.allow_tf32),
        "tf32_cudnn": str(torch.backends.cudnn.allow_tf32),
    }


def _component_dir(spec: GateSpec) -> Path:
    """Local checkout wins; otherwise fetch the index + only the shards holding
    this layer (or the single weight file for index-less checkpoints)."""
    if spec.model_root_env:
        root = os.environ.get(spec.model_root_env)
        if root:
            return Path(root) / spec.subfolder

    from huggingface_hub import hf_hub_download
    from huggingface_hub.errors import EntryNotFoundError

    index_name = f"{spec.subfolder}/{spec.weight_file}.index.json"
    prefix = spec.prefix_template.format(N=spec.layer)
    try:
        index_path = Path(hf_hub_download(spec.repo_id, index_name))
    except EntryNotFoundError:
        single = Path(hf_hub_download(spec.repo_id, f"{spec.subfolder}/{spec.weight_file}"))
        return single.parent
    weight_map = json.loads(index_path.read_text())["weight_map"]
    shards = sorted({shard for name, shard in weight_map.items() if name.startswith(prefix)})
    if not shards:
        pytest.fail(f"no shards found for prefix {prefix} in {spec.repo_id}", pytrace=False)
    for shard in shards:
        hf_hub_download(spec.repo_id, f"{spec.subfolder}/{shard}")
    return index_path.parent


def load_layer_state(spec: GateSpec, component_dir: Path) -> dict[str, torch.Tensor]:
    from safetensors import safe_open

    prefix = spec.prefix_template.format(N=spec.layer)
    renames = [(re.compile(pat), repl) for pat, repl in spec.renames]
    index_file = component_dir / f"{spec.weight_file}.index.json"

    by_shard: dict[str, list[str]] = {}
    if index_file.is_file():
        for name, shard in json.loads(index_file.read_text())["weight_map"].items():
            if name.startswith(prefix):
                by_shard.setdefault(shard, []).append(name)
    else:
        with safe_open(component_dir / spec.weight_file, framework="pt", device="cpu") as f:
            by_shard[spec.weight_file] = [n for n in f.keys() if n.startswith(prefix)]
    assert by_shard and any(by_shard.values()), f"no parameters found for prefix {prefix}"

    state: dict[str, torch.Tensor] = {}
    for shard, names in sorted(by_shard.items()):
        with safe_open(component_dir / shard, framework="pt", device="cpu") as f:
            for name in names:
                mapped = f".{name[len(prefix):]}"
                for pattern, repl in renames:
                    mapped = pattern.sub(repl, mapped)
                state[mapped[1:]] = f.get_tensor(name)
    return state


def _golden_relpath(spec: GateSpec) -> str:
    device_slug = re.sub(r"[^A-Za-z0-9]+", "_", torch.cuda.get_device_name(0)).strip("_")
    return f"{device_slug}/{spec.name}_layer{spec.layer}_{spec.attention_backend}_seed{spec.seed}.pt"


def _resolve_golden(spec: GateSpec) -> tuple[Path, bool]:
    local = GOLDEN_ROOT / _golden_relpath(spec)
    if local.exists():
        return local, True
    from huggingface_hub import hf_hub_download
    from huggingface_hub.errors import EntryNotFoundError
    try:
        remote = hf_hub_download(GOLDEN_REPO_ID, f"golden_gates/{_golden_relpath(spec)}", repo_type="dataset")
        return Path(remote), True
    except EntryNotFoundError:
        return local, False


def run_gate(spec: GateSpec) -> None:
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        pytest.skip("golden-gate requires a bf16-capable CUDA GPU")

    os.environ["FASTVIDEO_ATTENTION_BACKEND"] = spec.attention_backend
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False

    device = torch.device("cuda:0")
    component_dir = _component_dir(spec)

    from fastvideo.forward_context import set_forward_context

    block = spec.build_block(spec.layer)
    state = load_layer_state(spec, component_dir)
    missing, unexpected = block.load_state_dict(state, strict=True)
    assert not missing and not unexpected, (
        f"state mismatch for {spec.name}: missing={missing[:6]} unexpected={unexpected[:6]}")
    block = block.to(device=device, dtype=spec.dtype).eval()

    inputs = spec.make_inputs(device, spec.seed)
    with torch.inference_mode(), set_forward_context(current_timestep=0, attn_metadata=None):
        raw = block(**inputs)
    out = spec.postprocess(raw)
    out = out.detach().float().cpu()

    golden_path, golden_exists = _resolve_golden(spec)
    meta = {"name": spec.name, "layer": spec.layer, "seed": spec.seed,
            "shape": list(out.shape), "env": env_fingerprint(spec)}
    if not golden_exists:
        golden_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"output": out, "metadata": meta}, golden_path)
        pytest.fail(
            f"golden latent seeded at {golden_path} ({meta}). Verify with a second run, "
            f"then upload to {GOLDEN_REPO_ID} at golden_gates/{_golden_relpath(spec)}",
            pytrace=False,
        )

    golden = torch.load(golden_path, weights_only=True)
    mismatches = {
        key: (value, meta["env"].get(key))
        for key, value in golden["metadata"]["env"].items()
        if meta["env"].get(key) != value
    }
    assert not mismatches, (
        f"environment changed since golden was minted — re-mint deliberately rather "
        f"than chasing bit drift. golden -> current: {mismatches}")

    drift = (out - golden["output"]).abs()
    print(f"golden-gate[{spec.name}] drift: max_abs={drift.max().item():.10f} "
          f"mean_abs={drift.mean().item():.10f}", flush=True)
    assert_close(out, golden["output"], atol=0.0, rtol=0.0)


@pytest.fixture(scope="module")
def distributed_runtime():
    """Single-process distributed init — required by DistributedAttention."""
    if not torch.cuda.is_available():
        yield
        return
    from fastvideo.distributed import (
        cleanup_dist_env_and_memory,
        maybe_init_distributed_environment_and_model_parallel,
    )
    maybe_init_distributed_environment_and_model_parallel(1, 1)
    yield
    cleanup_dist_env_and_memory()
