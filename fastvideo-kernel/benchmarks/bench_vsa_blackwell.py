#!/usr/bin/env python3
"""Block-sparse attention micro-benchmark: flashinfer ``vsa_blackwell`` and
FA4 CuTe-DSL BSA, on identical block masks where the backends can represent
them (GB200 / SM100).

This harness originated in FlashInfer issue #4554 and its benchmark gist:
  https://github.com/flashinfer-ai/flashinfer/issues/4554
  https://gist.github.com/SolitaryThinker/90a1d1447929fc38dc509c1852e76532
The imported baseline is pinned to immutable gist revision
``e15ac9066f23ef3690e33e1cc1fdac45b4b9099f``.

Measures achieved sparse-aware algorithmic TFLOP/s over a sequence-length x
sparsity x (Q block, KV block) grid. Every successful arm is checked against
the same fp32 masked-SDPA reference for that cell. FlashInfer's plan()/run()
split is timed separately (VSA masks are data-dependent and change every
layer x step, so mask-per-call deployments pay both).

The default ``--mask_mode exact256`` samples one Q256/KV256 mask and expands
it into each finer block shape. Thus all shapes select exactly the same token
pairs and their output and relative-efficiency comparisons are apples-to-
apples. ``--mask_mode native`` samples independently at each requested shape.

Arms:
  flashinfer  — BlockSparseAttentionWrapper(backend="vsa_blackwell"),
                R=C=128. It runs only when the requested logical mask is
                exactly representable by 128-token tiles.
  cutedsl256  — original FastVideo 256x256 wrapper arm from the pinned gist;
                used as a measured relative-efficiency reference.
  fa4_wrapper — FastVideo's direct fine-grained BSHD wrapper, including
                block-map conversion and variable-block-size mask plumbing.
  fa4         — direct ``flash_attn.cute.interface._flash_attn_fwd`` call.
                Sparse Q and KV block sizes are configured independently.

For the stock 256x256 FA4 path, a logical KV256 edge expands into two physical
KV128 edges, matching ``block_sparse_attn_256``. The 128x64 and 64x64 paths
call FA4 with physical tiles (128, 64) and (64, 64), respectively. Mask-index
construction and output allocation are outside the raw FA4 timing region.
The direct ``fa4_wrapper`` 64x64 row is therefore the native Q64/KV64 kernel;
it is not the public VSA-256 ``FASTVIDEO_VSA_FA4_BLOCK_SHAPE=64x64`` adapter,
which safely pairs identical Q64 children onto physical Q128/KV64 and is
measured by selecting that environment variable on the ``cutedsl256`` arm.

The vendored FA4 source is a flat ``flash_attn.cute`` package. It can be used
without installation by putting that directory on ``PYTHONPATH``; this script
detects the flat layout and mounts it under the expected import namespace:

  PYTHONPATH="$PWD/fastvideo-kernel/fa4${PYTHONPATH:+:$PYTHONPATH}" \\
    uv run --no-sync python \\
    fastvideo-kernel/benchmarks/bench_vsa_blackwell.py --quick

MFU is reported separately from TFLOP/s. Its default denominator is the
official 2,500 TFLOP/s dense BF16 tensor-core peak for one GB200 GPU and is
explicitly overrideable with ``--peak_bf16_tflops``. Relative efficiency
against measured 256x256 baselines does not depend on that nominal peak.

Examples:
  python bench_vsa_blackwell.py
  python bench_vsa_blackwell.py --quick
  python bench_vsa_blackwell.py --seq_lens 32768 --sparsities dense 90 --block_shapes 256x256 128x64 64x64
  python bench_vsa_blackwell.py --mask_mode native --block_shapes 128x64 64x64
  torchrun --standalone --nproc-per-node=4 bench_vsa_blackwell.py --quick --out /tmp/fa4_tray.json
"""

from __future__ import annotations

import argparse
import importlib
import importlib.util
import inspect
import json
import math
import os
from pathlib import Path
import random
import subprocess
import sys
import time
import traceback
import types

os.environ.setdefault("FASTVIDEO_VSA_CUTEDSL", "1")

import numpy as np
import torch

FI_BLOCK = 128  # flashinfer vsa_blackwell R=C
DEFAULT_BLOCK_SHAPES = ((256, 256), (128, 64), (64, 64))
DEFAULT_GB200_BF16_DENSE_TFLOPS = 2500.0
GB200_DATASHEET_URL = (
    "https://dam-cdn.nvd.orangelogic.com/AssetLink/"
    "y441155802qub41q118b2852i557jem5.pdf"
)
ISSUE_URL = "https://github.com/flashinfer-ai/flashinfer/issues/4554"
GIST_URL = "https://gist.github.com/SolitaryThinker/90a1d1447929fc38dc509c1852e76532"
GIST_REVISION = "e15ac9066f23ef3690e33e1cc1fdac45b4b9099f"

KEEP_FRAC = {
    "dense": 1.0,
    "50": 0.5,
    "60": 0.4,
    "70": 0.3,
    "75": 0.25,
    "80": 0.2,
    "87.5": 0.125,
    "90": 0.1,
}


class UnsupportedConfig(ValueError):
    """The backend cannot exactly represent the requested logical mask."""


def parse_block_shape(value: str) -> tuple[int, int]:
    parts = value.lower().replace(",", "x").split("x")
    if len(parts) != 2:
        raise argparse.ArgumentTypeError(f"expected QxKV, got {value!r}")
    try:
        q_block, kv_block = (int(part) for part in parts)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"expected integer QxKV, got {value!r}") from exc
    if q_block <= 0 or kv_block <= 0:
        raise argparse.ArgumentTypeError("Q and KV block sizes must be positive")
    return q_block, kv_block


def flops_sparse_attention(
    bs: int,
    d: int,
    selected_edges: int,
    q_block: int,
    kv_block: int,
) -> float:
    """QK^T + PV FLOPs over selected token pairs only.

    ``selected_edges`` already includes the head dimension. Each selected
    logical edge covers ``q_block * kv_block`` token pairs, and each of QK^T
    and PV costs two FLOPs per head-dimension element.
    """
    return 4.0 * bs * d * selected_edges * q_block * kv_block


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def make_logical_mask(h: int, nq: int, nkv: int, keep: int, seed: int) -> torch.Tensor:
    """[H, NQ, NKV] bool with one diagonal anchor and random other edges.

    CPU generator so the pattern is device-independent and fully seeded.
    """
    if not 1 <= keep <= nkv:
        raise ValueError(f"keep must be in [1, {nkv}], got {keep}")
    g = torch.Generator(device="cpu").manual_seed(seed)
    scores = torch.rand(h, nq, nkv, generator=g)
    q_rows = torch.arange(nq)
    # For equal sequence lengths this is the KV block containing the first
    # token of the Q block. It keeps every softmax row nonempty even when
    # Q and KV block sizes differ.
    diagonal_anchor = torch.div(q_rows * nkv, nq, rounding_mode="floor").clamp_max(nkv - 1)
    scores[:, q_rows, diagonal_anchor] = 2.0  # rand() < 1, so this always wins topk
    idx = torch.topk(scores, keep, dim=-1).indices
    m = torch.zeros(h, nq, nkv, dtype=torch.bool)
    m.scatter_(-1, idx, True)
    assert int(m.sum(-1).min()) == keep and int(m.sum(-1).max()) == keep
    return m.cuda()


def expand_exact256_mask(base_mask: torch.Tensor, q_block: int, kv_block: int) -> torch.Tensor:
    """Expand a Q256/KV256 mask without changing its selected token pairs."""
    if 256 % q_block or 256 % kv_block:
        raise ValueError(f"QxKV block shape {q_block}x{kv_block} must divide 256")
    return base_mask.repeat_interleave(256 // q_block, dim=1).repeat_interleave(
        256 // kv_block, dim=2
    )


def ref_masked_sdpa_fp32(q, k, v, logical_mask, q_block, kv_block, scale, qchunk=2048):
    """fp32 SDPA with the expanded boolean token mask, chunked over q.

    q,k,v: [1,S,H,D] bf16. Returns [S,H,D] fp32.
    TF32 is disabled in main().
    """
    _, seq_len, heads, head_dim = q.shape
    if seq_len % q_block or k.shape[1] % kv_block:
        raise ValueError("reference requires sequence lengths divisible by their logical block sizes")
    qchunk = max(q_block, qchunk // q_block * q_block)
    q32 = q[0].permute(1, 0, 2).float()  # [H,S,D]
    k32 = k[0].permute(1, 0, 2).float()
    v32 = v[0].permute(1, 0, 2).float()
    out = torch.empty(heads, seq_len, head_dim, dtype=torch.float32, device=q.device)
    for i in range(0, seq_len, qchunk):
        j = min(i + qchunk, seq_len)
        scores = torch.matmul(q32[:, i:j], k32.transpose(-1, -2)).mul_(scale)
        mask_rows = logical_mask[:, i // q_block:j // q_block]
        mask_tokens = mask_rows.repeat_interleave(q_block, dim=1)
        mask_tokens = mask_tokens.repeat_interleave(kv_block, dim=2)  # [H,c,S]
        scores.masked_fill_(~mask_tokens, float("-inf"))
        out[:, i:j] = torch.matmul(torch.softmax(scores, dim=-1), v32)
        del scores, mask_tokens
    return out.permute(1, 0, 2).contiguous()  # [S,H,D]


def arm_cutedsl(q, k, v, logical_mask, q_block, kv_block):
    """FA4-lineage CuTe-DSL 256 path. Returns (bench_fn, out [S,H,D], extra).

    Timed as the full wrapper call (mask expansion + map->index included) —
    that is its per-call hot path in a mask-per-call deployment.
    """
    if (q_block, kv_block) != (256, 256):
        raise UnsupportedConfig("cutedsl256 is defined only for logical Q256/KV256")
    _mount_flat_fa4_from_pythonpath()
    from fastvideo_kernel import block_sparse_attn_256 as wrapper_module

    block_sparse_attn_256_bshd = wrapper_module.block_sparse_attn_256_bshd

    nkv = logical_mask.shape[-1]
    vbs = torch.full((nkv, ), kv_block, dtype=torch.int32, device=q.device)
    mask = logical_mask.unsqueeze(0)  # [1,H,NQ,NKV]

    def fn():
        return block_sparse_attn_256_bshd(q, k, v, mask, vbs)

    out, _lse = fn()
    interface = importlib.import_module("flash_attn.cute.interface")
    return fn, out[0], {
        "fa4_interface": str(Path(interface.__file__).resolve()),
        "fastvideo_wrapper": str(Path(wrapper_module.__file__).resolve()),
        "fa4_import_mode": _FA4_IMPORT_MODE,
        "route_semantics": "public_vsa256",
        "requested_vsa256_block_shape": os.environ.get(
            "FASTVIDEO_VSA_FA4_BLOCK_SHAPE", "256x256"
        ),
    }


def arm_fa4_wrapper(q, k, v, logical_mask, q_block, kv_block):
    """FastVideo's direct fine-grained BSHD wrapper.

    This arm deliberately includes boolean-map conversion and the VBS
    ``mask_mod``/aux-tensor plumbing in the timed region. Fixed-length cells
    use fully valid KV blocks. In particular, its 64x64 row stays on native
    Q64/KV64 rather than entering the public VSA-256 coalescing adapter.
    """
    if (q_block, kv_block) not in ((128, 64), (64, 64)):
        raise UnsupportedConfig("fa4_wrapper supports Q128/KV64 and Q64/KV64")

    _mount_flat_fa4_from_pythonpath()
    from fastvideo_kernel import block_sparse_attn_cute_fwd as wrapper_module

    block_sparse_attn_cute_fwd_bshd = wrapper_module.block_sparse_attn_cute_fwd_bshd

    nkv = logical_mask.shape[-1]
    vbs = torch.full((nkv,), kv_block, dtype=torch.int32, device=q.device)
    mask = logical_mask.unsqueeze(0)

    def fn():
        return block_sparse_attn_cute_fwd_bshd(q, k, v, mask, vbs)

    out, _lse = fn()
    interface = importlib.import_module("flash_attn.cute.interface")
    return fn, out[0], {
        "fa4_interface": str(Path(interface.__file__).resolve()),
        "fastvideo_wrapper": str(Path(wrapper_module.__file__).resolve()),
        "fa4_import_mode": _FA4_IMPORT_MODE,
        "route_semantics": "direct_native_fine_grained",
        "physical_tile": [q_block, kv_block],
    }


def arm_flashinfer(q, k, v, logical_mask, q_block, kv_block):
    """flashinfer vsa_blackwell (blk128). Returns (bench_fn, out, extra).

    run() timed alone (flashinfer's documented plan-once/run-many model);
    steady-state plan() wall time recorded separately per cell.
    """
    if q_block % FI_BLOCK or kv_block % FI_BLOCK:
        raise UnsupportedConfig(
            "flashinfer vsa_blackwell R=C=128 cannot exactly represent this logical mask"
        )
    from flashinfer.sparse import BlockSparseAttentionWrapper

    if "block_mask" not in inspect.signature(BlockSparseAttentionWrapper.plan).parameters:
        raise UnsupportedConfig(
            "installed FlashInfer lacks the per-head block_mask planning API used by "
            "issue #4554 (the issue used flashinfer 0.6.16.post2)"
        )

    _, seq_len, heads, head_dim = q.shape
    mask128 = logical_mask.repeat_interleave(q_block // FI_BLOCK, dim=1).repeat_interleave(
        kv_block // FI_BLOCK, dim=2
    )
    workspace = torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device=q.device)
    wrapper = BlockSparseAttentionWrapper(workspace, backend="vsa_blackwell")

    def do_plan():
        wrapper.plan(
            None,
            None,
            M=seq_len,
            N=seq_len,
            R=FI_BLOCK,
            C=FI_BLOCK,
            num_qo_heads=heads,
            num_kv_heads=heads,
            head_dim=head_dim,
            block_mask=mask128,
            q_data_type=torch.bfloat16,
            o_data_type=torch.bfloat16,
        )

    torch.cuda.synchronize()
    t0 = time.perf_counter()
    do_plan()
    torch.cuda.synchronize()
    plan_first = (time.perf_counter() - t0) * 1e3  # includes JIT on first cell
    plan_steady = []
    for _ in range(3):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        do_plan()
        torch.cuda.synchronize()
        plan_steady.append((time.perf_counter() - t0) * 1e3)
    qn, kn, vn = q[0], k[0], v[0]  # NHD views of the same storage

    def fn():
        return wrapper.run(qn, kn, vn)

    out = fn()
    return fn, out, {
        "plan_ms_first": round(plan_first, 2),
        "plan_ms": round(min(plan_steady), 2),
        "keep_alive": wrapper,
    }


_FA4_IMPORT_MODE = "normal"


def _flat_fa4_pythonpath() -> Path | None:
    """Find FastVideo's flat ``flash_attn.cute`` source on PYTHONPATH."""
    for entry in os.environ.get("PYTHONPATH", "").split(os.pathsep):
        if not entry:
            continue
        candidate = Path(entry).resolve()
        if all((candidate / name).is_file() for name in ("UPSTREAM.md", "interface.py", "block_sparsity.py")):
            return candidate
    return None


def _mount_flat_fa4_from_pythonpath() -> None:
    """Mount the flat vendored tree as ``flash_attn.cute`` without installing.

    A regular upstream checkout already has ``flash_attn/cute`` and needs no
    special handling. FastVideo intentionally vendors only that subpackage,
    so its source root itself is the package directory.
    """
    global _FA4_IMPORT_MODE
    source = _flat_fa4_pythonpath()
    if source is None:
        return

    try:
        current = importlib.util.find_spec("flash_attn.cute.interface")
    except (ImportError, AttributeError, ValueError):
        current = None
    if current is not None and current.origin is not None and Path(current.origin).resolve().parent == source:
        _FA4_IMPORT_MODE = "local-pythonpath"
        return

    # Load the package under its real namespace so the vendored source's
    # absolute ``flash_attn.cute.*`` imports resolve back to the same tree.
    try:
        import flash_attn
    except ModuleNotFoundError as exc:
        if exc.name != "flash_attn":
            raise
        # The fork is a standalone ``flash_attn.cute`` distribution. Create
        # its namespace parent when FA2's top-level package is not installed.
        flash_attn = types.ModuleType("flash_attn")
        flash_attn.__path__ = []
        sys.modules["flash_attn"] = flash_attn

    for module_name in tuple(sys.modules):
        if module_name == "flash_attn.cute" or module_name.startswith("flash_attn.cute."):
            del sys.modules[module_name]
    spec = importlib.util.spec_from_file_location(
        "flash_attn.cute",
        source / "__init__.py",
        submodule_search_locations=[str(source)],
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"could not create an import spec for local FA4 source {source}")
    package = importlib.util.module_from_spec(spec)
    sys.modules["flash_attn.cute"] = package
    setattr(flash_attn, "cute", package)
    spec.loader.exec_module(package)
    _FA4_IMPORT_MODE = "flat-pythonpath"


def _load_fa4_symbols():
    _mount_flat_fa4_from_pythonpath()
    block_sparsity = importlib.import_module("flash_attn.cute.block_sparsity")
    interface = importlib.import_module("flash_attn.cute.interface")
    return block_sparsity.BlockSparseTensorsTorch, interface._flash_attn_fwd, interface


def _fa4_physical_tiles(q_block: int, kv_block: int) -> tuple[int, int]:
    """Map logical sparse blocks to FA4's physical MMA tiles.

    KV256 is the historical logical VSA block and expands to KV128. Smaller
    requested KV blocks remain physical. Q256 is two Q128 stages; Q128 and
    Q64 select the single-stage paths being optimized.
    """
    tile_m = min(q_block, 128)
    tile_n = min(kv_block, 128)
    if q_block % tile_m or kv_block % tile_n:
        raise UnsupportedConfig(
            f"logical QxKV={q_block}x{kv_block} is not divisible by physical "
            f"FA4 tile {tile_m}x{tile_n}"
        )
    return tile_m, tile_n


def arm_fa4(q, k, v, logical_mask, q_block, kv_block):
    """Direct FA4 block-sparse forward; mask preparation is not timed."""
    BlockSparseTensorsTorch, _flash_attn_fwd, interface = _load_fa4_symbols()
    tile_m, tile_n = _fa4_physical_tiles(q_block, kv_block)
    kv_expansion = kv_block // tile_n
    physical_mask = logical_mask.repeat_interleave(kv_expansion, dim=-1).contiguous()
    heads, nq, nkv_physical = physical_mask.shape
    selected_per_row = int(physical_mask[0, 0].sum().item())
    if selected_per_row <= 0 or not bool((physical_mask.sum(dim=-1) == selected_per_row).all()):
        raise ValueError("direct FA4 arm requires the same positive selected-block count per row")

    physical_indices = torch.arange(nkv_physical, dtype=torch.int32, device=q.device)
    physical_indices = physical_indices.view(1, 1, -1).expand_as(physical_mask)
    full_idx = physical_indices.masked_select(physical_mask).view(1, heads, nq, selected_per_row)
    full_cnt = torch.full((1, heads, nq), selected_per_row, dtype=torch.int32, device=q.device)
    # Every selected logical block is fully valid in this fixed-length benchmark.
    # Keep a compact empty partial-block representation to exercise FA4's full path.
    mask_cnt = torch.zeros((1, heads, nq), dtype=torch.int32, device=q.device)
    mask_idx = torch.zeros((1, heads, nq, 1), dtype=torch.int32, device=q.device)
    sparse_tensors = BlockSparseTensorsTorch(
        mask_block_cnt=mask_cnt,
        mask_block_idx=mask_idx,
        full_block_cnt=full_cnt,
        full_block_idx=full_idx.contiguous(),
        block_size=(q_block, tile_n),
    )

    batch, seq_len, _, _ = q.shape
    out_buffer = torch.empty_like(q)
    lse_buffer = torch.empty((batch, heads, seq_len), dtype=torch.float32, device=q.device)
    scale = 1.0 / math.sqrt(q.shape[-1])

    def fn():
        return _flash_attn_fwd(
            q,
            k,
            v,
            out=out_buffer,
            lse=lse_buffer,
            softmax_scale=scale,
            tile_mn=(tile_m, tile_n),
            num_splits=1,
            pack_gqa=False,
            block_sparse_tensors=sparse_tensors,
            causal=False,
            return_lse=True,
        )[:2]

    out, _lse = fn()
    return fn, out[0], {
        "fa4_interface": str(Path(interface.__file__).resolve()),
        "fa4_import_mode": _FA4_IMPORT_MODE,
        "fa4_tile_m": tile_m,
        "fa4_tile_n": tile_n,
        "fa4_sparse_q_block": q_block,
        "fa4_sparse_kv_block": tile_n,
        "fa4_kv_expansion": kv_expansion,
        "keep_alive": (sparse_tensors, out_buffer, lse_buffer),
    }


ARMS = {
    "flashinfer": arm_flashinfer,
    "cutedsl256": arm_cutedsl,
    "fa4_wrapper": arm_fa4_wrapper,
    "fa4": arm_fa4,
}


def detect_arms(requested):
    if requested != ["auto"]:
        return requested
    arms = []
    try:
        from flashinfer.sparse import BlockSparseAttentionWrapper  # noqa: F401
        if "block_mask" not in inspect.signature(BlockSparseAttentionWrapper.plan).parameters:
            raise RuntimeError(
                "per-head block_mask planning API unavailable; issue #4554 used "
                "flashinfer 0.6.16.post2"
            )
        arms.append("flashinfer")
    except Exception as exc:  # noqa: BLE001
        print(f"# arm flashinfer unavailable: {exc!r}", flush=True)
    fa4_available = False
    try:
        _load_fa4_symbols()
        fa4_available = True
    except Exception as exc:  # noqa: BLE001
        print(f"# arm fa4 unavailable: {exc!r}", flush=True)
    if fa4_available:
        try:
            from fastvideo_kernel.block_sparse_attn_256 import block_sparse_attn_256_bshd  # noqa: F401

            arms.append("cutedsl256")
        except Exception as exc:  # noqa: BLE001
            print(f"# arm cutedsl256 unavailable (optional): {exc!r}", flush=True)
        try:
            from fastvideo_kernel.block_sparse_attn_cute_fwd import (  # noqa: F401
                block_sparse_attn_cute_fwd_bshd,
            )

            arms.append("fa4_wrapper")
        except Exception as exc:  # noqa: BLE001
            print(f"# arm fa4_wrapper unavailable (optional): {exc!r}", flush=True)
        arms.append("fa4")
    if not arms:
        raise RuntimeError(
            "no benchmark arm importable — need flashinfer-python, fastvideo_kernel, and/or FA4"
        )
    return arms


def print_table(rows, arms):
    header = "| seq | sparsity | Q blk | KV blk | keep/NKV |"
    separator = "|---|---|---|---|---|"
    for arm in arms:
        header += f" {arm} ms | {arm} TFLOP/s | {arm} MFU % |"
        separator += "---|---|---|"
        if arm == "flashinfer":
            header += " plan ms |"
            separator += "---|"
        if arm == "fa4":
            header += " vs raw 256 % | vs cutedsl256 % |"
            separator += "---|---|"
        if arm == "fa4_wrapper":
            header += " vs cutedsl256 % |"
            separator += "---|"
    print("\n" + header + "\n" + separator)
    for row in rows:
        line = (
            f"| {row['seq_len']} | {row['sparsity']} | {row['q_block']} | {row['kv_block']} | "
            f"{row['keep_kv_blocks']}/{row['nkv_blocks']} |"
        )
        for arm in arms:
            cell = row.get(arm, {})
            if cell.get("status") == "ok":
                line += f" {cell['fwd_ms']:.3f} | {cell['tflops']:.1f} | {cell['mfu_pct']:.1f} |"
                if arm == "flashinfer":
                    line += f" {cell.get('plan_ms', float('nan')):.2f} |"
                if arm == "fa4":
                    raw_relative = cell.get("vs_fa4_256_pct", float("nan"))
                    wrapper_relative = cell.get("vs_cutedsl256_pct", float("nan"))
                    line += f" {raw_relative:.1f} | {wrapper_relative:.1f} |"
                if arm == "fa4_wrapper":
                    wrapper_relative = cell.get("vs_cutedsl256_pct", float("nan"))
                    line += f" {wrapper_relative:.1f} |"
            else:
                status = cell.get("status", "FAILED")
                line += f" {status} | — | — |"
                if arm == "flashinfer":
                    line += " — |"
                if arm == "fa4":
                    line += " — | — |"
                if arm == "fa4_wrapper":
                    line += " — |"
        print(line)
    print()


def add_relative_efficiency(cell_rows):
    """Attach denominator-free efficiency relative to the 256x256 baselines."""
    baseline = next(
        (row for row in cell_rows if (row["q_block"], row["kv_block"]) == (256, 256)),
        None,
    )
    if baseline is None:
        return
    raw_tflops = baseline.get("fa4", {}).get("tflops")
    wrapper_tflops = baseline.get("cutedsl256", {}).get("tflops")
    for row in cell_rows:
        fa4_result = row.get("fa4", {})
        if fa4_result.get("status") == "ok":
            if raw_tflops:
                fa4_result["vs_fa4_256_pct"] = round(
                    100.0 * fa4_result["tflops"] / raw_tflops,
                    2,
                )
            if wrapper_tflops:
                fa4_result["vs_cutedsl256_pct"] = round(
                    100.0 * fa4_result["tflops"] / wrapper_tflops,
                    2,
                )
        fine_wrapper_result = row.get("fa4_wrapper", {})
        if fine_wrapper_result.get("status") == "ok" and wrapper_tflops:
            fine_wrapper_result["vs_cutedsl256_pct"] = round(
                100.0 * fine_wrapper_result["tflops"] / wrapper_tflops,
                2,
            )


def main():
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size > 1:
        if local_rank >= torch.cuda.device_count():
            raise RuntimeError(
                f"LOCAL_RANK={local_rank} exceeds visible CUDA device count={torch.cuda.device_count()}"
            )
        torch.cuda.set_device(local_rank)

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--seq_lens", type=int, nargs="+", default=[4096, 8192, 16384, 32768, 49152, 65536])
    parser.add_argument("--sparsities", type=str, nargs="+", default=list(KEEP_FRAC), choices=list(KEEP_FRAC))
    parser.add_argument("--arms", type=str, nargs="+", default=["auto"], choices=["auto", *ARMS])
    parser.add_argument(
        "--block_shapes",
        type=parse_block_shape,
        nargs="+",
        default=list(DEFAULT_BLOCK_SHAPES),
        metavar="QxKV",
        help="independent logical QxKV sparse block pairs (default: 256x256 128x64 64x64)",
    )
    parser.add_argument(
        "--mask_mode",
        choices=("exact256", "native"),
        default="exact256",
        help=(
            "exact256 expands one shared Q256/KV256 mask into every shape; "
            "native samples each shape independently (default: exact256)"
        ),
    )
    parser.add_argument(
        "--peak_bf16_tflops",
        type=float,
        default=DEFAULT_GB200_BF16_DENSE_TFLOPS,
        help=(
            "per-GPU dense BF16 tensor-core peak used only for MFU %% "
            f"(default: official GB200 {DEFAULT_GB200_BF16_DENSE_TFLOPS:.0f} TFLOP/s)"
        ),
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="sanity grid: {8192,32768} x {dense,90} x requested block shapes",
    )
    parser.add_argument("--out", default="bsa_bench_results.json")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--rep", type=int, default=20)
    parser.add_argument("--num_heads", type=int, default=12)
    parser.add_argument("--head_dim", type=int, default=128)
    args = parser.parse_args()
    if args.quick:
        args.seq_lens, args.sparsities = [8192, 32768], ["dense", "90"]
    if "auto" in args.arms and args.arms != ["auto"]:
        parser.error("--arms auto cannot be combined with explicit arms")
    if args.peak_bf16_tflops <= 0:
        parser.error("--peak_bf16_tflops must be positive")
    args.block_shapes = list(dict.fromkeys(args.block_shapes))
    for seq_len in args.seq_lens:
        if args.mask_mode == "exact256" and seq_len % 256:
            parser.error(f"exact256 mask mode requires sequence length divisible by 256; got {seq_len}")
        for q_block, kv_block in args.block_shapes:
            if seq_len % q_block or seq_len % kv_block:
                parser.error(
                    f"sequence length {seq_len} must be divisible by QxKV block shape "
                    f"{q_block}x{kv_block}"
                )
            if args.mask_mode == "exact256" and (256 % q_block or 256 % kv_block):
                parser.error(
                    "exact256 mask mode requires both block sizes to divide 256; "
                    f"got {q_block}x{kv_block}"
                )

    from triton.testing import do_bench

    # The fp32 reference must be true fp32, not TF32.
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    arms = detect_arms(args.arms)
    batch, heads, head_dim = 1, args.num_heads, args.head_dim
    scale = 1.0 / math.sqrt(head_dim)

    meta = {
        "device": torch.cuda.get_device_name(),
        "cuda_device_index": torch.cuda.current_device(),
        "rank": rank,
        "local_rank": local_rank,
        "world_size": world_size,
        "capability": list(torch.cuda.get_device_capability()),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "seed": args.seed,
        "warmup": args.warmup,
        "rep": args.rep,
        "batch": batch,
        "heads": heads,
        "head_dim": head_dim,
        "dtype": "bfloat16",
        "arms": arms,
        "block_shapes": [list(shape) for shape in args.block_shapes],
        "mask_mode": args.mask_mode,
        "mask_base_block": 256 if args.mask_mode == "exact256" else None,
        "peak_bf16_dense_tflops": args.peak_bf16_tflops,
        "peak_note": (
            "official single-GB200 dense BF16 tensor-core peak; override for clock-derived diagnostics"
        ),
        "peak_source": GB200_DATASHEET_URL,
        "flop_formula": "4*bs*d*selected_logical_edges*q_block*kv_block",
        "mfu_formula": "100*sparse_aware_algorithmic_tflops/peak_bf16_dense_tflops",
        "correctness_gate": "finite and max_abs_vs_fp32 <= max(1e-2, 8*bf16_rounding_floor)",
        "timing_scope": {
            "flashinfer": "run only; plan reported separately",
            "cutedsl256": "full FastVideo wrapper including mask expansion and map-to-index",
            "fa4_wrapper": (
                "FastVideo direct fine-grained BSHD wrapper including map-to-index and VBS mask "
                "plumbing; 64x64 is native Q64/KV64, not the public VSA-256 coalescing adapter"
            ),
            "fa4": "raw _flash_attn_fwd dispatch; mask/index preparation and output allocation excluded",
        },
        "provenance": {
            "issue": ISSUE_URL,
            "gist": GIST_URL,
            "gist_revision": GIST_REVISION,
        },
        "fa4_env": {
            name: os.environ.get(name)
            for name in (
                "FASTVIDEO_FA4_VSA_DUAL_STREAM",
                "FASTVIDEO_FA4_VSA_SP_DOUBLE_BUFFER",
                "FASTVIDEO_VSA_FA4_BLOCK_SHAPE",
            )
        },
    }
    try:
        meta["driver"] = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"], text=True
        ).split()[0]
    except Exception:  # noqa: BLE001
        meta["driver"] = None
    for module in ("flashinfer", "triton", "fastvideo_kernel"):
        try:
            meta[module] = __import__(module).__version__
        except Exception:  # noqa: BLE001
            pass
    if any(arm in arms for arm in ("cutedsl256", "fa4_wrapper", "fa4")):
        try:
            _, _, fa4_interface = _load_fa4_symbols()
            meta["fa4_interface"] = str(Path(fa4_interface.__file__).resolve())
            meta["fa4_import_mode"] = _FA4_IMPORT_MODE
        except Exception as exc:  # noqa: BLE001
            meta["fa4_import_error"] = f"{type(exc).__name__}: {exc}"
    print("META " + json.dumps(meta), flush=True)

    rows = []
    for seq_len in args.seq_lens:
        # Same q,k,v for every sparsity cell of this seq len (isolates sparsity).
        set_seed(args.seed)
        q = torch.randn(batch, seq_len, heads, head_dim, dtype=torch.bfloat16, device="cuda")
        k = torch.randn(batch, seq_len, heads, head_dim, dtype=torch.bfloat16, device="cuda")
        v = torch.randn(batch, seq_len, heads, head_dim, dtype=torch.bfloat16, device="cuda")

        for label in args.sparsities:
            fraction = KEEP_FRAC[label]
            base_mask_seed = args.seed + seq_len + int(fraction * 1000)
            if args.mask_mode == "exact256":
                base_nblocks_256 = seq_len // 256
                base_keep_256 = (
                    base_nblocks_256
                    if label == "dense"
                    else max(1, round(fraction * base_nblocks_256))
                )
                base_mask_256 = make_logical_mask(
                    heads,
                    base_nblocks_256,
                    base_nblocks_256,
                    base_keep_256,
                    base_mask_seed,
                )
            else:
                base_nblocks_256 = None
                base_keep_256 = None
                base_mask_256 = None
            exact_ref = (
                ref_masked_sdpa_fp32(
                    q,
                    k,
                    v,
                    base_mask_256,
                    256,
                    256,
                    scale,
                )
                if base_mask_256 is not None
                else None
            )
            cell_rows = []
            fa4_shape_outputs = {}
            for q_block, kv_block in args.block_shapes:
                nq_blocks, nkv_blocks = seq_len // q_block, seq_len // kv_block
                if args.mask_mode == "exact256":
                    if base_mask_256 is None:  # pragma: no cover - guarded above
                        raise AssertionError("exact256 mode requires a base mask")
                    keep = base_keep_256 * (256 // kv_block)
                    mask_seed = base_mask_seed
                    logical_mask = expand_exact256_mask(base_mask_256, q_block, kv_block)
                else:
                    keep = nkv_blocks if label == "dense" else max(1, round(fraction * nkv_blocks))
                    mask_seed = (
                        base_mask_seed
                        + q_block * 1_000_003
                        + kv_block * 9_176
                    )
                    logical_mask = make_logical_mask(
                        heads,
                        nq_blocks,
                        nkv_blocks,
                        keep,
                        mask_seed,
                    )
                selected_edges = heads * nq_blocks * keep
                flops = flops_sparse_attention(
                    batch,
                    head_dim,
                    selected_edges,
                    q_block,
                    kv_block,
                )
                row = {
                    "seq_len": seq_len,
                    "sparsity": label,
                    "mask_mode": args.mask_mode,
                    "base_nblocks_256": base_nblocks_256,
                    "base_keep_256": base_keep_256,
                    "base_mask_seed": base_mask_seed,
                    "q_block": q_block,
                    "kv_block": kv_block,
                    "nq_blocks": nq_blocks,
                    "nkv_blocks": nkv_blocks,
                    "keep_kv_blocks": keep,
                    "selected_logical_edges": selected_edges,
                    "actual_sparsity": round(1.0 - keep / nkv_blocks, 4),
                    "mask_seed": mask_seed,
                    "flops": flops,
                }

                ref = (
                    exact_ref
                    if exact_ref is not None
                    else ref_masked_sdpa_fp32(
                        q,
                        k,
                        v,
                        logical_mask,
                        q_block,
                        kv_block,
                        scale,
                    )
                )
                # Irreducible bf16 output-rounding floor: even a bit-exact kernel
                # emitting bf16 cannot beat this vs the fp32 reference.
                row["bf16_floor_max_abs"] = float(
                    (ref - ref.to(torch.bfloat16).float()).abs().max()
                )
                row["correctness_max_abs_limit"] = max(
                    1e-2,
                    8.0 * row["bf16_floor_max_abs"],
                )

                outputs = {}
                for name in arms:
                    result = {}
                    try:
                        fn, out, extra = ARMS[name](
                            q,
                            k,
                            v,
                            logical_mask,
                            q_block,
                            kv_block,
                        )
                        torch.cuda.synchronize()
                        result["max_abs_vs_fp32"] = float((out.float() - ref).abs().max())
                        if not bool(torch.isfinite(out).all()):
                            raise RuntimeError(f"{name} produced NaN or Inf")
                        if result["max_abs_vs_fp32"] > row["correctness_max_abs_limit"]:
                            raise RuntimeError(
                                f"{name} max_abs_vs_fp32={result['max_abs_vs_fp32']:.6g} "
                                f"exceeds limit={row['correctness_max_abs_limit']:.6g}"
                            )
                        outputs[name] = out
                        elapsed_ms = float(
                            do_bench(fn, warmup=args.warmup, rep=args.rep, quantiles=None)
                        )
                        achieved_tflops = flops / elapsed_ms * 1e-9
                        result["fwd_ms"] = round(elapsed_ms, 4)
                        result["tflops"] = round(achieved_tflops, 2)
                        result["mfu_pct"] = round(
                            100.0 * achieved_tflops / args.peak_bf16_tflops,
                            2,
                        )
                        for key, value in extra.items():
                            if key != "keep_alive":
                                result[key] = value
                        result["status"] = "ok"
                        del fn, extra
                    except UnsupportedConfig as exc:
                        result["status"] = "SKIPPED"
                        result["error"] = str(exc)
                    except Exception as exc:  # noqa: BLE001
                        result["status"] = "FAILED"
                        result["error"] = f"{type(exc).__name__}: {exc}"[:500]
                        traceback.print_exc()
                    row[name] = result
                    torch.cuda.empty_cache()

                output_names = list(outputs)
                if len(output_names) > 1:
                    row["cross_max_abs"] = {}
                    for lhs_idx, lhs in enumerate(output_names):
                        for rhs in output_names[lhs_idx + 1:]:
                            row["cross_max_abs"][f"{lhs}_vs_{rhs}"] = float(
                                (outputs[lhs].float() - outputs[rhs].float()).abs().max()
                            )
                if args.mask_mode == "exact256" and "fa4" in outputs:
                    fa4_shape_outputs[(q_block, kv_block)] = outputs["fa4"]
                outputs.clear()
                del ref, logical_mask
                torch.cuda.empty_cache()
                cell_rows.append(row)

            if args.mask_mode == "exact256":
                baseline_output = fa4_shape_outputs.get((256, 256))
                if baseline_output is not None:
                    for row in cell_rows:
                        shape = (row["q_block"], row["kv_block"])
                        target_output = fa4_shape_outputs.get(shape)
                        if target_output is not None:
                            row["fa4"]["max_abs_vs_fa4_256"] = float(
                                (target_output.float() - baseline_output.float()).abs().max()
                            )
                fa4_shape_outputs.clear()
            add_relative_efficiency(cell_rows)
            for row in cell_rows:
                print("ROW " + json.dumps(row), flush=True)
            rows.extend(cell_rows)
            del base_mask_256, exact_ref
        del q, k, v
        torch.cuda.empty_cache()

    output_path = Path(args.out)
    if world_size > 1:
        output_path = output_path.with_name(
            f"{output_path.stem}.rank{rank}{output_path.suffix}"
        )
    with output_path.open("w") as output_file:
        json.dump({"meta": meta, "rows": rows}, output_file, indent=2)
    print_table(rows, arms)
    print(f"wrote {output_path}")
    return 0


if __name__ == "__main__":
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required")
    sys.exit(main())
