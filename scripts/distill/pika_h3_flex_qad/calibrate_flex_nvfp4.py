# SPDX-License-Identifier: Apache-2.0
"""Static NVFP4 FFN activation scales for Pika H3 Flex: collect, candidates, MSE, choose.

The QAD student quantizes every block's ``ff.fc_in`` / ``ff.fc_out`` input under a static global scale
``448 * 6 / amax``. This picks that amax per layer on real Ref2VA (or T2VA) rows walked by the bf16 Flex teacher
over its six-forward ladder, the same recipe as the FastH3 OmniRef calibration:

1. ``collect`` (torchrun, one launch per tray): forward pre-hooks record each row's input amax per FFN layer,
   split into target video, target audio and conditioning rows (max over the sequence-parallel group).
2. ``candidates`` (CPU): per layer, the max over rows plus clipped alternatives (per-row p99 / p90 and
   0.7 / 0.5 / 0.35 / 0.25 x max); writes the plain-max table too.
3. ``mse`` (torchrun): on disjoint rows plus the rows that set the maxima, forward hooks compare each
   candidate's deployed NVFP4 GEMM (``nvfp4_qad.nvfp4_gemm``) with the bf16 output, per group.
4. ``choose`` (CPU): per layer, the candidate with the lowest all-token relative output MSE (``amax-mse.json``,
   the OmniRef criterion), and the video/audio-balanced variant (``amax-mse-balanced.json``). The step-0 PTQ
   score of the QAD harness then picks the table QAD pins, as the OmniRef sweep compared its arms.

Launch ``collect`` / ``mse`` with the QAD config of the adapter (``--config``) and ``--shard i --num-shards n`` so
each tray takes a disjoint slice; ``torchrun --nproc_per_node 4`` per tray (sequence parallel 4,
``--training.distributed.num_gpus 4 --training.distributed.hsdp_shard_dim 4``). Unknown ``--dotted.key value`` pairs
override the config. ``collect`` takes the first rows of the QAD plan (e.g. ``--models.student.data.per_group.480p
160 --models.student.data.per_group.768p 40``); ``mse`` runs on the config's larger plan and skips collected rows.
"""
from __future__ import annotations

import argparse
import json
import math
import re
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist

FFN_LAYER = re.compile(r"transformer_blocks\.(\d+)\.ff\.(fc_in|fc_out)$")
GROUP_NAMES = ("video", "audio", "condition")  # pdd_qad_recon.packed_row_groups ids 0, 1, 2
MULTIPLIERS = (0.7, 0.5, 0.35, 0.25)
FP4_RANGE = 448.0 * 6.0


def amax_key(module_name: str) -> str | None:
    """``...transformer_blocks.3.ff.fc_in`` -> ``b3.ff.fc_in`` (the QAD / converter table key)."""
    match = FFN_LAYER.search(module_name.replace("_checkpoint_wrapped_module.", "").replace("_fsdp_wrapped_module.",
                                                                                          ""))
    return f"b{match.group(1)}.ff.{match.group(2)}" if match else None


def percentile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    position = q * (len(ordered) - 1)
    low, high = math.floor(position), math.ceil(position)
    return ordered[low] + (ordered[high] - ordered[low]) * (position - low)


def candidate_table(rows: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    """Per layer: candidate name -> amax, from the per-row amax records of ``collect``."""
    keys = sorted({key for row in rows for key in row["amax"]}, key=lambda k: (int(k.split(".")[0][1:]), k))
    table: dict[str, dict[str, float]] = {}
    for key in keys:
        per_row = [max(row["amax"][key]) for row in rows]
        full = max(per_row)
        if not (math.isfinite(full) and full > 0):
            raise ValueError(f"{key}: non-positive or non-finite amax {full}")
        candidates = {"max": full, "row_p99": percentile(per_row, 0.99), "row_p90": percentile(per_row, 0.90)}
        candidates.update({f"{m}x_max": m * full for m in MULTIPLIERS})
        table[key] = candidates
    return table


def outlier_row_ids(rows: list[dict[str, Any]], limit: int) -> list[str]:
    """Rows that set some layer's maximum (most layers first): the MSE pass must see the outliers it clips."""
    counts: dict[str, int] = {}
    keys = {key for row in rows for key in row["amax"]}
    for key in keys:
        top = max(rows, key=lambda row: max(row["amax"][key]))
        counts[top["id"]] = counts.get(top["id"], 0) + 1
    return [row_id for row_id, _ in sorted(counts.items(), key=lambda item: -item[1])[:limit]]


def choose_table(mse: dict[str, dict[str, list[float]]], candidates: dict[str, dict[str, float]],
                 objective: str = "all") -> tuple[dict[str, float], dict[str, Any]]:
    """Per layer, the candidate with the lowest error.

    ``objective="all"`` is the FastH3 OmniRef criterion (relative output MSE over every token); ``"balanced"``
    is the mean of the target-video and target-audio relative errors (audio is ~1% of the tokens, so the
    all-token MSE barely sees it).
    """
    if objective not in ("all", "balanced"):
        raise ValueError(f"objective must be 'all' or 'balanced', got {objective!r}")
    table: dict[str, float] = {}
    report: dict[str, Any] = {}
    for key, cands in candidates.items():
        energy = mse[key]["ref"]
        scores: dict[str, dict[str, float]] = {}
        for name in cands:
            err = mse[key][name]
            rel = {g: err[i] / energy[i] for i, g in enumerate(GROUP_NAMES) if energy[i] > 0}
            target = [rel[g] for g in ("video", "audio") if g in rel]
            scores[name] = {**rel, "all": sum(err) / sum(energy), "balanced": sum(target) / len(target)}
        best = min(scores, key=lambda name: scores[name][objective])
        table[key] = cands[best]
        report[key] = {"best": best, "amax": cands[best], "max": cands["max"], "scores": scores}
    return table, report


# ---------------------------------------------------------------------- GPU passes
def _sum_over_sp(tensor: torch.Tensor, op: Any) -> torch.Tensor:
    from fastvideo.distributed import get_sp_group, get_sp_world_size
    if get_sp_world_size() > 1:
        dist.all_reduce(tensor, op=op, group=get_sp_group().device_group)
    return tensor


class FFNHooks:
    """Hooks on every FFN block linear of ``transformer``; ``groups`` is this SP rank's row-group ids."""

    def __init__(self, transformer: torch.nn.Module) -> None:
        self.modules = {}
        for name, module in transformer.named_modules():
            key = amax_key(name)
            if key is not None:
                self.modules[key] = module
        if len(self.modules) != 100:
            raise ValueError(f"expected 100 FFN block linears (50 blocks x fc_in/fc_out), found {len(self.modules)}")
        self.groups: torch.Tensor | None = None
        self._handles: list[Any] = []

    def _local(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        flat = x.detach().reshape(-1, x.shape[-1])
        assert self.groups is not None
        if flat.shape[0] != self.groups.numel():
            raise RuntimeError(f"FFN input has {flat.shape[0]} rows, the row groups {self.groups.numel()}")
        return flat, self.groups

    @contextmanager
    def attached(self, pre: Any = None, post: Any = None) -> Iterator[None]:
        for key, module in self.modules.items():
            if pre is not None:
                self._handles.append(module.register_forward_pre_hook(pre(key)))
            if post is not None:
                self._handles.append(module.register_forward_hook(post(key)))
        try:
            yield
        finally:
            for handle in self._handles:
                handle.remove()
            self._handles = []


def _group_reduce(values: torch.Tensor, groups: torch.Tensor, reduce: str) -> torch.Tensor:
    out = torch.zeros(len(GROUP_NAMES), dtype=torch.float32, device=values.device)
    valid = groups >= 0
    return out.scatter_reduce(0, groups[valid], values[valid].float(), reduce=reduce, include_self=True)


def _weight(module: torch.nn.Module) -> torch.Tensor:
    weight = module.weight
    full = getattr(weight, "full_tensor", None)
    return full() if callable(full) else weight


def walk_teacher(teacher: Any, prepared: Any, hooks: FFNHooks) -> None:
    """The teacher's own six-forward trajectory of ``prepared`` (no grad), with the row groups set for hooks."""
    from fastvideo.train.methods.knowledge_distillation.pdd_qad_metrics import step_packed
    from fastvideo.train.methods.knowledge_distillation.pdd_qad_recon import local_row_groups, packed_row_groups

    layout = prepared.layout
    hooks.groups = local_row_groups(packed_row_groups(layout, int(layout.sequence_length)), teacher.device)
    schedulers = teacher.schedulers(prepared)
    timesteps = (schedulers[0].timesteps, schedulers[1].timesteps)
    video, audio = prepared.video, prepared.audio
    with torch.no_grad():
        for rung in range(teacher.num_rungs):
            with teacher.rung_scope(prepared, rung, timesteps) as rt:
                out = teacher.forward_rung(prepared, video, audio, rt)
            video, audio = step_packed(schedulers, prepared, rung, video, audio, *out)


def collect_row(teacher: Any, prepared: Any, hooks: FFNHooks) -> dict[str, list[float]]:
    found: dict[str, torch.Tensor] = {}

    def pre(key: str) -> Any:

        def hook(module: torch.nn.Module, args: tuple[Any, ...]) -> None:
            flat, groups = hooks._local(args[0])
            amax = _group_reduce(flat.abs().amax(dim=-1), groups, "amax")
            found[key] = amax if key not in found else torch.maximum(found[key], amax)

        return hook

    with hooks.attached(pre=pre):
        walk_teacher(teacher, prepared, hooks)
    return {key: _sum_over_sp(value, dist.ReduceOp.MAX).tolist() for key, value in sorted(found.items())}


def mse_row(teacher: Any, prepared: Any, hooks: FFNHooks, candidates: dict[str, dict[str, float]],
            sums: dict[str, dict[str, torch.Tensor]]) -> None:
    from fastvideo.layers.quantization.nvfp4_qad import nvfp4_gemm

    def post(key: str) -> Any:

        def hook(module: torch.nn.Module, args: tuple[Any, ...], output: Any) -> None:
            x = args[0]
            reference = (output[0] if isinstance(output, tuple) else output).detach()
            flat_ref = reference.reshape(-1, reference.shape[-1]).float()
            _, groups = hooks._local(x)
            bias = None if getattr(module, "skip_bias_add", False) else module.bias
            weight = _weight(module)
            entry = sums.setdefault(key, {})
            ref_energy = _group_reduce(flat_ref.pow(2).sum(dim=-1), groups, "sum")
            entry["ref"] = entry.get("ref", torch.zeros_like(ref_energy)) + ref_energy
            for name, amax in candidates[key].items():
                scale = torch.tensor(FP4_RANGE / amax, dtype=torch.float32, device=x.device)
                quant = nvfp4_gemm(x, weight, bias, scale).reshape(-1, flat_ref.shape[-1]).float()
                err = _group_reduce((quant - flat_ref).pow(2).sum(dim=-1), groups, "sum")
                entry[name] = entry.get(name, torch.zeros_like(err)) + err

        return hook

    with hooks.attached(post=post):
        walk_teacher(teacher, prepared, hooks)


# ---------------------------------------------------------------------- drivers
def _build(args: argparse.Namespace) -> tuple[Any, list[Any]]:
    """The QAD config's bf16 teacher and its training-row plan (the student is built without quantization)."""
    from fastvideo.distributed import maybe_init_distributed_environment_and_model_parallel
    from fastvideo.train.utils.builder import build_from_config
    from fastvideo.train.utils.config import load_run_config

    overrides = ["--models.student.qad.enabled", "false", "--training.tracker.trackers", "[]"] + list(args.override)
    cfg = load_run_config(args.config, overrides=overrides)
    tc = cfg.training
    maybe_init_distributed_environment_and_model_parallel(tc.distributed.tp_size, tc.distributed.sp_size)
    _, method, _, _ = build_from_config(cfg)
    return method.teacher, list(method.student.train_rows)


def _dp() -> tuple[int, int]:
    """(data-parallel rank, size): sequence-parallel ranks of one group walk the same rows."""
    from fastvideo.distributed import get_sp_world_size, get_world_group

    world = get_world_group()
    return world.rank // get_sp_world_size(), world.world_size // get_sp_world_size()


def _shard(rows: list[Any], shard: int, num_shards: int) -> list[Any]:
    """This group's rows; striding a hash order keeps resolutions mixed (the plan alternates 480p / 768p groups)."""
    import hashlib

    dp_rank, dp_size = _dp()
    ordered = sorted(rows, key=lambda row: hashlib.sha256(row.id.encode()).hexdigest())
    return ordered[shard * dp_size + dp_rank::num_shards * dp_size]


def _is_writer() -> bool:
    from fastvideo.distributed import get_sp_group
    return get_sp_group().rank_in_group == 0


def run_collect(args: argparse.Namespace) -> None:
    teacher, plan = _build(args)
    rows = _shard(plan[:args.max_rows] if args.max_rows else plan, args.shard, args.num_shards)
    hooks = FFNHooks(teacher.transformer)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"collect-shard{args.shard}-dp{_dp()[0]}.jsonl"
    done = {json.loads(line)["id"] for line in path.read_text().splitlines()} if path.exists() else set()
    for index, spec in enumerate(rows):
        if spec.id in done:
            continue
        prepared = teacher.load_row(spec)
        amax = collect_row(teacher, prepared, hooks)
        if _is_writer():
            record = {"id": spec.id, "case": spec.case, "resolution": spec.resolution, "amax": amax,
                      "tokens": int(prepared.layout.sequence_length)}
            with path.open("a") as handle:
                handle.write(json.dumps(record) + "\n")
            print(f"collect {index + 1}/{len(rows)} {spec.id} ({record['tokens']} tokens)", flush=True)


def _read_rows(directory: Path, pattern: str) -> list[dict[str, Any]]:
    rows = {}
    for path in sorted(directory.glob(pattern)):
        for line in path.read_text().splitlines():
            record = json.loads(line)
            rows[record["id"]] = record
    if not rows:
        raise ValueError(f"no {pattern} rows in {directory}")
    return [rows[key] for key in sorted(rows)]


def run_candidates(args: argparse.Namespace) -> None:
    out = Path(args.output_dir)
    rows = _read_rows(out, "collect-*.jsonl")
    table = candidate_table(rows)
    half = candidate_table(rows[:len(rows) // 2])
    gaps = {key: 1 - half[key]["max"] / table[key]["max"] for key in table}
    (out / "candidates.json").write_text(json.dumps(table, indent=1))
    (out / "amax-max.json").write_text(json.dumps({k: v["max"] for k, v in table.items()}, indent=1))
    (out / "outlier_rows.json").write_text(json.dumps(outlier_row_ids(rows, args.outlier_rows)))
    worst = sorted(gaps.items(), key=lambda item: -item[1])[:5]
    print(f"{len(rows)} rows, {len(table)} layers; first-half vs full max gap: worst {worst}")


def run_mse(args: argparse.Namespace) -> None:
    out = Path(args.output_dir)
    candidates = json.loads((out / "candidates.json").read_text())
    collected = {row["id"] for row in _read_rows(out, "collect-*.jsonl")}
    outliers = set(json.loads((out / "outlier_rows.json").read_text()))
    teacher, plan = _build(args)
    # Rows the max was not taken from, a few per (case, resolution), plus the rows that set the maxima.
    per_group: dict[tuple[str, str], int] = {}
    fresh = []
    for spec in plan:
        key = (spec.case, spec.resolution)
        if spec.id not in collected and per_group.get(key, 0) < args.mse_rows_per_group:
            per_group[key] = per_group.get(key, 0) + 1
            fresh.append(spec)
    rows = _shard(fresh + [spec for spec in plan if spec.id in outliers], args.shard, args.num_shards)
    hooks = FFNHooks(teacher.transformer)
    sums: dict[str, dict[str, torch.Tensor]] = {}
    for index, spec in enumerate(rows):
        mse_row(teacher, teacher.load_row(spec), hooks, candidates, sums)
        if _is_writer():
            print(f"mse {index + 1}/{len(rows)} {spec.id}", flush=True)
    reduced = {key: {name: _sum_over_sp(value, dist.ReduceOp.SUM).tolist() for name, value in entry.items()}
               for key, entry in sums.items()}
    if _is_writer():
        path = out / f"mse-shard{args.shard}-dp{_dp()[0]}.json"
        path.write_text(json.dumps({"rows": [spec.id for spec in rows], "sums": reduced}))


def run_choose(args: argparse.Namespace) -> None:
    out = Path(args.output_dir)
    candidates = json.loads((out / "candidates.json").read_text())
    total: dict[str, dict[str, list[float]]] = {}
    for path in sorted(out.glob("mse-*.json")):
        for key, entry in json.loads(path.read_text())["sums"].items():
            slot = total.setdefault(key, {})
            for name, values in entry.items():
                slot[name] = [a + b for a, b in zip(slot.get(name, [0.0] * len(values)), values, strict=True)]
    if set(total) != set(candidates):
        raise ValueError(f"MSE sums cover {len(total)} of {len(candidates)} layers")
    for objective, name in (("all", "amax-mse.json"), ("balanced", "amax-mse-balanced.json")):
        table, report = choose_table(total, candidates, objective)
        (out / name).write_text(json.dumps(table, indent=1))
        (out / name.replace("amax-", "choose-")).write_text(json.dumps(report, indent=1))
        picks: dict[str, int] = {}
        for entry in report.values():
            picks[entry["best"]] = picks.get(entry["best"], 0) + 1
        print(f"{name} ({objective}): {dict(sorted(picks.items(), key=lambda item: -item[1]))}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("stage", choices=("collect", "candidates", "mse", "choose"))
    parser.add_argument("--config", help="the adapter's QAD YAML (collect / mse)")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--max-rows", type=int, default=0, help="cap the calibration plan (smoke tests)")
    parser.add_argument("--mse-rows-per-group", type=int, default=3)
    parser.add_argument("--outlier-rows", type=int, default=8)
    # Any other ``--dotted.key value`` pair overrides the QAD config, as in fastvideo.train.entrypoint.train.
    args, overrides = parser.parse_known_args()
    args.override = overrides
    {"collect": run_collect, "candidates": run_candidates, "mse": run_mse, "choose": run_choose}[args.stage](args)


if __name__ == "__main__":
    main()
