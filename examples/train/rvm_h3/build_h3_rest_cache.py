# SPDX-License-Identifier: Apache-2.0
"""Build an immutable scored full-H3 trajectory cache for REST/AMD training.

Launch with ``torchrun``. The initial production contract intentionally uses
one sequence-parallel full-H3 replica (SP == world size); rank zero decodes,
scores, and writes while every rank participates in the dense teacher forward.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist

from fastvideo.dataset.h3_rest_cache import (
    H3_REST_CACHE_SCHEMA_VERSION,
    sha256_file,
    validate_h3_rest_cache,
)
from fastvideo.logger import init_logger
from fastvideo.train.methods.knowledge_distillation.h3_rest_sampler import (
    H3RESTSamplingConfig,
    MiniMaxH3RESTSampler,
)
from fastvideo.train.methods.knowledge_distillation.h3_rest_utils import (
    canonical_json_hash,
    group_relative_advantages,
    normalize_reward_weights,
)
from fastvideo.train.methods.rl.common.rvm_utils import visual_text_from_h3_prompt
from fastvideo.train.methods.rl.rewards import build_multi_reward_scorer
from fastvideo.train.models.minimax_h3.minimax_h3_rest import (
    MiniMaxH3RESTTeacherModel,
)
from fastvideo.train.utils.config import load_run_config
from fastvideo.train.utils.instantiate import instantiate

logger = init_logger(__name__)


def _extract_prompt(raw_batch: Mapping[str, Any]) -> str:
    infos = raw_batch.get("info_list")
    if isinstance(infos, list) and infos and isinstance(infos[0], Mapping):
        prompt = infos[0].get("prompt") or infos[0].get("caption")
        if prompt is not None:
            return str(prompt)
    captions = raw_batch.get("caption_text")
    if isinstance(captions, list) and captions:
        return str(captions[0])
    raise ValueError("Could not find an H3 prompt in info_list or caption_text")


def _positive_int(mapping: Mapping[str, Any], key: str, default: int) -> int:
    raw = mapping.get(key, default)
    value = int(default if raw is None else raw)
    if value <= 0:
        raise ValueError(f"method.{key} must be positive, got {raw!r}")
    return value


def _finite_float(mapping: Mapping[str, Any], key: str, default: float) -> float:
    raw = mapping.get(key, default)
    value = float(default if raw is None else raw)
    if not torch.isfinite(torch.tensor(value)):
        raise ValueError(f"method.{key} must be finite, got {raw!r}")
    return value


def reward_weights_from_config(reward_config: Mapping[str, Any]) -> dict[str, float]:
    raw_rewards = reward_config.get("rewards", reward_config)
    if not isinstance(raw_rewards, Mapping) or not raw_rewards:
        raise ValueError("method.reward_fn.rewards must be a nonempty mapping")
    weights: dict[str, float] = {}
    for raw_name, raw_spec in raw_rewards.items():
        name = str(raw_name).strip().lower()
        if isinstance(raw_spec, Mapping):
            if "weight" not in raw_spec:
                raise ValueError(f"Reward {name!r} must define weight")
            weights[name] = float(raw_spec["weight"])
        else:
            weights[name] = float(raw_spec)
    return normalize_reward_weights(weights)


def _git_provenance(*, allow_dirty: bool) -> dict[str, Any]:
    def run(*args: str) -> str:
        return subprocess.check_output(
            ["git", *args], stderr=subprocess.STDOUT, text=True
        ).strip()

    try:
        head = run("rev-parse", "HEAD")
        tree = run("rev-parse", "HEAD^{tree}")
        status = run("status", "--porcelain=v1", "--untracked-files=all")
    except (subprocess.CalledProcessError, FileNotFoundError) as exc:
        raise RuntimeError("Cache generation must run from a Git checkout") from exc
    if status and not allow_dirty:
        raise RuntimeError(
            "Refusing to build a reportable REST cache from a dirty source tree:\n"
            + status
        )
    return {
        "head": head,
        "tree": tree,
        "dirty": bool(status),
        "status": status.splitlines(),
    }


def _write_tensor_file(path: Path, payload: dict[str, torch.Tensor]) -> tuple[str, int]:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)
    return sha256_file(path), path.stat().st_size


def _jsonl_write(path: Path, entries: Sequence[Mapping[str, Any]]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for entry in entries:
            handle.write(
                json.dumps(
                    dict(entry),
                    sort_keys=True,
                    ensure_ascii=False,
                    allow_nan=False,
                )
                + "\n"
            )
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _prepare_output(output_dir: Path, *, overwrite: bool, rank: int) -> Path:
    staging = output_dir.with_name(output_dir.name + ".building")
    if rank == 0:
        if output_dir.exists():
            if not overwrite:
                raise FileExistsError(
                    f"REST cache already exists: {output_dir}. Pass --overwrite explicitly."
                )
            shutil.rmtree(output_dir)
        if staging.exists():
            if not overwrite:
                raise FileExistsError(
                    f"Stale REST cache staging directory exists: {staging}"
                )
            shutil.rmtree(staging)
        (staging / "prompts").mkdir(parents=True)
        (staging / "trajectories").mkdir()
    if dist.is_initialized():
        dist.barrier()
    return staging


def _cache_prompt_id(prompt_index: int, prompt: str) -> str:
    digest = canonical_json_hash({"index": prompt_index, "prompt": prompt})[:12]
    return f"p{prompt_index:06d}-{digest}"


def _score_one(
    scorer: Any,
    endpoint_media: torch.Tensor,
    visual_prompt: str,
    reward_names: Sequence[str],
) -> dict[str, float]:
    outputs = scorer(endpoint_media, [visual_prompt])
    missing = [name for name in reward_names if name not in outputs]
    if missing:
        raise RuntimeError(f"Reward scorer omitted configured outputs: {missing}")
    return {name: float(outputs[name][0].detach().cpu()) for name in reward_names}


def build_cache(
    config_path: str,
    *,
    output_override: str | None,
    overwrite: bool,
    overrides: list[str] | None,
) -> None:
    from fastvideo.distributed import (
        maybe_init_distributed_environment_and_model_parallel,
    )

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    cfg = load_run_config(config_path, overrides=overrides)
    tc = cfg.training
    if tc.distributed.tp_size != 1:
        raise ValueError("H3 REST cache generation currently requires tp_size=1")
    maybe_init_distributed_environment_and_model_parallel(
        tc.distributed.tp_size,
        tc.distributed.sp_size,
    )
    rank = dist.get_rank() if dist.is_initialized() else 0
    world_size = dist.get_world_size() if dist.is_initialized() else 1
    if int(tc.distributed.num_gpus) != world_size:
        raise ValueError(
            f"training.distributed.num_gpus={tc.distributed.num_gpus} but torchrun world_size={world_size}"
        )
    if int(tc.distributed.sp_size) != world_size:
        raise ValueError(
            "Initial REST cache generation requires one full-H3 SP replica: "
            f"sp_size={tc.distributed.sp_size}, world_size={world_size}"
        )

    method_cfg = dict(cfg.method)
    output_raw = output_override or method_cfg.get("cache_output_dir")
    if not isinstance(output_raw, str) or not output_raw.strip():
        raise ValueError("method.cache_output_dir or --output-dir is required")
    output_dir = Path(output_raw).expanduser().resolve()
    allow_dirty = bool(method_cfg.get("allow_dirty_source", False))
    git_info = _git_provenance(allow_dirty=allow_dirty)
    staging = _prepare_output(output_dir, overwrite=overwrite, rank=rank)

    model_cfg = cfg.models.get("student")
    if not isinstance(model_cfg, dict):
        raise ValueError("Cache config must define models.student as the frozen full H3")
    teacher = instantiate(model_cfg, training_config=tc)
    if not isinstance(teacher, MiniMaxH3RESTTeacherModel):
        raise TypeError(
            "Cache config models.student._target_ must resolve to MiniMaxH3RESTTeacherModel"
        )
    teacher.init_preprocessors(tc)
    if teacher.dataloader is None:
        raise RuntimeError("Full-H3 cache builder did not initialize a prompt dataloader")
    teacher.transformer.eval()

    student_timesteps_raw = method_cfg.get(
        "student_timesteps", [1000, 750, 500, 250, 0]
    )
    if not isinstance(student_timesteps_raw, Sequence) or isinstance(
        student_timesteps_raw, str
    ):
        raise ValueError("method.student_timesteps must be a sequence")
    sampling_config = H3RESTSamplingConfig(
        student_timesteps=tuple(float(value) for value in student_timesteps_raw),
        substeps_per_segment=_positive_int(
            method_cfg, "teacher_substeps_per_segment", 12
        ),
    )
    sampler = MiniMaxH3RESTSampler(sampling_config)
    samples_per_prompt = _positive_int(method_cfg, "samples_per_prompt", 8)
    if samples_per_prompt < 2:
        raise ValueError("method.samples_per_prompt must be at least two")
    max_prompts = _positive_int(method_cfg, "max_prompts", 100)
    base_seed = int(method_cfg.get("seed", tc.data.seed))
    advantage_eps = _finite_float(method_cfg, "advantage_eps", 1e-6)
    advantage_clip = _finite_float(method_cfg, "advantage_clip", 1.0)
    reward_config = method_cfg.get("reward_fn")
    if not isinstance(reward_config, Mapping) or not reward_config:
        raise ValueError("method.reward_fn must be a nonempty mapping")
    reward_weights = reward_weights_from_config(reward_config)
    reward_names = tuple(reward_weights)
    reward_device_spec = str(method_cfg.get("reward_device", "cuda") or "cuda")
    scorer = None
    if rank == 0:
        reward_device: torch.device | str = (
            teacher.device
            if reward_device_spec == "cuda"
            else torch.device(reward_device_spec)
        )
        scorer = build_multi_reward_scorer(reward_config, device=reward_device)

    manifest: list[dict[str, Any]] = []
    prompt_stream = iter(teacher.dataloader)
    seen_prompt_ids: set[str] = set()
    with torch.no_grad():
        for prompt_index in range(max_prompts):
            try:
                raw_batch = next(prompt_stream)
            except StopIteration as exc:
                raise RuntimeError(
                    f"Prompt dataset ended after {prompt_index}; requested {max_prompts}"
                ) from exc
            full_prompt = _extract_prompt(raw_batch)
            visual_prompt = visual_text_from_h3_prompt(full_prompt)
            prompt_id = _cache_prompt_id(prompt_index, full_prompt)
            if prompt_id in seen_prompt_ids:
                raise RuntimeError(f"Duplicate deterministic prompt id: {prompt_id}")
            seen_prompt_ids.add(prompt_id)

            layout_generator = torch.Generator(device=teacher.device).manual_seed(
                base_seed + 10_000_000 + prompt_index
            )
            prepared = teacher.prepare_batch(
                raw_batch,
                generator=layout_generator,
                latents_source="zeros",
            )
            prompt_relative = Path("prompts") / f"{prompt_id}.pt"
            prompt_hash = ""
            prompt_bytes = 0
            if rank == 0:
                text_embedding = raw_batch.get("text_embedding")
                text_attention_mask = raw_batch.get("text_attention_mask")
                if not torch.is_tensor(text_embedding) or not torch.is_tensor(
                    text_attention_mask
                ):
                    raise ValueError("Prompt batch lacks text_embedding/text_attention_mask")
                prompt_hash, prompt_bytes = _write_tensor_file(
                    staging / prompt_relative,
                    {
                        "text_embedding": text_embedding.detach().cpu().contiguous(),
                        "text_attention_mask": text_attention_mask.detach()
                        .cpu()
                        .contiguous(),
                    },
                )

            candidate_rollouts: list[torch.Tensor] = []
            candidate_scores: dict[str, list[float]] = {
                name: [] for name in reward_names
            }
            for candidate_index in range(samples_per_prompt):
                seed = base_seed + prompt_index * samples_per_prompt + candidate_index
                generator = torch.Generator(device=teacher.device).manual_seed(seed)
                result = sampler.sample(teacher, prepared, generator=generator)
                if rank == 0:
                    candidate_rollouts.append(
                        result.anchor_states.detach().to("cpu", dtype=torch.bfloat16).contiguous()
                    )
                    assert scorer is not None
                    media = teacher.decode_latents(result.endpoint).cpu()
                    scores = _score_one(
                        scorer,
                        media,
                        visual_prompt,
                        reward_names,
                    )
                    for name, value in scores.items():
                        candidate_scores[name].append(value)
                    del media
            if rank == 0:
                reward_tensors = {
                    name: torch.tensor(values, dtype=torch.float32)
                    for name, values in candidate_scores.items()
                }
                reward_advantages, mixed_advantage = group_relative_advantages(
                    reward_tensors,
                    reward_weights,
                    eps=advantage_eps,
                    clip=advantage_clip,
                )
                for candidate_index, anchor_states in enumerate(candidate_rollouts):
                    seed = base_seed + prompt_index * samples_per_prompt + candidate_index
                    trajectory_id = f"{prompt_id}-c{candidate_index:02d}"
                    trajectory_relative = (
                        Path("trajectories") / f"{trajectory_id}.pt"
                    )
                    trajectory_hash, trajectory_bytes = _write_tensor_file(
                        staging / trajectory_relative,
                        {
                            "anchor_states": anchor_states,
                            "anchor_timesteps": result.anchor_timesteps.cpu(),
                        },
                    )
                    manifest.append(
                        {
                            "trajectory_id": trajectory_id,
                            "prompt_id": prompt_id,
                            "prompt": full_prompt,
                            "visual_prompt": visual_prompt,
                            "candidate_index": candidate_index,
                            "seed": seed,
                            "prompt_file": str(prompt_relative),
                            "prompt_sha256": prompt_hash,
                            "prompt_bytes": prompt_bytes,
                            "trajectory_file": str(trajectory_relative),
                            "trajectory_sha256": trajectory_hash,
                            "trajectory_bytes": trajectory_bytes,
                            "reward_scores": {
                                name: float(reward_tensors[name][candidate_index])
                                for name in reward_names
                            },
                            "reward_advantages": {
                                name: float(reward_advantages[name][candidate_index])
                                for name in reward_names
                            },
                            "mixed_advantage": float(
                                mixed_advantage[candidate_index]
                            ),
                        }
                    )
                logger.info(
                    "REST cache prompt %d/%d: id=%s, reward means=%s",
                    prompt_index + 1,
                    max_prompts,
                    prompt_id,
                    {
                        name: float(values.mean())
                        for name, values in reward_tensors.items()
                    },
                )
                del candidate_rollouts
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

    if rank == 0:
        manifest_path = staging / "manifest.jsonl"
        _jsonl_write(manifest_path, manifest)
        metadata: dict[str, Any] = {
            "schema_version": H3_REST_CACHE_SCHEMA_VERSION,
            "num_prompts": max_prompts,
            "samples_per_prompt": samples_per_prompt,
            "num_trajectories": len(manifest),
            "num_segments": len(sampling_config.student_timesteps) - 1,
            "student_timesteps": list(sampling_config.student_timesteps),
            "teacher_dense_schedule": list(sampling_config.dense_schedule),
            "teacher_substeps_per_segment": sampling_config.substeps_per_segment,
            "reward_names": list(reward_names),
            "reward_weights": reward_weights,
            "advantage_eps": advantage_eps,
            "advantage_clip": advantage_clip,
            "manifest_sha256": sha256_file(manifest_path),
            "provenance": {
                "fastvideo_git": git_info,
                "config_path": str(Path(config_path).expanduser().resolve()),
                "config_hash": canonical_json_hash(cfg.raw),
                "teacher_model": model_cfg,
                "teacher_revision": method_cfg.get("teacher_revision"),
                "reward_config": reward_config,
                "reward_revisions": method_cfg.get("reward_revisions", {}),
                "prompt_data_path": tc.data.data_path,
                "seed": base_seed,
                "geometry": {
                    "num_frames": int(tc.data.num_frames),
                    "num_latent_t": int(tc.data.num_latent_t),
                    "height": int(tc.data.num_height),
                    "width": int(tc.data.num_width),
                },
                "distributed": {
                    "world_size": world_size,
                    "sp_size": int(tc.distributed.sp_size),
                    "tp_size": int(tc.distributed.tp_size),
                },
            },
        }
        metadata["fingerprint"] = canonical_json_hash(metadata)
        metadata_path = staging / "metadata.json"
        metadata_path.write_text(
            json.dumps(
                metadata,
                sort_keys=True,
                indent=2,
                ensure_ascii=False,
                allow_nan=False,
            )
            + "\n",
            encoding="utf-8",
        )
        (staging / "COMPLETE").write_text(
            str(metadata["fingerprint"]) + "\n", encoding="utf-8"
        )
        summary = validate_h3_rest_cache(staging, verify_file_hashes=True)
        os.replace(staging, output_dir)
        logger.info(
            "Completed H3 REST cache: %s (%d prompts, %d trajectories, %d examples, fingerprint=%s)",
            output_dir,
            summary.num_prompts,
            summary.num_trajectories,
            summary.num_examples,
            summary.fingerprint,
        )
    if dist.is_initialized():
        dist.barrier()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build immutable full-H3 scored trajectory cache for FastH3 REST"
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--overwrite", action="store_true")
    args, unknown = parser.parse_known_args(sys.argv[1:])
    build_cache(
        args.config,
        output_override=args.output_dir,
        overwrite=bool(args.overwrite),
        overrides=unknown or None,
    )


if __name__ == "__main__":
    main()
