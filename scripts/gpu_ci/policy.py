"""Reviewed lane policy and validation at the trusted dispatcher boundary."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Mapping

BACKENDS = ("slurm", "modal", "vllm")
SCOPES = ("fastcheck", "full", "merge", "direct", "scheduled")
DEFAULT_REPOSITORY = "https://github.com/hao-ai-lab/FastVideo.git"


def _lane(key: str, public_type: str, script: str, gpus: int = 1, *, fastcheck: bool = False,
          extras: tuple[str, ...] = ("test",), fa4: str = "0") -> dict[str, Any]:
    return {"key": key, "public_type": public_type, "script": script, "gpus": gpus,
            "fastcheck": fastcheck, "extras": list(extras), "fa4": fa4, "timeout_seconds": 5400,
            "kernel": key != "dreamverse",
            "modal_gpu": "H100" if key in {"encoder", "kernel-tests", "training-vsa"} else "L40S",
            "modal_fa4": "0" if key in {"transformer", "training", "distillation", "self-forcing",
                                        "lora-training", "training-vsa", "train-framework"} else "1"}


LANES = (
    _lane("encoder", "encoder", "lanes/encoder.sh", fastcheck=True),
    _lane("vae", "vae", "lanes/vae.sh", fastcheck=True),
    _lane("transformer", "transformer", "lanes/transformer.sh", fastcheck=True),
    _lane("kernel-tests", "kernel_tests", "lanes/kernel_tests.sh", fastcheck=True),
    _lane("unit", "unit_test", "unit_test.sh", fastcheck=True),
    _lane("dreamverse", "dreamverse_app", "lanes/dreamverse.sh", fastcheck=True,
          extras=("test", "dreamverse")),
    _lane("golden-gate", "golden_gate", "lanes/golden_gate.sh"),
    _lane("ssim", "ssim", "lanes/ssim.sh", 4, fa4="1"),
    _lane("lora-inference", "inference_lora", "lanes/inference_lora.sh"),
    _lane("lora-extraction", "lora_extraction", "lanes/lora_extraction.sh"),
    _lane("training", "training", "lanes/training.sh", 4),
    _lane("distillation", "distillation_dmd", "lanes/distillation_dmd.sh", 2),
    _lane("self-forcing", "self_forcing", "lanes/self_forcing.sh", 2),
    _lane("lora-training", "training_lora", "lanes/training_lora.sh", 2),
    _lane("training-vsa", "training_vsa", "lanes/training_vsa.sh", 2),
    _lane("inference-vmoba", "inference_vmoba", "lanes/inference_vmoba.sh"),
    _lane("performance", "performance", "lanes/performance.sh", 2),
    _lane("api-server", "api_server", "lanes/api_server.sh"),
    _lane("train-framework", "train_framework", "lanes/train_framework.sh"),
    _lane("eval", "eval", "lanes/eval.sh", extras=("test", "eval-full")),
)
LANES_BY_KEY = {lane["key"]: lane for lane in LANES}


def backend_name(value: str | None, default: str = "slurm") -> str:
    backend = value or default
    if backend not in BACKENDS:
        raise ValueError(f"Unknown CI_GPU_BACKEND: {backend!r}")
    return backend


def _identifier(value: str, label: str) -> str:
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,99}", value):
        raise ValueError(f"Invalid {label}")
    return value


def build_request(env: Mapping[str, str], config: Mapping[str, Any], root: Path) -> dict[str, Any]:
    """Copy only the explicitly accepted Buildkite fields; never forward its token."""
    backend = backend_name(env.get("CI_GPU_BACKEND"), config.get("default_backend", "slurm"))
    if backend == "slurm":
        raise ValueError("Slurm uses the existing trusted uploader and dispatcher")
    repository = config.get("repository", DEFAULT_REPOSITORY)
    if not re.fullmatch(r"https://github\.com/[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+\.git", repository):
        raise ValueError("repository must be a fixed HTTPS GitHub .git URL")
    supplied_repo = env.get("BUILDKITE_REPO", "")
    ssh_repo = "git@github.com:" + repository.removeprefix("https://github.com/")
    if supplied_repo not in (repository, ssh_repo):
        raise ValueError("Buildkite repository does not match operator policy")
    commit = env.get("BUILDKITE_COMMIT", "")
    if not re.fullmatch(r"[0-9a-f]{40}", commit):
        raise ValueError("BUILDKITE_COMMIT must be an immutable 40-character SHA")
    build_id = _identifier(env.get("BUILDKITE_BUILD_ID", ""), "BUILDKITE_BUILD_ID")
    job_id = _identifier(env.get("BUILDKITE_JOB_ID", ""), "BUILDKITE_JOB_ID")
    pr = env.get("BUILDKITE_PULL_REQUEST", "false")
    if pr == "false" and env.get("PR_NUMBER"):
        pr = env["PR_NUMBER"]
    if pr not in ("", "false") and not re.fullmatch(r"[1-9][0-9]*", pr):
        raise ValueError("Invalid PR number")
    scope = env.get("TEST_SCOPE") or "fastcheck"
    if scope not in SCOPES:
        raise ValueError(f"Invalid TEST_SCOPE: {scope!r}")
    # Reference publication requires a separately trusted publishing job.
    if env.get("FASTVIDEO_SSIM_BOOTSTRAP_MODE", "0") not in ("", "0", "false"):
        raise ValueError("SSIM bootstrap publication is not supported by the isolated GPU runner")
    request = {"build_id": f"{build_id}.{job_id}", "buildkite_build_id": build_id,
               "repository": repository, "commit": commit, "pr_number": pr or "false",
               "pr_key": f"{repository}#{pr}" if pr not in ("", "false") else None,
               "backend": backend, "scope": scope, "env": {"TEST_SCOPE": scope,
               "BUILDKITE_PULL_REQUEST": pr or "false", "BUILDKITE_COMMIT": commit,
               "BUILDKITE_REPO": repository, "BUILDKITE_BUILD_ID": build_id,
               "BUILDKITE_JOB_ID": job_id}}
    for key in ("BUILDKITE_BRANCH", "BUILDKITE_SOURCE", "BUILDKITE_BUILD_URL"):
        value = env.get(key, "")
        if len(value) > 2048 or any(ord(c) < 32 for c in value):
            raise ValueError(f"Invalid {key}")
        request["env"][key] = value
    if scope == "direct":
        requested = env.get("TEST_TYPE", "").removesuffix("_ci")
        selected = [lane for lane in LANES if requested in (lane["public_type"], lane["key"])]
        if len(selected) != 1:
            raise ValueError(f"Unknown direct TEST_TYPE: {requested!r}")
    elif scope == "merge":
        encoded = env.get("MERGE_TEST_PLAN", "")
        if not re.fullmatch(r",(?:none|[a-z]+(?:-[a-z]+)*(?:,[a-z]+(?:-[a-z]+)*)*),", encoded):
            raise ValueError("Missing or malformed trusted MERGE_TEST_PLAN")
        names = encoded[1:-1].split(",")
        if names == ["none"]:
            names = []
        if any(name not in LANES_BY_KEY or LANES_BY_KEY[name]["fastcheck"] for name in names):
            raise ValueError("MERGE_TEST_PLAN contains an unknown or non-integration lane")
        if len(names) != len(set(names)):
            raise ValueError("Duplicate MERGE_TEST_PLAN lane")
        selected = [lane for lane in LANES if lane["key"] in names]
    elif scope == "scheduled":
        requested = env.get("TEST_TYPE", "ssim").removesuffix("_ci")
        if requested != "ssim":
            raise ValueError("Scheduled scope supports SSIM; performance uses direct scope")
        selected = [LANES_BY_KEY[requested]]
    else:
        selected = [lane for lane in LANES if scope == "full" or lane["fastcheck"]]
    for key, lane_key, directory in (("FASTVIDEO_GOLDEN_TEST_FILES", "golden-gate", "golden_gate"),
                                     ("FASTVIDEO_SSIM_TEST_FILES", "ssim", "ssim")):
        if lane_key not in {lane["key"] for lane in selected}:
            continue
        merge_key = "MERGE_GOLDEN_TESTS" if lane_key == "golden-gate" else "MERGE_SSIM_TESTS"
        value = env.get(merge_key, "") if scope == "merge" else "all"
        if not value:
            raise ValueError(f"Missing trusted {key} for merge scope")
        if value != "all":
            names = value.split(",")
            if len(names) != len(set(names)) or any(
                not re.fullmatch(r"test_[a-z0-9_]+\.py", name) for name in names
            ):
                raise ValueError(f"Invalid or unknown {key}")
        # Existence belongs to the isolated exact-SHA checkout. The trusted
        # controller need not be redeployed whenever a PR adds a test file.
        request["env"][key] = value
    request["lanes"] = [{**lane, "script": ".buildkite/scripts/" + lane["script"]} for lane in selected]
    return request
