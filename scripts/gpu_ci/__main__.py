"""Operator CLI. Use the isolated entrypoint from an immutable installation."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import threading
from pathlib import Path
from typing import Any

from .dispatcher import (cancellation_signals, execute, make_store, recover,
                         run_directory, upload_artifacts)
from .pipeline import render_pipeline
from .policy import DEFAULT_REPOSITORY, backend_name, build_request

ROOT = Path(__file__).resolve().parents[2]


def load_config(path: Path) -> dict[str, Any]:
    config = json.loads(path.read_text())
    if not isinstance(config, dict):
        raise ValueError("Operator config must be a JSON object")
    for key in ("state_path", "artifacts_dir"):
        value = config.get(key)
        if not isinstance(value, str) or not Path(value).is_absolute():
            raise ValueError(f"{key} must be an absolute local path")
    for key, default in (("max_active_prs", 2), ("max_gpus_per_pr", 4), ("max_gpus", 8),
                         ("queue_timeout_seconds", 21600)):
        value = config.get(key, default)
        if type(value) is not int or value < 1:
            raise ValueError(f"{key} must be a positive integer")
        config[key] = value
    # The initial rollout deliberately cannot raise the user's agreed ceilings.
    for key, ceiling in (("max_active_prs", 2), ("max_gpus_per_pr", 4), ("max_gpus", 8)):
        if config[key] > ceiling:
            raise ValueError(f"{key} exceeds the reviewed ceiling {ceiling}")
    if config.get("repository", DEFAULT_REPOSITORY) != DEFAULT_REPOSITORY:
        raise ValueError("This worker installation is allowlisted only for hao-ai-lab/FastVideo")
    backend_name(config.get("default_backend"))
    return config


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    for name in ("run", "render", "upload", "status", "recover"):
        sub = subparsers.add_parser(name)
        sub.add_argument("--config", type=Path, required=True)
        if name in ("render", "upload"):
            sub.add_argument("--canonical", type=Path, default=ROOT / ".buildkite/pipeline.yml")
        if name == "recover":
            sub.add_argument("--build-id", required=True)
    args = parser.parse_args(argv)
    try:
        config = load_config(args.config)
        if args.command == "status":
            print(json.dumps(make_store(config).snapshot(), indent=2))
            return 0
        if args.command == "recover":
            recover(config, args.build_id)
            return 0
        backend = backend_name(os.environ.get("CI_GPU_BACKEND"), config.get("default_backend", "slurm"))
        if args.command == "upload" and backend == "slurm":
            if config.get("default_backend", "slurm") != "slurm":
                raise ValueError("Slurm overrides are disabled after promoting another backend: legacy statuses are unqualified")
            legacy = config.get("slurm_uploader")
            if (not isinstance(legacy, list) or not legacy or
                    not all(isinstance(x, str) and x for x in legacy) or not Path(legacy[0]).is_absolute()):
                raise ValueError("Configure slurm_uploader as the existing trusted uploader argv")
            return subprocess.run(legacy, check=False).returncode
        if args.command in ("render", "upload"):
            import yaml
            canonical = yaml.safe_load(args.canonical.read_text())
            rendered = render_pipeline(canonical, backend, os.environ.get("TEST_SCOPE", ""),
                                       config.get("queue", "gpu-ci-dispatch"))
            payload = json.dumps(rendered, indent=2) + "\n"
            if args.command == "render":
                print(payload, end="")
                return 0
            return subprocess.run(["buildkite-agent", "pipeline", "upload", "--no-interpolation"],
                                  input=payload, text=True, check=False).returncode
        request = build_request(os.environ, config, ROOT)
        cancel = threading.Event()
        with cancellation_signals(cancel):
            code = execute(request, config, cancel=cancel)
        if config.get("upload_artifacts", True):
            upload_artifacts(run_directory(config, request["build_id"]))
        return code
    except (OSError, ValueError, RuntimeError, KeyError, subprocess.SubprocessError) as error:
        print(f"GPU CI dispatcher error: {error}", file=sys.stderr)
        return 97


if __name__ == "__main__":
    raise SystemExit(main())
