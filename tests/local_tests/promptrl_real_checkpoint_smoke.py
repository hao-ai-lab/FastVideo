# SPDX-License-Identifier: Apache-2.0
"""One-GPU PromptRL smoke test with the real Wan and Qwen checkpoints.

By default this uses a deterministic local reward judge so the smoke targets
the complete training path (checkpoint loading, prompt refinement, Wan rollout,
VAE decode, HTTP reward transport, backward, optimizer step, and bundle
export). Pass ``--real-videoscore2`` in a two-GPU allocation to put the official
VideoScore2 judge on the second GPU and validate the complete production path.

Run on a GPU host from the repository root::

    python -m tests.local_tests.promptrl_real_checkpoint_smoke
"""

from __future__ import annotations

import argparse
import contextlib
import socket
import tempfile
import threading
import time
from pathlib import Path
from collections.abc import Sequence

import torch


class _DeterministicJudge:
    """Cheap scorer with non-constant group rewards for GRPO."""

    def score_batch(
        self,
        videos: Sequence[bytes],
        prompts: Sequence[str],
        *,
        fps: int,
    ) -> list[dict[str, float]]:
        del fps
        if len(videos) != len(prompts):
            raise ValueError("video/prompt cardinality mismatch")
        if any(not video for video in videos):
            raise ValueError("received an empty encoded video")
        return [
            {
                "composite": float(index + 1),
                "visual_quality": float(index + 1),
                "text_alignment": float(index + 1),
                "physical_consistency": float(index + 1),
            }
            for index in range(len(videos))
        ]


def _wait_for_port(port: int, timeout_sec: float = 15.0) -> None:
    deadline = time.monotonic() + timeout_sec
    while time.monotonic() < deadline:
        with socket.socket() as sock:
            sock.settimeout(0.2)
            if sock.connect_ex(("127.0.0.1", port)) == 0:
                return
        time.sleep(0.05)
    raise TimeoutError(f"reward service did not listen on port {port}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="examples/train/configs/rl/wan/promptrl_prompt_only.yaml",
    )
    parser.add_argument("--port", type=int, default=18100)
    parser.add_argument("--real-videoscore2", action="store_true")
    parser.add_argument("--judge-model-id", default="TIGER-Lab/VideoScore2")
    parser.add_argument("--judge-device", default="cuda:1")
    parser.add_argument(
        "--mode",
        choices=("prompt_only", "joint"),
        default="prompt_only",
    )
    parser.add_argument("--work-dir", default="")
    parser.add_argument("--max-train-steps", type=int, default=1)
    parser.add_argument("--checkpoint-every", type=int, default=0)
    parser.add_argument("--resume-from-checkpoint", default="")
    args = parser.parse_args()

    import uvicorn

    from fastvideo.train.entrypoint.train import run_training_from_config
    from fastvideo.train.methods.rl.promptrl.rewards.service import (
        VideoScore2Judge,
        create_reward_app,
    )

    judge = (
        VideoScore2Judge(args.judge_model_id, device=args.judge_device)
        if args.real_videoscore2
        else _DeterministicJudge()
    )
    server = uvicorn.Server(
        uvicorn.Config(
            create_reward_app(judge, max_wait_sec=300.0),
            host="127.0.0.1",
            port=args.port,
            log_level="warning",
        )
    )
    server_thread = threading.Thread(target=server.run, daemon=True)
    server_thread.start()
    _wait_for_port(args.port)

    try:
        work_context = (
            contextlib.nullcontext(args.work_dir)
            if args.work_dir
            else tempfile.TemporaryDirectory(prefix="promptrl_real_smoke_")
        )
        with work_context as work_dir:
            root = Path(work_dir)
            root.mkdir(parents=True, exist_ok=True)
            prompts = root / "prompts.jsonl"
            prompts.write_text(
                '{"id":"smoke-1","prompt":"a red cube rolling on a table",'
                '"reward_tag":"smoke"}\n',
                encoding="utf-8",
            )
            output_dir = root / "output"
            bundle_dir = output_dir / "bundle"
            reward_timeout = "600" if args.real_videoscore2 else "60"
            latent_frames = "2" if args.real_videoscore2 else "1"
            decoded_frames = "5" if args.real_videoscore2 else "1"
            sde_steps = "1" if args.mode == "joint" else "0"
            format_reward_weight = "0" if args.mode == "joint" else "1"
            overrides = [
                "--method.mode",
                args.mode,
                "--method.format_reward_weight",
                format_reward_weight,
                "--method.group_size",
                "2",
                "--method.retained_originals",
                "1",
                "--method.data.data_path",
                str(prompts),
                "--method.reward.endpoint_url",
                f"http://127.0.0.1:{args.port}",
                "--method.reward.timeout_sec",
                reward_timeout,
                "--method.reward.retries",
                "0",
                "--method.reward.fps",
                "4",
                "--method.rollout.num_steps",
                "1",
                "--method.rollout.sde_steps",
                sde_steps,
                "--method.refiner.max_new_tokens",
                "16",
                "--models.refiner.max_prompt_tokens",
                "256",
                "--training.distributed.num_gpus",
                "1",
                "--training.distributed.sp_size",
                "1",
                "--training.distributed.hsdp_shard_dim",
                "1",
                "--training.data.num_latent_t",
                latent_frames,
                "--training.data.num_height",
                "64",
                "--training.data.num_width",
                "64",
                "--training.data.num_frames",
                decoded_frames,
                "--training.dit_precision",
                "bf16",
                "--training.loop.max_train_steps",
                str(args.max_train_steps),
                "--training.checkpoint.output_dir",
                str(output_dir),
                "--training.checkpoint.training_state_checkpointing_steps",
                str(args.checkpoint_every),
                "--training.tracker.trackers",
                "[none]",
                "--method.validation.every_steps",
                "0",
                "--callbacks.promptrl_export.output_dir",
                str(bundle_dir),
            ]
            if args.resume_from_checkpoint:
                overrides.extend([
                    "--training.checkpoint.resume_from_checkpoint",
                    args.resume_from_checkpoint,
                ])
            run_training_from_config(args.config, overrides=overrides)

            manifest = bundle_dir / "manifest.json"
            if not manifest.is_file():
                raise RuntimeError("PromptRL training completed without bundle export")
            if args.mode == "joint":
                from safetensors.torch import load_file

                generator_lora = (
                    bundle_dir
                    / "generator"
                    / "promptrl_generator_lora.safetensors"
                )
                if not generator_lora.is_file():
                    raise RuntimeError(
                        "joint PromptRL smoke did not export generator LoRA"
                    )
                state = load_file(str(generator_lora))
                lora_b = [
                    tensor
                    for name, tensor in state.items()
                    if name.endswith(".lora_B")
                ]
                if not lora_b or not any(
                    bool(torch.count_nonzero(tensor).item())
                    for tensor in lora_b
                ):
                    raise RuntimeError(
                        "joint PromptRL optimizer step left every Wan LoRA B "
                        "weight at zero"
                    )
            if args.checkpoint_every > 0:
                checkpoint = (
                    output_dir
                    / f"checkpoint-{args.max_train_steps}"
                    / "dcp"
                )
                if not checkpoint.is_dir():
                    raise RuntimeError(
                        f"PromptRL smoke did not write expected {checkpoint}"
                    )
            judge_name = "videoscore2" if args.real_videoscore2 else "deterministic"
            print(
                "PROMPTRL_REAL_CHECKPOINT_SMOKE_OK "
                f"mode={args.mode} judge={judge_name} "
                f"steps={args.max_train_steps} "
                f"resume={args.resume_from_checkpoint or 'none'} "
                f"manifest={manifest}",
                flush=True,
            )
    finally:
        server.should_exit = True
        server_thread.join(timeout=10.0)


if __name__ == "__main__":
    main()
