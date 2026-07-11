#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import csv
import tempfile
from pathlib import Path
from typing import Any

import yaml


REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = REPO_ROOT / "examples/train/configs/ablation/wan_causal_mixkit21"
DEFAULT_TEMPLATE = CONFIG_DIR / "tf_2k_template.yaml"
DEFAULT_MATRIX = CONFIG_DIR / "experiment_matrix.tsv"
DEFAULT_VALIDATION = REPO_ROOT / "examples/training/finetune/Wan2.1-VSA/Wan-Syn-Data/validation_4.json"

EXPECTED_CONDITIONS = {
    "A01": ("sink0_local21_absolute", 0, 21, "absolute", "node0"),
    "A02": ("sink0_local21_relative", 0, 21, "relativistic", "node1"),
    "A03": ("sink1_local21_relative", 1, 21, "relativistic", "node0"),
    "A04": ("sink0_local6_relative", 0, 6, "relativistic", "node0"),
    "A05": ("sink1_local6_relative", 1, 6, "relativistic", "node0"),
    "A06": ("sink0_local12_relative", 0, 12, "relativistic", "node1"),
    "A07": ("sink1_local12_relative", 1, 12, "relativistic", "node1"),
    "A08": ("sink3_local12_relative", 3, 12, "relativistic", "node1"),
}


def load_matrix(path: Path) -> dict[str, dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    matrix = {row["id"]: row for row in rows}
    if len(matrix) != len(rows):
        raise ValueError("experiment matrix contains duplicate IDs")
    return matrix


def validate_matrix(matrix: dict[str, dict[str, str]]) -> None:
    if set(matrix) != set(EXPECTED_CONDITIONS):
        raise ValueError(
            f"matrix IDs differ: expected {sorted(EXPECTED_CONDITIONS)}, got {sorted(matrix)}"
        )
    for experiment_id, expected in EXPECTED_CONDITIONS.items():
        row = matrix[experiment_id]
        actual = (
            row["condition"],
            int(row["sink_size"]),
            int(row["local_attn_size"]),
            row["rope_cache_policy"],
            row["lane"],
        )
        if actual != expected:
            raise ValueError(f"{experiment_id}: expected {expected}, got {actual}")
        sink_size = int(row["sink_size"])
        local_attn_size = int(row["local_attn_size"])
        if sink_size and sink_size + 3 > local_attn_size:
            raise ValueError(
                f"{experiment_id}: local={local_attn_size} cannot hold sink={sink_size} plus a 3-frame block"
            )


def render_config(
    template_path: Path,
    row: dict[str, str],
    *,
    output_path: Path,
    run_root: Path,
    project_name: str,
    run_name: str,
    max_train_steps: int,
    checkpoint_steps: int,
    validation_every_steps: int,
    validation_sampling_steps: int,
    validation_dataset_file: Path,
) -> dict[str, Any]:
    stage_dir = run_root / "tf"
    replacements = {
        "__MAX_TRAIN_STEPS__": str(max_train_steps),
        "__CHECKPOINT_STEPS__": str(checkpoint_steps),
        "__CHECKPOINT_DIR__": str(stage_dir / "checkpoints"),
        "__VALIDATION_DIR__": str(stage_dir / "validation"),
        "__PROJECT_NAME__": project_name,
        "__WANDB_RUN_NAME__": run_name,
        "__VALIDATION_DATASET_FILE__": str(validation_dataset_file),
        "__VALIDATION_EVERY_STEPS__": str(validation_every_steps),
        "__VALIDATION_SAMPLING_STEPS__": str(validation_sampling_steps),
        "__LOCAL_ATTN_SIZE__": row["local_attn_size"],
        "__SINK_SIZE__": row["sink_size"],
        "__ROPE_CACHE_POLICY__": row["rope_cache_policy"],
    }
    text = template_path.read_text(encoding="utf-8")
    for placeholder, value in replacements.items():
        text = text.replace(placeholder, value)
    if "__" in text:
        raise ValueError(f"unresolved placeholder in rendered config for {row['id']}")
    config = yaml.safe_load(text)
    validate_rendered_config(config, row)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(text, encoding="utf-8")
    return config


def validate_rendered_config(config: dict[str, Any], row: dict[str, str]) -> None:
    training = config["training"]
    data = training["data"]
    validation = config["callbacks"]["validation"]
    dit = config["pipeline"]["dit_config"]

    expected = {
        "max_train_steps": 2000,
        "num_latent_t": 21,
        "num_frames": 81,
        "num_height": 480,
        "num_width": 832,
        "validation_num_frames": 249,
        "validation_latent_frames": 63,
        "causal_train_attention": "triton",
    }
    if int(training["loop"]["max_train_steps"]) not in {1, expected["max_train_steps"]}:
        raise ValueError("max_train_steps must be 1 (smoke) or 2000 (full)")
    for key in ("num_latent_t", "num_frames", "num_height", "num_width"):
        if int(data[key]) != expected[key]:
            raise ValueError(f"training.data.{key} must be {expected[key]}")
    if int(validation["num_frames"]) != expected["validation_num_frames"]:
        raise ValueError("validation must use 249 pixel frames")
    validation_latents = (int(validation["num_frames"]) - 1) // 4 + 1
    if validation_latents != expected["validation_latent_frames"]:
        raise ValueError("249 validation frames must map to 63 latent frames")
    if dit["causal_train_attention"] != expected["causal_train_attention"]:
        raise ValueError("all experiments must use the fused Triton causal kernel")
    axis_values = (
        int(dit["sink_size"]),
        int(dit["local_attn_size"]),
        str(dit["rope_cache_policy"]),
    )
    expected_axes = (
        int(row["sink_size"]),
        int(row["local_attn_size"]),
        row["rope_cache_policy"],
    )
    if axis_values != expected_axes:
        raise ValueError(f"{row['id']}: expected axes {expected_axes}, got {axis_values}")


def normalized_config(config: dict[str, Any]) -> dict[str, Any]:
    normalized = copy.deepcopy(config)
    normalized["pipeline"]["dit_config"].update(
        local_attn_size="<axis>", sink_size="<axis>", rope_cache_policy="<axis>"
    )
    normalized["training"]["checkpoint"]["output_dir"] = "<run>"
    normalized["training"]["tracker"]["run_name"] = "<run>"
    normalized["callbacks"]["validation"]["output_dir"] = "<run>"
    return normalized


def validate_all(template_path: Path, matrix_path: Path) -> None:
    from fastvideo.train.utils.config import load_run_config

    matrix = load_matrix(matrix_path)
    validate_matrix(matrix)
    baseline: dict[str, Any] | None = None
    with tempfile.TemporaryDirectory(prefix="mixkit21-ablation-") as temp_dir:
        temp = Path(temp_dir)
        for experiment_id in sorted(matrix):
            row = matrix[experiment_id]
            output_path = temp / f"{experiment_id}.yaml"
            config = render_config(
                template_path,
                row,
                output_path=output_path,
                run_root=temp / experiment_id,
                project_name="causal_forcing_mixkit21_kernel_tf2k_long249_ablation",
                run_name=f"validate_{experiment_id}",
                max_train_steps=2000,
                checkpoint_steps=1000,
                validation_every_steps=200,
                validation_sampling_steps=40,
                validation_dataset_file=DEFAULT_VALIDATION,
            )
            resolved = load_run_config(str(output_path))
            arch = resolved.training.pipeline_config.dit_config.arch_config
            resolved_axes = (
                int(arch.sink_size),
                int(arch.local_attn_size),
                str(arch.rope_cache_policy),
                str(arch.causal_train_attention),
            )
            expected_axes = (
                int(row["sink_size"]),
                int(row["local_attn_size"]),
                row["rope_cache_policy"],
                "triton",
            )
            if resolved_axes != expected_axes:
                raise ValueError(
                    f"{experiment_id}: resolved axes {resolved_axes} differ from {expected_axes}"
                )
            current = normalized_config(config)
            if baseline is None:
                baseline = current
            elif current != baseline:
                raise ValueError(
                    f"{experiment_id}: a non-ablation hyperparameter differs from A01"
                )
            print(
                f"{experiment_id}\t{row['condition']}\tsink={row['sink_size']}\t"
                f"local={row['local_attn_size']}\trope={row['rope_cache_policy']}\tconfig_ok"
            )
    print("ALL_8_CONFIGS_VALID")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--template", type=Path, default=DEFAULT_TEMPLATE)
    parser.add_argument("--matrix", type=Path, default=DEFAULT_MATRIX)
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("validate")
    render = subparsers.add_parser("render")
    render.add_argument("--condition", required=True)
    render.add_argument("--output", type=Path, required=True)
    render.add_argument("--run-root", type=Path, required=True)
    render.add_argument("--project-name", required=True)
    render.add_argument("--run-name", required=True)
    render.add_argument("--max-train-steps", type=int, default=2000)
    render.add_argument("--checkpoint-steps", type=int, default=1000)
    render.add_argument("--validation-every-steps", type=int, default=200)
    render.add_argument("--validation-sampling-steps", type=int, default=40)
    render.add_argument("--validation-dataset-file", type=Path, default=DEFAULT_VALIDATION)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "validate":
        validate_all(args.template, args.matrix)
        return
    matrix = load_matrix(args.matrix)
    validate_matrix(matrix)
    if args.condition not in matrix:
        raise ValueError(f"unknown condition ID: {args.condition}")
    config = render_config(
        args.template,
        matrix[args.condition],
        output_path=args.output,
        run_root=args.run_root,
        project_name=args.project_name,
        run_name=args.run_name,
        max_train_steps=args.max_train_steps,
        checkpoint_steps=args.checkpoint_steps,
        validation_every_steps=args.validation_every_steps,
        validation_sampling_steps=args.validation_sampling_steps,
        validation_dataset_file=args.validation_dataset_file,
    )
    print(
        f"rendered={args.output} condition={args.condition} "
        f"sink={config['pipeline']['dit_config']['sink_size']} "
        f"local={config['pipeline']['dit_config']['local_attn_size']} "
        f"rope={config['pipeline']['dit_config']['rope_cache_policy']}"
    )


if __name__ == "__main__":
    main()
