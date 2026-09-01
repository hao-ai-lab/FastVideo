# pyright: reportAttributeAccessIssue=false
"""Launch one- or four-GPU FastH3 RVM correctness pilots on Modal.

The launcher follows the operational pattern used by the recent FastVideo
VPTD/PT-PDD and DiffusionNFT assets: clone the exact experiment branch at
runtime, use persistent model/data and run volumes, execute strict preflights
before optimization, and commit logs/checkpoints even when a run fails.

Examples:

    # Download/cache assets, run public FastH3 inference, load every reward,
    # build the real training config, and stop before optimization.
    modal run examples/train/rvm_h3/modal_h3_rvm.py \
        --gpus 1 --mode preflight

    # Four optimizer updates on one B200. Includes baseline validation at step 0.
    modal run examples/train/rvm_h3/modal_h3_rvm.py \
        --gpus 1 --mode smoke

    # Production-capacity LoRA and SP4 on four B200s, but only ten updates.
    modal run examples/train/rvm_h3/modal_h3_rvm.py \
        --gpus 4 --mode pilot --max-steps 10 --eval-prompts 8

    # Resume the same persistent run directory.
    modal run examples/train/rvm_h3/modal_h3_rvm.py \
        --gpus 4 --mode pilot --run-name my-rvm-pilot --resume
"""

from __future__ import annotations

from contextlib import suppress
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import sys
from typing import Any

import modal


FASTVIDEO_REPOSITORY = "https://github.com/Abecid/FastVideo.git"
DEFAULT_BRANCH = "adam/h3-rvm-posttraining"
DEFAULT_IMAGE = "ghcr.io/hao-ai-lab/fastvideo/fastvideo-dev:latest"
DEFAULT_1GPU_CONFIG = "examples/train/configs/rl/minimax_h3/rvm_h3_modal_1gpu.yaml"
DEFAULT_4GPU_CONFIG = "examples/train/configs/rl/minimax_h3/rvm_h3_modal_4gpu.yaml"

IMAGE_REF = os.environ.get("FASTVIDEO_MODAL_IMAGE", DEFAULT_IMAGE)
GPU_1 = os.environ.get("H3_RVM_MODAL_GPU_1", "B200")
GPU_4 = os.environ.get("H3_RVM_MODAL_GPU_4", "B200:4")
SECRET_NAMES = os.environ.get(
    "H3_RVM_MODAL_SECRETS",
    os.environ.get("H3_RVM_MODAL_SECRET", "fastvideo-training"),
)

image = (
    modal.Image.from_registry(IMAGE_REF, add_python="3.12")
    .env(
        {
            "H3_RVM_MODAL_GPU_1": GPU_1,
            "H3_RVM_MODAL_GPU_4": GPU_4,
            "H3_RVM_MODAL_SECRETS": SECRET_NAMES,
        }
    )
    .apt_install(
        "ffmpeg",
        "git",
        "git-lfs",
        "libgl1",
        "libglib2.0-0",
        "ninja-build",
    )
    .run_commands("python -m pip install --upgrade uv")
)

app = modal.App("fastvideo-h3-rvm-pilot")
asset_volume = modal.Volume.from_name(
    "fastvideo-h3-rvm-assets",
    create_if_missing=True,
)
run_volume = modal.Volume.from_name(
    "fastvideo-h3-rvm-runs",
    create_if_missing=True,
)
training_secrets = [
    modal.Secret.from_name(name.strip())
    for name in SECRET_NAMES.split(",")
    if name.strip()
]
if not training_secrets:
    raise ValueError("H3_RVM_MODAL_SECRETS must contain at least one secret name")

_ALLOWED_MODES = {"prepare", "preflight", "smoke", "pilot", "all"}


def _run_logged(
    command: list[str],
    *,
    cwd: Path,
    env: dict[str, str],
    log_path: Path,
) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    print("+", shlex.join(command), flush=True)
    with log_path.open("a", encoding="utf-8") as log:
        log.write(f"\n$ {shlex.join(command)}\n")
        log.flush()
        process = subprocess.Popen(
            command,
            cwd=cwd,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="", flush=True)
            log.write(line)
            log.flush()
        return_code = process.wait()
    if return_code != 0:
        raise subprocess.CalledProcessError(return_code, command)


def _capture(
    command: list[str],
    *,
    cwd: Path,
    env: dict[str, str],
) -> str:
    print("+", shlex.join(command), flush=True)
    return subprocess.check_output(
        command,
        cwd=cwd,
        env=env,
        text=True,
        stderr=subprocess.STDOUT,
    ).strip()


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _safe_run_name(value: str, *, gpu_count: int) -> str:
    text = value.strip()
    if not text or text == "auto":
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
        text = f"h3-rvm-{gpu_count}gpu-{timestamp}"
    sanitized = re.sub(r"[^A-Za-z0-9._-]+", "-", text).strip("-")
    if not sanitized or sanitized in {".", ".."}:
        raise ValueError(f"Invalid run_name: {value!r}")
    return sanitized


def _checkout_fastvideo(
    *,
    workspace: Path,
    branch: str,
    commit: str,
    env: dict[str, str],
    log_dir: Path,
) -> tuple[Path, str]:
    destination = workspace / "FastVideo"
    shutil.rmtree(destination, ignore_errors=True)
    command = [
        "git",
        "clone",
        "--filter=blob:none",
        "--no-tags",
        "--branch",
        branch,
        FASTVIDEO_REPOSITORY,
        str(destination),
    ]
    _run_logged(
        command,
        cwd=workspace,
        env=env,
        log_path=log_dir / "00_clone.log",
    )
    if commit.strip():
        _run_logged(
            ["git", "fetch", "--depth", "1", "origin", commit],
            cwd=destination,
            env=env,
            log_path=log_dir / "00_clone.log",
        )
        _run_logged(
            ["git", "checkout", "--detach", commit],
            cwd=destination,
            env=env,
            log_path=log_dir / "00_clone.log",
        )
    resolved = _capture(
        ["git", "rev-parse", "HEAD"],
        cwd=destination,
        env=env,
    )
    if commit.strip() and resolved != commit:
        raise RuntimeError(f"Requested commit {commit}, checked out {resolved}")
    _apply_runtime_fixes(destination)
    return destination, resolved


def _apply_runtime_fixes(repository: Path) -> None:
    """Apply the reviewed local fixes while Modal still clones the PR commit."""
    path = repository / "fastvideo/pipelines/preprocess/preprocess_minimax_h3_text_only.py"
    original = "    model_index = verify_model_config_and_directory(str(model_path))\n"
    replacement = (
        "    model_index = verify_model_config_and_directory(\n"
        "        str(model_path),\n"
        "        required_component_dirs=(\"tokenizer\", \"processor\", \"text_encoder\"),\n"
        "    )\n"
    )
    source = path.read_text(encoding="utf-8")
    if replacement not in source:
        if original not in source:
            raise RuntimeError(f"Cannot apply the text-only preprocessing fix to {path}")
        path.write_text(source.replace(original, replacement, 1), encoding="utf-8")

    public_runner = repository / "examples/inference/basic/basic_minimax_h3_t2v.py"
    runner_source = public_runner.read_text(encoding="utf-8")
    steps_line = '    parser.add_argument("--steps", type=int, default=50)\n'
    sparsity_lines = (
        steps_line
        + '    parser.add_argument("--vsa-sparsity", type=float, default=0.0)\n'
        + '    parser.add_argument("--dit-cpu-offload", action="store_true")\n'
    )
    if sparsity_lines not in runner_source:
        if steps_line not in runner_source:
            raise RuntimeError(f"Cannot add VSA sparsity to {public_runner}")
        runner_source = runner_source.replace(steps_line, sparsity_lines, 1)
    dit_offload_default = "                    dit=False,\n"
    dit_offload_option = "                    dit=args.dit_cpu_offload,\n"
    if dit_offload_option not in runner_source:
        if dit_offload_default not in runner_source:
            raise RuntimeError(f"Cannot wire DiT CPU offload in {public_runner}")
        runner_source = runner_source.replace(dit_offload_default, dit_offload_option, 1)
    generator_end = "        ))\n    try:\n"
    generator_with_sparsity = "        ))\n    generator.fastvideo_args.VSA_sparsity = args.vsa_sparsity\n    try:\n"
    if generator_with_sparsity not in runner_source:
        if generator_end not in runner_source:
            raise RuntimeError(f"Cannot wire VSA sparsity in {public_runner}")
        runner_source = runner_source.replace(generator_end, generator_with_sparsity, 1)
    public_runner.write_text(runner_source, encoding="utf-8")

    smoke_script = repository / "examples/train/rvm_h3/03_public_inference_smoke.sh"
    smoke_source = smoke_script.read_text(encoding="utf-8")
    old_command = """python examples/inference/basic/fasth3.py \\
    --model-path "${FASTH3_MODEL_DIR}" \\
    --prompt "${PROMPT}" \\
    --output "${OUTPUT_DIR}" \\
    --profile strict \\
    --height 480 \\
    --width 832 \\
    --num-frames 124 \\
    --steps 5 \\
    --seed 1000 \\
    --repeats 1 \\
    --num-gpus 1 \\
    --vsa-sparsity 0.9 \\
    --vsa-tile-size 64 \\
    --vsa-kernel triton \\
    --no-fa4 \\
    --no-h3-fusions \\
    --no-inference-torch-compile \\
    --no-compile-vae \\
    --no-parallel-vae
"""
    new_command = """FASTVIDEO_DMD_DENOISING_STEPS=1000,750,500,250 \\
python examples/inference/basic/basic_minimax_h3_t2v.py \\
    --model-path "${FASTH3_MODEL_DIR}" \\
    --prompt "${PROMPT}" \\
    --output "${OUTPUT_DIR}" \\
    --height 480 \\
    --width 832 \\
    --num-frames 124 \\
    --steps 5 \\
    --seed 1000 \\
    --repeats 1 \\
    --num-gpus 1 \\
    --vsa-sparsity 0.9 \\
    --dit-cpu-offload
"""
    if new_command not in smoke_source:
        if old_command not in smoke_source:
            raise RuntimeError(f"Cannot repair the public H3 smoke command in {smoke_script}")
        smoke_source = smoke_source.replace(old_command, new_command, 1)
    smoke_script.write_text(smoke_source, encoding="utf-8")

    preflight_script = repository / "examples/train/rvm_h3/03_preflight_1gpu.sh"
    preflight_source = preflight_script.read_text(encoding="utf-8")
    stale_tests = """    fastvideo/tests/train/methods/test_minimax_h3_rvm_sampler.py \\
    fastvideo/tests/train/methods/test_rvm_configs.py \\
    fastvideo/tests/train/utils/test_lora_context.py
"""
    existing_tests = """    fastvideo/tests/train/methods/test_minimax_h3_dmd2.py \\
    fastvideo/tests/train/methods/test_rvm_configs.py \\
    fastvideo/tests/inference/lora/test_merge_lora_math.py
"""
    if existing_tests not in preflight_source:
        if stale_tests not in preflight_source:
            raise RuntimeError(f"Cannot repair stale test paths in {preflight_script}")
        preflight_source = preflight_source.replace(stale_tests, existing_tests, 1)
    preflight_script.write_text(preflight_source, encoding="utf-8")

    config_test = repository / "fastvideo/tests/train/methods/test_rvm_configs.py"
    config_test_source = config_test.read_text(encoding="utf-8")
    stale_count = "    assert len(paths) == 5\n"
    exact_configs = """    assert {path.name for path in paths} == {
        "rvm_h3_1gpu_smoke.yaml",
        "rvm_h3_8gpu_audio_anchor.yaml",
        "rvm_h3_8gpu_exact.yaml",
        "rvm_h3_8gpu_full.yaml",
        "rvm_h3_8gpu_full_anchor.yaml",
        "rvm_h3_modal_1gpu.yaml",
        "rvm_h3_modal_4gpu.yaml",
    }
"""
    if exact_configs not in config_test_source:
        if stale_count not in config_test_source:
            raise RuntimeError(f"Cannot repair the RVM config inventory in {config_test}")
        config_test_source = config_test_source.replace(stale_count, exact_configs, 1)
    stale_target = '    assert method["_target_"].endswith("RVMMethod")\n'
    accepted_targets = (
        '    assert method["_target_"].endswith(("RVMMethod", "RVMWithLocalMetricsMethod"))\n'
    )
    if accepted_targets not in config_test_source:
        if stale_target not in config_test_source:
            raise RuntimeError(f"Cannot repair the RVM method targets in {config_test}")
        config_test_source = config_test_source.replace(stale_target, accepted_targets, 1)
    config_test.write_text(config_test_source, encoding="utf-8")

    videoalign_adapter = repository / "fastvideo/train/methods/rl/rewards/videoalign.py"
    videoalign_source = videoalign_adapter.read_text(encoding="utf-8")
    load_patch = """    cls.load_state_dict = wrapped
    cls._fastvideo_qwen2vl_key_remap = True


def _select_frame_indices"""
    init_patch = """    cls.load_state_dict = wrapped
    cls._fastvideo_qwen2vl_key_remap = True


def _patch_reward_model_init(cls: Any) -> None:
    \"\"\"Bridge Transformers 5 config changes to the pinned VideoAlign model.\"\"\"
    if getattr(cls, \"_fastvideo_transformers5_init\", False):
        return
    original = cls.__init__

    def wrapped(self, config, *args, use_cache=None, **kwargs):
        if use_cache is not None:
            config.use_cache = bool(use_cache)
        if not hasattr(config, \"hidden_size\"):
            text_config = getattr(config, \"text_config\", None)
            if text_config is None or not hasattr(text_config, \"hidden_size\"):
                raise AttributeError(\"Qwen2-VL config does not expose a text hidden size\")
            config.hidden_size = text_config.hidden_size
        return original(self, config, *args, **kwargs)

    cls.__init__ = wrapped
    cls._fastvideo_transformers5_init = True


def _patch_reward_model_from_pretrained(cls: Any) -> None:
    \"\"\"Apply the official Qwen2-VL v5 key conversion to the custom subclass.\"\"\"
    if getattr(cls, \"_fastvideo_transformers5_from_pretrained\", False):
        return
    original = cls.from_pretrained.__func__

    def wrapped(subclass, *args, **kwargs):
        kwargs.setdefault(
            \"key_mapping\",
            {
                r\"^visual\": \"model.visual\",
                r\"^model(?!\\.(language_model|visual))\": \"model.language_model\",
            },
        )
        return original(subclass, *args, **kwargs)

    cls.from_pretrained = classmethod(wrapped)
    cls._fastvideo_transformers5_from_pretrained = True


def _patch_reward_model_forward(cls: Any) -> None:
    \"\"\"Run the pinned reward head through Transformers 5's Qwen2-VL model.\"\"\"
    if getattr(cls, \"_fastvideo_transformers5_forward\", False):
        return
    original = cls.forward

    def wrapped(
        self,
        input_ids=None,
        attention_mask=None,
        position_ids=None,
        past_key_values=None,
        inputs_embeds=None,
        labels=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        pixel_values=None,
        pixel_values_videos=None,
        image_grid_thw=None,
        video_grid_thw=None,
        rope_deltas=None,
        mm_token_type_ids=None,
        **kwargs,
    ):
        del labels, rope_deltas
        if not hasattr(self.model, \"language_model\"):
            return original(
                self,
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                pixel_values=pixel_values,
                pixel_values_videos=pixel_values_videos,
                image_grid_thw=image_grid_thw,
                video_grid_thw=video_grid_thw,
            )
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            pixel_values=pixel_values,
            pixel_values_videos=pixel_values_videos,
            image_grid_thw=image_grid_thw,
            video_grid_thw=video_grid_thw,
            mm_token_type_ids=mm_token_type_ids,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            **kwargs,
        )
        head_dtype = next(self.rm_head.parameters()).dtype
        logits = self.rm_head(outputs[0].to(head_dtype))
        batch_size = input_ids.shape[0] if input_ids is not None else inputs_embeds.shape[0]
        if self.config.pad_token_id is None and batch_size != 1:
            raise ValueError(\"Cannot handle batch sizes > 1 if no padding token is defined.\")
        if self.config.pad_token_id is None:
            sequence_lengths: Any = -1
        elif input_ids is not None:
            sequence_lengths = torch.eq(input_ids, self.config.pad_token_id).int().argmax(-1) - 1
            sequence_lengths = sequence_lengths % input_ids.shape[-1]
            sequence_lengths = sequence_lengths.to(logits.device)
        else:
            sequence_lengths = -1
        if self.reward_token == \"last\":
            pooled_logits = logits[torch.arange(batch_size, device=logits.device), sequence_lengths]
        elif self.reward_token == \"mean\":
            valid_lengths = torch.clamp(sequence_lengths, min=0, max=logits.size(1) - 1)
            pooled_logits = torch.stack([logits[i, :valid_lengths[i]].mean(dim=0) for i in range(batch_size)])
        elif self.reward_token == \"special\":
            special_token_mask = torch.zeros_like(input_ids, dtype=torch.bool)
            for special_token_id in self.special_token_ids:
                special_token_mask |= input_ids == special_token_id
            special_tokens_per_sample = len(self.special_token_ids)
            pooled_logits = logits[special_token_mask].view(batch_size, special_tokens_per_sample, -1)
            if self.output_dim == special_tokens_per_sample:
                pooled_logits = pooled_logits.diagonal(dim1=1, dim2=2)
            pooled_logits = pooled_logits.view(batch_size, -1)
        else:
            raise ValueError(f\"Invalid reward_token: {self.reward_token}\")
        return {\"logits\": pooled_logits}

    cls.forward = wrapped
    cls._fastvideo_transformers5_forward = True


def _select_frame_indices"""
    if init_patch not in videoalign_source:
        if load_patch not in videoalign_source:
            raise RuntimeError(f"Cannot add the Transformers 5 init bridge to {videoalign_adapter}")
        videoalign_source = videoalign_source.replace(load_patch, init_patch, 1)
    old_runtime_patch = "    _patch_load_state_dict(trainer_mod.Qwen2VLRewardModelBT)\n"
    new_runtime_patch = (
        "    _patch_reward_model_init(trainer_mod.Qwen2VLRewardModelBT)\n"
        "    _patch_reward_model_from_pretrained(trainer_mod.Qwen2VLRewardModelBT)\n"
        "    _patch_reward_model_forward(trainer_mod.Qwen2VLRewardModelBT)\n"
        "    _patch_load_state_dict(trainer_mod.Qwen2VLRewardModelBT)\n"
    )
    if new_runtime_patch not in videoalign_source:
        if old_runtime_patch not in videoalign_source:
            raise RuntimeError(f"Cannot activate the Transformers 5 init bridge in {videoalign_adapter}")
        videoalign_source = videoalign_source.replace(old_runtime_patch, new_runtime_patch, 1)
    videoalign_adapter.write_text(videoalign_source, encoding="utf-8")

    hpsv3_adapter = repository / "fastvideo/train/methods/rl/rewards/hpsv3.py"
    hpsv3_source = hpsv3_adapter.read_text(encoding="utf-8")
    old_hpsv3_imports = """import os
import tempfile"""
    new_hpsv3_imports = """import os
import sys
import tempfile
import types"""
    if new_hpsv3_imports not in hpsv3_source:
        if old_hpsv3_imports not in hpsv3_source:
            raise RuntimeError(f"Cannot add HPSv3 compatibility imports to {hpsv3_adapter}")
        hpsv3_source = hpsv3_source.replace(old_hpsv3_imports, new_hpsv3_imports, 1)
    hpsv3_marker = """_INFERENCERS: dict[str, Any] = {}


def _get_inferencer"""
    hpsv3_helpers = """_INFERENCERS: dict[str, Any] = {}


def _patch_transformers5_imports() -> None:
    \"\"\"Expose HPSv3's training-only legacy imports without downgrading Transformers.\"\"\"
    from transformers import trainer, trainer_pt_utils

    if not hasattr(trainer, \"nested_concat\"):
        trainer.nested_concat = trainer_pt_utils.nested_concat

    class _TrainingOnlyLegacyUtility:
        def __init__(self, *args, **kwargs) -> None:
            del args, kwargs
            raise RuntimeError(\"This removed Transformers utility is only available in HPSv3 training code\")

    for name in (\"DistributedTensorGatherer\", \"SequentialDistributedSampler\"):
        if not hasattr(trainer, name):
            setattr(trainer, name, _TrainingOnlyLegacyUtility)
    differentiable_module = \"hpsv3.model.differentiable_image_processor\"
    if differentiable_module not in sys.modules:
        module = types.ModuleType(differentiable_module)
        module.Qwen2VLImageProcessor = _TrainingOnlyLegacyUtility
        sys.modules[differentiable_module] = module


def _patch_hpsv3_model() -> None:
    from hpsv3.model import qwen2vl_trainer

    from fastvideo.train.methods.rl.rewards.videoalign import (
        _patch_load_state_dict,
        _patch_reward_model_forward,
        _patch_reward_model_from_pretrained,
        _patch_reward_model_init,
    )

    cls = qwen2vl_trainer.Qwen2VLRewardModelBT
    _patch_reward_model_init(cls)
    _patch_reward_model_from_pretrained(cls)
    _patch_reward_model_forward(cls)
    _patch_load_state_dict(cls)


def _get_inferencer"""
    if hpsv3_helpers not in hpsv3_source:
        if hpsv3_marker not in hpsv3_source:
            raise RuntimeError(f"Cannot add the Transformers 5 HPSv3 bridge to {hpsv3_adapter}")
        hpsv3_source = hpsv3_source.replace(hpsv3_marker, hpsv3_helpers, 1)
    old_hpsv3_import = """        try:
            from hpsv3 import HPSv3RewardInferencer
        except ImportError as exc:"""
    new_hpsv3_import = """        try:
            _patch_transformers5_imports()
            from hpsv3 import HPSv3RewardInferencer
            _patch_hpsv3_model()
        except ImportError as exc:"""
    if new_hpsv3_import not in hpsv3_source:
        if old_hpsv3_import not in hpsv3_source:
            raise RuntimeError(f"Cannot activate the Transformers 5 HPSv3 bridge in {hpsv3_adapter}")
        hpsv3_source = hpsv3_source.replace(old_hpsv3_import, new_hpsv3_import, 1)
    old_hpsv3_call = "                raw = inferencer.reward(all_prompts, image_paths=all_paths)\n"
    new_hpsv3_call = "                raw = inferencer.reward(all_paths, all_prompts)\n"
    if new_hpsv3_call not in hpsv3_source:
        if old_hpsv3_call not in hpsv3_source:
            raise RuntimeError(f"Cannot repair the HPSv3 reward call in {hpsv3_adapter}")
        hpsv3_source = hpsv3_source.replace(old_hpsv3_call, new_hpsv3_call, 1)
    unbounded_hps_init = '''        device: torch.device | str = "cuda",
        max_frames: int | None = 53,
    ) -> None:
        self.device = torch.device(device)
        self.max_frames = None if max_frames is None else int(max_frames)
'''
    bounded_hps_init = '''        device: torch.device | str = "cuda",
        max_frames: int | None = 53,
        batch_size: int = 16,
    ) -> None:
        self.device = torch.device(device)
        self.max_frames = None if max_frames is None else int(max_frames)
        self.batch_size = int(batch_size)
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")
'''
    if bounded_hps_init not in hpsv3_source:
        if unbounded_hps_init not in hpsv3_source:
            raise RuntimeError(f"Cannot bound HPSv3 reward batches in {hpsv3_adapter}")
        hpsv3_source = hpsv3_source.replace(unbounded_hps_init, bounded_hps_init, 1)
    unbounded_hps_call = '''            with torch.autocast(device_type=self.device.type, enabled=self.device.type == "cuda"):
                raw = inferencer.reward(all_paths, all_prompts)
'''
    bounded_hps_call = '''            with torch.autocast(device_type=self.device.type, enabled=self.device.type == "cuda"):
                raw = []
                for start in range(0, len(all_paths), self.batch_size):
                    raw.extend(
                        inferencer.reward(
                            all_paths[start:start + self.batch_size],
                            all_prompts[start:start + self.batch_size],
                        )
                    )
'''
    if bounded_hps_call not in hpsv3_source:
        if unbounded_hps_call not in hpsv3_source:
            raise RuntimeError(f"Cannot chunk HPSv3 reward calls in {hpsv3_adapter}")
        hpsv3_source = hpsv3_source.replace(unbounded_hps_call, bounded_hps_call, 1)
    hpsv3_adapter.write_text(hpsv3_source, encoding="utf-8")

    reward_registry = repository / "fastvideo/train/methods/rl/rewards/__init__.py"
    reward_registry_source = reward_registry.read_text(encoding="utf-8")
    reward_builder_header = (
        "def _build_scorer(name: str, *, device: torch.device | str, "
        "options: dict[str, Any]) -> RewardScorer:\n"
    )
    reward_builder_with_override = (
        f"{reward_builder_header}"
        "    options = dict(options)\n"
        "    scorer_device = options.pop(\"device\", device)\n"
    )
    if reward_builder_with_override not in reward_registry_source:
        if reward_builder_header not in reward_registry_source:
            raise RuntimeError(f"Cannot add per-reward device override in {reward_registry}")
        reward_registry_source = reward_registry_source.replace(
            reward_builder_header,
            reward_builder_with_override,
            1,
        )
    for scorer_class in (
        "VideoAlignTextAlignmentScorer",
        "VideoAlignMotionQualityScorer",
        "VideoAlignVisualQualityScorer",
        "HPSv3GeneralScorer",
        "HPSv3PercentileScorer",
        "DynamicTrackingScorer",
    ):
        reward_registry_source = reward_registry_source.replace(
            f"return {scorer_class}(device=device, **options)",
            f"return {scorer_class}(device=scorer_device, **options)",
            1,
        )
    old_reward_parser = '''def _parse_reward_specs(raw: Mapping[str, Any]) -> tuple[dict[str, float], dict[str, dict[str, Any]]]:
    if "rewards" in raw:
        raw = raw["rewards"]
    if not isinstance(raw, Mapping) or not raw:
        raise ValueError("reward config must be a nonempty mapping")
    weights: dict[str, float] = {}
    options: dict[str, dict[str, Any]] = {}
    for raw_name, value in raw.items():
        name = str(raw_name).strip().lower()
        if isinstance(value, Mapping):
            spec = dict(value)
            if "weight" not in spec:
                raise ValueError(f"reward {name!r} must define a numeric weight")
            weights[name] = float(spec.pop("weight"))
            options[name] = spec
        else:
            weights[name] = float(value)
            options[name] = {}
    return weights, options
'''
    merged_reward_parser = '''def _parse_reward_specs(raw: Mapping[str, Any]) -> tuple[dict[str, float], dict[str, dict[str, Any]]]:
    configured_options: Mapping[str, Any] = {}
    if "rewards" in raw:
        candidate_options = raw.get("options", {})
        if not isinstance(candidate_options, Mapping):
            raise ValueError("reward config options must be a mapping")
        configured_options = candidate_options
        raw = raw["rewards"]
    if not isinstance(raw, Mapping) or not raw:
        raise ValueError("reward config must be a nonempty mapping")
    weights: dict[str, float] = {}
    options: dict[str, dict[str, Any]] = {}
    for raw_name, value in raw.items():
        name = str(raw_name).strip().lower()
        base_options = configured_options.get(name, {})
        if not isinstance(base_options, Mapping):
            raise ValueError(f"reward options for {name!r} must be a mapping")
        merged_options = dict(base_options)
        if isinstance(value, Mapping):
            spec = dict(value)
            if "weight" not in spec:
                raise ValueError(f"reward {name!r} must define a numeric weight")
            weights[name] = float(spec.pop("weight"))
            merged_options.update(spec)
        else:
            weights[name] = float(value)
        options[name] = merged_options
    return weights, options
'''
    if merged_reward_parser not in reward_registry_source:
        if old_reward_parser not in reward_registry_source:
            raise RuntimeError(f"Cannot merge top-level reward options in {reward_registry}")
        reward_registry_source = reward_registry_source.replace(old_reward_parser, merged_reward_parser, 1)
    reward_registry.write_text(reward_registry_source, encoding="utf-8")

    dynamic_tracking = repository / "fastvideo/train/methods/rl/rewards/dynamic_tracking.py"
    dynamic_source = dynamic_tracking.read_text(encoding="utf-8")
    old_dynamic_init = '''        device: torch.device | str = "cuda",
        num_pairs: int = 8,
        top_fraction: float = 0.05,
        variant: str = "large",
        pair_batch_size: int = 4,
    ) -> None:
        self.device = torch.device(device)
        self.num_pairs = int(num_pairs)
        self.top_fraction = float(top_fraction)
        self.variant = str(variant).strip().lower()
        self.pair_batch_size = int(pair_batch_size)
        if not 0.0 < self.top_fraction <= 1.0:
            raise ValueError("top_fraction must be in (0, 1]")
        if self.pair_batch_size <= 0:
            raise ValueError("pair_batch_size must be positive")
'''
    compatible_dynamic_init = '''        device: torch.device | str = "cuda",
        num_pairs: int = 8,
        frame_pairs: int | None = None,
        top_fraction: float = 0.05,
        variant: str = "large",
        pair_batch_size: int = 4,
        resize_short_edge: int | None = None,
        pretrained: bool = True,
    ) -> None:
        self.device = torch.device(device)
        self.num_pairs = int(frame_pairs if frame_pairs is not None else num_pairs)
        self.top_fraction = float(top_fraction)
        self.variant = str(variant).strip().lower()
        self.pair_batch_size = int(pair_batch_size)
        self.resize_short_edge = None if resize_short_edge is None else int(resize_short_edge)
        if not pretrained:
            raise ValueError("Dynamic tracking requires pretrained RAFT weights")
        if not 0.0 < self.top_fraction <= 1.0:
            raise ValueError("top_fraction must be in (0, 1]")
        if self.pair_batch_size <= 0:
            raise ValueError("pair_batch_size must be positive")
        if self.resize_short_edge is not None and self.resize_short_edge <= 0:
            raise ValueError("resize_short_edge must be positive")
'''
    if compatible_dynamic_init not in dynamic_source:
        if old_dynamic_init not in dynamic_source:
            raise RuntimeError(f"Cannot bridge H3 dynamic reward options in {dynamic_tracking}")
        dynamic_source = dynamic_source.replace(old_dynamic_init, compatible_dynamic_init, 1)
    old_dynamic_geometry = '''        batch_size, _, num_frames, height, width = videos.shape
        pairs = _pair_indices(num_frames, self.num_pairs)
'''
    resized_dynamic_geometry = '''        batch_size, _, num_frames, height, width = videos.shape
        if self.resize_short_edge is not None and min(height, width) != self.resize_short_edge:
            scale = self.resize_short_edge / min(height, width)
            resized_height = max(8, 8 * round(height * scale / 8))
            resized_width = max(8, 8 * round(width * scale / 8))
            videos = torch.nn.functional.interpolate(
                videos.permute(0, 2, 1, 3, 4).flatten(0, 1),
                size=(resized_height, resized_width),
                mode="bilinear",
                align_corners=False,
            ).unflatten(0, (batch_size, num_frames)).permute(0, 2, 1, 3, 4)
            height, width = resized_height, resized_width
        pairs = _pair_indices(num_frames, self.num_pairs)
'''
    if resized_dynamic_geometry not in dynamic_source:
        if old_dynamic_geometry not in dynamic_source:
            raise RuntimeError(f"Cannot apply configured dynamic resize in {dynamic_tracking}")
        dynamic_source = dynamic_source.replace(old_dynamic_geometry, resized_dynamic_geometry, 1)
    dynamic_tracking.write_text(dynamic_source, encoding="utf-8")

    rvm_method = repository / "fastvideo/train/methods/rl/rvm.py"
    rvm_source = rvm_method.read_text(encoding="utf-8")
    cached_reward_boundary = """                    )

        self._collection_count += 1
"""
    clean_reward_boundary = """                    )

        if torch.cuda.is_available():
            # Release VAE/reward cache before NCCL metric reductions allocate
            # their communication buffers, not only before the DiT forward.
            torch.cuda.empty_cache()
        self._collection_count += 1
"""
    if clean_reward_boundary not in rvm_source:
        if cached_reward_boundary not in rvm_source:
            raise RuntimeError(f"Cannot clear scorer cache at the training boundary in {rvm_method}")
        rvm_source = rvm_source.replace(cached_reward_boundary, clean_reward_boundary, 1)
    rvm_method.write_text(rvm_source, encoding="utf-8")

    moduleloader = repository / "fastvideo/train/utils/moduleloader.py"
    moduleloader_source = moduleloader.read_text(encoding="utf-8")
    old_module_validation = "    config = verify_model_config_and_directory(local_model_path)\n"
    new_module_validation = """    config = verify_model_config_and_directory(
        local_model_path,
        required_component_dirs=(module_type,),
    )
"""
    if new_module_validation not in moduleloader_source:
        if old_module_validation not in moduleloader_source:
            raise RuntimeError(f"Cannot scope component validation in {moduleloader}")
        moduleloader_source = moduleloader_source.replace(old_module_validation, new_module_validation, 1)
    moduleloader.write_text(moduleloader_source, encoding="utf-8")

    lora_linear = repository / "fastvideo/layers/lora/linear.py"
    lora_source = lora_linear.read_text(encoding="utf-8")
    compiled_forward = "    @torch.compile()\n    def forward(self, x: torch.Tensor) -> torch.Tensor:\n"
    eager_forward = "    def forward(self, x: torch.Tensor) -> torch.Tensor:\n"
    if compiled_forward in lora_source:
        lora_source = lora_source.replace(compiled_forward, eager_forward, 1)
    elif eager_forward not in lora_source:
        raise RuntimeError(f"Cannot disable the incompatible compiled LoRA forward in {lora_linear}")
    lora_source = lora_source.replace(
        "                            dtype=self.base_layer.weight.dtype))",
        "                            dtype=torch.float32))",
        2,
    )
    lora_linear.write_text(lora_source, encoding="utf-8")

    h3_dmd_model = repository / "fastvideo/train/models/minimax_h3/minimax_h3_dmd.py"
    h3_dmd_source = h3_dmd_model.read_text(encoding="utf-8")
    full_vae_validation = "        model_index = verify_model_config_and_directory(self._init_from)\n"
    scoped_vae_validation = (
        "        model_index = verify_model_config_and_directory(\n"
        "            self._init_from,\n"
        "            required_component_dirs=(\"vae\",),\n"
        "        )\n"
    )
    if scoped_vae_validation not in h3_dmd_source:
        if full_vae_validation not in h3_dmd_source:
            raise RuntimeError(f"Cannot scope VAE validation in {h3_dmd_model}")
        h3_dmd_source = h3_dmd_source.replace(full_vae_validation, scoped_vae_validation, 1)
    single_batch_pack = """        return torch.cat(
            (video_latents.reshape(1, -1), audio_latents.reshape(1, -1)),
            dim=1,
        )
"""
    batched_pack = """        if video_latents.shape[0] != audio_latents.shape[0]:
            raise ValueError("Video and audio latent batches must match, got "
                             f"{video_latents.shape[0]} and {audio_latents.shape[0]}")
        batch_size = video_latents.shape[0]
        return torch.cat(
            (video_latents.reshape(batch_size, -1), audio_latents.reshape(batch_size, -1)),
            dim=1,
        )
"""
    if batched_pack not in h3_dmd_source:
        if single_batch_pack not in h3_dmd_source:
            raise RuntimeError(f"Cannot generalize packed H3 batches in {h3_dmd_model}")
        h3_dmd_source = h3_dmd_source.replace(single_batch_pack, batched_pack, 1)
    single_batch_unpack = """        if packed.shape != (1, split + math.prod(audio_shape)):
            raise ValueError("Packed latents must have shape "
                             f"[1, {split + math.prod(audio_shape)}], got {tuple(packed.shape)}")
        return (
            packed[:, :split].reshape(video_shape),
            packed[:, split:].reshape(audio_shape),
        )
"""
    batched_unpack = """        packed_width = split + math.prod(audio_shape)
        if packed.ndim != 2 or packed.shape[1] != packed_width:
            raise ValueError("Packed latents must have shape "
                             f"[B, {packed_width}], got {tuple(packed.shape)}")
        batch_size = packed.shape[0]
        return (
            packed[:, :split].reshape(batch_size, *video_shape[1:]),
            packed[:, split:].reshape(batch_size, *audio_shape[1:]),
        )
"""
    if batched_unpack not in h3_dmd_source:
        if single_batch_unpack not in h3_dmd_source:
            raise RuntimeError(f"Cannot generalize unpacked H3 batches in {h3_dmd_model}")
        h3_dmd_source = h3_dmd_source.replace(single_batch_unpack, batched_unpack, 1)
    vae_cpu_line = "        vae.to(\"cpu\")\n"
    h100_safe_tiling = """        if int(self.training_config.distributed.num_gpus) == 1:
            from fastvideo.hooks.layerwise_offload import enable_layerwise_offload
            enable_layerwise_offload(vae.decoder)
            vae.enable_tiling(
                tile_sample_min_height=128,
                tile_sample_min_width=128,
            )
        vae.to("cpu")
"""
    if h100_safe_tiling not in h3_dmd_source:
        if vae_cpu_line not in h3_dmd_source:
            raise RuntimeError(f"Cannot configure one-GPU VAE tiling in {h3_dmd_model}")
        h3_dmd_source = h3_dmd_source.replace(vae_cpu_line, h100_safe_tiling, 1)
    cached_vae_move = """        vae = self._load_vis_vae()
        vae.to(self.device)
"""
    cache_safe_vae_move = """        vae = self._load_vis_vae()
        if self.device.type == "cuda":
            torch.cuda.empty_cache()
        vae.to(self.device)
"""
    if cache_safe_vae_move not in h3_dmd_source:
        if cached_vae_move not in h3_dmd_source:
            raise RuntimeError(f"Cannot clear fragmented cache before VAE materialization in {h3_dmd_model}")
        h3_dmd_source = h3_dmd_source.replace(cached_vae_move, cache_safe_vae_move, 1)
    h3_dmd_model.write_text(h3_dmd_source, encoding="utf-8")

    h3_rvm_model = repository / "fastvideo/train/models/minimax_h3/minimax_h3_rvm.py"
    h3_rvm_source = h3_rvm_model.read_text(encoding="utf-8")
    if "import os\n" not in h3_rvm_source:
        h3_rvm_source = h3_rvm_source.replace(
            "from __future__ import annotations\n\n",
            "from __future__ import annotations\n\nimport os\n",
            1,
        )
    unbounded_decode = """        decoded = torch.from_numpy(self.decode_vis_latents(packed))
        return decoded.permute(0, 2, 1, 3, 4).contiguous()
"""
    bounded_decode = """        decode_batch_size = int(
            os.environ.get("FASTVIDEO_RVM_VAE_DECODE_BATCH_SIZE", packed.shape[0]))
        if decode_batch_size <= 0:
            raise ValueError("FASTVIDEO_RVM_VAE_DECODE_BATCH_SIZE must be positive")
        decoded = torch.cat([
            torch.from_numpy(self.decode_vis_latents(chunk))
            for chunk in packed.split(decode_batch_size, dim=0)
        ])
        return decoded.permute(0, 2, 1, 3, 4).contiguous()
"""
    if bounded_decode not in h3_rvm_source:
        if unbounded_decode not in h3_rvm_source:
            raise RuntimeError(f"Cannot bound reward VAE decode batches in {h3_rvm_model}")
        h3_rvm_source = h3_rvm_source.replace(unbounded_decode, bounded_decode, 1)
    h3_rvm_model.write_text(h3_rvm_source, encoding="utf-8")

    one_gpu_config = repository / DEFAULT_1GPU_CONFIG
    one_gpu_source = one_gpu_config.read_text(encoding="utf-8")
    full_reward_block = '''  reward_fn:
    rewards:
      videoalign_ta: 1.5
      videoalign_mq: 1.0
      hpsv3_general: 0.1
      hpsv3_percentile: 0.1
      dynamic_tracking: 0.7
    options:
      dynamic_tracking:
        frame_pairs: 4
        top_fraction: 0.05
        resize_short_edge: 256
        pretrained: true
'''
    smoke_reward_block = '''  reward_fn:
    rewards:
      mean_luminance: 1.0
'''
    if smoke_reward_block not in one_gpu_source:
        if full_reward_block not in one_gpu_source:
            raise RuntimeError(f"Cannot select the lightweight H100 smoke reward in {one_gpu_config}")
        one_gpu_source = one_gpu_source.replace(full_reward_block, smoke_reward_block, 1)
    if "    run_at_start: false\n" not in one_gpu_source:
        one_gpu_source = one_gpu_source.replace("    run_at_start: true\n", "    run_at_start: false\n", 1)
    one_gpu_config.write_text(one_gpu_source, encoding="utf-8")


def _install_runtime(
    *,
    repository: Path,
    env: dict[str, str],
    log_dir: Path,
) -> None:
    _run_logged(
        ["uv", "pip", "install", "--system", "-e", ".[eval,test]"],
        cwd=repository,
        env=env,
        log_path=log_dir / "01_install.log",
    )
    _run_logged(
        [
            "uv",
            "pip",
            "install",
            "--system",
            "decord",
            "accelerate>=1.1",
            "fire",
            "liger-kernel",
            "qwen-vl-utils",
            "safetensors",
            "trl==0.8.6",
        ],
        cwd=repository,
        env=env,
        log_path=log_dir / "01_install.log",
    )
    _run_logged(
        [
            "uv",
            "pip",
            "install",
            "--system",
            "--no-deps",
            "hpsv3==1.0.0",
        ],
        cwd=repository,
        env=env,
        log_path=log_dir / "01_install.log",
    )


def _link_artifacts(repository: Path, artifact_root: Path) -> None:
    link = repository / "artifacts" / "rvm_h3"
    link.parent.mkdir(parents=True, exist_ok=True)
    if link.is_symlink() or link.is_file():
        link.unlink()
    elif link.exists():
        shutil.rmtree(link)
    link.symlink_to(artifact_root, target_is_directory=True)


def _parquet_rows(root: Path) -> int:
    files = sorted(root.rglob("*.parquet")) if root.is_dir() else []
    if not files:
        return 0
    import pyarrow.parquet as pq

    return sum(int(pq.ParquetFile(path).metadata.num_rows) for path in files)


def _prepare_assets(
    *,
    repository: Path,
    env: dict[str, str],
    log_dir: Path,
    gpu_count: int,
    max_train_prompts: int,
    eval_prompts: int,
    smoke_prompts: int,
    force_prepare: bool,
) -> dict[str, int]:
    artifact_root = Path(env["RVM_ARTIFACT_ROOT"])
    model = Path(env["FASTH3_MODEL_DIR"])
    reward_checkpoint = Path(env["VIDEOALIGN_CHECKPOINT_PATH"])
    reward_runtime = Path(env["VIDEOALIGN_RUNTIME_PATH"])

    models_ready = (
        (model / "transformer").is_dir()
        and (model / "vae").is_dir()
        and (model / "text_encoder").is_dir()
        and reward_checkpoint.is_dir()
        and (reward_runtime / "inference.py").is_file()
    )
    if force_prepare or not models_ready:
        _run_logged(
            ["bash", "examples/train/rvm_h3/01_download_models.sh"],
            cwd=repository,
            env=env,
            log_path=log_dir / "02_download_models.log",
        )
        asset_volume.commit()

    expected = {
        "train": int(max_train_prompts),
        "eval": int(eval_prompts),
        "smoke": int(smoke_prompts),
    }
    roots = {
        "train": Path(env["RVM_TRAIN_DATA"]),
        "eval": Path(env["RVM_EVAL_DATA"]),
        "smoke": Path(env["RVM_SMOKE_DATA"]),
    }
    current = {name: _parquet_rows(path) for name, path in roots.items()}
    needs_data = force_prepare or any(current[name] < expected[name] for name in expected)
    if needs_data:
        data_env = dict(env)
        data_env.update(
            {
                "RVM_EVAL_PROMPTS": str(eval_prompts),
                "RVM_FORCE_PREPROCESS": "1",
                "RVM_MAX_TRAIN_PROMPTS": str(max_train_prompts),
                "RVM_PREPROCESS_GPUS": str(gpu_count),
                "RVM_SMOKE_PROMPTS": str(smoke_prompts),
            }
        )
        _run_logged(
            ["bash", "examples/train/rvm_h3/02_prepare_dataset.sh"],
            cwd=repository,
            env=data_env,
            log_path=log_dir / "03_prepare_dataset.log",
        )
        asset_volume.commit()
        current = {name: _parquet_rows(path) for name, path in roots.items()}

    missing = {
        name: {"required": expected[name], "found": current[name]}
        for name in expected
        if current[name] < expected[name]
    }
    if missing:
        raise RuntimeError(f"Prepared prompt datasets are incomplete: {missing}")
    _link_artifacts(repository, artifact_root)
    return current


def _run_preflight(
    *,
    repository: Path,
    env: dict[str, str],
    log_dir: Path,
    run_dir: Path,
    gpu_count: int,
) -> None:
    preflight_env = dict(env)
    preflight_env["RVM_PREFLIGHT_REWARD_OUTPUT"] = str(
        run_dir / "preflight_reward_scores.json"
    )
    inference_root = Path(env["RVM_ARTIFACT_ROOT"]) / "inference_smoke"
    if not any(inference_root.glob("*.mp4")):
        _run_logged(
            ["bash", "examples/train/rvm_h3/03_public_inference_smoke.sh"],
            cwd=repository,
            env=preflight_env,
            log_path=log_dir / "04_public_inference.log",
        )
        asset_volume.commit()
    else:
        (log_dir / "04_public_inference.log").write_text(
            f"Reused deterministic inference smoke artifacts from {inference_root}\n",
            encoding="utf-8",
        )
    if inference_root.is_dir():
        shutil.copytree(
            inference_root,
            run_dir / "public_inference_smoke",
            dirs_exist_ok=True,
        )
    _run_logged(
        ["bash", "examples/train/rvm_h3/03_preflight_1gpu.sh"],
        cwd=repository,
        env=preflight_env,
        log_path=log_dir / "05_preflight.log",
    )

    if gpu_count == 4:
        dry_run_env = dict(env)
        dry_run_env["NUM_GPUS"] = "4"
        _run_logged(
            [
                "bash",
                "examples/train/run.sh",
                DEFAULT_4GPU_CONFIG,
                "--dry-run",
                "--training.distributed.num_gpus",
                "4",
                "--training.distributed.sp_size",
                "4",
                "--training.distributed.tp_size",
                "1",
                "--training.distributed.hsdp_replicate_dim",
                "1",
                "--training.distributed.hsdp_shard_dim",
                "4",
            ],
            cwd=repository,
            env=dry_run_env,
            log_path=log_dir / "05_preflight_4gpu_config.log",
        )


def _checkpoint_exists(output_dir: Path) -> bool:
    return any(
        (path / "dcp" / ".metadata").is_file()
        for path in output_dir.glob("checkpoint-*")
        if path.is_dir()
    )


def _training_command(
    *,
    gpu_count: int,
    config: str,
    output_dir: Path,
    run_name: str,
    max_steps: int,
    eval_prompts: int,
    learning_rate: float,
    resume: bool,
    env: dict[str, str],
) -> list[str]:
    data_path = env["RVM_SMOKE_DATA"] if gpu_count == 1 else env["RVM_TRAIN_DATA"]
    checkpoint_interval = max(1, max_steps // 2)
    command = [
        "bash",
        "examples/train/run.sh",
        config,
        "--models.student.init_from",
        env["FASTH3_MODEL_DIR"],
        "--training.model_path",
        env["FASTH3_MODEL_DIR"],
        "--training.data.data_path",
        data_path,
        "--method.validation.data_path",
        env["RVM_EVAL_DATA"],
        "--method.validation.num_prompts",
        str(min(100, eval_prompts)),
        "--method.validation.log_sample_limit",
        str(min(8, eval_prompts)),
        "--method.validation.every_steps",
        str(max_steps),
        "--training.loop.max_train_steps",
        str(max_steps),
        "--training.checkpoint.output_dir",
        str(output_dir),
        "--training.checkpoint.training_state_checkpointing_steps",
        str(checkpoint_interval),
        "--training.tracker.run_name",
        run_name,
        "--training.distributed.num_gpus",
        str(gpu_count),
        "--training.distributed.sp_size",
        str(gpu_count),
        "--training.distributed.tp_size",
        "1",
        "--training.distributed.hsdp_replicate_dim",
        "1",
        "--training.distributed.hsdp_shard_dim",
        str(gpu_count),
    ]
    if gpu_count == 1:
        command.extend([
            # A strict 80GB H100 cannot hold the 35B model's full 124-frame
            # training graph. Keep this as a real, shorter end-to-end update;
            # the production four-GPU recipe stays 124 frames and rank 128.
            "--models.student.lora.rank",
            "1",
            "--models.student.lora.alpha",
            "1",
            "--training.data.num_latent_t",
            "12",
            "--training.data.num_frames",
            "39",
            "--training.data.num_height",
            "320",
            "--training.data.num_width",
            "576",
            "--method.reward_fn.options.videoalign_ta.device",
            "cpu",
            "--method.reward_fn.options.videoalign_mq.device",
            "cpu",
            "--method.reward_fn.options.hpsv3_general.device",
            "cpu",
            "--method.reward_fn.options.hpsv3_general.max_frames",
            "4",
            "--method.reward_fn.options.hpsv3_percentile.device",
            "cpu",
            "--method.reward_fn.options.hpsv3_percentile.max_frames",
            "4",
            "--method.reward_fn.options.dynamic_tracking.device",
            "cpu",
        ])
    if learning_rate > 0:
        command.extend(
            [
                "--training.optimizer.learning_rate",
                str(learning_rate),
            ]
        )
    if resume and _checkpoint_exists(output_dir):
        command.extend(
            [
                "--training.checkpoint.resume_from_checkpoint",
                "latest",
            ]
        )
    return command


def _environment_report(
    *,
    repository: Path,
    env: dict[str, str],
) -> dict[str, Any]:
    report: dict[str, Any] = {}
    for name, command in {
        "git_head": ["git", "rev-parse", "HEAD"],
        "gpu": ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
        "python": ["python", "--version"],
    }.items():
        try:
            report[name] = _capture(command, cwd=repository, env=env)
        except Exception as exc:
            report[name] = f"ERROR: {exc}"
    try:
        versions = _capture(
            [
                "python",
                "-c",
                (
                    "import json, torch, transformers, diffusers; "
                    "print(json.dumps({'torch': torch.__version__, "
                    "'cuda': torch.version.cuda, "
                    "'transformers': transformers.__version__, "
                    "'diffusers': diffusers.__version__}, sort_keys=True))"
                ),
            ],
            cwd=repository,
            env=env,
        )
        report["packages"] = json.loads(versions.splitlines()[-1])
    except Exception as exc:
        report["packages"] = {"error": str(exc)}
    return report


def _run_job(
    *,
    gpu_count: int,
    mode: str,
    branch: str,
    commit: str,
    config: str,
    run_name: str,
    max_steps: int,
    eval_prompts: int,
    max_train_prompts: int,
    smoke_prompts: int,
    learning_rate: float,
    force_prepare: bool,
    resume: bool,
    skip_preflight: bool,
) -> dict[str, Any]:
    if gpu_count not in {1, 4}:
        raise ValueError("gpu_count must be one or four")
    mode = mode.strip().lower()
    if mode not in _ALLOWED_MODES:
        raise ValueError(f"mode must be one of {sorted(_ALLOWED_MODES)}")
    if max_steps < 0:
        raise ValueError("max_steps must be zero (use config default) or positive")
    if not 1 <= eval_prompts <= 100:
        raise ValueError("eval_prompts must be in [1, 100]")
    if max_train_prompts <= 0 or smoke_prompts <= 0:
        raise ValueError("prompt counts must be positive")
    if learning_rate < 0:
        raise ValueError("learning_rate must be zero (use config) or positive")

    effective_steps = max_steps or (4 if gpu_count == 1 else 10)
    selected_config = config.strip() or (
        DEFAULT_1GPU_CONFIG if gpu_count == 1 else DEFAULT_4GPU_CONFIG
    )
    safe_name = _safe_run_name(run_name, gpu_count=gpu_count)
    run_dir = Path("/runs/h3-rvm") / safe_name
    log_dir = run_dir / "logs"
    run_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    with suppress(Exception):
        asset_volume.reload()
    with suppress(Exception):
        run_volume.reload()

    workspace = Path("/workspace")
    workspace.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env.pop("HF_HUB_ENABLE_HF_TRANSFER", None)
    env.update(
        {
            "FASTH3_MODEL_DIR": "/cache/rvm_h3/models/fasth3",
            "FASTVIDEO_ATTENTION_BACKEND": "VIDEO_SPARSE_ATTN_H3",
            "FASTVIDEO_MINIMAX_H3_FUSIONS": "0",
            "FASTVIDEO_RVM_VAE_DECODE_BATCH_SIZE": "1",
            "FASTVIDEO_VSA_CUTEDSL": "0",
            "FASTVIDEO_VSA_SM100A": "0",
            "HF_HOME": "/cache/huggingface",
            "LOG_DIR": str(log_dir),
            "NCCL_ASYNC_ERROR_HANDLING": "1",
            "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
            "RVM_ARTIFACT_ROOT": "/cache/rvm_h3",
            "RVM_EVAL_DATA": "/cache/rvm_h3/data/eval",
            "RVM_PROMPT_DIR": "/cache/rvm_h3/prompts",
            "RVM_REWARD_ROOT": "/cache/rvm_h3/rewards",
            "RVM_SKIP_CONDA": "1",
            "RVM_SMOKE_DATA": "/cache/rvm_h3/data/train_smoke",
            "RVM_TRAIN_DATA": "/cache/rvm_h3/data/train",
            "TOKENIZERS_PARALLELISM": "false",
            "TORCH_NCCL_ASYNC_ERROR_HANDLING": "1",
            "TRANSFORMERS_CACHE": "/cache/huggingface/hub",
            "VIDEOALIGN_CHECKPOINT_PATH": "/cache/rvm_h3/rewards/VideoReward",
            "VIDEOALIGN_RUNTIME_PATH": "/cache/rvm_h3/rewards/VideoAlign",
            "WANDB_DIR": str(run_dir / "wandb"),
            "WANDB_MODE": (
                "online" if (env.get("WANDB_API_KEY") or "").strip() else "offline"
            ),
            "WANDB_RESUME": "allow",
            "WANDB_RUN_ID": hashlib.sha1(safe_name.encode("utf-8")).hexdigest()[:16],
        }
    )

    status: dict[str, Any] = {
        "branch": branch,
        "commit_requested": commit or None,
        "config": selected_config,
        "eval_prompts": int(eval_prompts),
        "gpu_count": int(gpu_count),
        "learning_rate_override": float(learning_rate),
        "max_steps": int(effective_steps),
        "mode": mode,
        "run_dir": str(run_dir),
        "run_name": safe_name,
        "status": "starting",
    }
    _write_json(run_dir / "modal_manifest.json", status)

    repository: Path | None = None
    try:
        repository, resolved = _checkout_fastvideo(
            workspace=workspace,
            branch=branch,
            commit=commit,
            env=env,
            log_dir=log_dir,
        )
        env["PYTHONPATH"] = str(repository)
        status["commit_resolved"] = resolved
        status["status"] = "installing"
        _write_json(run_dir / "modal_manifest.json", status)

        _install_runtime(repository=repository, env=env, log_dir=log_dir)
        _link_artifacts(repository, Path(env["RVM_ARTIFACT_ROOT"]))
        status["environment"] = _environment_report(repository=repository, env=env)

        status["status"] = "preparing_assets"
        _write_json(run_dir / "modal_manifest.json", status)
        status["dataset_rows"] = _prepare_assets(
            repository=repository,
            env=env,
            log_dir=log_dir,
            gpu_count=gpu_count,
            max_train_prompts=max_train_prompts,
            eval_prompts=eval_prompts,
            smoke_prompts=smoke_prompts,
            force_prepare=force_prepare,
        )

        should_preflight = mode in {"preflight", "smoke", "pilot", "all"} and not skip_preflight
        if should_preflight:
            status["status"] = "preflight"
            _write_json(run_dir / "modal_manifest.json", status)
            _run_preflight(
                repository=repository,
                env=env,
                log_dir=log_dir,
                run_dir=run_dir,
                gpu_count=gpu_count,
            )

        should_train = mode in {"smoke", "pilot", "all"}
        if should_train:
            status["status"] = "training"
            _write_json(run_dir / "modal_manifest.json", status)
            train_env = dict(env)
            train_env["NUM_GPUS"] = str(gpu_count)
            train_env["RVM_SP_SIZE"] = str(gpu_count)
            command = _training_command(
                gpu_count=gpu_count,
                config=selected_config,
                output_dir=run_dir,
                run_name=safe_name,
                max_steps=effective_steps,
                eval_prompts=eval_prompts,
                learning_rate=learning_rate,
                resume=resume,
                env=env,
            )
            _run_logged(
                command,
                cwd=repository,
                env=train_env,
                log_path=log_dir / "06_training.log",
            )

        # The repository is cloned after Modal imports this launcher, so make it
        # importable in this process as well as in the training subprocesses.
        repository_path = str(repository)
        if repository_path not in sys.path:
            sys.path.insert(0, repository_path)
        from fastvideo.train.methods.rl.rvm_local_metrics import (
            collect_initial_reward_results,
        )

        reward_results = collect_initial_reward_results(run_dir)
        _write_json(run_dir / "initial_reward_results.json", reward_results)
        status["initial_reward_results"] = reward_results
        status["status"] = "succeeded"
        _write_json(run_dir / "modal_manifest.json", status)
        return status
    except Exception as exc:
        status["status"] = "failed"
        status["error_type"] = type(exc).__name__
        status["error"] = str(exc)
        _write_json(run_dir / "modal_manifest.json", status)
        raise
    finally:
        with suppress(Exception):
            run_volume.commit()
        with suppress(Exception):
            asset_volume.commit()


@app.function(
    image=image,
    gpu=GPU_1,
    cpu=32,
    memory=131_072,
    ephemeral_disk=524_288,
    timeout=43_000,
    startup_timeout=4_800,
    secrets=training_secrets,
    volumes={"/cache": asset_volume, "/runs": run_volume},
)
def run_1gpu(
    *,
    mode: str = "smoke",
    branch: str = DEFAULT_BRANCH,
    commit: str = "",
    config: str = "",
    run_name: str = "auto",
    max_steps: int = 0,
    eval_prompts: int = 8,
    max_train_prompts: int = 64,
    smoke_prompts: int = 16,
    learning_rate: float = 0.0,
    force_prepare: bool = False,
    resume: bool = False,
    skip_preflight: bool = False,
) -> dict[str, Any]:
    return _run_job(
        gpu_count=1,
        mode=mode,
        branch=branch,
        commit=commit,
        config=config,
        run_name=run_name,
        max_steps=max_steps,
        eval_prompts=eval_prompts,
        max_train_prompts=max_train_prompts,
        smoke_prompts=smoke_prompts,
        learning_rate=learning_rate,
        force_prepare=force_prepare,
        resume=resume,
        skip_preflight=skip_preflight,
    )


@app.function(
    image=image,
    gpu=GPU_4,
    cpu=64,
    memory=262_144,
    ephemeral_disk=524_288,
    timeout=43_000,
    startup_timeout=4_800,
    secrets=training_secrets,
    volumes={"/cache": asset_volume, "/runs": run_volume},
)
def run_4gpu(
    *,
    mode: str = "pilot",
    branch: str = DEFAULT_BRANCH,
    commit: str = "",
    config: str = "",
    run_name: str = "auto",
    max_steps: int = 0,
    eval_prompts: int = 8,
    max_train_prompts: int = 64,
    smoke_prompts: int = 16,
    learning_rate: float = 0.0,
    force_prepare: bool = False,
    resume: bool = False,
    skip_preflight: bool = False,
) -> dict[str, Any]:
    return _run_job(
        gpu_count=4,
        mode=mode,
        branch=branch,
        commit=commit,
        config=config,
        run_name=run_name,
        max_steps=max_steps,
        eval_prompts=eval_prompts,
        max_train_prompts=max_train_prompts,
        smoke_prompts=smoke_prompts,
        learning_rate=learning_rate,
        force_prepare=force_prepare,
        resume=resume,
        skip_preflight=skip_preflight,
    )


@app.local_entrypoint()
def main(
    gpus: int = 1,
    mode: str = "smoke",
    branch: str = DEFAULT_BRANCH,
    commit: str = "",
    config: str = "",
    run_name: str = "auto",
    max_steps: int = 0,
    eval_prompts: int = 8,
    max_train_prompts: int = 64,
    smoke_prompts: int = 16,
    learning_rate: float = 0.0,
    force_prepare: bool = False,
    resume: bool = False,
    skip_preflight: bool = False,
) -> None:
    kwargs = {
        "mode": mode,
        "branch": branch,
        "commit": commit,
        "config": config,
        "run_name": run_name,
        "max_steps": max_steps,
        "eval_prompts": eval_prompts,
        "max_train_prompts": max_train_prompts,
        "smoke_prompts": smoke_prompts,
        "learning_rate": learning_rate,
        "force_prepare": force_prepare,
        "resume": resume,
        "skip_preflight": skip_preflight,
    }
    if gpus == 1:
        result = run_1gpu.remote(**kwargs)
    elif gpus == 4:
        result = run_4gpu.remote(**kwargs)
    else:
        raise ValueError("--gpus must be 1 or 4")
    print(json.dumps(result, indent=2, sort_keys=True))
