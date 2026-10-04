# SPDX-License-Identifier: Apache-2.0
"""Resolution steps that fill typed fields from files that a checkpoint bundles.

Distilled LTX-2 checkpoints carry ``fastvideo_refine_*`` defaults in ``model_index.json``, and distilled MiniMax-H3
checkpoints carry their trained DMD schedule in ``fastvideo_inference.json``. The steps here read those files during
resolution, so the frozen config already holds the checkpoint's values when the pipeline loads. Each field fills only
while it is ``None``: a value from the input always wins.

A local ``model_path`` is read from disk. A Hub id downloads only the files that the steps need: the manifest through
``fastvideo.registry.maybe_download_model_index`` (the download that the test isolation blocks), and the schedule file
or the component ``config.json`` files through ``huggingface_hub``. When the manifest download fails, the steps leave
their fields unset; the pipeline's own download reports the error.
"""
from __future__ import annotations

import json
import os
from typing import Any

from fastvideo.api.resolution import ResolutionStep, ResolutionView
from fastvideo.logger import init_logger
from fastvideo.utils import _split_hf_repo_subfolder, maybe_download_model, verify_model_config_and_directory

logger = init_logger(__name__)

# Refine component paths, by the model_index.json key that supplies the checkpoint's default.
LTX2_REFINE_PATH_KEYS = {
    "pipeline.ltx2.refine.transformer_path": "fastvideo_refine_transformer_path",
    "pipeline.ltx2.refine.lora_path": "fastvideo_refine_lora_path",
    "pipeline.ltx2.refine.noise_path": "fastvideo_refine_noise_path",
    "pipeline.ltx2.refine.audio_noise_path": "fastvideo_refine_audio_noise_path",
}
# Refine switches, by the model_index.json key that supplies the checkpoint's default and the value's type.
LTX2_REFINE_SWITCH_KEYS = (
    ("pipeline.ltx2.refine.num_inference_steps", "fastvideo_refine_num_inference_steps", int),
    ("pipeline.ltx2.refine.guidance_scale", "fastvideo_refine_guidance_scale", float),
    ("pipeline.ltx2.refine.add_noise", "fastvideo_refine_add_noise", bool),
)
# Directory names under which published LTX-2 checkpoints bundle the refine upsampler without declaring it.
_UPSAMPLER_DIRECTORY_NAMES = ("spatial_upscaler", "spatial_upsampler")
# The MiniMax-H3 schedule file; the pipeline validates its contents against the loaded schedulers.
H3_SCHEDULE_FILENAME = "fastvideo_inference.json"


def _local_manifest(model_path: str) -> dict[str, Any] | None:
    """The manifest of a local checkpoint directory, or ``None`` when the directory has no manifest file."""
    manifest_names = ("model_index.json", "modular_model_index.json")
    if not any(os.path.isfile(os.path.join(model_path, name)) for name in manifest_names):
        return None
    return verify_model_config_and_directory(model_path, required_component_dirs=[])


def _hub_manifest(model_path: str, revision: str | None) -> dict[str, Any] | None:
    """The manifest of a Hub checkpoint, or ``None`` when its download fails.

    The download goes through ``fastvideo.registry`` so that a test isolation which blocks registry downloads blocks
    this one too; the steps then leave their fields unset.
    """
    import fastvideo.registry as registry

    try:
        return registry.maybe_download_model_index(model_path, revision=revision)
    except Exception as error:
        logger.debug("Skipping the checkpoint defaults of %s: the manifest is unavailable (%s)", model_path, error)
        return None


def _checkpoint_manifest(model_path: str, revision: str | None) -> dict[str, Any] | None:
    """The checkpoint manifest of ``model_path``, or ``None`` when it cannot be read."""
    if os.path.exists(model_path):
        return _local_manifest(model_path)
    return _hub_manifest(model_path, revision)


def _resolve_refine_path(root: str, value: str | None) -> str | None:
    """A refine component path; a relative value that exists inside the checkpoint root becomes that path."""
    if value is None:
        return None
    if os.path.isabs(value):
        return value
    candidate = os.path.join(root, value)
    if os.path.exists(candidate):
        return candidate
    return value


def _resolve_refine_upsampler_path(root: str, model_index: dict[str, Any]) -> str | None:
    """The refine upsampler directory of a checkpoint.

    The ``model_index.json`` keys ``fastvideo_refine_upsampler_path`` and then ``spatial_upsampler`` are the documented
    override and win when present. Otherwise the checkpoint root is probed: published LTX-2 distilled checkpoints
    bundle the upsampler weights on disk (spelled ``spatial_upscaler``) without declaring them in ``model_index.json``.
    """
    path = _resolve_refine_path(root, model_index.get("fastvideo_refine_upsampler_path"))
    if path is None and "spatial_upsampler" in model_index:
        path = _resolve_refine_path(root, "spatial_upsampler")
    if path is None:
        for dirname in _UPSAMPLER_DIRECTORY_NAMES:
            candidate = os.path.join(root, dirname)
            if os.path.isdir(candidate):
                return candidate
    return path


def _ltx2_checkpoint_root(model_path: str, revision: str | None, model_index: dict[str, Any]) -> str:
    """The directory that relative refine paths of ``model_index`` resolve against.

    A local ``model_path`` is that directory. For a Hub id, a partial snapshot with every component ``config.json``
    and every relative refine path of the manifest is downloaded; its directory is the one that the pipeline's full
    download fills later, so the resolved paths stay valid.
    """
    if os.path.exists(model_path):
        return model_path
    relative_values = [
        value for key in (*LTX2_REFINE_PATH_KEYS.values(), "fastvideo_refine_upsampler_path")
        for value in (model_index.get(key), ) if isinstance(value, str) and not os.path.isabs(value)
    ]
    return maybe_download_model(model_path, revision=revision, allow_patterns=["*/config.json", *relative_values])


def ltx2_refine_checkpoint_step(defaults: Any) -> ResolutionStep:
    """Build the step that fills the unset LTX-2 refine fields from the checkpoint's ``model_index.json``.

    ``defaults`` is the model's ``PipelineConfig``; the step decides nothing for a model that is not LTX-2.
    ``fastvideo_refine_enabled`` can only turn refine on. The upsampler falls back to a directory of the checkpoint.
    """

    def fill_ltx2_refine_from_checkpoint(view: ResolutionView) -> dict[str, Any]:
        from fastvideo.pipelines.basic.ltx2.pipeline_configs import LTX2T2VConfig

        if not isinstance(defaults, LTX2T2VConfig):
            return {}
        model_path, revision = view.get("model_path"), view.get("revision")
        model_index = _checkpoint_manifest(model_path, revision)
        if model_index is None:
            return {}
        root = _ltx2_checkpoint_root(model_path, revision, model_index)
        values: dict[str, Any] = {}
        if model_index.get("fastvideo_refine_enabled") is True and view.get("pipeline.ltx2.refine.enabled") is None:
            values["pipeline.ltx2.refine.enabled"] = True
        if view.get("pipeline.components.upsampler_weights") is None:
            upsampler_path = _resolve_refine_upsampler_path(root, model_index)
            if upsampler_path is not None:
                values["pipeline.components.upsampler_weights"] = upsampler_path
        for path, key in LTX2_REFINE_PATH_KEYS.items():
            if view.get(path) is None:
                component_path = _resolve_refine_path(root, model_index.get(key))
                if component_path is not None:
                    values[path] = component_path
        for path, key, convert in LTX2_REFINE_SWITCH_KEYS:
            if view.get(path) is None and model_index.get(key) is not None:
                values[path] = convert(model_index[key])
        return values

    fill_ltx2_refine_from_checkpoint.__qualname__ = "fill_ltx2_refine_from_checkpoint"
    return fill_ltx2_refine_from_checkpoint


def _checkpoint_json(model_path: str, revision: str | None, filename: str) -> Any:
    """The parsed JSON file ``filename`` of the checkpoint, or ``None`` when the checkpoint does not have it.

    For a Hub id, the manifest download gates the file download (see :func:`_hub_manifest`).
    """
    if os.path.exists(model_path):
        path = os.path.join(model_path, filename)
    else:
        if _hub_manifest(model_path, revision) is None:
            return None
        from huggingface_hub import hf_hub_download
        from huggingface_hub.utils import EntryNotFoundError

        repo_id, subfolder = _split_hf_repo_subfolder(model_path)
        try:
            path = hf_hub_download(repo_id=repo_id,
                                   filename=f"{subfolder}/{filename}" if subfolder else filename,
                                   revision=revision)
        except EntryNotFoundError:
            return None
    if not os.path.isfile(path):
        return None
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def dmd_schedule_checkpoint_step(defaults: Any) -> ResolutionStep:
    """Build the step that fills an unset ``pipeline.dmd_denoising_steps`` from a MiniMax-H3 checkpoint's
    ``fastvideo_inference.json``.

    ``defaults`` is the model's ``PipelineConfig``; the step decides nothing for a model that is not MiniMax-H3. The
    step copies the ``dmd_denoising_steps`` list as written; the pipeline validates the file against the loaded
    schedulers and raises on a malformed or conflicting schedule.
    """

    def fill_dmd_schedule_from_checkpoint(view: ResolutionView) -> dict[str, Any]:
        from fastvideo.configs.pipelines.minimax_h3 import MiniMaxH3PipelineConfig

        if not isinstance(defaults, MiniMaxH3PipelineConfig) or view.get("pipeline.dmd_denoising_steps") is not None:
            return {}
        contract = _checkpoint_json(view.get("model_path"), view.get("revision"), H3_SCHEDULE_FILENAME)
        steps = contract.get("dmd_denoising_steps") if isinstance(contract, dict) else None
        if not isinstance(steps, list) or not steps:
            return {}
        return {"pipeline.dmd_denoising_steps": list(steps)}

    fill_dmd_schedule_from_checkpoint.__qualname__ = "fill_dmd_schedule_from_checkpoint"
    return fill_dmd_schedule_from_checkpoint


__all__ = [
    "H3_SCHEDULE_FILENAME",
    "LTX2_REFINE_PATH_KEYS",
    "LTX2_REFINE_SWITCH_KEYS",
    "dmd_schedule_checkpoint_step",
    "ltx2_refine_checkpoint_step",
]
