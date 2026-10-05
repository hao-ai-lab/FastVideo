# SPDX-License-Identifier: Apache-2.0
"""Opt-in, released-weight pipeline parity through the public FastVideo API.

Each implementation runs in its own process so their weights never overlap in
VRAM. Synthetic references are the default; QWEN_IMAGE21_PARITY_CASES_JSON can
provide per-case prompts and ordered reference paths from the original source.
QWEN_IMAGE21_PARITY_DIAGNOSTICS=1 records conditioning, scheduler and per-step
latent boundaries without changing either implementation's numeric operations.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image, ImageDraw, ImageOps

ROOT = Path(__file__).resolve().parents[3]
REFERENCE_REVISION = "578c9b2c6636ab2424a0e56186268b83623656b2"
PARITY_SCOPE = "pipeline"
CASES = ("t2i", "i2i", "edit", "ref2img", "ten_refs", "rgba", "rgba_edit", "annotation", "mask_image", "true_cfg", "uncached")


def _cpu_snapshot(value):
    # Local imports keep cloudpickle from serializing Torch's backend modules.
    import torch

    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, (tuple, list)):
        return [_cpu_snapshot(item) for item in value]
    return value


def _runtime_metadata() -> dict:
    import torch

    return {
        "default_dtype": str(torch.get_default_dtype()),
        "float32_matmul_precision": torch.get_float32_matmul_precision(),
        "matmul_fp32_precision": getattr(torch.backends.cuda.matmul, "fp32_precision", None),
        "matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
        "cudnn_fp32_precision": getattr(torch.backends.cudnn, "fp32_precision", None),
        "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
    }


def _install_native_diagnostics(worker) -> None:
    """Observe the actual GPU worker; only ``extra`` survives executor transport."""
    import torch

    from fastvideo.pipelines.basic.qwen_image21.inputs import STATE_KEY, reference_pixels

    original_forward = worker.execute_forward
    scheduler = worker.pipeline.get_module("scheduler")
    original_step = scheduler.step

    def forward(forward_batch, fastvideo_args):
        captured = {"initial_latents": _cpu_snapshot(forward_batch.latents), "step_latents": [],
                    "noise_predictions": [], "runtime": _runtime_metadata()}

        def step(model_output, timestep, sample, *args, **kwargs):
            captured["noise_predictions"].append(_cpu_snapshot(model_output))
            result = original_step(model_output, timestep, sample, *args, **kwargs)
            previous = result[0] if isinstance(result, tuple) else result.prev_sample
            captured["step_latents"].append(_cpu_snapshot(previous))
            return result

        scheduler.step = step
        try:
            output = original_forward(forward_batch, fastvideo_args)
            state = output.extra[STATE_KEY]
            captured.update({
                "final_latents": _cpu_snapshot(output.latents),
                "prompt_embeds": _cpu_snapshot(output.prompt_embeds[0]),
                "prompt_attention_mask": _cpu_snapshot(output.prompt_attention_mask[0]),
                "image_pad_mask": _cpu_snapshot(state.image_pad_mask),
                "image_latents": _cpu_snapshot(output.image_latent),
                "reference_pixels": [reference_pixels(image) for image in state.references],
                "timesteps": _cpu_snapshot(output.timesteps),
                "sigmas": _cpu_snapshot(scheduler.sigmas),
                "img_shapes": _cpu_snapshot(state.img_shapes),
            })
            captured["reference_pixels_bf16"] = [pixels.to(torch.bfloat16)
                                                 for pixels in captured["reference_pixels"]]
            if output.do_classifier_free_guidance:
                captured.update({
                    "negative_prompt_embeds": _cpu_snapshot(output.negative_prompt_embeds[0]),
                    "negative_attention_mask": _cpu_snapshot(output.negative_attention_mask[0]),
                    "negative_image_pad_mask": _cpu_snapshot(state.negative_image_pad_mask),
                })
            output.extra["qwen_image21_diagnostics"] = captured
            return output
        finally:
            scheduler.step = original_step
            worker.execute_forward = original_forward

    worker.execute_forward = forward


def _install_reference_diagnostics(pipe, captured: dict):
    original_encode = pipe.encode_prompt
    original_prepare = pipe.prepare_latents
    original_step = pipe.scheduler.step
    captured.update({"step_latents": [], "noise_predictions": [], "runtime": _runtime_metadata()})

    def encode(*args, **kwargs):
        result = original_encode(*args, **kwargs)
        keys = (("prompt_embeds", "prompt_attention_mask", "image_pad_mask") if "prompt_embeds" not in captured
                else ("negative_prompt_embeds", "negative_attention_mask", "negative_image_pad_mask"))
        captured.update(zip(keys, _cpu_snapshot(result), strict=True))
        return result

    def prepare(images, *args, **kwargs):
        captured["reference_pixels"] = _cpu_snapshot(images or [])
        captured["reference_pixels_bf16"] = [pixels.to(torch.bfloat16)
                                             for pixels in captured["reference_pixels"]]
        result = original_prepare(images, *args, **kwargs)
        captured["initial_latents"], captured["image_latents"] = _cpu_snapshot(result)
        for key in ("image_pad_mask", "negative_image_pad_mask"):
            if key in captured:
                mask = captured[key]
                slots = mask.new_ones(mask.shape[0], result[0].shape[1] // 4)
                captured[key] = torch.cat((mask, slots), dim=1)
        return result

    def step(model_output, timestep, sample, *args, **kwargs):
        captured["noise_predictions"].append(_cpu_snapshot(model_output))
        return original_step(model_output, timestep, sample, *args, **kwargs)

    def callback(pipeline, index, timestep, callback_kwargs):
        captured["step_latents"].append(_cpu_snapshot(callback_kwargs["latents"]))
        return callback_kwargs

    def restore():
        pipe.encode_prompt = original_encode
        pipe.prepare_latents = original_prepare
        pipe.scheduler.step = original_step

    pipe.encode_prompt = encode
    pipe.prepare_latents = prepare
    pipe.scheduler.step = step
    return callback, restore


def _worker(kind: str, spec_path: str, output_path: str) -> None:
    spec = json.loads(Path(spec_path).read_text())
    sys.path.insert(0, str(ROOT))
    sys.path.insert(0, str(Path(spec["reference"]) / "src"))
    height, width = spec.get("height", spec["resolution"]), spec.get("width", spec["resolution"])
    seed = spec.get("seed", 21)
    reference_resolution = spec.get("reference_resolution", 1024)
    sigmas = spec.get("sigmas", np.linspace(1.0, 1 / spec["steps"], spec["steps"]).tolist())
    noise = torch.load(spec["latents"], map_location="cpu", weights_only=True)
    assert noise.dtype == torch.bfloat16
    assert noise.shape == (1, (height // 16) * (width // 16), 64)
    torch.manual_seed(seed)
    diagnostics = spec.get("diagnostics", os.environ.get("QWEN_IMAGE21_PARITY_DIAGNOSTICS") == "1")
    captured = {}
    if kind == "native":
        from fastvideo import VideoGenerator
        from fastvideo.api import (EngineConfig, GenerationRequest, GeneratorConfig, InputConfig, OffloadConfig,
                                   OutputConfig, ParallelismConfig, PipelineSelection, SamplingConfig)

        generator = VideoGenerator.from_config(
            GeneratorConfig(
                model_path=spec["model"],
                engine=EngineConfig(num_gpus=1, parallelism=ParallelismConfig(tp_size=1, sp_size=1),
                                    offload=OffloadConfig(dit=True, dit_layerwise=True, text_encoder=True,
                                                          vae=True, lazy_module_load=True)),
                pipeline=PipelineSelection(workload_type="i2i" if spec["refs"] else "t2i"),
            ))
        original_execute = generator.executor.execute_forward
        try:
            if diagnostics:
                import cloudpickle

                generator.executor.collective_rpc(cloudpickle.dumps(_install_native_diagnostics))

                def execute(*args, **kwargs):
                    batch = original_execute(*args, **kwargs)
                    captured.update(batch.extra["qwen_image21_diagnostics"])
                    return batch

                generator.executor.execute_forward = execute
            result = generator.generate(
                GenerationRequest(
                    prompt=spec["prompt"], negative_prompt=spec["negative_prompt"],
                    inputs=InputConfig(references=spec["refs"] or None, latents=noise.clone()),
                    sampling=SamplingConfig(height=height, width=width, num_frames=1,
                                            fps=1, num_inference_steps=spec["steps"], guidance_scale=1,
                                            true_cfg_scale=spec["true_cfg_scale"], seed=seed,
                                            reference_resolution=reference_resolution,
                                            use_kv_cache=spec["cache"], sigmas=sigmas),
                    output=OutputConfig(output_path=str(Path(output_path).with_suffix(".png")),
                                        return_frames=True, save_video=True),
                ))
            torch.save({"pixels": result.samples.detach().float().cpu(), "png_path": result.video_path,
                        "peak_memory_mb": result.peak_memory_mb, "diagnostics": captured}, output_path)
        finally:
            generator.executor.execute_forward = original_execute
            generator.shutdown()
    else:
        from diffusers.pipelines.qwenimage21 import pipeline_qwenimage21

        expected = Path(spec["reference"]) / "src/diffusers/pipelines/qwenimage21/pipeline_qwenimage21.py"
        assert Path(pipeline_qwenimage21.__file__).resolve() == expected.resolve()
        pipe = pipeline_qwenimage21.QwenImage21Pipeline.from_pretrained(
            spec["model"], torch_dtype=torch.bfloat16, local_files_only=True)
        pipe.vae.enable_tiling()
        pipe.enable_sequential_cpu_offload()
        references = []
        for path in spec["refs"]:
            with Image.open(path) as image:
                references.append(ImageOps.exif_transpose(image).convert("RGBA"))
        callback, restore = _install_reference_diagnostics(pipe, captured) if diagnostics else (None, lambda: None)
        try:
            with torch.no_grad():
                output = pipe(prompt=spec["prompt"], image=references or None, height=height, width=width,
                              output_resolution=reference_resolution, num_inference_steps=spec["steps"], sigmas=sigmas,
                              negative_prompt=spec["negative_prompt"], true_cfg_scale=spec["true_cfg_scale"],
                              generator=torch.Generator(device="cpu").manual_seed(seed),
                              latents=noise.clone(), output_type="pt", use_kv_cache=spec["cache"],
                              callback_on_step_end=callback)
            if diagnostics:
                captured.update({"final_latents": captured["step_latents"][-1],
                                 "timesteps": _cpu_snapshot(pipe.scheduler.timesteps),
                                 "sigmas": _cpu_snapshot(pipe.scheduler.sigmas),
                                 "img_shapes": [[*[list((1, pixels.shape[-2] // 16, pixels.shape[-1] // 16))
                                                   for pixels in captured["reference_pixels"]],
                                                 [1, height // 16, width // 16]]]})
        finally:
            restore()
        torch.save({"pixels": output.images.unsqueeze(2).float().cpu(), "diagnostics": captured}, output_path)


def _assets(tmp_path: Path) -> list[str]:
    files = []
    for index in range(10):
        image = Image.new("RGBA", (256, 256), (230, 235, 240, 0 if index == 0 else 255))
        draw = ImageDraw.Draw(image)
        draw.ellipse((48 + index, 48, 208, 208), fill=(30 + index * 20, 80, 140, 200))
        if index == 1:
            draw.rectangle((24, 24, 232, 232), outline="red", width=8)
        path = tmp_path / f"reference_{index + 1}.png"
        image.save(path)
        files.append(str(path))
    mask = Image.new("RGBA", (256, 256), "black")
    ImageDraw.Draw(mask).ellipse((48, 48, 208, 208), fill="white")
    path = tmp_path / "mask.png"
    mask.save(path)
    return files + [str(path)]


def _case_overrides(case: str) -> dict:
    path = os.environ.get("QWEN_IMAGE21_PARITY_CASES_JSON")
    if not path:
        return {}
    manifest = Path(path).resolve()
    overrides = json.loads(manifest.read_text()).get(case, {})
    allowed = {"prompt", "refs", "height", "width", "steps", "seed", "reference_resolution", "negative_prompt",
               "true_cfg_scale", "cache", "sigmas"}
    assert isinstance(overrides, dict) and not (set(overrides) - allowed)
    if "refs" in overrides:
        assert isinstance(overrides["refs"], list) and len(overrides["refs"]) <= 10
        paths = [Path(value) for value in overrides["refs"]]
        paths = [value if value.is_absolute() else manifest.parent / value for value in paths]
        assert all(value.is_file() for value in paths), "Manifest references must be existing local files"
        overrides["refs"] = [str(value.resolve()) for value in paths]
    return overrides


def _difference_metrics(actual: torch.Tensor, expected: torch.Tensor, *, atol=0.04, rtol=0.02) -> dict[str, float]:
    actual, expected = actual.float(), expected.float()
    error = (actual - expected).abs()
    mean_error = error.mean().item()
    reference_mean = expected.abs().mean().item()
    rmse = error.square().mean().sqrt().item()
    return {
        "abs_mean": mean_error,
        "abs_max": error.max().item(),
        "rmse": rmse,
        "relative_l1": mean_error / max(reference_mean, 1e-12),
        "relative_l2": rmse / max(expected.square().mean().sqrt().item(), 1e-12),
        "actual_abs_mean": actual.abs().mean().item(),
        "reference_abs_mean": reference_mean,
        "fraction_outside_tolerance": (error > atol + rtol * expected.abs()).float().mean().item(),
    }


def _diagnostic_metrics(actual, expected):
    if isinstance(actual, torch.Tensor) and isinstance(expected, torch.Tensor):
        result = {"actual_shape": list(actual.shape), "reference_shape": list(expected.shape),
                  "actual_dtype": str(actual.dtype), "reference_dtype": str(expected.dtype),
                  "exact": torch.equal(actual, expected)}
        if actual.shape == expected.shape and actual.numel():
            result.update(_difference_metrics(actual, expected, atol=0, rtol=0))
        return result
    if isinstance(actual, torch.Tensor) or isinstance(expected, torch.Tensor):
        def describe(value):
            if isinstance(value, torch.Tensor):
                return {"shape": list(value.shape), "dtype": str(value.dtype)}
            return value

        return {"actual": describe(actual), "reference": describe(expected), "exact": False}
    if isinstance(actual, dict) and isinstance(expected, dict):
        return {key: _diagnostic_metrics(actual.get(key), expected.get(key))
                for key in sorted(set(actual) | set(expected))}
    if isinstance(actual, list) and isinstance(expected, list):
        return {"actual_length": len(actual), "reference_length": len(expected),
                "items": [_diagnostic_metrics(left, right)
                          for left, right in zip(actual, expected, strict=False)]}
    return {"actual": actual, "reference": expected, "exact": actual == expected}


def _run_parity(case, resolution, tmp_path):
    if os.environ.get("QWEN_IMAGE21_RUN_PIPELINE_PARITY") != "1":
        pytest.skip("Set QWEN_IMAGE21_RUN_PIPELINE_PARITY=1 on an allocated CUDA node")
    if not torch.cuda.is_available():
        pytest.fail("Qwen-Image-2.1 pipeline parity requires CUDA", pytrace=False)
    model = Path(os.environ.get("QWEN_IMAGE21_MODEL_ROOT", ROOT / "official_weights/Qwen-Image-2.1"))
    reference = Path(os.environ.get("QWEN_IMAGE21_DIFFUSERS_DIR", ROOT / "official_reference/diffusers"))
    if not (model / "model_index.json").is_file() or not (reference / "src").is_dir():
        pytest.fail("Provide QWEN_IMAGE21_MODEL_ROOT and the pinned QWEN_IMAGE21_DIFFUSERS_DIR", pytrace=False)
    revision = subprocess.check_output(["git", "-C", str(reference), "rev-parse", "HEAD"], text=True).strip()
    assert revision == REFERENCE_REVISION
    subprocess.run(["git", "-C", str(reference), "diff", "--exit-code", "HEAD", "--", "src/diffusers"], check=True)
    files = _assets(tmp_path)
    counts = dict(i2i=1, edit=1, ref2img=2, ten_refs=10, rgba_edit=1, annotation=2, mask_image=1, true_cfg=1)
    refs = files[:counts.get(case, 0)]
    prompt = "An elegant blue ceramic vase in a studio, with the word Qwen written on it."
    if refs:
        prompt = "Make a blue sculpture inspired by the shapes in the reference images."
    if case == "edit":
        prompt = "Change the object in image 1 into a blue ceramic sphere; retain the background and composition."
    if case in ("rgba", "rgba_edit"):
        prompt = "A blue sphere isolated on a transparent background."
    if case == "annotation":
        prompt = "Change the object enclosed by the red annotation in image 2 into a blue sphere in image 1."
    if case == "mask_image":
        refs.append(files[-1])
        prompt = "Replace the white masked area of image 2 in image 1 with flowers."
    noise_path = tmp_path / "initial_latents.pt"
    spec = dict(model=str(model.resolve()), reference=str(reference.resolve()), prompt=prompt, refs=refs,
                resolution=resolution, steps=int(os.environ.get("QWEN_IMAGE21_PARITY_STEPS", "4")),
                negative_prompt="" if case == "true_cfg" else None,
                true_cfg_scale=2 if case == "true_cfg" else 1, cache=case != "uncached", latents=str(noise_path),
                seed=21, reference_resolution=1024, height=resolution, width=resolution,
                diagnostics=os.environ.get("QWEN_IMAGE21_PARITY_DIAGNOSTICS") == "1")
    spec.update(_case_overrides(case))
    height, width = spec["height"], spec["width"]
    assert min(height, width, spec["reference_resolution"]) > 0
    assert all(value % 32 == 0 for value in (height, width, spec["reference_resolution"]))
    assert spec["steps"] > 0
    spec.setdefault("sigmas", np.linspace(1.0, 1 / spec["steps"], spec["steps"]).tolist())
    torch.save(torch.randn(1, (height // 16) * (width // 16), 64,
                           generator=torch.Generator(device="cpu").manual_seed(spec["seed"]),
                           dtype=torch.bfloat16), noise_path)
    spec_path = tmp_path / "spec.json"
    spec_path.write_text(json.dumps(spec))
    outputs = []
    for kind in ("native", "reference"):
        output = tmp_path / f"{kind}.pt"
        subprocess.run([sys.executable, str(Path(__file__).resolve()), "--worker", kind, str(spec_path), str(output)],
                       check=True, cwd=ROOT)
        outputs.append(torch.load(output, map_location="cpu", weights_only=True))
    actual, expected = outputs[0]["pixels"], outputs[1]["pixels"]
    assert actual.shape == expected.shape == (1, 4, 1, height, width)
    assert torch.isfinite(actual).all() and bool(((actual >= 0) & (actual <= 1)).all())
    assert torch.isfinite(expected).all() and bool(((expected >= 0) & (expected <= 1)).all())
    metrics = {"case": case, "steps": spec["steps"], "cache": spec["cache"], "seed": spec["seed"],
               "rgb": _difference_metrics(actual[:, :3], expected[:, :3]),
               "all_channels": _difference_metrics(actual, expected)}
    if spec["diagnostics"]:
        stage_metrics = _diagnostic_metrics(outputs[0]["diagnostics"], outputs[1]["diagnostics"])
        (tmp_path / "diagnostic_metrics.json").write_text(json.dumps(stage_metrics, indent=2) + "\n")
        print("Stage diagnostics: " + json.dumps(stage_metrics), flush=True)
    (tmp_path / "parity_metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    print(json.dumps(metrics), flush=True)
    # Provisional tolerances: calibrate from measured BF16 CUDA evidence before release.
    torch.testing.assert_close(actual, expected, atol=0.04, rtol=0.02)
    native_png = Image.open(outputs[0]["png_path"])
    assert native_png.mode == "RGBA"
    expected_u8 = (actual[0, :, 0].permute(1, 2, 0) * 255).clamp(0, 255).to(torch.uint8).numpy()
    np.testing.assert_array_equal(np.array(native_png), expected_u8)
    if outputs[0]["peak_memory_mb"] is not None:
        assert outputs[0]["peak_memory_mb"] <= 24 * 1024


@pytest.mark.parametrize("case", CASES)
def test_qwen_image21_modes_1024_pipeline_parity(case, tmp_path):
    _run_parity(case, 1024, tmp_path)


def test_qwen_image21_t2i_2k_pipeline_parity(tmp_path):
    if os.environ.get("QWEN_IMAGE21_RUN_2K_PARITY") != "1":
        pytest.skip("Set QWEN_IMAGE21_RUN_2K_PARITY=1 for the separate 2K gate")
    if os.environ.get("QWEN_IMAGE21_RUN_PIPELINE_PARITY") != "1":
        pytest.fail("The 2K gate also requires QWEN_IMAGE21_RUN_PIPELINE_PARITY=1", pytrace=False)
    _run_parity("t2i", 2048, tmp_path)


if __name__ == "__main__":
    if len(sys.argv) != 5 or sys.argv[1] != "--worker":
        raise SystemExit("This helper is launched by pytest")
    _worker(sys.argv[2], sys.argv[3], sys.argv[4])
