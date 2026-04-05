# SPDX-License-Identifier: Apache-2.0
# WAYPOINT_HF_DIFFUSERS_SMOKE=2  # verify deploy: ``grep SMOKE=2`` on the pod copy.
"""Hugging Face Diffusers path for Overworld Waypoint-1-Small (official repo).

This is **not** FastVideo ``VideoGenerator``; it runs ``diffusers`` ``ModularPipeline``
as in the upstream model card. Use it to smoke-test HF parity or to produce an MP4
for manual / SSIM comparison against FastVideo.

Requirements: CUDA, ``diffusers`` with modular pipelines (>=0.36 typical),
``tensordict`` (Overworld remote code), network, and ``trust_remote_code`` for
``Overworld/Waypoint-1-Small``. Install test extras: ``uv pip install -e ".[test]"``.

This path downloads **google/umt5-xl** (~10GiB) plus Waypoint weights into the HF
cache. Small root disks (common on RunPod) fill ``/root/.cache`` and fail with
cryptic ``NoneType`` / ``dtype`` errors. Point the cache at a large volume::

    export HF_HOME=/workspace/.cache/huggingface
    mkdir -p "$HF_HOME"

Run::

    pytest fastvideo/tests/ssim/test_waypoint_hf_diffusers.py -v

After copying to a remote box, confirm line 2 contains ``SMOKE=2``. To detect a
stale copy, ``grep 'st_size > 10_000'`` on this file should print nothing.

Optional env:

- ``WAYPOINT_HF_MODEL_ID`` — default ``Overworld/Waypoint-1-Small``
- ``WAYPOINT_HF_TEST_MP4`` — if set, write the smoke MP4 to this path (e.g.
  ``/workspace/waypoint_hf_smoke.mp4``) so you can ``scp`` or download it; otherwise
  the file lives only under pytest's ``tmp_path`` (hard to find after the run).
- ``FASTVIDEO_SKIP_WAYPOINT_HF_DIFFUSERS_TEST=1`` — skip even on CUDA (e.g. CI image
  without Hub access)
"""

from __future__ import annotations

import gc
import os
import shutil
from pathlib import Path

import pytest
import torch

# ModularPipeline pulls google/umt5-xl (~10GiB) + Waypoint; need headroom on the
# filesystem that hosts HF_HOME (not just inode space).
_MIN_FREE_BYTES = 14 * 1024 * 1024 * 1024

# Match ``test_waypoint_similarity`` defaults where applicable.
DEFAULT_PROMPT = "A first-person view of walking through a grassy field."
KEY_FORWARD = 17
NUM_FRAMES = 17
SEED = 1024
HEIGHT = 368
WIDTH = 640
FPS = 60


def _hf_home_dir() -> str:
    return os.environ.get(
        "HF_HOME",
        os.path.join(os.path.expanduser("~"), ".cache", "huggingface"),
    )


def _assert_mp4_has_expected_frames(path: str, expected: int) -> None:
    """Smoke-check encoded output; H.264 can be <10KiB for few near-static frames."""
    import imageio.v2 as imageio

    reader = imageio.get_reader(path, "ffmpeg")
    try:
        n = sum(1 for _ in reader)
    finally:
        reader.close()
    assert n == expected, (
        f"WAYPOINT_HF_SMOKE_V2: expected {expected} frames in {path!r}, got {n}"
    )


def _pil_from_state_image(obj: object):
    from PIL import Image

    if isinstance(obj, Image.Image):
        return obj
    import numpy as np

    # ModularPipeline may return a single-element batch (list/tuple).
    while isinstance(obj, (list, tuple)):
        if not obj:
            raise TypeError("empty images sequence")
        obj = obj[0]
    if isinstance(obj, Image.Image):
        return obj

    if isinstance(obj, torch.Tensor):
        t = obj.detach().float().cpu()
        if t.dim() == 4:
            t = t[0]
        if t.dim() == 3 and t.shape[0] in (1, 3):
            t = t.permute(1, 2, 0)
        tmin = float(t.min())
        tmax = float(t.max())
        # [0, 1] tensors often have tiny negative noise; (t+1)/2 would wash to white.
        # Only remap when the min clearly indicates VAE [-1, 1] decode.
        if tmax > 1.5:
            t = t / 255.0
        elif tmin <= -0.25:
            t = (t + 1.0) * 0.5
        arr = (t.clamp(0, 1) * 255).byte().numpy()
        return Image.fromarray(arr)
    if isinstance(obj, np.ndarray):
        return Image.fromarray(obj)
    raise TypeError(f"Unsupported image type: {type(obj)}")


def _run_hf_waypoint_to_mp4(
    *,
    output_path: str,
    model_id: str,
    prompt: str,
    num_frames: int,
    seed: int,
    height: int,
    width: int,
    fps: int,
    key_forward: int,
    seed_gray: int,
    dtype_bf16: bool,
) -> None:
    from PIL import Image
    from tqdm import tqdm

    from diffusers.modular_pipelines import ModularPipeline
    from diffusers.utils import export_to_video

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    image = Image.new("RGB", (width, height), color=(seed_gray, seed_gray, seed_gray))
    dtype = torch.bfloat16 if dtype_bf16 else torch.float16

    pipe = ModularPipeline.from_pretrained(
        model_id,
        trust_remote_code=True,
    )
    pipe.load_components(
        device_map="cuda",
        torch_dtype=dtype,
        trust_remote_code=True,
    )
    pipe.transformer.apply_inference_patches()

    button = {int(key_forward)}
    mouse = (0.0, 0.0)

    outputs: list = []
    state = pipe(
        prompt=prompt,
        image=image,
        button=button,
        mouse=mouse,
        output_type="pil",
    )
    outputs.append(_pil_from_state_image(state.values["images"]))
    state.values["image"] = None

    for _ in tqdm(range(1, num_frames), desc="hf waypoint frames"):
        state = pipe(
            state,
            prompt=prompt,
            button=button,
            mouse=mouse,
            output_type="pil",
        )
        outputs.append(_pil_from_state_image(state.values["images"]))

    out = os.path.abspath(output_path)
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    export_to_video(outputs, out, fps=fps)

    del pipe, state, outputs
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


@pytest.mark.skipif(
    os.environ.get("FASTVIDEO_SKIP_WAYPOINT_HF_DIFFUSERS_TEST", "").strip().lower()
    in {"1", "true", "yes", "on"},
    reason="FASTVIDEO_SKIP_WAYPOINT_HF_DIFFUSERS_TEST set",
)
@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="HF Waypoint ModularPipeline test requires CUDA",
)
@pytest.mark.parametrize("model_id", ["Waypoint-1-Small-Diffusers"])
def test_waypoint_hf_diffusers_smoke(tmp_path, model_id: str) -> None:
    """Load ``Overworld/Waypoint-1-Small`` via ModularPipeline and write an MP4.

    ``model_id`` matches FastVideo SSIM tests so ``FASTVIDEO_SSIM_MODEL_ID`` can
    select Waypoint runs; the HF repo id is ``WAYPOINT_HF_MODEL_ID`` (default
    ``Overworld/Waypoint-1-Small``).
    """
    assert model_id == "Waypoint-1-Small-Diffusers"
    try:
        from diffusers.modular_pipelines import ModularPipeline  # noqa: F401
    except ImportError:
        pytest.skip("diffusers modular_pipelines not available (install diffusers>=0.36)")
    try:
        import tensordict  # noqa: F401
    except ImportError:
        pytest.skip(
            "tensordict required for Overworld modular code "
            "(uv pip install tensordict or uv pip install -e '.[test]')",
        )

    hf_home = _hf_home_dir()
    os.makedirs(hf_home, exist_ok=True)
    free_b = shutil.disk_usage(hf_home).free
    if free_b < _MIN_FREE_BYTES:
        need_gib = _MIN_FREE_BYTES / (1024**3)
        free_gib = free_b / (1024**3)
        pytest.skip(
            f"HF cache disk too small: {free_gib:.1f} GiB free under {hf_home!r}, "
            f"need ~{need_gib:.0f} GiB for Overworld+umt5-xl. "
            "Set HF_HOME to a large volume (e.g. export HF_HOME=/workspace/.cache/huggingface).",
        )

    hf_model_id = os.environ.get(
        "WAYPOINT_HF_MODEL_ID",
        "Overworld/Waypoint-1-Small",
    ).strip()
    mp4_env = os.environ.get("WAYPOINT_HF_TEST_MP4", "").strip()
    if mp4_env:
        out = Path(os.path.abspath(mp4_env))
        out.parent.mkdir(parents=True, exist_ok=True)
    else:
        out = tmp_path / "waypoint_hf_diffusers.mp4"

    _run_hf_waypoint_to_mp4(
        output_path=str(out),
        model_id=hf_model_id,
        prompt=DEFAULT_PROMPT,
        num_frames=NUM_FRAMES,
        seed=SEED,
        height=HEIGHT,
        width=WIDTH,
        fps=FPS,
        key_forward=KEY_FORWARD,
        seed_gray=128,
        dtype_bf16=True,
    )

    assert out.is_file(), f"expected MP4 at {out}"
    assert out.stat().st_size > 64, (
        f"WAYPOINT_HF_SMOKE_V2: MP4 empty/truncated ({out.stat().st_size} B)"
    )
    _assert_mp4_has_expected_frames(str(out), NUM_FRAMES)
