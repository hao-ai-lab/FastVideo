# SPDX-License-Identifier: Apache-2.0
"""
Run official Overworld Waypoint-1-Small with diffusers (ModularPipeline).

Use this to compare sharpness with our FastVideo port. If diffusers output
is also blurry, the cause may be model/schedule/resolution; if diffusers is
sharp, the bug is in the FastVideo port.

Requirements (separate from FastVideo):
  pip install "torch>=2.9.0" "diffusers>=0.36.0" "transformers>=4.57.1" \
    einops tensordict regex ftfy imageio imageio-ffmpeg tqdm

Usage:
  python examples/inference/basic/basic_waypoint_diffusers_baseline.py
  On RunPod (to avoid disk quota): set HF_HOME, TMPDIR to /workspace (see below), then run.

  If the 12.5G transformer download keeps failing partway, pre-download with resume:
    huggingface-cli download Overworld/Waypoint-1-Small --local-dir /workspace/models/Waypoint-1-Small --resume-download
  Then: export WAYPOINT_DIFFUSERS_MODEL_PATH=/workspace/models/Waypoint-1-Small
  and run this script (it will load from that path).

Output: waypoint_diffusers_baseline.mp4 (and optional first-frame image)
"""

import os
import random
from dataclasses import dataclass, field
from typing import Set, Tuple

# Optional: reduce compile noise on RunPod
os.environ.setdefault("TORCH_COMPILE_DISABLE", "0")
os.environ.setdefault("TORCHDYNAMO_DISABLE", "0")

# RunPod: force HF cache and temp to /workspace to avoid disk quota (errno 122).
# Must be set before any huggingface/diffusers import. Use run_waypoint_diffusers_baseline.sh
# so env is set before Python starts if quota still occurs.
_workspace = "/workspace"
if os.path.isdir(_workspace):
    _hf = os.path.join(_workspace, ".cache", "huggingface")
    _hub = os.path.join(_hf, "hub")
    _xet = os.path.join(_hf, "xet")
    _tmp = os.path.join(_workspace, "tmp")
    for _d in (_hub, _xet, _tmp):
        os.makedirs(_d, exist_ok=True)
    os.environ["HF_HOME"] = _hf
    os.environ["HUGGINGFACE_HUB_CACHE"] = _hub
    os.environ["HF_HUB_CACHE"] = _hub
    os.environ["HF_XET_CACHE"] = _xet
    os.environ["TMPDIR"] = _tmp
    os.environ["HF_HUB_DISABLE_XET"] = "1"

def main():
    import torch
    from tqdm import tqdm

    try:
        from diffusers.modular_pipelines import ModularPipeline, ModularPipelineBlocks
        from diffusers.utils import load_image, export_to_video
    except ImportError as e:
        raise SystemExit(
            "Install diffusers and deps first, e.g.:\n"
            "  pip install 'diffusers>=0.36.0' 'transformers>=4.57.1' "
            "einops tensordict imageio imageio-ffmpeg tqdm"
        ) from e

    @dataclass
    class CtrlInput:
        button: Set[int] = field(default_factory=set)
        mouse: Tuple[float, float] = (0.0, 0.0)

    # Similar to RunPod: hold W (forward) for comparison
    KEY_W = 17
    ctrl = lambda: CtrlInput(button={KEY_W}, mouse=(0.0, 0.0))

    model_id = os.environ.get(
        "WAYPOINT_DIFFUSERS_MODEL_PATH", "Overworld/Waypoint-1-Small"
    )
    prompt = "A first-person gameplay video exploring a stylized world."
    num_frames = 16  # Short clip for quick comparison
    out_path = "waypoint_diffusers_baseline.mp4"

    print(f"Loading {model_id}...")
    # WorldEngineBlocks is custom (repo modular_blocks.py); load blocks with
    # trust_remote_code then init_pipeline so we don't look it up in diffusers.
    blocks = ModularPipelineBlocks.from_pretrained(model_id, trust_remote_code=True)
    pipe = blocks.init_pipeline(pretrained_model_name_or_path=model_id)
    pipe.load_components(
        device_map="cuda",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )
    if pipe.transformer is None:
        raise RuntimeError(
            "Failed to load transformer (and possibly other components). "
            "If the download stops at ~59%% and then .bin 404: pre-download with resume, then load from local:\n"
            "  huggingface-cli download Overworld/Waypoint-1-Small --local-dir /workspace/models/Waypoint-1-Small --resume-download\n"
            "  export WAYPOINT_DIFFUSERS_MODEL_PATH=/workspace/models/Waypoint-1-Small\n"
            "  python examples/inference/basic/basic_waypoint_diffusers_baseline.py\n"
            "Also set HF_HOME and TMPDIR to /workspace to avoid disk quota."
        )
    pipe.transformer.apply_inference_patches()
    # Optional: compile (can skip on first run to avoid long compile)
    # pipe.transformer.compile(fullgraph=True, mode="max-autotune", dynamic=False)
    # pipe.vae.bake_weight_norm()
    # pipe.vae.compile(fullgraph=True, mode="max-autotune")

    # First frame: no image = random start; or use an image for seeded start
    print("Generating first frame...")
    state = pipe(
        prompt=prompt,
        image=None,  # no seed image -> random first frame
        button=ctrl().button,
        mouse=ctrl().mouse,
        output_type="pil",
    )
    outputs = [state.values["images"]]
    state.values["image"] = None

    print(f"Generating {num_frames - 1} more frames...")
    for _ in tqdm(range(1, num_frames)):
        state = pipe(
            state,
            prompt=prompt,
            button=ctrl().button,
            mouse=ctrl().mouse,
            output_type="pil",
        )
        outputs.append(state.values["images"])

    export_to_video(outputs, out_path, fps=60)
    print(f"Saved: {os.path.abspath(out_path)}")
    print("Compare with FastVideo RunPod output to see if blur is port-specific.")


if __name__ == "__main__":
    main()
