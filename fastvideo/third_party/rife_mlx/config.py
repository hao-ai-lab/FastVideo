"""Version registry — pinned config for the RIFE checkpoint(s) we port.

Source of truth: the bundled `IFNet_HDv3.py` + `flownet.pkl` that ship together
per version (Google Drive), NOT the GitHub repo (which omits the version arch).
CONFIRM gate #3: pin the exact arch from the downloaded package, never guess.

This round pins **RIFE 4.25** (upstream's recommended default; "anime scenes
significantly improved"). Arch PINNED in P1 from the bundled IFNet_HDv3.py +
flownet.pkl shapes: 5 IFBlocks (the 4.25 "more flow blocks" note), channels
[192,128,96,64,32], in_planes [15,28,28,28,28] (blocks 1-4 add the running
4-ch flow inside IFBlock.forward), scale_list [16,8,4,2,1] (÷ the --scale knob).
Activation is LeakyReLU(0.2). All F.interpolate use align_corners=False; warp
grid_sample uses align_corners=True. 40 teacher/caltime keys are train-only.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class VersionConfig:
    name: str               # internal key
    hf_name: str            # published HF repo name (mlx-community/<hf_name>)
    gdrive_id: str          # per-version package (model .py + flownet.pkl)
    # arch params — PINNED from bundled v4.25 IFNet_HDv3.py + flownet.pkl (P1)
    scale_list: tuple[float, ...] = (16, 8, 4, 2, 1)
    block_channels: tuple[int, ...] = (192, 128, 96, 64, 32)
    block_in_planes: tuple[int, ...] = (15, 28, 28, 28, 28)
    pad_to: int = 64  # 5 blocks downsample; pad H,W to a multiple of 64
    # warp / interpolate semantics (verified from source)
    grid_align_corners: bool = True      # warp grid_sample
    grid_padding_mode: str = "border"
    interp_align_corners: bool = False   # F.interpolate in IFBlock


VERSIONS: dict[str, VersionConfig] = {
    "4.25": VersionConfig(
        name="4.25",
        hf_name="RIFE-4.25",
        gdrive_id="1ZKjcbmt1hypiFprJPIKW0Tt0lr_2i7bg",
    ),
}

DEFAULT_VERSION = "4.25"
