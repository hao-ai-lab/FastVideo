# SPDX-License-Identifier: Apache-2.0
"""Run one Kandinsky6 SR bundle end to end and save the output frames (driver of the pipeline parity test).

Run in a subprocess per implementation, because the current and the previous implementation are both the package
``fastvideo``.  ``--legacy-steps`` passes the step count the way the previous implementation took it (``sr_num_steps``
Euler grid points instead of ``num_inference_steps`` DiT calls).
"""
import argparse
import json
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--cases", required=True, help="JSON list of {name, video, scale, steps}")
    parser.add_argument("--out", required=True)
    parser.add_argument("--legacy-steps", action="store_true")
    args = parser.parse_args()

    import fastvideo
    from fastvideo import VideoGenerator

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    generator = VideoGenerator.from_pretrained(args.model, num_gpus=1, dit_cpu_offload=False, vae_cpu_offload=False,
                                               dit_layerwise_offload=False)
    for case in json.loads(args.cases):
        kwargs = {"video_path": case["video"], "sr_resolution_scale": case["scale"]}
        if case.get("steps") is not None:
            if args.legacy_steps:
                kwargs["sr_num_steps"] = case["steps"] + 1
            else:
                kwargs["num_inference_steps"] = case["steps"]
        result = generator.generate_video(None, seed=42, save_video=False, return_frames=True, **kwargs)
        np.save(out / f"{case['name']}.npy", np.stack([np.asarray(frame) for frame in result["frames"]]))
    generator.shutdown()
    (out / "fastvideo_path.txt").write_text(str(Path(fastvideo.__file__).resolve().parent))


if __name__ == "__main__":
    main()
