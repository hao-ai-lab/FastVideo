"""VideoAlign runtime gate — the vendored PR #1476 reward stack scoring
fixed synthetic inputs on the real KwaiVGI/VideoReward checkpoint.

Code identity with the authority is proven byte-level in
``fastvideo2/tests/test_videoalign.py`` (vendored runtime sha256 == PR
blobs); this gate proves the stack RUNS on the cluster and is
deterministic: MQ/VQ/TA on 2 seeded videos + 1 image, scored twice —
run2 must match run1 exactly, all scores finite. First run writes the
scores as goldens (evidence/goldens/videoalign-pr1476/); later runs also
compare against the stored goldens.

Run (cluster): VIDEOALIGN_CHECKPOINT_PATH=... python -m
    fastvideo2.train.gates.videoalign_anchor
(without the env var it snapshot-downloads KwaiVGI/VideoReward)
"""
from __future__ import annotations

import json
import os
import sys


def _gold_dir() -> str:
    here = os.path.dirname(os.path.abspath(__file__))
    return os.path.normpath(os.path.join(here, "..", "..", "evidence",
                                         "goldens", "videoalign-pr1476"))


def fixed_inputs():
    """Deterministic media: low-frequency moving gradients (stable under
    the mp4v encode both sides share) + one static image."""
    import numpy as np
    import torch
    rng = np.random.default_rng(1476)
    t, h, w = 16, 64, 96
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    vids = []
    for v in range(2):
        phase = rng.uniform(0, 3.14, size=3)
        speed = rng.uniform(0.5, 2.0, size=3)
        frames = np.zeros((t, h, w, 3), dtype=np.float32)
        for i in range(t):
            for c in range(3):
                frames[i, :, :, c] = 0.5 + 0.5 * np.sin(
                    xx / w * 6.28 * (c + 1) + yy / h * 3.14
                    + phase[c] + speed[c] * i * 0.3)
        vids.append(frames)
    img = np.zeros((1, h, w, 3), dtype=np.float32)
    img[..., 0] = yy / h
    img[..., 1] = xx / w
    img[..., 2] = 0.25
    media = torch.from_numpy(np.stack(vids + [np.repeat(img, t, axis=0)]))
    prompts = ["a red fox running through fresh snow",
               "timelapse of a city skyline at night",
               "a colorful gradient test pattern"]
    return media, prompts  # [3, T, H, W, C] float in [0,1]


def main() -> int:
    import numpy as np
    import torch

    if not os.environ.get("VIDEOALIGN_CHECKPOINT_PATH"):
        from huggingface_hub import snapshot_download
        os.environ["VIDEOALIGN_CHECKPOINT_PATH"] = snapshot_download(
            "KwaiVGI/VideoReward", token=False)

    from fastvideo2.train.videoalign import (VideoAlignMotionQualityScorer,
                                             VideoAlignTextAlignmentScorer,
                                             VideoAlignVisualQualityScorer)

    media, prompts = fixed_inputs()
    scorers = {"MQ": VideoAlignMotionQualityScorer(),
               "VQ": VideoAlignVisualQualityScorer(),
               "TA": VideoAlignTextAlignmentScorer()}

    def run_once() -> dict[str, list[float]]:
        return {k: [round(float(v), 6) for v in s(media, prompts).tolist()]
                for k, s in scorers.items()}

    r1, r2 = run_once(), run_once()
    failed = []
    for k in scorers:
        det = r1[k] == r2[k]
        finite = all(np.isfinite(v) for v in r1[k])
        print(f"  {k}: {r1[k]} deterministic={det} finite={finite}")
        if not det:
            failed.append(f"{k}.determinism")
        if not finite:
            failed.append(f"{k}.finite")

    gold = _gold_dir()
    gold_file = os.path.join(gold, "scores.json")
    if os.path.exists(gold_file):
        with open(gold_file) as f:
            ref = json.load(f)["scores"]
        for k in scorers:
            if r1[k] != ref[k]:
                failed.append(f"{k}.vs_golden")
                print(f"  {k} vs golden MISMATCH: {r1[k]} != {ref[k]}")
    else:
        os.makedirs(gold, exist_ok=True)
        with open(gold_file, "w") as f:
            json.dump({"scores": r1,
                       "checkpoint": os.environ["VIDEOALIGN_CHECKPOINT_PATH"],
                       "authority": "PR #1476 maint/pr1476-runtime-compat "
                                    "@518aeab0b (vendored byte-identical, "
                                    "see test_videoalign.py)",
                       "torch": torch.__version__,
                       "gpu": torch.cuda.get_device_name(0)}, f, indent=2)
        print(f"  goldens written -> {gold_file}")

    verdict = "PASS" if not failed else f"FAIL ({', '.join(failed)})"
    print(f"anchor.videoalign-pr1476: {verdict}")
    from fastvideo2.verify import GateResult, append_ledger, env_fingerprint
    from fastvideo2.wan21.card import WAN21_T2V_1_3B as card
    append_ledger([GateResult(gate="anchor.videoalign-pr1476",
                              status="pass" if not failed else "fail",
                              model_id=card.model_id,
                              card_digest=card.digest(),
                              metrics={f"{k}.v{i}": v for k in r1
                                       for i, v in enumerate(r1[k])},
                              tolerances={"determinism": "exact",
                                          "vs_golden": "exact"},
                              env=env_fingerprint(),
                              detail="MQ/VQ/TA on 3 fixed media, "
                                     "KwaiVGI/VideoReward, use_norm=True")])
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
