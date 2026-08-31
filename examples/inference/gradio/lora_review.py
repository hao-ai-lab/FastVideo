# SPDX-License-Identifier: Apache-2.0
"""Review whether a FastH3 adapter reproduces the checkpoint it was extracted from.

The question this page answers is not "is the video good" but "does base + adapter land
where the real checkpoint lands". So each row is one checkpoint, and the two players in
it are the checkpoint itself and base MiniMax-H3 with that checkpoint's adapter merged
in. They share a seed and a prompt, so anything you can see between them is the
adapter's approximation error and nothing else.

The base model at four steps sits at the top as the floor. It is not distilled, so it
should look clearly worse than everything below it -- if an adapter row looks like the
floor instead of like its checkpoint, the adapter did not land.

    python examples/inference/gradio/lora_review.py --runs /path/to/lora_review

Expects one directory per arm, each holding ``<index>_<case_id>.mp4``:

    <runs>/base/            <runs>/v1-true/     <runs>/v1-lora-r64/    ...
"""
from __future__ import annotations

import argparse
import json
import subprocess
from functools import lru_cache
from pathlib import Path

import gradio as gr

FLOOR_ARM = "base"


def discover_pairs(arms: list[str]) -> tuple[list[tuple[str, str, str]], list[str]]:
    """Split the arms present into checkpoint/adapter pairs and everything else.

    Pairs are found by name -- ``<x>-true`` next to ``<x>-lora-<rank>`` -- rather than
    listed, so adding an arm to the render directory is enough to get it on the page.
    Arms that pair with nothing (a third-party adapter with no checkpoint to compare
    against) still get shown, on their own row, instead of being silently dropped.
    """
    pairs, used = [], set()
    for arm in sorted(arms):
        if not arm.endswith("-true"):
            continue
        stem = arm[:-len("-true")]
        partner = next((a for a in arms if a.startswith(f"{stem}-lora")), None)
        if partner is None:
            continue
        pairs.append((stem, arm, partner))
        used.update({arm, partner})
    standalone = [a for a in sorted(arms) if a not in used and a != FLOOR_ARM]
    return pairs, standalone


def probe(path: Path) -> str:
    """`WxH · Nf · Ds · MiB`, so a truncated or mis-sized render is visible as text."""
    try:
        out = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
             "stream=width,height,nb_read_packets,duration", "-count_packets", "-of", "json",
             str(path)],
            capture_output=True, text=True, check=True).stdout
        stream = json.loads(out)["streams"][0]
        frames = stream.get("nb_read_packets", "?")
        duration = float(stream.get("duration", 0) or 0)
        return (f"{stream['width']}x{stream['height']} · {frames}f · {duration:.1f}s · "
                f"{path.stat().st_size / 2**20:.1f} MiB")
    except (subprocess.CalledProcessError, KeyError, IndexError, json.JSONDecodeError):
        return f"{path.stat().st_size / 2**20:.1f} MiB"


class Runs:
    """Which prompts rendered, and where each arm's clip for them lives."""

    def __init__(self, runs_dir: Path, prompts_file: Path | None) -> None:
        self.root = runs_dir
        self.arms = sorted(d.name for d in runs_dir.iterdir() if d.is_dir())
        self.prompts: dict[str, str] = {}
        if prompts_file and prompts_file.exists():
            with prompts_file.open() as handle:
                for index, line in enumerate(handle):
                    line = line.strip()
                    if line:
                        self.prompts[f"{index:03d}"] = json.loads(line).get("prompt", "")

        self.clips: dict[str, dict[str, Path]] = {}
        for arm in self.arms:
            for mp4 in sorted((runs_dir / arm).glob("*.mp4")):
                self.clips.setdefault(mp4.stem.split("_")[0], {})[arm] = mp4
        if not self.clips:
            raise SystemExit(f"no clips under {runs_dir}")

    def label(self, index: str) -> str:
        head = " ".join(self.prompts.get(index, "").split())[:90]
        return f"[{index}] {head}..." if head else f"[{index}]"

    def by_label(self, label: str) -> str:
        return next(i for i in self.clips if self.label(i) == label)


def build(runs: Runs, height: int) -> gr.Blocks:

    @lru_cache(maxsize=512)
    def cached_probe(path: str) -> str:
        return probe(Path(path))

    def player_update(index: str, arm: str, prefix: str):
        mp4 = runs.clips.get(index, {}).get(arm)
        if mp4 is None:
            return gr.update(value=None, label=f"{prefix} — not rendered")
        return gr.update(value=str(mp4), label=f"{prefix} — {cached_probe(str(mp4))}")

    with gr.Blocks(title="FastH3 adapter review") as demo:
        gr.Markdown(
            "# FastH3 adapter review\n"
            "Each row is one checkpoint: **left** is the real checkpoint, **right** is base "
            "MiniMax-H3 with that checkpoint's rank-64 adapter merged in. Same prompt, same "
            "seed, same sampler. Differences between the two are the adapter's approximation "
            "error.\n\n"
            "The top player is undistilled base MiniMax-H3 at four steps — the floor. Every "
            "row below it should look clearly better than that; an adapter that landed looks "
            "like its own left-hand player, not like the floor.")

        prompt_dd = gr.Dropdown(choices=[runs.label(i) for i in runs.clips],
                                value=runs.label(next(iter(runs.clips))),
                                label="Prompt")
        prompt_box = gr.Textbox(label="Prompt", lines=4, max_lines=6, interactive=False, show_copy_button=True)

        with gr.Row():
            floor = gr.Video(label="base MiniMax-H3, 4 steps (floor)", height=height, loop=True,
                             autoplay=False, interactive=False)

        pairs, standalone = discover_pairs(runs.arms)
        players: list[tuple[gr.Video, str, str]] = []
        for row_label, true_arm, lora_arm in pairs:
            gr.Markdown(f"### {row_label}")
            with gr.Row():
                left = gr.Video(label=f"{row_label} — checkpoint", height=height, loop=True,
                                autoplay=False, interactive=False)
                right = gr.Video(label=f"{row_label} — base + adapter", height=height, loop=True,
                                 autoplay=False, interactive=False)
            players.append((left, true_arm, "checkpoint"))
            players.append((right, lora_arm, "base + adapter"))

        if standalone:
            gr.Markdown("### Other adapters (no matching checkpoint to compare against)")
            with gr.Row():
                for arm in standalone:
                    players.append((gr.Video(label=arm, height=height, loop=True, autoplay=False,
                                             interactive=False), arm, arm))

        def show(label: str):
            index = runs.by_label(label)
            return [
                player_update(index, FLOOR_ARM, "base, 4 steps"),
                gr.update(value=runs.prompts.get(index, "")),
                *[player_update(index, arm, prefix) for _, arm, prefix in players],
            ]

        outputs = [floor, prompt_box, *[p for p, _, _ in players]]
        gr.on(triggers=[prompt_dd.change, demo.load], fn=show, inputs=prompt_dd, outputs=outputs)

    return demo


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--runs", default="/mnt/lustre/vlm-s4duan/arena_arms/lora_review")
    parser.add_argument("--prompts-file", default="/mnt/lustre/vlm-s4duan/FastVideo/prompts.jsonl")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=7865)
    parser.add_argument("--share", action="store_true")
    parser.add_argument("--video-height", type=int, default=420)
    args = parser.parse_args()

    runs = Runs(Path(args.runs).resolve(), Path(args.prompts_file))
    print(f"arms: {runs.arms}")
    print(f"prompts with output: {sorted(runs.clips)}")
    build(runs, args.video_height).launch(server_name=args.host, server_port=args.port,
                                          share=args.share, allowed_paths=[str(runs.root)])


if __name__ == "__main__":
    main()
