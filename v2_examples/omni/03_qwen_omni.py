#!/usr/bin/env python3
"""03 — Qwen-Omni: thinker → talker → vocoder (three separate experts).

The cascade topology (designv4 §9.5): three *distinct* experts on three loop types in one request —
the `thinker` (`ar_decode` → text), the `talker` (`ar_decode` → speech codec tokens, conditioned on the
thinker's tokens + hidden state), and the `vocoder` (`audio_decode` → waveform). Produces both text and
audio; the vocoder streams `AUDIO_CHUNK` units. This is the topological opposite of MoT: no shared
weights, but the same Card/Loop/Program vocabulary.

Run:  python3 v2_examples/omni/03_qwen_omni.py
"""
from __future__ import annotations

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

import numpy as np

from v2.recipes import build_omni_engine
from v2.core.request import OutputSpec, TaskType, make_request


def main() -> None:
    eng = build_omni_engine()
    out = eng.run(make_request(
        TaskType.T2A, "qwen-omni-tts", "Explain audio generation in fifteen words.",
        outputs=OutputSpec(stream={"audio": True})))

    samples = np.asarray(out.artifacts["audio"].samples)
    print("Qwen-Omni (thinker → talker → vocoder, three separate experts / three loops):")
    print(f"  thinker text   : {out.artifacts['text'].text}")
    print(f"  speech tokens  : {out.artifacts['speech_tokens'].tensor}")
    print(f"  audio waveform : shape={samples.shape} sample_rate={out.artifacts['audio'].sample_rate} "
          f"range=[{samples.min():.3f}, {samples.max():.3f}]")
    print(f"  vocoder chunks : {out.metrics['audio_chunks']:.0f} AUDIO_CHUNK units "
          f"(total streamed events incl. AR text: {out.metrics.get('stream_chunks', 0):.0f})")


if __name__ == "__main__":
    main()
