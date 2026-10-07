# SPDX-License-Identifier: Apache-2.0
"""Generate the repository-owned MiniMax H3 Ref2VA synthetic fixture."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import wave

import av
import numpy as np
from PIL import Image, ImageDraw, __version__ as PILLOW_VERSION

ROOT = Path(__file__).resolve().parent
VIDEO_PATH = ROOT / "targets" / "geometric_motion.mp4"
IMAGE_PATH = ROOT / "refs" / "images" / "geometric_reference.png"
AUDIO_PATH = ROOT / "refs" / "audios" / "synthetic_tones.wav"
PROVENANCE_PATH = ROOT / "PROVENANCE.json"

WIDTH = 224
HEIGHT = 128
FPS = 24
NUM_FRAMES = 124
SAMPLE_RATE = 32_000
NUM_SAMPLES = math.ceil(NUM_FRAMES * SAMPLE_RATE / FPS)
MEDIA_PATHS = (VIDEO_PATH, IMAGE_PATH, AUDIO_PATH)


def _pixels(frame_index: int) -> np.ndarray:
    x = np.arange(WIDTH, dtype=np.uint16)[None, :]
    y = np.arange(HEIGHT, dtype=np.uint16)[:, None]
    phase = frame_index * 3
    pixels = np.empty((HEIGHT, WIDTH, 3), dtype=np.uint8)
    pixels[..., 0] = ((x + phase) % 256).astype(np.uint8)
    pixels[..., 1] = ((2 * y + phase) % 256).astype(np.uint8)
    pixels[..., 2] = (((x // 2) + (y // 2) + phase) % 256).astype(np.uint8)

    image = Image.fromarray(pixels)
    draw = ImageDraw.Draw(image)
    for grid_x in range(0, WIDTH, 32):
        draw.line((grid_x, 0, grid_x, HEIGHT - 1), fill=(235, 240, 255), width=1)
    for grid_y in range(0, HEIGHT, 32):
        draw.line((0, grid_y, WIDTH - 1, grid_y), fill=(235, 240, 255), width=1)
    center_x = 20 + (frame_index * 3) % (WIDTH - 40)
    center_y = HEIGHT // 2 + round(22 * math.sin(frame_index * 2 * math.pi / NUM_FRAMES))
    draw.ellipse((center_x - 12, center_y - 12, center_x + 12, center_y + 12),
                 fill=(20, 70, 220), outline=(255, 255, 255), width=2)
    return np.asarray(image)


def _stereo_samples() -> np.ndarray:
    sample = np.arange(NUM_SAMPLES, dtype=np.float64)
    seconds = sample / SAMPLE_RATE
    phase = 2 * np.pi * (220.0 * seconds + 0.5 * 110.0 * seconds**2 / (NUM_SAMPLES / SAMPLE_RATE))
    left = 0.24 * np.sin(phase) + 0.08 * np.sin(2 * np.pi * 440.0 * seconds)
    right = 0.24 * np.sin(phase + np.pi / 3) + 0.08 * np.sin(2 * np.pi * 330.0 * seconds)
    return np.round(np.stack((left, right), axis=1) * 32767).astype(np.int16)


def _write_image() -> None:
    image = Image.fromarray(_pixels(0))
    image.save(IMAGE_PATH, format="PNG", compress_level=9, optimize=False)


def _write_audio(samples: np.ndarray) -> None:
    with wave.open(str(AUDIO_PATH), "wb") as output:
        output.setnchannels(2)
        output.setsampwidth(2)
        output.setframerate(SAMPLE_RATE)
        output.writeframes(samples.astype("<i2", copy=False).tobytes())


def _write_video(samples: np.ndarray) -> None:
    with av.open(str(VIDEO_PATH), "w") as container:
        container.metadata.clear()
        video_stream = container.add_stream("libx264", rate=FPS, options={"preset": "slow", "crf": "18"})
        video_stream.width = WIDTH
        video_stream.height = HEIGHT
        video_stream.pix_fmt = "yuv420p"
        video_stream.codec_context.thread_count = 1
        audio_stream = container.add_stream("aac", rate=SAMPLE_RATE)
        audio_stream.codec_context.layout = "stereo"

        for frame_index in range(NUM_FRAMES):
            frame = av.VideoFrame.from_ndarray(_pixels(frame_index), format="rgb24")
            frame.pts = frame_index
            container.mux(video_stream.encode(frame))
        container.mux(video_stream.encode())

        interleaved = samples.reshape(1, -1)
        audio_frame = av.AudioFrame.from_ndarray(interleaved, format="s16", layout="stereo")
        audio_frame.sample_rate = SAMPLE_RATE
        resampler = av.audio.resampler.AudioResampler(format="fltp", layout="stereo", rate=SAMPLE_RATE)
        pts = 0
        for frame in resampler.resample(audio_frame):
            frame.pts = pts
            pts += frame.samples
            container.mux(audio_stream.encode(frame))
        container.mux(audio_stream.encode())


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _relative(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def generate() -> None:
    for path in MEDIA_PATHS:
        path.parent.mkdir(parents=True, exist_ok=True)
    samples = _stereo_samples()
    _write_image()
    _write_audio(samples)
    _write_video(samples)
    provenance = {
        "schema_version": "fastvideo_synthetic_fixture_v1",
        "generator": "generate_fixture.py",
        "license": "Apache-2.0",
        "copyright": "FastVideo contributors",
        "third_party_content": False,
        "human_subjects": False,
        "description": "Deterministic geometric animation and mathematically synthesized tones.",
        "generation_environment": {
            "numpy": np.__version__,
            "pillow": PILLOW_VERSION,
            "pyav": av.__version__,
            "libav": {name: ".".join(str(part) for part in version)
                      for name, version in sorted(av.library_versions.items())},
            "video_encoder": "libx264",
            "audio_encoder": "aac",
        },
        "parameters": {
            "width": WIDTH,
            "height": HEIGHT,
            "fps": FPS,
            "num_frames": NUM_FRAMES,
            "sample_rate": SAMPLE_RATE,
        },
        "sha256": {_relative(path): _sha256(path) for path in MEDIA_PATHS},
    }
    PROVENANCE_PATH.write_text(json.dumps(provenance, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def verify() -> None:
    provenance = json.loads(PROVENANCE_PATH.read_text(encoding="utf-8"))
    expected = provenance.get("sha256", {})
    actual = {_relative(path): _sha256(path) for path in MEDIA_PATHS}
    if actual != expected:
        raise SystemExit(f"Synthetic fixture hash mismatch:\nexpected={expected}\nactual={actual}")
    print(json.dumps(actual, indent=2, sort_keys=True))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify", action="store_true", help="Verify checked-in media against PROVENANCE.json")
    args = parser.parse_args()
    if args.verify:
        verify()
    else:
        generate()
        verify()


if __name__ == "__main__":
    main()
