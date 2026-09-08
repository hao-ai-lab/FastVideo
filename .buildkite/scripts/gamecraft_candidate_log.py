#!/usr/bin/env python3
"""Emit and recover a small GameCraft MP4 through a Buildkite job log."""

from __future__ import annotations

import argparse
import base64
import binascii
import hashlib
import html
import json
import re
import sys
from pathlib import Path

MARKER = "FV_GAMECRAFT_FA4_MP4"
CHUNK_BYTES = 3 * 1024
MAX_MEDIA_BYTES = 1024 * 1024


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def emit(media_path: Path) -> None:
    if media_path.suffix.lower() != ".mp4":
        raise ValueError(f"Candidate must be an MP4: {media_path}")

    data = media_path.read_bytes()
    size = len(data)
    if not 0 < size <= MAX_MEDIA_BYTES:
        raise ValueError(f"Candidate size must be between 1 and {MAX_MEDIA_BYTES} bytes; got {size}")

    digest = _sha256(data)
    chunks = [data[offset:offset + CHUNK_BYTES] for offset in range(0, size, CHUNK_BYTES)]
    print(
        f"{MARKER}_BEGIN version=1 sha256={digest} size={size} "
        f"chunks={len(chunks)} chunk_bytes={CHUNK_BYTES}",
        flush=True,
    )
    for index, chunk in enumerate(chunks):
        encoded = base64.b64encode(chunk).decode("ascii")
        print(f"{MARKER}_CHUNK index={index:06d} data={encoded}", flush=True)
    print(
        f"{MARKER}_END version=1 sha256={digest} size={size} chunks={len(chunks)}",
        flush=True,
    )


def _buildkite_output(text: str) -> str:
    """Unwrap Buildkite's public JSON and reverse its HTML entity escaping."""
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        return html.unescape(text)
    if isinstance(payload, dict) and isinstance(payload.get("output"), str):
        return html.unescape(payload["output"])
    return html.unescape(text)


def decode(log_text: str) -> tuple[bytes, str]:
    log_text = _buildkite_output(log_text)
    begin_pattern = re.compile(
        rf"{MARKER}_BEGIN version=1 sha256=([0-9a-f]{{64}}) size=([0-9]+) "
        rf"chunks=([0-9]+) chunk_bytes=([0-9]+)"
    )
    end_pattern = re.compile(
        rf"{MARKER}_END version=1 sha256=([0-9a-f]{{64}}) size=([0-9]+) chunks=([0-9]+)"
    )
    chunk_pattern = re.compile(rf"{MARKER}_CHUNK index=([0-9]{{6}}) data=([A-Za-z0-9+/]+={{0,2}})")

    begin_matches = begin_pattern.findall(log_text)
    end_matches = end_pattern.findall(log_text)
    if len(begin_matches) != 1 or len(end_matches) != 1:
        raise ValueError(
            "Expected exactly one candidate envelope; "
            f"found {len(begin_matches)} begin and {len(end_matches)} end markers"
        )

    digest, size_text, count_text, chunk_bytes_text = begin_matches[0]
    end_digest, end_size_text, end_count_text = end_matches[0]
    if (digest, size_text, count_text) != (end_digest, end_size_text, end_count_text):
        raise ValueError("Candidate begin/end metadata does not match")

    size = int(size_text)
    count = int(count_text)
    chunk_bytes = int(chunk_bytes_text)
    if not 0 < size <= MAX_MEDIA_BYTES:
        raise ValueError(f"Candidate size is outside the accepted range: {size}")
    if chunk_bytes != CHUNK_BYTES:
        raise ValueError(f"Unexpected chunk size: {chunk_bytes}")

    matches = chunk_pattern.findall(log_text)
    if len(matches) != count:
        raise ValueError(f"Expected {count} candidate chunks; found {len(matches)}")

    encoded_chunks: dict[int, str] = {}
    for index_text, encoded in matches:
        index = int(index_text)
        if index in encoded_chunks:
            raise ValueError(f"Duplicate candidate chunk: {index}")
        encoded_chunks[index] = encoded
    if sorted(encoded_chunks) != list(range(count)):
        raise ValueError("Candidate chunk sequence is incomplete or out of range")

    try:
        data = b"".join(base64.b64decode(encoded_chunks[index], validate=True) for index in range(count))
    except binascii.Error as error:
        raise ValueError(f"Candidate chunk is not valid base64: {error}") from error
    if len(data) != size:
        raise ValueError(f"Decoded candidate size mismatch: expected {size}, got {len(data)}")
    actual_digest = _sha256(data)
    if actual_digest != digest:
        raise ValueError(f"Decoded candidate sha256 mismatch: expected {digest}, got {actual_digest}")
    return data, digest


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    emit_parser = subparsers.add_parser("emit", help="Encode one MP4 to stdout between strict log markers")
    emit_parser.add_argument("media_path", type=Path)

    decode_parser = subparsers.add_parser("decode", help="Recover and verify one MP4 from a raw or public JSON log")
    decode_parser.add_argument("--input", type=Path, help="Log file (default: stdin)")
    decode_parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    if args.command == "emit":
        emit(args.media_path)
        return 0

    log_text = args.input.read_text() if args.input else sys.stdin.read()
    data, digest = decode(log_text)
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite existing output: {args.output}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(data)
    print(f"Recovered {len(data)} bytes with sha256={digest} to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
