"""Input-shape helpers: paths → samples list.

The :class:`Evaluator` and every metric consume the same internal
representation — a ``list[dict]`` with one dict per sample.  Building
that list by hand is the largest source of ceremony in user scripts.
:func:`samples_from` is a pure function that turns path-style inputs
(generated videos / audio, optional paired references, optional
per-sample prompts / fps / metadata) into the canonical samples list,
ready to hand to :meth:`Evaluator.evaluate`.

Design rules:

* No Evaluator state, no metric introspection, no I/O beyond reading
  the prompts file when given.  Just dict assembly.
* "Extra" keys on a sample are free — metrics each read what they need
  and ignore the rest.  So one fat samples list naturally serves many
  metrics in one Evaluator.
* No "primary modality" axis.  Each modality is a named kwarg
  (``video=``, ``audio=``); pass whichever you have.  Both is fine.
* No ``mode`` axis.  The shape of the output is determined by
  cardinality: when ``|gen| == |ref|`` the samples list is pair-zipped;
  when ``|gen| < |ref|`` the unmatched references become role-tagged
  set samples for corpus-shaped metrics (FVD / FAD) without disturbing
  per-sample paired metrics (LPIPS / PSNR / SSIM / gt_optical_flow).
"""
from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import torch

from fastvideo.eval.types import Video

_VIDEO_EXTS = (".mp4", ".avi", ".mov", ".mkv", ".gif", ".webm")
_AUDIO_EXTS = (".wav", ".mp3", ".flac", ".m4a", ".ogg", ".opus")

PathSpec = str | Path | Iterable[str | Path]


def as_video(x: str | Path | torch.Tensor | Video) -> Video:
    """Coerce path/tensor/Video → :class:`Video` for the pool to decode.

    Path strings and :class:`pathlib.Path` become ``Video(source=str(x))``;
    the pool then calls :func:`load_video` on first use.  Tensors become
    ``Video(source=None, frames=x)`` — the pool sees ``.frames`` already
    populated and forwards untouched.  :class:`Video` instances pass through.
    """
    if isinstance(x, Video):
        return x
    if isinstance(x, str | Path):
        return Video(source=str(x))
    if isinstance(x, torch.Tensor):
        return Video(source=None, frames=x)
    raise TypeError(f"Cannot coerce {type(x).__name__} to Video")


def samples_from(
    *,
    # Modality inputs — pass whichever you have, in any combination.
    video: PathSpec | None = None,
    reference: PathSpec | None = None,
    audio: PathSpec | None = None,
    reference_audio: PathSpec | None = None,
    # Per-sample attachments.  Scalars broadcast to every sample; lists
    # / jsonl paths zip per-sample.
    text_prompt: str | None = None,
    text_prompts: str | Path | list[str] | None = None,
    fps: float | None = None,
    auxiliary_info: dict | list[dict] | None = None,
    # Catch-all for exotic metric inputs (physics_iq scenarios,
    # synthetic_optical_flow actions, ...).  Single dict broadcasts;
    # list of dicts zips.  Keys merge into each sample dict.
    extras: dict | list[dict] | None = None,
    # Sugar: pull audio off the video sources via PyAV.  Pass a path
    # to use a persistent on-disk cache, ``True`` for a system tempdir.
    extract_audio: bool | str | Path = False,
    extract_workers: int = 4,
) -> list[dict]:
    """Build a samples list from path-style inputs.

    Parameters
    ----------
    video, reference, audio, reference_audio :
        File path, directory of files (sorted by name), or any iterable
        of paths.  Pass whichever modalities apply to the metrics you
        plan to run — they attach to ``sample["video"]`` /
        ``sample["reference"]`` / ``sample["audio"]`` /
        ``sample["reference_audio"]`` respectively.  Video paths are
        wrapped in :class:`Video` so :class:`VideoPool` decodes them
        lazily in parallel; audio paths stay as strings (audio metrics
        each load with their own resample / preprocess).
    text_prompt :
        A single prompt string broadcast onto every sample.
    text_prompts :
        A list of strings (one per sample), or a path to a ``.jsonl`` /
        ``.json`` file containing per-sample prompts.
    fps :
        Scalar fps broadcast onto every sample.
    auxiliary_info :
        Single dict (broadcast) or list of dicts (zipped) for
        ``sample["auxiliary_info"]`` — vbench structured-prompt
        metrics read this.
    extras :
        Catch-all per-sample attachments.  Use for metric-specific keys
        the dedicated kwargs don't cover (``scenario``, ``view``,
        ``actions``, ``calibration``, ``reference_take2``, ...).  Pass
        a single dict to broadcast or a list-of-dicts to zip; the keys
        merge into each sample dict.
    extract_audio :
        If truthy, auto-extract audio from each ``video`` /
        ``reference`` source into ``.wav`` files via PyAV and attach
        the paths under ``sample["audio"]`` / ``sample["reference_audio"]``.
        Pass a path for a persistent cache, ``True`` for a tempdir.
        Skipped silently for videos with no audio stream; ignored
        wherever ``audio`` / ``reference_audio`` is already explicit.
    extract_workers :
        Parallel workers for ``extract_audio``.

    Returns
    -------
    list[dict]
        Canonical samples shape — hand directly to
        :meth:`Evaluator.evaluate`.

    Cardinality and shape
    ---------------------
    Let ``N = len(generated inputs)`` (the agreed length of whichever
    of ``video`` / ``audio`` you passed).  References are attached
    1:1 onto the first N samples; any extras (when ``|ref| > N``)
    become standalone role-tagged samples at the end of the list, so
    set metrics like FVD see the full reference corpus while per-sample
    paired metrics like LPIPS only run on the first N pairs.

    Notes
    -----
    "Missing" keys are simply absent from the sample dict.  Metrics
    handle them per their own contract (``sample.get(...)`` for
    optional, ``sample[...]`` raises for required, or
    :meth:`BaseMetric._skip` for opt-in skip behavior).  One fat samples
    list with many keys can serve many metrics — each reads its subset.
    """
    gen_video = _expand(video, _VIDEO_EXTS) if video is not None else None
    gen_audio = _expand(audio, _AUDIO_EXTS) if audio is not None else None
    ref_video = _expand(reference, _VIDEO_EXTS) if reference is not None else None
    ref_audio = _expand(reference_audio, _AUDIO_EXTS) if reference_audio is not None else None

    if gen_video is None and gen_audio is None:
        raise ValueError("samples_from: pass at least one of video= or audio=")

    # All "generated" inputs must agree on length — they describe the
    # same N samples in different modalities.
    n_candidates = [len(x) for x in (gen_video, gen_audio) if x is not None]
    if len({*n_candidates}) != 1:
        raise ValueError(f"samples_from: generated inputs have inconsistent lengths {n_candidates}")
    n = n_candidates[0]

    prompts = _broadcast(text_prompts if text_prompts is not None else text_prompt, n, loader=_load_prompts)
    aux = _broadcast(auxiliary_info, n)
    extras_per_sample = _broadcast(extras, n)
    if (text_prompts is not None) and (text_prompt is not None):
        raise ValueError("Pass either text_prompt (broadcast) or text_prompts (per-sample), not both.")

    samples: list[dict[str, Any]] = []
    for i in range(n):
        s: dict[str, Any] = {}
        if gen_video is not None:
            s["video"] = as_video(gen_video[i])
        if gen_audio is not None:
            s["audio"] = str(gen_audio[i])
        if ref_video is not None and i < len(ref_video):
            s["reference"] = as_video(ref_video[i])
        if ref_audio is not None and i < len(ref_audio):
            s["reference_audio"] = str(ref_audio[i])
        if prompts is not None:
            s["text_prompt"] = prompts[i]
        if fps is not None:
            s["fps"] = fps
        if aux is not None:
            s["auxiliary_info"] = aux[i]
        if extras_per_sample is not None:
            s.update(extras_per_sample[i])
        samples.append(s)

    # Unmatched references → role-tagged set samples.  Per-sample paired
    # metrics skip these via the worker's role-skip rule; set metrics
    # (FVD, FAD) accumulate the features.
    if ref_video is not None and len(ref_video) > n:
        for v in ref_video[n:]:
            samples.append({"video": as_video(v), "role": "reference"})
    if ref_audio is not None and len(ref_audio) > n:
        for a in ref_audio[n:]:
            samples.append({"audio": str(a), "role": "reference"})

    if extract_audio:
        _attach_extracted_audio(samples, extract_audio, extract_workers)
    return samples


# ------------------------------------------------------------------
# Internal helpers
# ------------------------------------------------------------------


def _expand(spec: PathSpec, exts: tuple[str, ...]) -> list[Path]:
    """Resolve a path-style input into a sorted list of file paths.

    * dir → sorted list of files whose extension is in *exts*.
    * file → ``[Path(spec)]``.
    * iterable → ``[Path(x) for x in spec]`` (no sort, caller's order).
    """
    if isinstance(spec, str | Path):
        p = Path(spec)
        if p.is_dir():
            files = sorted(f for f in p.iterdir() if f.suffix.lower() in exts)
            if not files:
                raise FileNotFoundError(f"No files with extension in {exts} under {p}")
            return files
        if p.is_file():
            return [p]
        raise FileNotFoundError(f"{p} is neither a file nor a directory.")
    out = [x if isinstance(x, Path) else Path(x) for x in spec]
    if not out:
        raise ValueError("Empty iterable passed where a non-empty path list was expected.")
    return out


def _broadcast(value: Any, n: int, *, loader: Any = None) -> list[Any] | None:
    """Turn *value* into a length-*n* list.

    Single dict, scalar (str / int / float), or ``None`` → broadcast.
    Path to a ``.json`` / ``.jsonl`` file → load via *loader* (when
    given) and treat as per-sample.  Iterable (list, tuple, ...) → zip
    per-sample (must already be length *n*).
    """
    if value is None:
        return None
    if isinstance(value, str | int | float):
        if loader is not None and isinstance(value, str) and Path(value).suffix in (".jsonl", ".json"):
            items = loader(value)
        else:
            return [value] * n
    elif isinstance(value, Path):
        if loader is None:
            raise ValueError(f"Got a Path ({value}) where no loader is available")
        items = loader(value)
    elif isinstance(value, dict):
        return [value] * n
    else:
        items = list(value)
    if len(items) != n:
        raise ValueError(f"per-sample value has {len(items)} entries; need {n} to match generated")
    return items


def _attach_extracted_audio(
    samples: list[dict],
    extract_audio: bool | str | Path,
    workers: int,
) -> None:
    """For each sample with a video source but no audio key set, run
    PyAV audio extraction in parallel and attach the resulting .wav path.

    Mutates *samples* in place.  Silently skips videos whose source has
    no audio stream (the metric will skip on the missing key just as it
    would have without auto-extract).  Mirrors the logic onto
    ``sample["reference"]`` → ``sample["reference_audio"]``.
    """
    from concurrent.futures import ThreadPoolExecutor
    import tempfile
    from fastvideo.eval.io.audio import extract_audio_track, NoAudioStreamError

    if isinstance(extract_audio, str | Path):
        out_dir: Path = Path(extract_audio)
    else:
        out_dir = Path(tempfile.mkdtemp(prefix="fv_extracted_audio_"))

    # Build the work list — (sample_idx, dest_audio_key, source_path)
    work: list[tuple[int, str, str]] = []
    for i, s in enumerate(samples):
        for vkey, akey in (("video", "audio"), ("reference", "reference_audio")):
            if akey in s:
                continue
            v = s.get(vkey)
            if isinstance(v, Video) and v.source is not None:
                work.append((i, akey, str(v.source)))
    if not work:
        return

    def _do(item: tuple[int, str, str]) -> tuple[int, str, str | None]:
        idx, akey, src = item
        try:
            return idx, akey, str(extract_audio_track(src, output_dir=out_dir))
        except NoAudioStreamError:
            return idx, akey, None

    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        for idx, akey, audio_path in pool.map(_do, work):
            if audio_path is not None:
                samples[idx][akey] = audio_path


def _load_prompts(path: str | Path) -> list[str]:
    """Load per-sample text prompts from a ``.jsonl`` or ``.json`` file.

    * ``.jsonl`` — one JSON value per line.  String values are taken
      as-is; objects are searched for a ``"prompt"`` or ``"text_prompt"``
      key.
    * ``.json`` — a top-level list (same per-item rules).
    """
    p = Path(path)
    if p.suffix == ".jsonl":
        items = [json.loads(line) for line in p.read_text().splitlines() if line.strip()]
    elif p.suffix == ".json":
        items = json.loads(p.read_text())
        if not isinstance(items, list):
            raise ValueError(f"{p}: expected a top-level JSON list of prompts.")
    else:
        raise ValueError(f"{p}: text prompts file must be .jsonl or .json, got {p.suffix!r}")
    out: list[str] = []
    for item in items:
        if isinstance(item, str):
            out.append(item)
        elif isinstance(item, dict):
            v = item.get("prompt") or item.get("text_prompt")
            if not isinstance(v, str):
                raise ValueError(f"{p}: dict entry missing 'prompt'/'text_prompt' string field: {item!r}")
            out.append(v)
        else:
            raise ValueError(f"{p}: prompt entries must be strings or dicts, got {type(item).__name__}")
    return out
