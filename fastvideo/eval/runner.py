"""End-to-end eval orchestrator.

:class:`EvalRunner` is the top-level glue that connects a prompt dataset,
an optional :class:`fastvideo.VideoGenerator`, and an
:class:`fastvideo.eval.Evaluator` into a single generate-then-score
pipeline. It owns:

* the videos directory and filename convention,
* per-run config (fps, seed, n_samples override, generation kwargs),
* the generation loop and its manifest,
* loading videos back from disk into evaluator-ready kwargs.

The Evaluator below it stays focused on scoring tensors. Multi-GPU
fan-out is delegated to the Evaluator (which holds N
:class:`fastvideo.eval.worker.EvalWorker` replicas internally) — the
runner just drives the work.

Three named constructors cover the common shapes::

    # benchmark: prompts → generate → score
    EvalRunner.from_dataset(dataset, videos_dir, generator=gen,
                            num_gpus=4).run()

    # score-only: prompts → score existing videos
    EvalRunner.from_dataset(dataset, videos_dir, num_gpus=4).score()

    # quick multi-GPU scoring of a flat folder
    EvalRunner.from_videos(["a.mp4", "b.mp4"], num_gpus=2).score()

    # power user: pre-built eval kwargs
    EvalRunner.from_samples(my_kwargs_list, num_gpus=4).score()
"""
from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Iterable

from fastvideo.eval.evaluator import Evaluator, create_evaluator
from fastvideo.eval.types import EvalResult

if TYPE_CHECKING:
    from fastvideo.eval.datasets.base import PromptDataset

log = logging.getLogger(__name__)

# Filesystem-unsafe characters mirrored from VideoGenerator's output-path
# sanitizer so on-disk filenames match what the generator writes.
_INVALID_CHARS = re.compile(r'[\\/:*?"<>|]')


def sanitize_prompt(prompt: str, max_len: int = 100) -> str:
    """Default prompt → safe filename stem."""
    s = _INVALID_CHARS.sub("", prompt[:max_len]).strip().strip(".")
    return re.sub(r"\s+", " ", s) or "output"


def _default_filename(row: dict, idx: int) -> str:
    """``<sanitized-prompt>-<idx>.mp4`` — VBench-style."""
    return f"{sanitize_prompt(row['prompt'])}-{idx}.mp4"


def _default_eval_kwargs(row: dict, video_path: Path, *,
                         fps: float) -> dict[str, Any]:
    """Build evaluator kwargs from a sample row + a video on disk."""
    from fastvideo.eval.io.video import load_video

    video = load_video(str(video_path))            # (T, C, H, W) in [0, 1]
    kwargs: dict[str, Any] = {
        "video": video.unsqueeze(0),               # (1, T, C, H, W)
        "fps": fps,
    }
    if "prompt" in row:
        kwargs["text_prompt"] = [row["prompt"]]
    aux = row.get("auxiliary_info")
    if aux:
        kwargs["auxiliary_info"] = [aux]
    return kwargs


class EvalRunner:
    """Composes ``(dataset?, generator?, evaluator)`` for an eval run.

    Construct via one of :meth:`from_dataset`, :meth:`from_videos`, or
    :meth:`from_samples`. Use the run methods (:meth:`generate`,
    :meth:`score`, :meth:`run`) to drive work.

    Owned config:
        videos_dir, fps, seed, n_samples (override), gen_kwargs,
        filename_fn (row, idx → name), eval_kwargs_fn (row, path → kwargs).

    The evaluator is either supplied directly or built lazily from
    ``metrics`` + ``num_gpus``.
    """

    def __init__(
        self,
        *,
        videos_dir: str | Path | None,
        evaluator: Evaluator,
        dataset: "PromptDataset | None" = None,
        generator: Any = None,                       # VideoGenerator
        sample_rows: list[dict] | None = None,       # pre-built (from_samples)
        fps: float = 24.0,
        seed: int = 0,
        n_samples: int | None = None,
        gen_kwargs: dict | None = None,
        filename_fn: Callable[[dict, int], str] | None = None,
        eval_kwargs_fn: Callable[[dict, Path], dict[str, Any]] | None = None,
    ) -> None:
        self.videos_dir = Path(videos_dir) if videos_dir is not None else None
        self.evaluator = evaluator
        self.dataset = dataset
        self.generator = generator
        self._sample_rows = sample_rows
        self.fps = fps
        self.seed = seed
        self.n_samples = n_samples
        self.gen_kwargs = dict(gen_kwargs or {})
        self.filename_fn = filename_fn or _default_filename
        self._eval_kwargs_fn = eval_kwargs_fn

    # ----------------------------------------------------------------- ctors

    @classmethod
    def from_dataset(
        cls,
        dataset: "PromptDataset",
        videos_dir: str | Path,
        *,
        evaluator: Evaluator | None = None,
        generator: Any = None,
        metrics: list[str] | str = "all",
        num_gpus: int = 1,
        fps: float = 24.0,
        seed: int = 0,
        n_samples: int | None = None,
        gen_kwargs: dict | None = None,
        filename_fn: Callable[[dict, int], str] | None = None,
        eval_kwargs_fn: Callable[[dict, Path], dict[str, Any]] | None = None,
    ) -> "EvalRunner":
        """Run against a :class:`PromptDataset`. With *generator*, supports
        :meth:`generate` + :meth:`score` (and :meth:`run`); without it,
        only :meth:`score` (videos must already be on disk under
        *videos_dir* matching *filename_fn*)."""
        ev = evaluator or create_evaluator(metrics=metrics, num_gpus=num_gpus)
        return cls(
            videos_dir=videos_dir,
            evaluator=ev,
            dataset=dataset,
            generator=generator,
            fps=fps,
            seed=seed,
            n_samples=n_samples,
            gen_kwargs=gen_kwargs,
            filename_fn=filename_fn,
            eval_kwargs_fn=eval_kwargs_fn,
        )

    @classmethod
    def from_videos(
        cls,
        video_paths: Iterable[str | Path],
        *,
        prompts: list[str] | None = None,
        evaluator: Evaluator | None = None,
        metrics: list[str] | str = "all",
        num_gpus: int = 1,
        fps: float = 24.0,
        eval_kwargs_fn: Callable[[dict, Path], dict[str, Any]] | None = None,
    ) -> "EvalRunner":
        """Score a flat list of video files. Optional *prompts* (same
        length) attach a text prompt to each video for prompt-using
        metrics."""
        ev = evaluator or create_evaluator(metrics=metrics, num_gpus=num_gpus)
        paths = [Path(p) for p in video_paths]
        if prompts is not None and len(prompts) != len(paths):
            raise ValueError(
                f"prompts length {len(prompts)} != video_paths length "
                f"{len(paths)}")

        # Build a synthetic row per video so the score loop is uniform.
        rows: list[dict] = []
        for i, p in enumerate(paths):
            row: dict = {"_video_path": p}
            if prompts is not None:
                row["prompt"] = prompts[i]
            rows.append(row)

        return cls(
            videos_dir=None,
            evaluator=ev,
            sample_rows=rows,
            fps=fps,
            eval_kwargs_fn=eval_kwargs_fn,
        )

    @classmethod
    def from_samples(
        cls,
        samples: list[dict[str, Any]],
        *,
        evaluator: Evaluator | None = None,
        metrics: list[str] | str = "all",
        num_gpus: int = 1,
    ) -> "EvalRunner":
        """Score a list of pre-built evaluator-kwargs dicts. Each dict is
        passed verbatim to :meth:`Evaluator.evaluate` via the list-form
        API. Use this when you already have video tensors in memory."""
        ev = evaluator or create_evaluator(metrics=metrics, num_gpus=num_gpus)
        return cls(
            videos_dir=None,
            evaluator=ev,
            sample_rows=[{"_eval_kwargs": s} for s in samples],
        )

    # ------------------------------------------------------------- generate

    def generate(self, *, skip_existing: bool = True) -> dict[str, list[Path]]:
        """Drive the generator over the dataset. Returns ``{prompt: [paths]}``.

        Requires ``dataset`` and ``generator`` to have been supplied.
        Writes a ``manifest.json`` alongside the videos.
        """
        if self.dataset is None or self.generator is None:
            raise RuntimeError(
                "generate() requires both dataset and generator. Use "
                "from_dataset(dataset, videos_dir, generator=...).")
        if self.videos_dir is None:
            raise RuntimeError("generate() requires videos_dir.")

        self.videos_dir.mkdir(parents=True, exist_ok=True)
        total = sum((self.n_samples or row.get("n_samples", 1))
                    for row in self.dataset)
        log.info("[gen] %d prompts -> %d videos in %s",
                 len(self.dataset), total, self.videos_dir)

        global_idx = 0
        manifest: dict[str, list[str]] = {}
        for row in self.dataset:
            n = self.n_samples or row.get("n_samples", 1)
            paths: list[str] = []
            for k in range(n):
                target = self.videos_dir / self.filename_fn(row, k)
                paths.append(str(target))
                seed = self.seed + global_idx
                global_idx += 1
                if skip_existing and target.exists():
                    log.info("[gen %d/%d] SKIP exists: %s",
                             global_idx, total, target.name)
                    continue
                log.info("[gen %d/%d] prompt=%r seed=%d -> %s",
                         global_idx, total, row["prompt"][:60], seed,
                         target.name)
                self.generator.generate_video(
                    prompt=row["prompt"],
                    output_path=str(target),
                    save_video=True,
                    seed=seed,
                    **self.gen_kwargs,
                )
                if target.exists():
                    log.info("[gen %d/%d] OK %.1f MB",
                             global_idx, total, target.stat().st_size / 1e6)
                else:
                    log.warning("[gen %d/%d] missing after generate: %s",
                                global_idx, total, target)
            manifest[row["prompt"]] = paths
        (self.videos_dir / "manifest.json").write_text(
            json.dumps(manifest, indent=2))
        return {p: [Path(x) for x in v] for p, v in manifest.items()}

    # ----------------------------------------------------------------- score

    def score(self, *, verbose: bool = True) -> EvalResult:
        """Score every (eval_kwargs, meta) job for this runner.

        Iteration order: input order. Multi-GPU fan-out happens inside
        :meth:`Evaluator.evaluate` when called with the list form.
        """
        jobs = list(self._iter_jobs())
        if not jobs:
            log.warning("[score] no jobs found")
            return EvalResult.from_raw({}, [])

        kwargs_list = [j[0] for j in jobs]
        metas = [j[1] for j in jobs]

        # Single dispatch — Evaluator handles the multi-GPU fan-out.
        results = self.evaluator.evaluate(kwargs_list)

        by_metric: dict[str, list[float]] = {}
        per_video: list[dict] = []
        for meta, res in zip(metas, results, strict=True):
            scores: dict[str, float] = {}
            for name, mr in res.items():
                if mr is not None and mr.score is not None:
                    scores[name] = float(mr.score)
                    by_metric.setdefault(name, []).append(scores[name])
            per_video.append({**meta, "scores": scores})
            if verbose:
                log.info("[score] %s -> %s",
                         meta.get("video", "<no video>"),
                         {k: round(v, 4) for k, v in scores.items()})
        if verbose:
            log.info("[score] done. videos_scored=%d", len(per_video))
        return EvalResult.from_raw(by_metric, per_video)

    def run(self) -> EvalResult:
        """Generate then score. Convenience for full benchmark runs."""
        self.generate()
        return self.score()

    # ----------------------------------------------------------- iteration

    def _iter_jobs(self):
        """Yield ``(eval_kwargs, meta)`` pairs for every video to score."""
        if self._sample_rows is not None:
            for row in self._sample_rows:
                # from_samples: kwargs are pre-built.
                if "_eval_kwargs" in row:
                    kw = row["_eval_kwargs"]
                    meta = {"video": "<in-memory>"}
                    yield kw, meta
                    continue
                # from_videos: load the file, build kwargs.
                vp: Path = row["_video_path"]
                base_row = {k: v for k, v in row.items()
                            if not k.startswith("_")}
                kw = self._build_eval_kwargs(base_row, vp)
                meta = {"video": str(vp), "prompt": base_row.get("prompt")}
                yield kw, meta
            return

        if self.dataset is None or self.videos_dir is None:
            raise RuntimeError(
                "score() requires either a dataset+videos_dir or "
                "from_videos/from_samples construction.")

        for row in self.dataset:
            for vp in self._glob_row_videos(row):
                meta = {
                    "prompt": row.get("prompt"),
                    "video": str(vp),
                    "dimensions": list(row.get("dimensions", [])),
                }
                yield self._build_eval_kwargs(row, vp), meta

    def _glob_row_videos(self, row: dict) -> list[Path]:
        """Find every generated video for *row* under videos_dir, sorted."""
        assert self.videos_dir is not None
        # Strip trailing -<idx>.mp4 from the filename_fn(row, 0) output to
        # get a glob pattern. Falls back to the row's own k=0..k=N enumeration
        # if the convention isn't ``<stem>-<idx>.mp4``.
        first = self.filename_fn(row, 0)
        m = re.match(r"^(?P<stem>.*)-0(?P<ext>\.[^.]+)$", first)
        if m:
            pattern = f"{m['stem']}-*{m['ext']}"
            files = list(self.videos_dir.glob(pattern))

            def _idx(p: Path) -> int:
                try:
                    return int(p.stem.rsplit("-", 1)[1])
                except (IndexError, ValueError):
                    return -1

            return sorted(files, key=_idx)

        # Non-indexed convention: enumerate explicitly.
        n = self.n_samples or row.get("n_samples", 1)
        out: list[Path] = []
        for k in range(n):
            p = self.videos_dir / self.filename_fn(row, k)
            if p.exists():
                out.append(p)
        return out

    def _build_eval_kwargs(self, row: dict, vp: Path) -> dict[str, Any]:
        if self._eval_kwargs_fn is not None:
            return self._eval_kwargs_fn(row, vp)
        return _default_eval_kwargs(row, vp, fps=self.fps)
