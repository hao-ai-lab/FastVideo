"""Frechet Audio Distance over PaSST embeddings (``FD_PaSST``).

Corpus-vs-corpus Fréchet distance between Gaussian moments of two PaSST
768-d embedding sets. Ports ``av_bench.metrics.fad.compute_fd`` 1:1.

References are supplied per-sample via ``reference_audio`` /
``role="reference"``, or once from a ``.pt`` cache at
``$FASTVIDEO_FAD_REF_FEATURES``. Skips when either side has fewer than
two finite embeddings.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import numpy as np
import torch
from scipy import linalg

from fastvideo.eval.metrics.base import BaseMetric
from fastvideo.eval.registry import register
from fastvideo.eval.types import MetricResult

PASST_SAMPLING_RATE = 32000
PASST_WIN_SAMPLES = 320000  # 10 s at 32 kHz — av-benchmark's truncate/pad target
PASST_EMBED_DIM = 768  # PaSST mode="all" returns 527+768 = 1295; embed is the trailing 768

REF_FEATURES_ENV = "FASTVIDEO_FAD_REF_FEATURES"


def _frechet_distance(mu1: np.ndarray,
                      sigma1: np.ndarray,
                      mu2: np.ndarray,
                      sigma2: np.ndarray,
                      eps: float = 1e-6) -> float:
    """``d^2 = ||mu1 - mu2||^2 + Tr(sigma1 + sigma2 - 2 sqrt(sigma1 sigma2))``."""
    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)
    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)
    assert mu1.shape == mu2.shape
    assert sigma1.shape == sigma2.shape

    sigma1 = sigma1 + eps * np.eye(sigma1.shape[0])
    sigma2 = sigma2 + eps * np.eye(sigma2.shape[0])

    diff = mu1 - mu2
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            raise ValueError(f"imaginary component {np.max(np.abs(covmean.imag))}")
        covmean = covmean.real

    return float(diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * np.trace(covmean))


def _passt_embed(model: Any, audio_path: str, device: torch.device) -> np.ndarray:
    """Run PaSST (mode='all') over a 10-s window and return the 768-d embedding."""
    import librosa
    import pyloudnorm as pyln

    audio, _ = librosa.load(audio_path, sr=PASST_SAMPLING_RATE, mono=True)
    audio = pyln.normalize.peak(audio, -1.0)
    if len(audio) >= PASST_WIN_SAMPLES:
        audio = audio[:PASST_WIN_SAMPLES]
    else:
        padded = np.zeros(PASST_WIN_SAMPLES, dtype=np.float32)
        padded[:len(audio)] = audio
        audio = padded

    wav = torch.from_numpy(np.asarray(audio, dtype=np.float32)).unsqueeze(0).to(device)
    with torch.no_grad():
        out = model(wav)  # (1, 1295) — logits[:527] | embed[527:]
    embed = out[0, 527:].detach().cpu().numpy()
    return embed


@register("audio.frechet_distance")
class FrechetAudioDistanceMetric(BaseMetric):
    """Corpus-vs-corpus Frechet Audio Distance with PaSST embeddings."""

    name = "audio.frechet_distance"
    requires_reference = True
    higher_is_better = False
    needs_gpu = True
    is_set_metric = True
    backbone = "passt"
    dependencies = ["hear21passt", "librosa", "pyloudnorm"]

    def __init__(self) -> None:
        super().__init__()
        self._model: Any = None
        self._gen_buf: list[np.ndarray] = []
        self._ref_buf: list[np.ndarray] = []
        self._cached_ref_path: str | None = os.environ.get(REF_FEATURES_ENV)
        self._cached_ref_mu: np.ndarray | None = None
        self._cached_ref_sigma: np.ndarray | None = None
        self._n_cached_ref: int = 0

    def to(self, device):
        super().to(device)
        if self._model is not None:
            self._model = self._model.to(self.device)
        return self

    def setup(self) -> None:
        if self._model is None:
            from hear21passt.base import get_basic_model
            model = get_basic_model(mode="all")
            model.eval()
            model = model.to(self.device)
            self._model = model
        if self._cached_ref_path and self._cached_ref_mu is None:
            self._load_cached_ref()

    def _load_cached_ref(self) -> None:
        assert self._cached_ref_path is not None
        path = Path(self._cached_ref_path)
        if not path.exists():
            raise FileNotFoundError(f"{REF_FEATURES_ENV} set to {path}, but the file does not exist. "
                                    f"Either unset the env var to use sample-supplied references or "
                                    f"pre-compute the reference features file.")
        ref = torch.load(str(path), weights_only=True, map_location="cpu")
        if hasattr(ref, "numpy"):
            ref = ref.numpy()
        ref = np.asarray(ref)
        if ref.ndim != 2 or ref.shape[1] != PASST_EMBED_DIM:
            raise ValueError(f"Expected a 2-D tensor of shape (N, {PASST_EMBED_DIM}) "
                             f"at {path}; got shape {ref.shape}.")
        finite = np.isfinite(ref).all(axis=1)
        ref = ref[finite]
        if ref.shape[0] < 2:
            raise ValueError(f"Cached reference features at {path} have only "
                             f"{ref.shape[0]} finite rows; need >= 2.")
        self._cached_ref_mu = ref.mean(axis=0)
        self._cached_ref_sigma = np.cov(ref, rowvar=False)
        self._n_cached_ref = int(ref.shape[0])

    def reset(self) -> None:
        self._gen_buf.clear()
        self._ref_buf.clear()

    def accumulate(self, sample: dict) -> None:
        if self._model is None:
            self.setup()
        if sample.get("role") == "reference":
            ref_path = sample.get("audio") or sample.get("reference_audio")
            if ref_path is not None:
                self._ref_buf.append(_passt_embed(self._model, ref_path, self.device))
            return
        gen_path = sample.get("audio")
        if gen_path is not None:
            self._gen_buf.append(_passt_embed(self._model, gen_path, self.device))
        if self._cached_ref_mu is not None:
            return
        ref_path = sample.get("reference_audio")
        if ref_path is not None:
            self._ref_buf.append(_passt_embed(self._model, ref_path, self.device))

    def merge_from(self, other: FrechetAudioDistanceMetric) -> None:
        self._gen_buf.extend(other._gen_buf)
        self._ref_buf.extend(other._ref_buf)

    def finalize(self) -> MetricResult:
        gen_all = np.stack(self._gen_buf) if self._gen_buf else np.empty((0, ))
        gen = gen_all[np.isfinite(gen_all).all(axis=1)] if gen_all.size else gen_all
        n_gen = int(gen.shape[0])
        n_gen_dropped = len(self._gen_buf) - n_gen

        if self._cached_ref_mu is not None:
            mu_r = self._cached_ref_mu
            sigma_r = self._cached_ref_sigma
            n_ref = self._n_cached_ref
            n_ref_dropped = 0
            ref_source = "cached"
        else:
            ref_all = np.stack(self._ref_buf) if self._ref_buf else np.empty((0, ))
            ref = ref_all[np.isfinite(ref_all).all(axis=1)] if ref_all.size else ref_all
            n_ref = int(ref.shape[0])
            n_ref_dropped = len(self._ref_buf) - n_ref
            ref_source = "samples"
            if n_ref < 2 or n_gen < 2:
                return MetricResult(
                    name=self.name,
                    score=None,
                    details={
                        "skipped":
                        f"FAD needs >=2 finite-embed samples per side "
                        f"(got n_gen={n_gen} valid of {len(self._gen_buf)}, "
                        f"n_ref={n_ref} valid of {len(self._ref_buf)})",
                        "n_gen":
                        n_gen,
                        "n_ref":
                        n_ref,
                        "n_gen_dropped_nonfinite":
                        n_gen_dropped,
                        "n_ref_dropped_nonfinite":
                        n_ref_dropped,
                        "ref_source":
                        ref_source,
                    },
                )
            mu_r, sigma_r = ref.mean(axis=0), np.cov(ref, rowvar=False)

        if n_gen < 2:
            return MetricResult(
                name=self.name,
                score=None,
                details={
                    "skipped": f"FAD needs >=2 finite-embed gen samples "
                    f"(got {n_gen} valid of {len(self._gen_buf)})",
                    "n_gen": n_gen,
                    "n_ref": n_ref,
                    "n_gen_dropped_nonfinite": n_gen_dropped,
                    "n_ref_dropped_nonfinite": n_ref_dropped,
                    "ref_source": ref_source,
                },
            )

        mu_g, sigma_g = gen.mean(axis=0), np.cov(gen, rowvar=False)
        fd = _frechet_distance(mu_g, sigma_g, mu_r, sigma_r)
        return MetricResult(
            name=self.name,
            score=fd,
            details={
                "n_gen": n_gen,
                "n_ref": n_ref,
                "n_gen_dropped_nonfinite": n_gen_dropped,
                "n_ref_dropped_nonfinite": n_ref_dropped,
                "ref_source": ref_source,
            },
        )
