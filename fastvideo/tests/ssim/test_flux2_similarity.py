# SPDX-License-Identifier: Apache-2.0
"""SSIM regression slot for Flux 2 (Klein + base).

This file reserves a CI slot for Flux 2 SSIM regression testing. The actual
parity assertions live in ``tests/local_tests/pipelines/test_flux2_pipeline_parity.py``
(1077 LOC, asserts allclose at ATOL/RTOL=1e-4 on TP1 H100), which is not in CI.

Seeding workflow:
1. Run the local parity test on Modal L40S / H100 to generate reference latents/videos.
2. Use the ``seed-ssim-references`` skill to upload references to the
   ``FastVideo/ssim-reference-videos`` HF dataset.
3. Replace the ``pytest.skip()`` below with a call to ``run_text_to_video_similarity_test``
   (or the latent-only equivalent) following sibling tests in this directory.

Until then this slot exists so the CI matrix surfaces "flux2 SSIM = SKIPPED"
rather than silently omitting Flux 2 from quality regression coverage.
"""

import pytest


def test_flux2_klein_similarity_placeholder() -> None:
    pytest.skip(
        "Flux 2 SSIM references not yet seeded to HF; "
        + "run the seed-ssim-references skill against "
        + "tests/local_tests/pipelines/test_flux2_pipeline_parity.py to populate."
    )
