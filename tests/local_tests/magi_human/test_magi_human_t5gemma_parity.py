# SPDX-License-Identifier: Apache-2.0
"""Parity test: FastVideo T5GemmaEncoderModel vs direct HF
`T5GemmaEncoderModel.from_pretrained(...)`.

FastVideo's wrapper is intentionally thin — it lazy-loads the same HF
class on the same gated repo (`google/t5gemma-9b-9b-ul2`) that the
upstream MagiHuman pipeline uses (see
daVinci-MagiHuman/inference/model/t5_gemma/t5_gemma_model.py). This
test guards against future regressions in the wrapper (e.g. accidental
mutation of `last_hidden_state`, wrong dtype cast, forgetting to pass
attention_mask) by comparing wrapper forward output against a direct HF
forward on the same model.

Skips when the T5-Gemma repo isn't accessible (gated — requires user's
HF token with accepted terms of use).
"""
from __future__ import annotations

import os

import pytest
import torch
from torch.testing import assert_close


_T5GEMMA_ID = "google/t5gemma-9b-9b-ul2"


def _hf_token():
    for k in ("HF_TOKEN", "HUGGINGFACE_HUB_TOKEN", "HF_API_KEY"):
        v = os.environ.get(k)
        if v:
            return v
    return None


def _can_access_t5gemma() -> bool:
    token = _hf_token()
    if token is None:
        return False
    try:
        from huggingface_hub import hf_hub_download
        hf_hub_download(
            repo_id=_T5GEMMA_ID, filename="config.json", token=token,
        )
        return True
    except Exception:
        return False


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="MagiHuman T5-Gemma parity requires CUDA (encoder is 9B params).",
)
@pytest.mark.skipif(
    not _can_access_t5gemma(),
    reason=(f"{_T5GEMMA_ID} not accessible — gated Google repo; set "
            f"HF_TOKEN / HF_API_KEY and accept the terms of use."),
)
def test_magi_human_t5gemma_wrapper_parity():
    # Alias any of the three token env vars to HF_TOKEN (what transformers
    # reads) before constructing models.
    for src in ("HF_TOKEN", "HUGGINGFACE_HUB_TOKEN", "HF_API_KEY"):
        v = os.environ.get(src)
        if v:
            os.environ.setdefault("HF_TOKEN", v)
            os.environ.setdefault("HUGGINGFACE_HUB_TOKEN", v)
            break

    device = torch.device("cuda:0")

    # --- Upstream / direct HF path (matches the reference pipeline's
    #     `T5GemmaEncoder` wrapper exactly: see
    #     daVinci-MagiHuman/inference/model/t5_gemma/t5_gemma_model.py) ---
    from transformers import AutoTokenizer
    from transformers.models.t5gemma import T5GemmaEncoderModel as HFEncoder

    tokenizer = AutoTokenizer.from_pretrained(_T5GEMMA_ID)
    ref_model = HFEncoder.from_pretrained(
        _T5GEMMA_ID, is_encoder_decoder=False, dtype=torch.bfloat16,
    ).to(device).eval()

    # --- FastVideo wrapper path ---
    from fastvideo.configs.models.encoders.t5gemma import T5GemmaEncoderConfig
    from fastvideo.models.encoders.t5gemma import T5GemmaEncoderModel as FVEncoder

    fv_config = T5GemmaEncoderConfig()
    fv_config.arch_config.t5gemma_model_path = _T5GEMMA_ID
    fv_model = FVEncoder(fv_config)

    # --- Identical input ---
    prompt = (
        "A warm afternoon scene: a person sits on a park bench reading "
        "a book, surrounded by softly swaying trees."
    )
    inputs = tokenizer(
        [prompt], return_tensors="pt", padding=True, truncation=False,
    ).to(device)

    with torch.inference_mode():
        ref_out = ref_model(**inputs)
        ref_hidden = ref_out["last_hidden_state"].detach().float().cpu()

        # FastVideo wrapper: forward through the adapter; it lazy-loads the
        # encoder on first call and moves it to the input's device.
        fv_out = fv_model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs.get("attention_mask"),
        )
        fv_hidden = fv_out.last_hidden_state.detach().float().cpu()

    print(
        f"ref_hidden shape={tuple(ref_hidden.shape)} "
        f"abs_mean={ref_hidden.abs().mean().item():.6f}"
    )
    print(
        f"fv_hidden  shape={tuple(fv_hidden.shape)} "
        f"abs_mean={fv_hidden.abs().mean().item():.6f}"
    )
    diff = (ref_hidden - fv_hidden).abs()
    print(
        f"diff max={diff.max().item():.6e} "
        f"mean={diff.mean().item():.6e}"
    )

    assert ref_hidden.shape == fv_hidden.shape
    # Both sides run the exact same HF model on the exact same inputs;
    # drift is bounded by nondeterminism in SDPA + bf16 matmul. This
    # should be <= 1e-3 end-to-end.
    assert_close(fv_hidden, ref_hidden, atol=1e-3, rtol=1e-3)
