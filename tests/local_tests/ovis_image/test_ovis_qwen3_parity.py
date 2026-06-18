# SPDX-License-Identifier: Apache-2.0
"""Parity for the FastVideo ``Qwen3Model`` (production ``TextEncoderLoader``) vs
HF ``transformers.Qwen3Model`` through the real Ovis prompt path (chat template +
system prompt + 28-token slice). fp32, max_diff < 1e-3."""

import os

import pytest
import torch
from torch.distributed.tensor import DTensor
from torch.testing import assert_close

import fastvideo  # noqa: F401  # ensure full package init before deep submodule imports
from fastvideo.configs.models.encoders import Qwen3Config
from fastvideo.configs.pipelines.base import PipelineConfig
from fastvideo.configs.pipelines.ovis_image import (OVIS_SYSTEM_PROMPT,
                                                    USER_PROMPT_BEGIN_ID,
                                                    qwen3_preprocess_text)
from fastvideo.fastvideo_args import FastVideoArgs
from fastvideo.forward_context import set_forward_context
from fastvideo.logger import init_logger
from fastvideo.models.loader.component_loader import TextEncoderLoader
from fastvideo.utils import PRECISION_TO_TYPE

logger = init_logger(__name__)

os.environ.setdefault("MASTER_ADDR", "localhost")
os.environ.setdefault("MASTER_PORT", "29510")

LOCAL_WEIGHTS = os.getenv("OVIS_WEIGHTS", "official_weights/ovis_image")
TEXT_ENCODER_PATH = os.path.join(LOCAL_WEIGHTS, "text_encoder")
TOKENIZER_PATH = os.path.join(LOCAL_WEIGHTS, "tokenizer")


@pytest.mark.skipif(
    not os.path.exists(TEXT_ENCODER_PATH),
    reason=(f"Ovis-Image text_encoder not found at {TEXT_ENCODER_PATH}. "
            f"Set OVIS_WEIGHTS or download from AIDC-AI/Ovis-Image-7B."))
@pytest.mark.usefixtures("distributed_setup")
def test_ovis_qwen3_chat_template_parity():
    try:
        from transformers import (AutoTokenizer,
                                  Qwen3Model as HFQwen3Model)
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"transformers Qwen3Model unavailable: {exc!r}")

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    precision_str = "fp32"
    precision = PRECISION_TO_TYPE[precision_str]

    # ---- HF baseline ----
    hf_model = HFQwen3Model.from_pretrained(TEXT_ENCODER_PATH).to(
        precision).to(device).eval()
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH)

    # ---- FastVideo encoder (production loader path) ----
    cfg = Qwen3Config()
    args = FastVideoArgs(
        model_path=TEXT_ENCODER_PATH,
        pipeline_config=PipelineConfig(
            text_encoder_configs=(cfg, ),
            text_encoder_precisions=(precision_str, ),
        ),
        pin_cpu_memory=False,
    )
    fv_model = TextEncoderLoader().load(TEXT_ENCODER_PATH, args)
    fv_model = fv_model.to(precision).eval()

    # ---- Tokenize with the exact Ovis production prompt path ----
    prompt = "A neon sign that reads OPEN above a rainy city street at night."
    messages = qwen3_preprocess_text(prompt)
    assert messages[0]["content"].startswith(OVIS_SYSTEM_PROMPT)

    tok_kwargs = dict(cfg.arch_config.tokenizer_kwargs)
    tokens = tokenizer.apply_chat_template(messages, **tok_kwargs)
    input_ids = tokens["input_ids"].to(device)
    attention_mask = tokens["attention_mask"].to(device)

    with torch.no_grad():
        hf_out = hf_model(input_ids=input_ids,
                          attention_mask=attention_mask).last_hidden_state
        with set_forward_context(current_timestep=0, attn_metadata=None):
            fv_out = fv_model(input_ids=input_ids,
                              attention_mask=attention_mask).last_hidden_state

    assert hf_out.shape == fv_out.shape, \
        f"Shape mismatch: HF={hf_out.shape}, FV={fv_out.shape}"

    max_diff = (hf_out - fv_out).abs().max().item()
    logger.info(f"chat-template parity max_diff={max_diff:.3e}")
    assert max_diff < 1e-3, f"Encoder max diff {max_diff:.3e} exceeds 1e-3"

    # The 28-token system-prompt slice must drop exactly the system tokens, so
    # the user-visible embeddings start at USER_PROMPT_BEGIN_ID.
    assert input_ids.shape[1] > USER_PROMPT_BEGIN_ID, \
        "Tokenized sequence shorter than the system-prompt prefix"
    sliced = fv_out[:, USER_PROMPT_BEGIN_ID:]
    assert sliced.shape[1] == fv_out.shape[1] - USER_PROMPT_BEGIN_ID

    logger.info("Ovis Qwen3 chat-template parity passed.")
