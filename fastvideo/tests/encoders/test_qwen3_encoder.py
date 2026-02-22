# SPDX-License-Identifier: Apache-2.0
"""
Parity test: FastVideo Qwen3Model vs HuggingFace Qwen3Model for Ovis-Image.

Mirrors the pattern of test_qwen2_5_encoder.py:
  - Loads the real Ovis2.5-2B text encoder from local weights
  - Compares FastVideo's Qwen3Model output against the HF baseline
  - Checks key weight values and final hidden state numerically

Set OVIS_WEIGHTS env var to the local model root, e.g.
    OVIS_WEIGHTS=official_weights/ovis_image \
        pytest fastvideo/tests/encoders/test_qwen3_encoder.py -vs
"""

import os

import pytest
import torch
from torch.distributed.tensor import DTensor
from torch.testing import assert_close
from transformers import AutoConfig, AutoTokenizer, Qwen3Model as HFQwen3Model

from fastvideo.configs.models.encoders import Qwen3Config
from fastvideo.configs.pipelines.base import PipelineConfig
from fastvideo.fastvideo_args import FastVideoArgs
from fastvideo.forward_context import set_forward_context
from fastvideo.logger import init_logger
from fastvideo.models.loader.component_loader import TextEncoderLoader
from fastvideo.utils import PRECISION_TO_TYPE

logger = init_logger(__name__)

os.environ["MASTER_ADDR"] = "localhost"
os.environ["MASTER_PORT"] = "29509"

LOCAL_WEIGHTS = os.getenv("OVIS_WEIGHTS", "official_weights/ovis_image")
TEXT_ENCODER_PATH = os.path.join(LOCAL_WEIGHTS, "text_encoder")
TOKENIZER_PATH = os.path.join(LOCAL_WEIGHTS, "tokenizer")


@pytest.fixture
def qwen3_model_paths():
    return TEXT_ENCODER_PATH, TOKENIZER_PATH


@pytest.mark.skipif(
    not os.path.exists(TEXT_ENCODER_PATH),
    reason=(f"Ovis-Image text_encoder not found at {TEXT_ENCODER_PATH}. "
            f"Set OVIS_WEIGHTS env var or download from AIDC-AI/Ovis-Image-7B."))
@pytest.mark.usefixtures("distributed_setup")
def test_qwen3_encoder(qwen3_model_paths):
    """
    Load Qwen3 via FastVideo's TextEncoderLoader and verify its last_hidden_state
    matches the HuggingFace Qwen3Model baseline (fp32, atol=1e-3).
    """
    text_encoder_path, tokenizer_path = qwen3_model_paths
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    precision_str = "fp32"
    precision = PRECISION_TO_TYPE[precision_str]

    hf_config = AutoConfig.from_pretrained(text_encoder_path)
    logger.info(f"Qwen3 config: hidden_size={hf_config.hidden_size}, "
                f"layers={hf_config.num_hidden_layers}")

    # ---- HF baseline ----
    hf_model = HFQwen3Model.from_pretrained(text_encoder_path).to(
        precision).to(device).eval()
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

    # ---- FastVideo model ----
    args = FastVideoArgs(
        model_path=text_encoder_path,
        pipeline_config=PipelineConfig(
            text_encoder_configs=(Qwen3Config(),),
            text_encoder_precisions=(precision_str,),
        ),
        pin_cpu_memory=False,
    )
    loader = TextEncoderLoader()
    fv_model = loader.load(text_encoder_path, args)
    fv_model = fv_model.to(precision)
    fv_model.eval()

    # ---- Weight spot-check ----
    logger.info("Spot-checking weights...")
    params_hf = dict(hf_model.named_parameters())
    params_fv = dict(fv_model.named_parameters())

    weight_names = [
        "norm.weight",
        "layers.0.input_layernorm.weight",
        "layers.0.post_attention_layernorm.weight",
        "layers.0.mlp.down_proj.weight",
    ]
    for name in weight_names:
        if name not in params_hf or name not in params_fv:
            logger.warning(f"Weight {name} not present in both models, skipping")
            continue
        p_hf = params_hf[name].to(device)
        p_fv = params_fv[name]
        p_fv = (p_fv.to_local() if isinstance(p_fv, DTensor) else p_fv).to(p_hf)
        assert p_hf.shape == p_fv.shape, \
            f"Shape mismatch for {name}: HF={p_hf.shape}, FV={p_fv.shape}"
        assert_close(p_hf, p_fv, atol=1e-7, rtol=1e-7,
                     msg=f"Weight mismatch for {name}")
    logger.info("Weight spot-check passed.")

    # ---- Forward-pass parity ----
    prompts = [
        "A vibrant sunset over the ocean with vivid colors.",
        "The quick brown fox jumps over the lazy dog.",
    ]
    for prompt in prompts:
        logger.info(f"Testing prompt: {prompt!r}")
        tokens = tokenizer(
            prompt,
            return_tensors="pt",
            padding="max_length",
            max_length=128,
            truncation=True,
        ).to(device)

        with torch.no_grad():
            hf_out = hf_model(
                input_ids=tokens.input_ids,
                attention_mask=tokens.attention_mask,
            ).last_hidden_state

            with set_forward_context(current_timestep=0, attn_metadata=None):
                fv_out = fv_model(
                    input_ids=tokens.input_ids,
                    attention_mask=tokens.attention_mask,
                ).last_hidden_state

        assert hf_out.shape == fv_out.shape, \
            f"Output shape mismatch: HF={hf_out.shape}, FV={fv_out.shape}"

        max_diff = (hf_out - fv_out).abs().max().item()
        mean_diff = (hf_out - fv_out).abs().mean().item()
        logger.info(f"  max_diff={max_diff:.3e}  mean_diff={mean_diff:.3e}")

        atol = 1e-3 if precision_str == "fp32" else 5e-2
        assert max_diff < atol, \
            f"Output max diff {max_diff:.3e} > {atol} for prompt: {prompt!r}"

    logger.info("Qwen3 encoder parity test passed.")
