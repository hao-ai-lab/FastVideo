# SPDX-License-Identifier: Apache-2.0
import os
import sys
from pathlib import Path

import pytest
import torch
from safetensors.torch import load_file
from torch.testing import assert_close

from fastvideo.configs.models.encoders import LTX2GemmaConfig
from fastvideo.configs.pipelines import PipelineConfig
from fastvideo.fastvideo_args import FastVideoArgs
from fastvideo.forward_context import set_forward_context
from fastvideo.models.loader.component_loader import TextEncoderLoader

repo_root = Path(__file__).resolve().parents[3]
ltx_core_path = repo_root / "LTX-2" / "packages" / "ltx-core" / "src"
if ltx_core_path.exists() and str(ltx_core_path) not in sys.path:
    sys.path.insert(0, str(ltx_core_path))


def _load_connector_weights(path: str) -> dict[str, torch.Tensor]:
    weights = load_file(path)
    mapped: dict[str, torch.Tensor] = {}
    for name, tensor in weights.items():
        if name == "aggregate_embed.weight":
            mapped["feature_extractor_linear.aggregate_embed.weight"] = tensor
        elif name.startswith("embeddings_connector."):
            mapped[name] = tensor
        elif name.startswith("audio_embeddings_connector."):
            mapped[name] = tensor
    return mapped


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="LTX-2 Gemma encoder parity test requires CUDA.",
)
def test_ltx2_gemma_text_encoder_parity():
    diffusers_root = Path(
        os.getenv("LTX2_DIFFUSERS_PATH", "converted/ltx2_diffusers")
    )
    text_encoder_path = os.getenv(
        "LTX2_TEXT_ENCODER_PATH",
        str(diffusers_root / "text_encoder"),
    )
    gemma_model_path = str(Path(text_encoder_path) / "gemma")
    if not os.path.isdir(text_encoder_path):
        pytest.skip(f"LTX-2 text encoder weights not found at {text_encoder_path}")
    if not gemma_model_path or not os.path.isdir(gemma_model_path):
        pytest.skip("Gemma weights not found in text_encoder/gemma.")

    try:
        from ltx_core.text_encoders.gemma.embeddings_connector import (
            Embeddings1DConnector,
        )
        from ltx_core.text_encoders.gemma.encoders.av_encoder import (
            AVGemmaTextEncoderModel,
        )
        from ltx_core.text_encoders.gemma.feature_extractor import (
            GemmaFeaturesExtractorProjLinear,
        )
        from ltx_core.text_encoders.gemma.tokenizer import LTXVGemmaTokenizer
        from transformers import Gemma3ForConditionalGeneration
    except Exception as exc:
        pytest.skip(f"LTX-2 Gemma import failed: {exc}")

    device = torch.device("cuda:0")
    precision = torch.bfloat16

    tokenizer = LTXVGemmaTokenizer(gemma_model_path, max_length=1024)
    gemma_model = Gemma3ForConditionalGeneration.from_pretrained(
        gemma_model_path,
        local_files_only=True,
        torch_dtype=precision,
    ).to(device)
    gemma_model.eval()

    ref_model = AVGemmaTextEncoderModel(
        feature_extractor_linear=GemmaFeaturesExtractorProjLinear(),
        embeddings_connector=Embeddings1DConnector(),
        audio_embeddings_connector=Embeddings1DConnector(),
        tokenizer=tokenizer,
        model=gemma_model,
        dtype=precision,
    ).to(device)
    ref_model.eval()

    connector_weights = _load_connector_weights(
        os.path.join(text_encoder_path, "model.safetensors")
    )
    ref_model.load_state_dict(connector_weights, strict=False)

    prompt = "A fast moving train in a snowy landscape."
    token_pairs = tokenizer.tokenize_with_weights(prompt)["gemma"]
    input_ids = torch.tensor(
        [[t[0] for t in token_pairs]], device=device, dtype=torch.long
    )
    attention_mask = torch.tensor(
        [[t[1] for t in token_pairs]], device=device, dtype=torch.long
    )

    args = FastVideoArgs(
        model_path=text_encoder_path,
        pipeline_config=PipelineConfig(
            text_encoder_configs=(LTX2GemmaConfig(),),
            text_encoder_precisions=("bf16",),
        ),
    )
    loader = TextEncoderLoader()
    fastvideo_model = loader.load(text_encoder_path, args).to(device)
    fastvideo_model.eval()

    with torch.no_grad():
        ref_video, ref_audio, ref_mask = ref_model(prompt, padding_side="left")
        with set_forward_context(current_timestep=0, attn_metadata=None):
            fastvideo_out = fastvideo_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
            )

    assert_close(ref_video, fastvideo_out.last_hidden_state, atol=1e-2, rtol=1e-2)
    assert_close(ref_audio, fastvideo_out.hidden_states[0], atol=1e-2, rtol=1e-2)
    assert torch.equal(ref_mask, fastvideo_out.attention_mask)
