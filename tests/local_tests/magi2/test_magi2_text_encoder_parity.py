# SPDX-License-Identifier: Apache-2.0
"""Strict prompt and Qwen3.5 encoder parity for MAGI-2 Preview.

Coverage scope: both. The official side uses
``inference.pipeline.inference_engine.initialize_text_encoder`` with the
published Qwen3.5 checkpoint. The FastVideo side targets
``fastvideo.models.encoders.qwen3_5.Magi2Qwen35TextEncoder`` with the same checkpoint.
The comparison covers structured-prompt normalization, CJK token splitting,
token IDs, attention masks, the skip-layer-2 hidden state, and ``<Figure 1>``
token pooling.
"""
from __future__ import annotations

import gc
import json
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from fastvideo.models.encoders.qwen3_5 import (
    Magi2Qwen35TextEncoder,
    json_to_compact_markdown,
)
from tests.local_tests.magi2._parity_utils import (
    LOCAL_WEIGHTS_DIR,
    OFFICIAL_REF_DIR,
    assert_tensor_exact,
    import_official_module,
    require_complete_safetensor_index,
)


PARITY_COVERAGE = "both"
TEXT_ENCODER_DIR = LOCAL_WEIGHTS_DIR / "text_encoder"


def _structured_i2v_prompt() -> str:
    """Return one deterministic prompt that exercises JSON, CJK, and figure tokens."""
    prompt = {
        "global_layer": {
            "context": "夜晚的城市街道",
            "description": "A cyclist passes a quiet café.",
            "aesthetics": {
                "style": "documentary",
                "mood_atmosphere": "calm",
                "color_scheme": "blue and amber",
            },
        },
        "reference_layer": ["The first frame refers to <Figure 1>"],
    }
    return json.dumps(prompt, ensure_ascii=False)


def _tokenize(
    tokenizer,
    max_length: int,
    normalized_prompt: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Tokenize with the exact keyword arguments used by the official encoder."""
    tokens = tokenizer(
        [normalized_prompt],
        return_tensors="pt",
        padding="longest",
        max_length=max_length,
        truncation=True,
    )
    return tokens["input_ids"], tokens["attention_mask"]


def _load_official_encoder(model_path: str, device: torch.device):
    """Instantiate Qwen3.5 through the official component implementation."""
    qwen35_module = import_official_module("inference.model.qwen35")
    return qwen35_module.Qwen35TextEncoder(
        model_path=model_path,
        device=str(device),
        precision=torch.bfloat16,
        skip_layer=2,
    )


def test_iter_indexed_language_model_weights_reads_only_language_model_tensors(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Read only indexed language-model keys and ignore unrelated missing shards."""
    from fastvideo.models.encoders import qwen3_5 as qwen35_module

    language_shards = ["language-a.safetensors", "language-b.safetensors"]
    for shard_name in language_shards:
        (tmp_path / shard_name).touch()
    weight_map = {
        "model.language_model.layers.0.weight": language_shards[1],
        "model.visual.weight": "missing-visual.safetensors",
        "model.language_model.embed_tokens.weight": language_shards[0],
        "lm_head.weight": "missing-output.safetensors",
    }
    tensors_by_shard = {
        language_shards[0]: {
            "model.language_model.embed_tokens.weight": torch.tensor([1.0]),
        },
        language_shards[1]: {
            "model.language_model.layers.0.weight": torch.tensor([2.0]),
        },
    }
    opened_shards: list[str] = []
    read_requests: list[tuple[str, str]] = []

    @contextmanager
    def recording_safe_open(filename: str, *, framework: str, device: str):
        """Record each shard and reject tensor reads from the wrong shard."""
        shard_name = Path(filename).name
        opened_shards.append(shard_name)
        assert framework == "pt"
        assert device == "cpu"

        def read_tensor(checkpoint_name: str) -> torch.Tensor:
            read_requests.append((shard_name, checkpoint_name))
            return tensors_by_shard[shard_name][checkpoint_name]

        yield SimpleNamespace(get_tensor=read_tensor)

    monkeypatch.setattr(qwen35_module, "safe_open", recording_safe_open)
    loaded_weights = list(
        qwen35_module._iter_indexed_language_model_weights(
            tmp_path,
            weight_map,
            torch.device("cpu"),
        )
    )

    assert opened_shards == language_shards
    assert read_requests == [
        (language_shards[0], "model.language_model.embed_tokens.weight"),
        (language_shards[1], "model.language_model.layers.0.weight"),
    ]
    assert [name for name, _ in loaded_weights] == [name for _, name in read_requests]
    assert [tensor.item() for _, tensor in loaded_weights] == [1.0, 2.0]


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="MAGI-2 Qwen3.5 parity requires CUDA.",
)
def test_magi2_text_encoder_structured_i2v_prompt_exact_parity() -> None:
    """Require exact tokenizer, hidden-state, and figure-token outputs."""
    model_dir = require_complete_safetensor_index(TEXT_ENCODER_DIR)
    device = torch.device("cuda:0")
    prompt = _structured_i2v_prompt()

    official_encoder = _load_official_encoder(
        str(model_dir),
        torch.device("cpu"),
    ).to(device)
    official_normalized_prompt = official_encoder._normalize_prompt(prompt)
    official_input_ids, official_attention_mask = _tokenize(
        official_encoder.tokenizer,
        official_encoder.max_length,
        official_normalized_prompt,
    )
    official_embedding = official_encoder.encode(prompt).detach().cpu()
    release_prompt = (
        OFFICIAL_REF_DIR / "assets" / "sample_enhanced_t2v.json"
    ).read_text(encoding="utf-8").strip()
    official_release_embedding = official_encoder.encode(release_prompt).detach().cpu()
    official_figure_embedding = official_encoder.get_special_token(
        prompt,
        ["<Figure 1>"],
        official_embedding,
    )

    del official_encoder
    gc.collect()
    torch.cuda.empty_cache()

    from fastvideo.configs.models.encoders.qwen3_5 import Magi2Qwen35Config
    fastvideo_encoder = Magi2Qwen35TextEncoder.from_pretrained_local(
        model_path=str(model_dir),
        model_config=Magi2Qwen35Config(),
        dtype=torch.bfloat16,
        device=torch.device("cpu"),
    ).to(device)
    fastvideo_normalized_prompt = json_to_compact_markdown(prompt)
    fastvideo_input_ids, fastvideo_attention_mask = _tokenize(
        fastvideo_encoder.tokenizer,
        fastvideo_encoder.config.text_len,
        fastvideo_normalized_prompt,
    )
    fastvideo_embedding = fastvideo_encoder.encode(prompt).detach().cpu()
    fastvideo_release_embedding = fastvideo_encoder.encode(
        release_prompt
    ).detach().cpu()
    fastvideo_figure_embedding = fastvideo_encoder.get_special_token(
        prompt,
        ["<Figure 1>"],
        fastvideo_embedding,
    )

    assert fastvideo_normalized_prompt == official_normalized_prompt
    assert_tensor_exact(fastvideo_input_ids, official_input_ids, "token IDs")
    assert_tensor_exact(
        fastvideo_attention_mask,
        official_attention_mask,
        "attention mask",
    )
    assert_tensor_exact(fastvideo_embedding, official_embedding, "skip-layer-2 hidden state")
    assert_tensor_exact(
        fastvideo_release_embedding,
        official_release_embedding,
        "release-prompt skip-layer-2 hidden state",
    )
    assert_tensor_exact(
        fastvideo_figure_embedding,
        official_figure_embedding,
        "<Figure 1> pooled embedding",
    )
