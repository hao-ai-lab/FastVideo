# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import gc
import os
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

_UPSTREAM_ENV = "INTERLEAVETHINKER_UPSTREAM_REPO"
_REAL_PARITY_ENV = "INTERLEAVETHINKER_REAL_PARITY"


pytestmark = pytest.mark.skipif(
    not os.environ.get(_UPSTREAM_ENV),
    reason=f"{_UPSTREAM_ENV} must point to an official InterleaveThinker checkout",
)


def _upstream_root() -> Path:
    root = Path(os.environ[_UPSTREAM_ENV]).expanduser().resolve()
    if not (root / "demo_klein.py").is_file():
        raise RuntimeError(f"{_UPSTREAM_ENV} does not look like InterleaveThinker: {root}")
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    return root


def _strip_line_end_whitespace(text: str) -> str:
    return "\n".join(line.rstrip() for line in text.splitlines())


def _reset_torch_rng(seed: int = 0) -> None:
    import torch

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def test_prompt_templates_match_official_reference() -> None:
    _upstream_root()

    from UEval.system import (  # type: ignore[import-not-found]
        GUIDANCE_GLOBAL_PROMPT_JSON,
        Iterative_T2I_PROMPT_QWEN,
        NARRATIVE_PROMPT_JSON,
    )
    from fastvideo.train.models.interleave_thinker import (
        INTERLEAVE_GUIDANCE_PLANNER_PROMPT,
        INTERLEAVE_PLANNER_PROMPT,
    )
    from fastvideo.train.models.interleave_thinker.critic import INTERLEAVE_CRITIC_PROMPT

    assert _strip_line_end_whitespace(INTERLEAVE_PLANNER_PROMPT) == _strip_line_end_whitespace(
        NARRATIVE_PROMPT_JSON)
    assert _strip_line_end_whitespace(INTERLEAVE_GUIDANCE_PLANNER_PROMPT) == _strip_line_end_whitespace(
        GUIDANCE_GLOBAL_PROMPT_JSON)

    original_instruction = "draw a blue square"
    rewritten_prompt = "a crisp blue square centered on a white background"
    upstream_critic_prompt = ("<image><image>\n" + Iterative_T2I_PROMPT_QWEN).replace(
        "{original_instruction}",
        original_instruction,
    ).replace(
        "{rewritten_prompt}",
        rewritten_prompt,
    )
    fastvideo_critic_prompt = INTERLEAVE_CRITIC_PROMPT.format(
        original_instruction=original_instruction,
        rewritten_prompt=rewritten_prompt,
    )
    assert _strip_line_end_whitespace(fastvideo_critic_prompt) == _strip_line_end_whitespace(upstream_critic_prompt)


def test_text_image_messages_match_official_demo_constructor() -> None:
    _upstream_root()

    sys.modules.setdefault("json_repair", SimpleNamespace(loads=lambda text: text))
    from demo_klein import SingleSampleGenerator  # type: ignore[import-not-found]
    from fastvideo.train.models.interleave_thinker import InterleaveThinkerPlannerModel

    upstream = SingleSampleGenerator.__new__(SingleSampleGenerator)
    fastvideo = InterleaveThinkerPlannerModel(load_backend=False)

    cases = [
        ("plain text prompt", None),
        ("<image> edit this image", ["one.png"]),
        ("before <image> middle <image> after", ["one.png", "two.png"]),
        ("<image><image><image><image><image> summarize the sequence", [
            "one.png",
            "two.png",
            "three.png",
            "four.png",
            "five.png",
        ]),
    ]
    for prompt, images in cases:
        assert fastvideo.build_text_image_messages(prompt, images) == upstream.construct_msgs(prompt, images)


def test_qwen_response_generation_matches_official_predict_with_fake_backend() -> None:
    _upstream_root()

    import torch
    from UEval.qwen3_vl_api import predict as upstream_predict  # type: ignore[import-not-found]
    from fastvideo.train.models.interleave_thinker.qwen_actor import Qwen3VLActorBase

    class Batch(dict):

        def __getattr__(self, name: str):
            return self[name]

        def to(self, device):
            self["device"] = str(device)
            return self

    class FakeProcessor:

        def __init__(self) -> None:
            self.tokenizer = SimpleNamespace(pad_token_id=0, eos_token_id=2)

        def apply_chat_template(
            self,
            messages,
            *,
            tokenize,
            add_generation_prompt,
            return_dict,
            return_tensors,
        ):
            del messages, tokenize, add_generation_prompt, return_dict, return_tensors
            return Batch({
                "input_ids": torch.tensor([[1, 2, 3]], dtype=torch.long),
                "attention_mask": torch.ones(1, 3, dtype=torch.long),
            })

        def batch_decode(self, sequences, **kwargs):
            del kwargs
            return [" ".join(str(int(token)) for token in sequence.flatten()) for sequence in sequences]

    class FakeQwen(torch.nn.Module):

        def __init__(self) -> None:
            super().__init__()
            self.device = torch.device("cpu")

        def generate(self, **kwargs):
            input_ids = kwargs["input_ids"]
            num_return_sequences = int(kwargs.get("num_return_sequences", 1))
            suffix = torch.tensor([[9, 10]], dtype=input_ids.dtype)
            return torch.cat(
                [
                    input_ids.repeat(num_return_sequences, 1),
                    suffix.repeat(num_return_sequences, 1),
                ],
                dim=1,
            )

    class FakeActor(Qwen3VLActorBase):

        @property
        def device(self):
            return torch.device("cpu")

    messages = [{"role": "user", "content": [{"type": "text", "text": "hello"}]}]
    processor = FakeProcessor()
    qwen = FakeQwen()

    upstream_response = upstream_predict(qwen, processor, messages, max_new_tokens=7)

    actor = FakeActor(init_from="unused", load_backend=False, trainable=False)
    actor.processor = processor
    actor.transformer = qwen
    fastvideo_response = actor.generate_qwen_responses(
        messages,
        num_generations=1,
        temperature=0.0,
        top_p=1.0,
        max_new_tokens=7,
    )[0]

    assert fastvideo_response == upstream_response


@pytest.mark.skipif(
    os.environ.get(_REAL_PARITY_ENV) != "1",
    reason=f"{_REAL_PARITY_ENV}=1 is required for real checkpoint parity",
)
def test_real_planner_generation_matches_official_predict() -> None:
    _upstream_root()

    import torch
    from UEval.qwen3_vl_api import predict as upstream_predict  # type: ignore[import-not-found]
    from fastvideo.train.models.interleave_thinker import InterleaveThinkerPlannerModel

    model = InterleaveThinkerPlannerModel(
        init_from=os.environ.get("INTERLEAVETHINKER_PLANNER_CKPT", "InterleaveThinker/InterleaveThinker-Planner-8B"),
        processor_from=os.environ.get("INTERLEAVETHINKER_PROCESSOR_CKPT", "Qwen/Qwen3-VL-8B-Instruct"),
        trainable=False,
        torch_dtype=os.environ.get("INTERLEAVETHINKER_TORCH_DTYPE", "auto"),
        device_map=os.environ.get("INTERLEAVETHINKER_DEVICE_MAP", "cuda:0"),
        attn_implementation=os.environ.get("INTERLEAVETHINKER_ATTN_IMPL", "sdpa"),
    )
    try:
        messages = model.build_messages({"instruction": "How to draw a cat step by step?"})
        _reset_torch_rng()
        upstream_response = upstream_predict(
            model.transformer,
            model.processor,
            messages,
            max_new_tokens=64,
        )
        _reset_torch_rng()
        fastvideo_response = model.generate_qwen_responses(
            messages,
            num_generations=1,
            temperature=0.0,
            top_p=1.0,
            max_new_tokens=64,
        )[0]

        assert fastvideo_response == upstream_response
    finally:
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


@pytest.mark.skipif(
    os.environ.get(_REAL_PARITY_ENV) != "1",
    reason=f"{_REAL_PARITY_ENV}=1 is required for real checkpoint parity",
)
def test_real_critic_generation_matches_official_predict(tmp_path: Path) -> None:
    _upstream_root()

    import torch
    from PIL import Image
    from UEval.qwen3_vl_api import predict as upstream_predict  # type: ignore[import-not-found]
    from fastvideo.train.models.interleave_thinker import InterleaveThinkerCriticModel

    before_path = tmp_path / "before.png"
    after_path = tmp_path / "after.png"
    Image.new("RGB", (128, 128), "white").save(before_path)
    Image.new("RGB", (128, 128), "blue").save(after_path)

    model = InterleaveThinkerCriticModel(
        init_from=os.environ.get("INTERLEAVETHINKER_CRITIC_CKPT", "InterleaveThinker/Critic-SFT-8B"),
        processor_from=os.environ.get("INTERLEAVETHINKER_PROCESSOR_CKPT", "Qwen/Qwen3-VL-8B-Instruct"),
        trainable=False,
        torch_dtype=os.environ.get("INTERLEAVETHINKER_TORCH_DTYPE", "auto"),
        device_map=os.environ.get("INTERLEAVETHINKER_DEVICE_MAP", "cuda:0"),
        attn_implementation=os.environ.get("INTERLEAVETHINKER_ATTN_IMPL", "sdpa"),
    )
    try:
        messages = model.build_messages({
            "origin_prompt": "draw a blue square",
            "previous_prompt": "a crisp blue square centered on a white background",
            "previous_image_path": str(before_path),
            "edited_image_path": str(after_path),
        })
        _reset_torch_rng()
        upstream_response = upstream_predict(
            model.transformer,
            model.processor,
            messages,
            max_new_tokens=64,
        )
        _reset_torch_rng()
        fastvideo_response = model.generate_qwen_responses(
            messages,
            num_generations=1,
            temperature=0.0,
            top_p=1.0,
            max_new_tokens=64,
        )[0]

        assert fastvideo_response == upstream_response
    finally:
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
