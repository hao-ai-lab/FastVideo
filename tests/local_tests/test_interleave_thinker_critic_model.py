import json

from fastvideo.train.models.interleave_thinker import (
    INTERLEAVE_CRITIC_PROMPT,
    InterleaveThinkerCriticModel,
)
from fastvideo.train.utils.training_config import (
    DataConfig,
    TrainingConfig,
)


def test_interleave_thinker_critic_builds_qwen_vl_messages_without_loading_backend():
    model = InterleaveThinkerCriticModel(load_backend=False)

    messages = model.build_messages({
        "origin_prompt": "draw a vase",
        "previous_prompt": "a vase on a table",
        "origin_image_path": "before.png",
        "edited_image_path": "after.png",
    })

    content = messages[0]["content"]
    images = [part for part in content if part["type"] == "image"]
    text = "\n".join(part["text"] for part in content if part["type"] == "text")
    assert messages[0]["role"] == "user"
    assert images == [{
        "type": "image",
        "image": "before.png",
    }, {
        "type": "image",
        "image": "after.png",
    }]
    assert "draw a vase" in text
    assert "a vase on a table" in text


def test_interleave_thinker_critic_initializes_jsonl_dataloader(tmp_path):
    data_path = tmp_path / "critic_rl.jsonl"
    data_path.write_text(
        json.dumps({
            "origin_prompt": "draw a chair",
            "rewritten_prompt": "a wooden chair",
            "origin_image_path": "before.png",
            "edited_image_path": "after.png",
            "evaluation": {
                "success": True,
                "semantics": 7.0,
                "quality": 8.0,
            },
        }) + "\n")
    model = InterleaveThinkerCriticModel(load_backend=False)

    model.init_preprocessors(TrainingConfig(data=DataConfig(data_path=str(data_path), train_batch_size=1)))
    batch = next(iter(model.dataloader))

    assert batch["items"][0]["origin_prompt"] == "draw a chair"
    assert batch["items"][0]["evaluation"]["success"] is True
    assert "{original_instruction}" in INTERLEAVE_CRITIC_PROMPT
