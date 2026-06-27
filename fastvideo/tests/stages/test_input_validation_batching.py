# SPDX-License-Identifier: Apache-2.0
from types import SimpleNamespace

from fastvideo.pipelines import ForwardBatch
from fastvideo.pipelines.stages.input_validation import InputValidationStage


def test_input_validation_preserves_explicit_dynamic_batch_seeds() -> None:
    batch = ForwardBatch(
        data_type="video",
        prompt=["one", "two"],
        seed=100,
        seeds=[17, 23],
        height=8,
        width=8,
        num_frames=1,
        num_inference_steps=1,
    )

    InputValidationStage()._generate_seeds(batch, SimpleNamespace())

    assert batch.seeds == [17, 23]
    assert [generator.initial_seed() for generator in batch.generator] == [17, 23]


def test_input_validation_generates_one_seed_per_prompt() -> None:
    batch = ForwardBatch(
        data_type="video",
        prompt=["one", "two"],
        seed=100,
        height=8,
        width=8,
        num_frames=1,
        num_inference_steps=1,
    )

    InputValidationStage()._generate_seeds(batch, SimpleNamespace())

    assert batch.seeds == [100, 101]
    assert [generator.initial_seed() for generator in batch.generator] == [100, 101]
