# SPDX-License-Identifier: Apache-2.0
"""Public API contracts for the Helios-Distilled pipeline."""

from argparse import ArgumentParser
from dataclasses import fields

from fastvideo.api.compat import normalize_generation_request, request_to_sampling_param
from fastvideo.api.sampling_param import SamplingParam
from fastvideo.api.schema import SamplingConfig
from fastvideo.pipelines.pipeline_batch_info import ForwardBatch


HELIOS_SAMPLING_VALUES = {
    "pyramid_num_inference_steps_list": [2, 2, 2],
    "history_sizes": [16, 2, 1],
    "num_latent_frames_per_chunk": 9,
    "keep_first_frame": True,
    "is_skip_first_chunk": False,
    "use_zero_init": True,
    "zero_steps": 1,
    "is_amplify_first_chunk": True,
}


def test_helios_public_sampling_fields_reach_forward_batch() -> None:
    sampling_fields = {item.name for item in fields(SamplingParam)}
    typed_fields = {item.name for item in fields(SamplingConfig)}
    batch_fields = {item.name for item in fields(ForwardBatch)}
    expected = set(HELIOS_SAMPLING_VALUES)

    assert expected <= sampling_fields
    assert expected <= typed_fields
    assert expected <= batch_fields


def test_helios_typed_request_maps_sampling_fields() -> None:
    request = normalize_generation_request({"sampling": HELIOS_SAMPLING_VALUES})
    sampling = request_to_sampling_param(
        request,
        model_path="BestWishYsh/Helios-Distilled",
    )

    for name, expected in HELIOS_SAMPLING_VALUES.items():
        assert getattr(sampling, name) == expected


def test_helios_sampling_fields_are_available_from_cli() -> None:
    parser = ArgumentParser()
    SamplingParam.add_cli_args(parser)

    args = parser.parse_args([
        "--pyramid-num-inference-steps-list",
        "3",
        "2",
        "1",
        "--history-sizes",
        "8",
        "2",
        "1",
        "--num-latent-frames-per-chunk",
        "9",
        "--keep-first-frame",
        "true",
        "--is-skip-first-chunk",
        "false",
        "--use-zero-init",
        "true",
        "--zero-steps",
        "2",
        "--is-amplify-first-chunk",
        "true",
    ])

    assert args.pyramid_num_inference_steps_list == [3, 2, 1]
    assert args.history_sizes == [8, 2, 1]
    assert args.num_latent_frames_per_chunk == 9
    assert args.keep_first_frame is True
    assert args.is_skip_first_chunk is False
    assert args.use_zero_init is True
    assert args.zero_steps == 2
    assert args.is_amplify_first_chunk is True
