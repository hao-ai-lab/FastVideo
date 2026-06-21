# SPDX-License-Identifier: Apache-2.0

from fastvideo.pipelines.pipeline_batch_info import PipelineLoggingInfo
from fastvideo.tests.performance.test_inference_performance import _extract_component_times


def test_extract_component_times_handles_pipeline_logging_info_object():
    logging_info = PipelineLoggingInfo()
    logging_info.add_stage_execution_time("prompt_encoding_stage", 1.25)
    logging_info.add_stage_metric("prompt_encoding_stage", "stage_class", "TextEncodingStage")

    assert _extract_component_times({"logging_info": logging_info}) == {
        "text_encoder_time_s": 1.25,
        "dit_time_s": None,
        "vae_decode_time_s": None,
    }


def test_extract_component_times_uses_stage_class_for_pipeline_stage_keys():
    result = {
        "logging_info": {
            "stages": {
                "prompt_encoding_stage": {
                    "execution_time": 1.2,
                    "stage_class": "TextEncodingStage",
                },
                "denoising_stage": {
                    "execution_time": 3.4,
                    "stage_class": "DenoisingStage",
                },
                "decoding_stage": {
                    "execution_time": 0.8,
                    "stage_class": "DecodingStage",
                },
            },
        },
    }

    assert _extract_component_times(result) == {
        "text_encoder_time_s": 1.2,
        "dit_time_s": 3.4,
        "vae_decode_time_s": 0.8,
    }


def test_extract_component_times_keeps_legacy_class_name_keys():
    result = {
        "logging_info": {
            "stages": {
                "TextEncodingStage": {"execution_time": 1.0},
                "DenoisingStage": {"execution_time": 2.0},
                "DecodingStage": {"execution_time": 3.0},
            },
        },
    }

    assert _extract_component_times(result) == {
        "text_encoder_time_s": 1.0,
        "dit_time_s": 2.0,
        "vae_decode_time_s": 3.0,
    }


def test_extract_component_times_accumulates_duplicate_component_classes():
    result = {
        "logging_info": {
            "stages": {
                "base_denoising_stage": {
                    "execution_time": 2.0,
                    "stage_class": "DenoisingStage",
                },
                "refine_denoising_stage": {
                    "execution_time": 3.5,
                    "stage_class": "DenoisingStage",
                },
            },
        },
    }

    assert _extract_component_times(result) == {
        "text_encoder_time_s": None,
        "dit_time_s": 5.5,
        "vae_decode_time_s": None,
    }


def test_extract_component_times_ignores_unmapped_stages():
    result = {
        "logging_info": {
            "stages": {
                "PostDecodeFrameProcessStage": {"execution_time": 0.2},
                "VideoSaveStage": {"execution_time": 0.4},
                "AudioMuxStage": {"execution_time": 0.1},
            },
        },
    }

    assert _extract_component_times(result) == {
        "text_encoder_time_s": None,
        "dit_time_s": None,
        "vae_decode_time_s": None,
    }


def test_extract_component_times_skips_malformed_stage_data():
    result = {
        "logging_info": {
            "stages": {
                "prompt_encoding_stage": None,
                "denoising_stage": "not-a-stage-metric-dict",
                "decoding_stage": {
                    "execution_time": 0.8,
                    "stage_class": "DecodingStage",
                },
            },
        },
    }

    assert _extract_component_times(result) == {
        "text_encoder_time_s": None,
        "dit_time_s": None,
        "vae_decode_time_s": 0.8,
    }
