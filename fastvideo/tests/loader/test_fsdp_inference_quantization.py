# SPDX-License-Identifier: Apache-2.0
"""FSDP inference accepts unquantized transformers and NVFP4QATTrainConfig."""

from types import SimpleNamespace

import pytest
import torch

from fastvideo.layers.quantization.fp8_qat_train_config import FP8QATTrainConfig
from fastvideo.layers.quantization.mxfp8_config import MXFP8Config
from fastvideo.layers.quantization.nvfp4_config import NVFP4Config
from fastvideo.layers.quantization.nvfp4_qat_train_config import (
    NVFP4QATTrainConfig, )
from fastvideo.models.loader.fsdp_load import (
    _validate_fsdp_inference_quantization,
    maybe_load_fsdp_model,
)


def _init_params(quant_config: object | None) -> dict[str, object]:
    return {"config": SimpleNamespace(quant_config=quant_config)}


@pytest.mark.parametrize("quant_config", [MXFP8Config(), NVFP4Config(), FP8QATTrainConfig()])
def test_maybe_load_fsdp_model_unsupported_quantization_rejected_before_construction(quant_config: object) -> None:
    """Reject unsupported quantization before model construction or FSDP sharding."""
    with pytest.raises(NotImplementedError, match=type(quant_config).__name__):
        maybe_load_fsdp_model(
            model_cls=torch.nn.Module,
            init_params=_init_params(quant_config),
            weight_dir_list=[],
            device=torch.device("cpu"),
            hsdp_replicate_dim=1,
            hsdp_shard_dim=1,
            default_dtype=torch.bfloat16,
            param_dtype=torch.bfloat16,
            reduce_dtype=torch.float32,
            fsdp_inference=True,
            training_mode=False,
        )


def test_validate_fsdp_inference_quantization_nvfp4_qat_training_allowed() -> None:
    _validate_fsdp_inference_quantization(_init_params(NVFP4QATTrainConfig()), fsdp_inference=True)


def test_validate_fsdp_inference_quantization_unquantized_allowed() -> None:
    _validate_fsdp_inference_quantization(_init_params(None), fsdp_inference=True)


def test_validate_fsdp_inference_quantization_without_fsdp_allowed() -> None:
    _validate_fsdp_inference_quantization(_init_params(MXFP8Config()), fsdp_inference=False)
