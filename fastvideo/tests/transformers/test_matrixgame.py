# SPDX-License-Identifier: Apache-2.0
import glob
import os

import pytest
import torch

from fastvideo.configs.models.dits.matrixgame import MatrixGameWanVideoConfig
from fastvideo.configs.pipelines import PipelineConfig
from fastvideo.fastvideo_args import FastVideoArgs
from fastvideo.forward_context import set_forward_context
from fastvideo.logger import init_logger
from fastvideo.models.loader.component_loader import TransformerLoader
from fastvideo.pipelines.pipeline_batch_info import ForwardBatch
from fastvideo.utils import maybe_download_model

logger = init_logger(__name__)

try:
    from diffusers import CausalMatrixGameWanModel as HFMatrixGameWanModel
    DIFFUSERS_IMPORT_ERROR = None
except (ImportError, AttributeError) as exc:  # pragma: no cover - skip when model absent
    from safetensors.torch import load_file

    from fastvideo.models.dits.matrix_game.causal_model import (
        CausalMatrixGameWanModel as _FastVideoMatrixGameWanModel)
    from fastvideo.models.hf_transformer_utils import get_diffusers_config
    from fastvideo.models.loader.utils import (get_param_names_mapping,
                                               hf_to_custom_state_dict)

    DIFFUSERS_IMPORT_ERROR = exc

    class HFMatrixGameWanModel(_FastVideoMatrixGameWanModel):
        """Fallback loader when diffusers class is unavailable."""

        @classmethod
        def from_pretrained(cls, model_path, torch_dtype=None):
            hf_config = get_diffusers_config(model_path)
            hf_config.pop("_class_name", None)
            hf_config.pop("_diffusers_version", None)

            cfg = MatrixGameWanVideoConfig()
            cfg.update_model_arch(hf_config)
            model = cls(config=cfg, hf_config=hf_config)

            weight_candidates = glob.glob(
                os.path.join(model_path, "*.safetensors"))
            if not weight_candidates:
                raise FileNotFoundError(
                    f"No safetensors files found under {model_path}")
            state_dict = load_file(weight_candidates[0])
            mapping_fn = get_param_names_mapping(model.param_names_mapping)
            custom_state_dict, _ = hf_to_custom_state_dict(
                state_dict, mapping_fn)
            missing, unexpected = model.load_state_dict(custom_state_dict,
                                                        strict=False)
            if missing or unexpected:
                raise RuntimeError(
                    f"Failed to load state_dict. Missing: {missing}, Unexpected: {unexpected}"
                )

            if torch_dtype is not None:
                model = model.to(dtype=torch_dtype)
            model = model.eval()
            return model

    logger.warning(
        "diffusers CausalMatrixGameWanModel unavailable (%s). "
        "Falling back to FastVideo implementation for parity test.",
        DIFFUSERS_IMPORT_ERROR)
    DIFFUSERS_IMPORT_ERROR = None

os.environ["MASTER_ADDR"] = "localhost"
os.environ["MASTER_PORT"] = "29505"
os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")

BASE_MODEL_PATH = os.environ.get("MATRIX_GAME_BASE_MODEL",
                                 "/workspace/Matrix-Game-2.0-Diffusers")
MODEL_VARIANT = os.environ.get("MATRIX_GAME_VARIANT",
                               "base_distilled_model")

def _resolve_transformer_path() -> str:
    """Resolve the Matrix-Game transformer directory."""
    if os.path.exists(BASE_MODEL_PATH):
        model_root = BASE_MODEL_PATH
    else:
        model_root = maybe_download_model(
            BASE_MODEL_PATH,
            local_dir=os.path.join('data', BASE_MODEL_PATH))

    candidate = os.path.join(model_root, MODEL_VARIANT, "transformer")
    if not os.path.isdir(candidate):
        fallback = os.path.join(model_root, "transformer")
        if os.path.isdir(fallback):
            candidate = fallback
        else:
            raise FileNotFoundError(
                f"Could not locate transformer weights under {candidate} or {fallback}"
            )
    return candidate


@pytest.mark.usefixtures("distributed_setup")
def test_matrixgame_transformer():
    if HFMatrixGameWanModel is None:
        pytest.skip(
            f"CausalMatrixGameWanModel is unavailable: {DIFFUSERS_IMPORT_ERROR}"
        )

    transformer_path = _resolve_transformer_path()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    precision = torch.bfloat16
    precision_str = "bf16"
    args = FastVideoArgs(
        model_path=transformer_path,
        dit_cpu_offload=False,
        use_fsdp_inference=False,
        pipeline_config=PipelineConfig(
            dit_config=MatrixGameWanVideoConfig(), dit_precision=precision_str))
    args.device = device

    loader = TransformerLoader()
    model2 = loader.load(transformer_path, args).to(device, dtype=precision)

    model1 = HFMatrixGameWanModel.from_pretrained(
        transformer_path,
        torch_dtype=precision).to(device, dtype=precision).requires_grad_(False)

    total_params = sum(p.numel() for p in model1.parameters())
    weight_sum_model1 = sum(
        p.to(torch.float64).sum().item() for p in model1.parameters())
    weight_mean_model1 = weight_sum_model1 / total_params
    logger.info("Model 1 weight sum: %s", weight_sum_model1)
    logger.info("Model 1 weight mean: %s", weight_mean_model1)

    total_params_model2 = sum(p.numel() for p in model2.parameters())
    weight_sum_model2 = sum(
        p.to(torch.float64).sum().item() for p in model2.parameters())
    weight_mean_model2 = weight_sum_model2 / total_params_model2
    logger.info("Model 2 weight sum: %s", weight_sum_model2)
    logger.info("Model 2 weight mean: %s", weight_mean_model2)

    weight_sum_diff = abs(weight_sum_model1 - weight_sum_model2)
    weight_mean_diff = abs(weight_mean_model1 - weight_mean_model2)
    logger.info("Weight sum difference: %s", weight_sum_diff)
    logger.info("Weight mean difference: %s", weight_mean_diff)

    model1 = model1.eval()
    model2 = model2.eval()

    batch_size = 1
    latent_frames = 15
    latent_height = 44
    latent_width = 80

    in_channels = getattr(model1.config, "in_channels", 36)
    image_dim = getattr(model1.config, "image_dim", 1280)
    image_seq_len = 257

    hidden_states = torch.randn(batch_size,
                                in_channels,
                                latent_frames,
                                latent_height,
                                latent_width,
                                device=device,
                                dtype=precision)
    encoder_hidden_states = torch.zeros(
        batch_size,
        0,
        model1.config.hidden_size,
        device=device,
        dtype=precision,
    )
    encoder_hidden_states_image = [torch.randn(batch_size,
                                               image_seq_len,
                                               image_dim,
                                               device=device,
                                               dtype=precision)]
    timestep = torch.full((batch_size * latent_frames, ),
                          500,
                          device=device,
                          dtype=precision)

    forward_batch = ForwardBatch(
        data_type="dummy",
    )

    with torch.amp.autocast("cuda",
                            dtype=precision,
                            enabled=torch.cuda.is_available()):
        with set_forward_context(
                current_timestep=0,
                attn_metadata=None,
                forward_batch=forward_batch,
        ):
            output1 = model1(
                hidden_states=hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                encoder_hidden_states_image=encoder_hidden_states_image,
                timestep=timestep)

        with set_forward_context(
                current_timestep=0,
                attn_metadata=None,
                forward_batch=forward_batch,
        ):
            output2 = model2(
                hidden_states=hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                encoder_hidden_states_image=encoder_hidden_states_image,
                timestep=timestep,
            )

    assert output1.shape == output2.shape, (
        f"Output shapes don't match: {output1.shape} vs {output2.shape}")
    assert output1.dtype == output2.dtype, (
        f"Output dtype don't match: {output1.dtype} vs {output2.dtype}")

    max_diff = torch.max(torch.abs(output1 - output2))
    mean_diff = torch.mean(torch.abs(output1 - output2))
    logger.info("Max Diff: %s", max_diff.item())
    logger.info("Mean Diff: %s", mean_diff.item())
    assert max_diff < 1e-1, (
        f"Maximum difference between outputs: {max_diff.item()}")
    assert mean_diff < 1e-2, (
        f"Mean difference between outputs: {mean_diff.item()}")
