# SPDX-License-Identifier: Apache-2.0
import os
from pathlib import Path
import sys

import pytest
import torch
from torch.testing import assert_close

from safetensors.torch import load_file

from fastvideo.configs.models.encoders import LTX2GemmaConfig
from fastvideo.models.encoders.gemma import LTX2GemmaTextEncoderModel
from fastvideo.models.loader.component_loader import get_diffusers_config


repo_root = Path(__file__).resolve().parents[3]
ltx_core_path = repo_root / "LTX-2" / "packages" / "ltx-core" / "src"
if ltx_core_path.exists() and str(ltx_core_path) not in sys.path:
    sys.path.insert(0, str(ltx_core_path))


def _init_log_paths() -> tuple[Path, Path]:
    base_dir = Path(os.getenv("LTX2_DEBUG_DIR", "ltx2_debug"))
    fastvideo_log = Path(os.getenv("LTX2_FASTVIDEO_GEMMA_LOG", base_dir / "fastvideo_gemma.log"))
    reference_log = Path(os.getenv("LTX2_REFERENCE_GEMMA_LOG", base_dir / "reference_gemma.log"))
    fastvideo_log.parent.mkdir(parents=True, exist_ok=True)
    reference_log.parent.mkdir(parents=True, exist_ok=True)
    fastvideo_log.write_text("")
    reference_log.write_text("")
    return fastvideo_log, reference_log


def _log_line(path: Path, message: str) -> None:
    with path.open("a", encoding="utf-8") as f:
        f.write(message + "\n")


def _attach_encoder_logging(
    encoder: torch.nn.Module,
    log_path: Path,
    label: str,
) -> None:
    def _format_sum(tensor: torch.Tensor | None) -> str:
        if tensor is None:
            return "None"
        return f"{tensor.float().sum().item():.6f}"

    def _hook_factory(name: str):
        def _hook(_module, _inputs, outputs):  # noqa: ANN001
            out = outputs[0] if isinstance(outputs, tuple) else outputs
            out_sum = _format_sum(out if torch.is_tensor(out) else None)
            _log_line(log_path, f"{label}:{name}:sum={out_sum}")
        return _hook

    def _pre_hook_factory(name: str):
        def _hook(_module, inputs):  # noqa: ANN001
            tensor = inputs[0] if inputs else None
            in_sum = _format_sum(tensor if torch.is_tensor(tensor) else None)
            _log_line(log_path, f"{label}:{name}:in_sum={in_sum}")
        return _hook

    def _attach_block_detail(block: torch.nn.Module, prefix: str) -> None:
        block.register_forward_pre_hook(_pre_hook_factory(f"{prefix}:input"))
        block.register_forward_hook(_hook_factory(f"{prefix}:output"))
        if hasattr(block, "attn1"):
            block.attn1.register_forward_hook(
                _hook_factory(f"{prefix}:attn1"))
        if hasattr(block, "ff"):
            block.ff.register_forward_hook(_hook_factory(f"{prefix}:ff"))

    if hasattr(encoder, "feature_extractor_linear"):
        encoder.feature_extractor_linear.register_forward_pre_hook(
            _pre_hook_factory("feature_extractor_linear"))
        encoder.feature_extractor_linear.register_forward_hook(_hook_factory("feature_extractor_linear"))
    if hasattr(encoder, "embeddings_connector"):
        encoder.embeddings_connector.register_forward_hook(_hook_factory("embeddings_connector"))
        for idx, block in enumerate(encoder.embeddings_connector.transformer_1d_blocks):
            _attach_block_detail(block, f"embeddings_block_{idx}")
    if hasattr(encoder, "audio_embeddings_connector"):
        encoder.audio_embeddings_connector.register_forward_hook(_hook_factory("audio_embeddings_connector"))
        for idx, block in enumerate(encoder.audio_embeddings_connector.transformer_1d_blocks):
            _attach_block_detail(block, f"audio_embeddings_block_{idx}")


def _log_register_sums(encoder: torch.nn.Module, log_path: Path, label: str) -> None:
    def _sum_param(module: torch.nn.Module, name: str) -> float | None:
        if not hasattr(module, "learnable_registers"):
            return None
        param = getattr(module, "learnable_registers")
        if not torch.is_tensor(param):
            return None
        return param.float().sum().item()

    if hasattr(encoder, "embeddings_connector"):
        reg_sum = _sum_param(encoder.embeddings_connector, "learnable_registers")
        if reg_sum is not None:
            _log_line(log_path, f"{label}:embeddings_registers:sum={reg_sum:.6f}")
    if hasattr(encoder, "audio_embeddings_connector"):
        reg_sum = _sum_param(encoder.audio_embeddings_connector, "learnable_registers")
        if reg_sum is not None:
            _log_line(log_path, f"{label}:audio_registers:sum={reg_sum:.6f}")


def _log_param_sums(encoder: torch.nn.Module, log_path: Path, label: str) -> None:
    def _log_param(name: str, tensor: torch.Tensor | None) -> None:
        if tensor is None:
            _log_line(log_path, f"{label}:param:{name}:sum=None")
            return
        _log_line(log_path, f"{label}:param:{name}:sum={tensor.float().sum().item():.6f}")

    if hasattr(encoder, "feature_extractor_linear"):
        _log_param(
            "feature_extractor_linear.aggregate_embed.weight",
            encoder.feature_extractor_linear.aggregate_embed.weight,
        )
    if hasattr(encoder, "embeddings_connector"):
        block0 = encoder.embeddings_connector.transformer_1d_blocks[0]
        _log_param("embeddings_block0.attn1.to_q.weight", block0.attn1.to_q.weight)
        _log_param("embeddings_block0.attn1.to_k.weight", block0.attn1.to_k.weight)
        _log_param("embeddings_block0.attn1.to_v.weight", block0.attn1.to_v.weight)
        _log_param("embeddings_block0.ff.net.0.proj.weight", block0.ff.net[0].proj.weight)
    if hasattr(encoder, "audio_embeddings_connector"):
        block0 = encoder.audio_embeddings_connector.transformer_1d_blocks[0]
        _log_param("audio_block0.attn1.to_q.weight", block0.attn1.to_q.weight)
        _log_param("audio_block0.attn1.to_k.weight", block0.attn1.to_k.weight)
        _log_param("audio_block0.attn1.to_v.weight", block0.attn1.to_v.weight)
        _log_param("audio_block0.ff.net.0.proj.weight", block0.ff.net[0].proj.weight)


def _log_gemma_param_sums(
    gemma_model: torch.nn.Module | None,
    log_path: Path,
    label: str,
) -> None:
    if gemma_model is None:
        _log_line(log_path, f"{label}:gemma_param:embed_tokens.weight:sum=None")
        return
    tokens = None
    if hasattr(gemma_model, "get_input_embeddings"):
        try:
            tokens = gemma_model.get_input_embeddings()
        except Exception:
            tokens = None
    if tokens is None:
        embed = getattr(gemma_model, "model", None)
        if embed is not None:
            tokens = getattr(embed, "embed_tokens", None)
    if tokens is None or not hasattr(tokens, "weight"):
        _log_line(log_path, f"{label}:gemma_param:embed_tokens.weight:sum=None")
        return
    _log_line(
        log_path,
        f"{label}:gemma_param:embed_tokens.weight:sum={tokens.weight.float().sum().item():.6f}",
    )


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="LTX-2 Gemma parity test requires CUDA.",
)
def test_ltx2_gemma_parity():
    os.environ["FASTVIDEO_ATTENTION_BACKEND"] = "TORCH_SDPA"
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_math_sdp(True)
    fastvideo_log, reference_log = _init_log_paths()
    diffusers_root = Path(
        os.getenv("LTX2_DIFFUSERS_PATH", "converted/ltx2_diffusers")
    )
    official_path = Path(
        os.getenv(
            "LTX2_OFFICIAL_PATH",
            "official_ltx_weights/ltx-2-19b-distilled.safetensors",
        )
    )
    text_encoder_path = diffusers_root / "text_encoder"
    gemma_path = text_encoder_path / "gemma"

    if not official_path.exists():
        pytest.skip(f"LTX-2 weights not found at {official_path}")
    if not text_encoder_path.exists():
        pytest.skip(f"LTX-2 text encoder not found at {text_encoder_path}")
    if not gemma_path.exists():
        pytest.skip(f"LTX-2 Gemma weights not found at {gemma_path}")

    try:
        from ltx_core.loader.single_gpu_model_builder import SingleGPUModelBuilder
        from ltx_core.model.transformer.attention import Attention, AttentionFunction
        from ltx_core.text_encoders.gemma import (
            AV_GEMMA_TEXT_ENCODER_KEY_OPS,
            AVGemmaTextEncoderModelConfigurator,
            module_ops_from_gemma_root,
        )
    except ImportError as exc:
        pytest.skip(f"LTX-2 import failed: {exc}")

    device = torch.device("cuda:0")
    precision = torch.bfloat16

    reference_builder = SingleGPUModelBuilder(
        model_path=str(official_path),
        model_class_configurator=AVGemmaTextEncoderModelConfigurator,
        model_sd_ops=AV_GEMMA_TEXT_ENCODER_KEY_OPS,
        module_ops=module_ops_from_gemma_root(str(gemma_path)),
    )
    reference_encoder = reference_builder.build(
        device=device, dtype=precision
    ).to(device=device, dtype=precision)
    if hasattr(reference_encoder.model, "config"):
        if hasattr(reference_encoder.model.config, "attn_implementation"):
            reference_encoder.model.config.attn_implementation = "sdpa"
        if hasattr(reference_encoder.model.config, "_attn_implementation"):
            reference_encoder.model.config._attn_implementation = "sdpa"
    for module in reference_encoder.modules():
        if isinstance(module, Attention):
            module.attention_function = AttentionFunction.PYTORCH
    reference_encoder.eval()
    _attach_encoder_logging(reference_encoder, reference_log, "reference")
    _log_register_sums(reference_encoder, reference_log, "reference")
    _log_param_sums(reference_encoder, reference_log, "reference")
    _log_gemma_param_sums(reference_encoder.model, reference_log, "reference")
    _log_line(
        reference_log,
        "reference:gemma_config:attn_impl="
        f"{getattr(reference_encoder.model.config, 'attn_implementation', None)} "
        f"dtype={reference_encoder.model.dtype}",
    )

    diffusers_config = get_diffusers_config(model=str(text_encoder_path))
    encoder_config = LTX2GemmaConfig()
    encoder_config.update_model_arch(diffusers_config)
    encoder_config.arch_config.gemma_model_path = str(gemma_path)
    fastvideo_encoder = LTX2GemmaTextEncoderModel(encoder_config).to(
        device=device, dtype=precision
    )
    if hasattr(fastvideo_encoder.gemma_model, "config"):
        if hasattr(fastvideo_encoder.gemma_model.config, "attn_implementation"):
            fastvideo_encoder.gemma_model.config.attn_implementation = "sdpa"
        if hasattr(fastvideo_encoder.gemma_model.config, "_attn_implementation"):
            fastvideo_encoder.gemma_model.config._attn_implementation = "sdpa"

    official_weights = load_file(str(official_path))
    fastvideo_weights: dict[str, torch.Tensor] = {}
    for name, tensor in official_weights.items():
        mapped_name = AV_GEMMA_TEXT_ENCODER_KEY_OPS.apply_to_key(name)
        if mapped_name is None:
            continue
        fastvideo_weights[mapped_name] = tensor
    fastvideo_encoder.load_weights(fastvideo_weights.items())
    fastvideo_encoder.eval()
    _attach_encoder_logging(fastvideo_encoder, fastvideo_log, "fastvideo")
    _log_register_sums(fastvideo_encoder, fastvideo_log, "fastvideo")
    _log_param_sums(fastvideo_encoder, fastvideo_log, "fastvideo")
    _log_gemma_param_sums(fastvideo_encoder.gemma_model, fastvideo_log, "fastvideo")
    _log_line(
        fastvideo_log,
        "fastvideo:gemma_config:attn_impl="
        f"{getattr(fastvideo_encoder.gemma_model.config, 'attn_implementation', None)} "
        f"dtype={fastvideo_encoder.gemma_model.dtype}",
    )

    prompt = "A curious raccoon peers through a vibrant field of yellow sunflowers."
    ref_tokenizer = reference_encoder.tokenizer
    if ref_tokenizer is None:
        pytest.skip("Reference tokenizer is not initialized.")
    token_pairs = ref_tokenizer.tokenize_with_weights(prompt)["gemma"]
    input_ids = torch.tensor([[t[0] for t in token_pairs]], device=device)
    attention_mask = torch.tensor([[w[1] for w in token_pairs]], device=device)

    with torch.no_grad(), torch.backends.cuda.sdp_kernel(
        enable_flash=False,
        enable_mem_efficient=False,
        enable_math=True,
    ):
        ref_outputs = reference_encoder.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True,
            use_cache=False,
        )
        _log_line(
            reference_log,
            "reference:gemma_hidden_last:sum="
            f"{ref_outputs.hidden_states[-1].float().sum().item():.6f}",
        )
        ref_projected = reference_encoder._run_feature_extractor(
            ref_outputs.hidden_states,
            attention_mask=attention_mask,
            padding_side="left",
        )
        # Compare rotary embeddings between implementations.
        from ltx_core.model.transformer.rope import precompute_freqs_cis
        from fastvideo.models.dits.ltx2 import precompute_ltx_freqs_cis

        seq_len = ref_projected.shape[1]
        indices_grid = torch.arange(seq_len, device=device, dtype=torch.float32)[None, None, :]
        ref_cos, ref_sin = precompute_freqs_cis(
            indices_grid=indices_grid,
            dim=reference_encoder.embeddings_connector.inner_dim,
            out_dtype=ref_projected.dtype,
            theta=reference_encoder.embeddings_connector.positional_embedding_theta,
            max_pos=reference_encoder.embeddings_connector.positional_embedding_max_pos,
            num_attention_heads=reference_encoder.embeddings_connector.num_attention_heads,
            rope_type=reference_encoder.embeddings_connector.rope_type,
        )
        fast_cos, fast_sin = precompute_ltx_freqs_cis(
            indices_grid=indices_grid,
            dim=fastvideo_encoder.embeddings_connector.inner_dim,
            out_dtype=ref_projected.dtype,
            theta=fastvideo_encoder.embeddings_connector.positional_embedding_theta,
            max_pos=fastvideo_encoder.embeddings_connector.positional_embedding_max_pos,
            num_attention_heads=fastvideo_encoder.embeddings_connector.num_attention_heads,
            rope_type=fastvideo_encoder.embeddings_connector.rope_type,
        )
        _log_line(
            reference_log,
            "reference:rope:cos_sum="
            f"{ref_cos.float().sum().item():.6f} sin_sum={ref_sin.float().sum().item():.6f} "
            f"theta={reference_encoder.embeddings_connector.positional_embedding_theta} "
            f"max_pos={reference_encoder.embeddings_connector.positional_embedding_max_pos} "
            f"rope_type={reference_encoder.embeddings_connector.rope_type}"
        )
        _log_line(
            fastvideo_log,
            "fastvideo:rope:cos_sum="
            f"{fast_cos.float().sum().item():.6f} sin_sum={fast_sin.float().sum().item():.6f} "
            f"theta={fastvideo_encoder.embeddings_connector.positional_embedding_theta} "
            f"max_pos={fastvideo_encoder.embeddings_connector.positional_embedding_max_pos} "
            f"rope_type={fastvideo_encoder.embeddings_connector.rope_type}"
        )
        ref_video, ref_audio, _ = reference_encoder._run_connectors(
            ref_projected, attention_mask
        )
        fast_video_from_ref, fast_audio_from_ref, _ = fastvideo_encoder._run_connectors(
            ref_projected, attention_mask
        )
        _log_line(
            fastvideo_log,
            "fastvideo:connector_on_ref:video_sum="
            f"{fast_video_from_ref.float().sum().item():.6f}",
        )
        _log_line(
            fastvideo_log,
            "fastvideo:connector_on_ref:audio_sum="
            f"{fast_audio_from_ref.float().sum().item():.6f}",
        )

        fast_outputs = fastvideo_encoder.gemma_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True,
            use_cache=False,
        )
        _log_line(
            fastvideo_log,
            "fastvideo:gemma_hidden_last:sum="
            f"{fast_outputs.hidden_states[-1].float().sum().item():.6f}",
        )
        fast_projected = fastvideo_encoder._run_feature_extractor(
            fast_outputs.hidden_states,
            attention_mask=attention_mask,
            padding_side=fastvideo_encoder.padding_side,
        )
        fast_video, fast_audio, _ = fastvideo_encoder._run_connectors(
            fast_projected, attention_mask
        )

    assert ref_video.shape == fast_video.shape
    assert ref_audio.shape == fast_audio.shape
    assert torch.isfinite(ref_video).all(), "Reference Gemma produced non-finite video embeddings."
    assert torch.isfinite(ref_audio).all(), "Reference Gemma produced non-finite audio embeddings."
    assert torch.isfinite(fast_video).all(), "FastVideo Gemma produced non-finite video embeddings."
    assert torch.isfinite(fast_audio).all(), "FastVideo Gemma produced non-finite audio embeddings."
    assert_close(ref_video, fast_video, atol=3e-1, rtol=5e-2)
    assert_close(ref_audio, fast_audio, atol=3e-1, rtol=5e-2)
