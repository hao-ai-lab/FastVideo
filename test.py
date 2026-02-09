# SPDX-License-Identifier: Apache-2.0
import importlib
from itertools import combinations
import os
from pathlib import Path
import sys
import types

import pytest
import torch
from torch.testing import assert_close

os.environ.setdefault("MASTER_ADDR", "localhost")
os.environ.setdefault("MASTER_PORT", "29517")
os.environ.setdefault("FASTVIDEO_ATTENTION_BACKEND", "TORCH_SDPA")


def _find_repo_root() -> Path:
    current = Path(__file__).resolve().parent
    for candidate in [current, *current.parents]:
        if (candidate / "fastvideo").exists():
            return candidate
    return current


repo_root = _find_repo_root()


def _find_official_root() -> Path | None:
    env_root = os.getenv("LINGBOT_WORLD_ROOT")
    if env_root:
        path = Path(env_root).expanduser().resolve()
        if (path / "wan" / "modules").exists():
            return path

    in_repo = repo_root / "lingbot-world"
    if (in_repo / "wan" / "modules").exists():
        return in_repo

    sibling = repo_root.parent / "lingbot-world"
    if (sibling / "wan" / "modules").exists():
        return sibling

    return None


def _ensure_diffusers_compat():
    try:
        importlib.import_module("diffusers.configuration_utils")
        importlib.import_module("diffusers.models.modeling_utils")
        return
    except Exception:
        pass

    diffusers_pkg = types.ModuleType("diffusers")
    diffusers_pkg.__path__ = []  # type: ignore[attr-defined]
    sys.modules["diffusers"] = diffusers_pkg

    configuration_utils = types.ModuleType("diffusers.configuration_utils")

    class ConfigMixin:
        pass

    def register_to_config(fn):  # noqa: ANN001
        return fn

    configuration_utils.ConfigMixin = ConfigMixin
    configuration_utils.register_to_config = register_to_config
    sys.modules["diffusers.configuration_utils"] = configuration_utils

    models_pkg = types.ModuleType("diffusers.models")
    models_pkg.__path__ = []  # type: ignore[attr-defined]
    sys.modules["diffusers.models"] = models_pkg

    modeling_utils = types.ModuleType("diffusers.models.modeling_utils")

    class ModelMixin(torch.nn.Module):
        def __init__(self, *args, **kwargs):  # noqa: ANN002, ANN003
            super().__init__()

    modeling_utils.ModelMixin = ModelMixin
    sys.modules["diffusers.models.modeling_utils"] = modeling_utils


def _load_official_wan_model_cls():
    official_root = _find_official_root()
    if official_root is None:
        pytest.skip(
            "Official LingBotWorld source not found. "
            "Expected one of: "
            f"{repo_root / 'lingbot-world'} or {repo_root.parent / 'lingbot-world'}; "
            "or set LINGBOT_WORLD_ROOT."
        )
    modules_dir = official_root / "wan" / "modules"

    if str(official_root) not in sys.path:
        sys.path.insert(0, str(official_root))

    _ensure_diffusers_compat()

    try:
        attention_mod = importlib.import_module("wan.modules.attention")
        model_mod = importlib.import_module("wan.modules.model")
    except Exception as exc:
        pytest.skip(f"Official LingBotWorld imports failed: {exc}")

    # Ensure official path does not require flash-attn in parity test.
    attention_mod.FLASH_ATTN_2_AVAILABLE = False
    attention_mod.FLASH_ATTN_3_AVAILABLE = False

    def _flash_attention_with_sdpa_fallback(*args, **kwargs):  # noqa: ANN001
        q = None
        if args:
            q = args[0]
        elif "q" in kwargs:
            q = kwargs["q"]
        if "version" in kwargs:
            kwargs = dict(kwargs)
            kwargs["fa_version"] = kwargs.pop("version")
        out = attention_mod.attention(*args, **kwargs)
        # Torch SDPA may return bf16 depending on kernel policy. Keep output
        # dtype aligned with query to avoid Linear dtype mismatch in official
        # WanModel when running parity in fp32.
        if isinstance(out, torch.Tensor) and isinstance(q, torch.Tensor):
            out = out.to(dtype=q.dtype)
        return out

    model_mod.flash_attention = _flash_attention_with_sdpa_fallback
    return model_mod.WanModel


def _assert_case_close(
    reference: torch.Tensor,
    target: torch.Tensor,
    *,
    case_name: str,
    atol: float = 1e-4,
    rtol: float = 1e-4,
) -> None:
    max_abs = (reference - target).abs().max().item()
    mean_abs = (reference - target).abs().mean().item()
    print(
        f"[LingBotWorld parity][{case_name}] "
        f"max_abs={max_abs:.6e}, mean_abs={mean_abs:.6e}"
    )
    assert_close(reference, target, atol=atol, rtol=rtol)


def _find_lingbotworld_examples_root() -> Path | None:
    candidates = [
        repo_root / "examples" / "inference" / "basic" / "lingbotworld_examples",
        repo_root.parent / "FastVideo" / "examples" / "inference" / "basic" / "lingbotworld_examples",
    ]
    for candidate in candidates:
        if (candidate / "00" / "poses.npy").exists() and (candidate / "00" / "intrinsics.npy").exists():
            return candidate
    return None


def _mean_abs_diff(a: torch.Tensor, b: torch.Tensor) -> float:
    return (a - b).abs().mean().item()


def _relative_l1_error(reference: torch.Tensor, target: torch.Tensor) -> float:
    num = (reference - target).abs().mean().item()
    den = reference.abs().mean().item() + 1e-8
    return num / den


def _seed_non_degenerate_output_heads(
    official_model: torch.nn.Module,
    fastvideo_model: torch.nn.Module,
) -> None:
    gen = torch.Generator(device="cpu")
    gen.manual_seed(20260209)
    with torch.no_grad():
        head_w = torch.randn(
            official_model.head.head.weight.shape,  # type: ignore[attr-defined]
            generator=gen,
            dtype=official_model.head.head.weight.dtype,  # type: ignore[attr-defined]
        )
        head_b = torch.randn(
            official_model.head.head.bias.shape,  # type: ignore[attr-defined]
            generator=gen,
            dtype=official_model.head.head.bias.dtype,  # type: ignore[attr-defined]
        )
        official_model.head.head.weight.copy_(  # type: ignore[attr-defined]
            head_w.to(device=official_model.head.head.weight.device)  # type: ignore[attr-defined]
        )
        official_model.head.head.bias.copy_(  # type: ignore[attr-defined]
            head_b.to(device=official_model.head.head.bias.device)  # type: ignore[attr-defined]
        )
        fastvideo_model.proj_out.weight.copy_(
            head_w.to(
                device=fastvideo_model.proj_out.weight.device,
                dtype=fastvideo_model.proj_out.weight.dtype,
            )
        )
        fastvideo_model.proj_out.bias.copy_(
            head_b.to(
                device=fastvideo_model.proj_out.bias.device,
                dtype=fastvideo_model.proj_out.bias.dtype,
            )
        )


def test_lingbotworld_transformer_parity():
    if not torch.cuda.is_available():
        pytest.skip("LingBotWorld parity test requires CUDA.")

    try:
        from fastvideo.configs.models.dits.lingbotworld import (
            LingBotWorldArchConfig,
            LingBotWorldVideoConfig,
        )
        from fastvideo.distributed import (
            cleanup_dist_env_and_memory,
            maybe_init_distributed_environment_and_model_parallel,
        )
        from fastvideo.forward_context import set_forward_context
        from fastvideo.models.dits.lingbotworld.model import (
            LingBotWorldTransformer3DModel,
        )
        from fastvideo.models.dits.lingbotworld.cam_utils import (
            prepare_camera_embedding,
        )
        from fastvideo.models.loader.utils import (
            get_param_names_mapping,
            hf_to_custom_state_dict,
        )
    except Exception as exc:
        pytest.skip(f"FastVideo LingBotWorld imports failed: {exc}")

    WanModel = _load_official_wan_model_cls()

    torch.manual_seed(42)
    device = torch.device("cuda:0")
    dtype = torch.float32

    maybe_init_distributed_environment_and_model_parallel(1, 1)
    try:
        arch = LingBotWorldArchConfig(
            patch_size=(1, 2, 2),
            num_attention_heads=2,
            attention_head_dim=8,
            in_channels=4,
            out_channels=4,
            text_dim=12,
            freq_dim=32,
            ffn_dim=32,
            num_layers=2,
            qk_norm="rms_norm_across_heads",
            cross_attn_norm=True,
            eps=1e-6,
        )
        arch.text_len = 6
        config = LingBotWorldVideoConfig(arch_config=arch)

        official_model = WanModel(
            model_type="t2v",
            patch_size=(1, 2, 2),
            text_len=6,
            in_dim=4,
            dim=16,
            ffn_dim=32,
            freq_dim=32,
            text_dim=12,
            out_dim=4,
            num_heads=2,
            num_layers=2,
            qk_norm=True,
            cross_attn_norm=True,
            eps=1e-6,
        ).to(device=device, dtype=dtype)
        fastvideo_model = LingBotWorldTransformer3DModel(
            config=config,
            hf_config={},
        ).to(device=device, dtype=dtype)

        mapping_fn = get_param_names_mapping(fastvideo_model.param_names_mapping)
        converted_sd, _ = hf_to_custom_state_dict(
            official_model.state_dict(),
            mapping_fn,
        )
        load_result = fastvideo_model.load_state_dict(converted_sd, strict=False)
        assert not load_result.missing_keys, (
            f"Unexpected missing keys: {load_result.missing_keys}"
        )
        assert not load_result.unexpected_keys, (
            f"Unexpected keys after mapping: {load_result.unexpected_keys}"
        )
        official_model.eval()
        fastvideo_model.eval()

        batch_size = 1
        in_channels = 4
        frames = 2
        height = 4
        width = 4
        text_len = 6
        text_dim = 12
        seq_len = frames * (height // 2) * (width // 2)

        hidden_states = torch.randn(
            batch_size,
            in_channels,
            frames,
            height,
            width,
            device=device,
            dtype=dtype,
        )
        encoder_hidden_states = torch.randn(
            batch_size,
            text_len,
            text_dim,
            device=device,
            dtype=dtype,
        )
        timestep = torch.randint(
            low=0,
            high=1000,
            size=(batch_size, seq_len),
            device=device,
            dtype=torch.long,
        )

        with torch.no_grad():
            ref_out_no_cam = official_model(
                x=[hidden_states[0]],
                t=timestep,
                context=[encoder_hidden_states[0]],
                seq_len=seq_len,
                dit_cond_dict=None,
            )[0].unsqueeze(0)
            with set_forward_context(
                current_timestep=0,
                attn_metadata=None,
                forward_batch=None,
            ):
                fast_out_no_cam = fastvideo_model(
                    hidden_states=hidden_states,
                    encoder_hidden_states=encoder_hidden_states,
                    timestep=timestep,
                    c2ws_plucker_emb=None,
                )
        _assert_case_close(ref_out_no_cam, fast_out_no_cam, case_name="no_cam")
        # Inject non-zero output heads after strict baseline parity check.
        _seed_non_degenerate_output_heads(official_model, fastvideo_model)

        with torch.no_grad():
            ref_out_no_cam = official_model(
                x=[hidden_states[0]],
                t=timestep,
                context=[encoder_hidden_states[0]],
                seq_len=seq_len,
                dit_cond_dict=None,
            )[0].unsqueeze(0)
            with set_forward_context(
                current_timestep=0,
                attn_metadata=None,
                forward_batch=None,
            ):
                fast_out_no_cam = fastvideo_model(
                    hidden_states=hidden_states,
                    encoder_hidden_states=encoder_hidden_states,
                    timestep=timestep,
                    c2ws_plucker_emb=None,
                )
        _assert_case_close(
            ref_out_no_cam,
            fast_out_no_cam,
            case_name="no_cam_seeded",
            atol=1.5e-2,
            rtol=1e-1,
        )

        examples_root = _find_lingbotworld_examples_root()
        if examples_root is None:
            pytest.skip(
                "LingBotWorld example actions not found. "
                "Expected examples/inference/basic/lingbotworld_examples/00~02."
            )

        # from examples
        cam_cases: dict[str, list[torch.Tensor]] = {}
        for case_id in ("00", "01", "02"):
            action_path = examples_root / case_id
            c2ws_plucker_emb, out_frames = prepare_camera_embedding(
                action_path=str(action_path),
                num_frames=5,
                height=32,
                width=32,
                spatial_scale=8,
            )
            assert out_frames == 5, (
                f"Unexpected aligned frame count for case {case_id}: {out_frames}"
            )
            cam_cases[case_id] = [
                t.to(device=device, dtype=dtype) for t in c2ws_plucker_emb
            ]

        ref_cam_outputs: dict[str, torch.Tensor] = {}
        fast_cam_outputs: dict[str, torch.Tensor] = {}
        for case_id, c2ws_plucker_emb in cam_cases.items():
            with torch.no_grad():
                ref_out_cam = official_model(
                    x=[hidden_states[0]],
                    t=timestep,
                    context=[encoder_hidden_states[0]],
                    seq_len=seq_len,
                    dit_cond_dict={"c2ws_plucker_emb": c2ws_plucker_emb},
                )[0].unsqueeze(0)
                with set_forward_context(
                    current_timestep=0,
                    attn_metadata=None,
                    forward_batch=None,
                ):
                    fast_out_cam = fastvideo_model(
                        hidden_states=hidden_states,
                        encoder_hidden_states=encoder_hidden_states,
                        timestep=timestep,
                        c2ws_plucker_emb=c2ws_plucker_emb,
                    )
            _assert_case_close(
                ref_out_cam,
                fast_out_cam,
                case_name=f"with_cam_{case_id}",
                atol=1.5e-2,
                rtol=1e-1,
            )
            ref_delta = ref_out_cam - ref_out_no_cam
            fast_delta = fast_out_cam - fast_out_no_cam
            rel_l1 = _relative_l1_error(ref_delta, fast_delta)
            print(
                f"[LingBotWorld parity][cam_delta_{case_id}] "
                f"rel_l1={rel_l1:.6e}"
            )
            assert rel_l1 < 0.2, (
                f"Camera delta mismatch too large for {case_id}: rel_l1={rel_l1:.6e}"
            )
            ref_cam_outputs[case_id] = ref_out_cam
            fast_cam_outputs[case_id] = fast_out_cam

        # each cam trajectory should change output vs no-cam
        min_effect_ratio = 0.8
        max_effect_ratio = 1.25
        for case_id in cam_cases:
            ref_effect = _mean_abs_diff(ref_cam_outputs[case_id], ref_out_no_cam)
            fast_effect = _mean_abs_diff(fast_cam_outputs[case_id], fast_out_no_cam)
            print(
                f"[LingBotWorld parity][cam_effect_{case_id}] "
                f"official={ref_effect:.6e}, fastvideo={fast_effect:.6e}"
            )
            assert ref_effect > 0.0, f"Official model has zero cam effect for case {case_id}"
            assert fast_effect >= min_effect_ratio * ref_effect, (
                f"FastVideo cam effect too weak for case {case_id}: "
                f"{fast_effect:.6e} < {min_effect_ratio:.2f} * {ref_effect:.6e}"
            )
            assert fast_effect <= max_effect_ratio * ref_effect, (
                f"FastVideo cam effect too strong for case {case_id}: "
                f"{fast_effect:.6e} > {max_effect_ratio:.2f} * {ref_effect:.6e}"
            )

        # different cam trajectories should produce different outputs
        for case_a, case_b in combinations(("00", "01", "02"), 2):
            ref_sep = _mean_abs_diff(ref_cam_outputs[case_a], ref_cam_outputs[case_b])
            fast_sep = _mean_abs_diff(fast_cam_outputs[case_a], fast_cam_outputs[case_b])
            print(
                f"[LingBotWorld parity][cam_sep_{case_a}_{case_b}] "
                f"official={ref_sep:.6e}, fastvideo={fast_sep:.6e}"
            )
            assert ref_sep > 0.0, f"Official outputs collapsed for {case_a} vs {case_b}"
            assert fast_sep >= min_effect_ratio * ref_sep, (
                f"FastVideo trajectory separation too weak for {case_a} vs {case_b}: "
                f"{fast_sep:.6e} < {min_effect_ratio:.2f} * {ref_sep:.6e}"
            )
            assert fast_sep <= max_effect_ratio * ref_sep, (
                f"FastVideo trajectory separation too strong for {case_a} vs {case_b}: "
                f"{fast_sep:.6e} > {max_effect_ratio:.2f} * {ref_sep:.6e}"
            )
    finally:
        cleanup_dist_env_and_memory()
