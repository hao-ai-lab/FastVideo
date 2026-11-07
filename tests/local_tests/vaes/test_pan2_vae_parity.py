# SPDX-License-Identifier: Apache-2.0
"""PAN2 reuse of FastVideo's Wan2.2 (48-channel) VAE.

Coverage scope: both. PAN2 ships a Diffusers ``AutoencoderKLWan`` in the
Wan2.2 layout (z_dim 48, residual blocks, 2x2 patchify, 16x spatial and 4x
temporal compression). The FastVideo side loads it through ``VAELoader``, the
path a pipeline uses, and is compared on CPU fp32 against
``diffusers.AutoencoderKLWan.from_pretrained``:

* posterior mode/logvar for a single first frame (PAN2's I2V conditioning),
* posterior mode/logvar for a 9-frame clip (4x temporal downsampling + cache),
* decode of 3 latent frames (first-chunk handling + temporal upsampling cache).

The normalization tests use a copy of the checkpoint with non-trivial
``latents_mean``/``latents_std`` (the tiny checkpoint ships 0/1, which would
hide a missing or doubled normalization) and pin where it happens: the raw
``encode``/``decode`` never normalize, ``PAN2ConditioningStage`` applies
PAN2's ``(z - mean) / std`` once, and ``DecodingStage`` applies
``z * std + mean`` once.

The default asset is the random-weight tiny PAN2 checkpoint, so this runs on a
CPU-only host::

    pytest tests/local_tests/vaes/test_pan2_vae_parity.py -v -s

Set ``PAN2_MODEL_PATH`` to a local PAN2 root containing ``vae/`` to use other
weights.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from safetensors.torch import load_file
from torch.testing import assert_close

try:
    from diffusers import AutoencoderKLWan as DiffusersAutoencoderKLWan
except ImportError:  # pragma: no cover - environment dependent
    pytest.skip("diffusers with AutoencoderKLWan is required", allow_module_level=True)

from fastvideo.models.loader.component_loader import VAELoader
from fastvideo.models.wan.vae import AutoencoderKLWan as FastVideoAutoencoderKLWan
from fastvideo.models.wan.vae_config import WanVAEConfig

PARITY_SCOPE = "both"
TINY_PAN2_REPO = "wuqing157/tiny-pan2-modular-pipe"
TINY_PAN2_REVISION = "3ff3130b4527f61aaab7bde01465c9884dc84cdd"
MODEL_PATH_ENV = "PAN2_MODEL_PATH"

# H and W must be multiples of 32 (16x spatial compression times the 2x2
# patchify inside the VAE).
HEIGHT, WIDTH = 64, 96
VIDEO_FRAMES = 9
LATENT_FRAMES = (VIDEO_FRAMES - 1) // 4 + 1
ATOL = RTOL = 1e-5

# Structural fields of Wan-AI/Wan2.2-TI2V-5B-Diffusers vae/config.json. The
# statistics do not affect parameter shapes, so placeholders keep this inline.
WAN22_TI2V_VAE_CONFIG = {
    "attn_scales": [],
    "base_dim": 160,
    "clip_output": False,
    "decoder_base_dim": 256,
    "dim_mult": [1, 2, 4, 4],
    "dropout": 0.0,
    "in_channels": 12,
    "is_residual": True,
    "latents_mean": [0.0] * 48,
    "latents_std": [1.0] * 48,
    "num_res_blocks": 2,
    "out_channels": 12,
    "patch_size": 2,
    "scale_factor_spatial": 16,
    "scale_factor_temporal": 4,
    "temperal_downsample": [False, True, True],
    "z_dim": 48,
}


def _pan2_root(subfolder: str) -> Path:
    """Resolve a PAN2 root that contains ``subfolder``; skip when unavailable."""
    override = os.getenv(MODEL_PATH_ENV)
    if override:
        root = Path(override)
    else:
        try:
            from huggingface_hub import snapshot_download
        except ImportError:
            pytest.skip("huggingface_hub is required to resolve the tiny PAN2 checkpoint")
        root = None
        last_error: Exception | None = None
        for local_files_only in (True, False):
            try:
                root = Path(
                    snapshot_download(
                        TINY_PAN2_REPO,
                        revision=TINY_PAN2_REVISION,
                        allow_patterns=[f"{subfolder}/*"],
                        local_files_only=local_files_only,
                    ))
                break
            except Exception as exc:  # noqa: BLE001 - offline/missing cache must skip.
                last_error = exc
        if root is None:
            pytest.skip(f"Tiny PAN2 checkpoint unavailable: {last_error}")
    if not (root / subfolder).is_dir():
        pytest.skip(f"PAN2 {subfolder}/ missing under {root}")
    return root


@pytest.fixture(scope="module")
def vae_dir() -> Path:
    path = _pan2_root("vae") / "vae"
    for name in ("config.json", "diffusion_pytorch_model.safetensors"):
        if not (path / name).is_file():
            pytest.skip(f"PAN2 VAE asset missing: {path / name}")
    return path


@pytest.fixture
def cpu_runtime(monkeypatch: pytest.MonkeyPatch) -> None:
    """Pin FastVideo's device helper to CPU; it reports ``mps`` on CPU-only Linux."""
    for module in (
            "fastvideo.models.loader.component_loader",
            "fastvideo.pipelines.stages.decoding",
            "fastvideo.pipelines.basic.pan2.stages",
    ):
        monkeypatch.setattr(f"{module}.get_local_torch_device", lambda: torch.device("cpu"))


def _pan2_vae_config() -> WanVAEConfig:
    """The VAE config a PAN2 pipeline config would carry (arch filled by the loader)."""
    config = WanVAEConfig()
    config.load_encoder = True
    config.load_decoder = True
    return config


def _load_fastvideo(vae_dir: Path) -> FastVideoAutoencoderKLWan:
    """Load through the production VAELoader, exactly as a pipeline does."""
    fastvideo_args = SimpleNamespace(
        model_paths={},
        pipeline_config=SimpleNamespace(vae_config=_pan2_vae_config(), vae_precision="fp32"),
        vae_cpu_offload=False,
    )
    vae = VAELoader().load(str(vae_dir), fastvideo_args)
    assert isinstance(vae, FastVideoAutoencoderKLWan)
    assert fastvideo_args.model_paths["vae"] == str(vae_dir)
    assert all(param.dtype == torch.float32 and param.device.type == "cpu" for param in vae.parameters())
    return vae


def _load_diffusers(vae_dir: Path) -> DiffusersAutoencoderKLWan:
    model, info = DiffusersAutoencoderKLWan.from_pretrained(
        str(vae_dir),
        torch_dtype=torch.float32,
        local_files_only=True,
        output_loading_info=True,
    )
    problems = {key: value for key, value in info.items() if value}
    assert not problems, f"Diffusers reference did not load strictly: {problems}"
    return model.eval()


def _pixels(num_frames: int, seed: int) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    return torch.rand((1, 3, num_frames, HEIGHT, WIDTH), generator=generator) * 2.0 - 1.0


def _latents(seed: int) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    return torch.randn((1, 48, LATENT_FRAMES, HEIGHT // 16, WIDTH // 16), generator=generator)


def _report(name: str, actual: torch.Tensor, expected: torch.Tensor, clamp_bound: float | None = None) -> None:
    assert actual.shape == expected.shape, f"{name}: {tuple(actual.shape)} != {tuple(expected.shape)}"
    diff = (actual.float() - expected.float()).abs()
    message = (f"[pan2-vae] {name}: shape={tuple(actual.shape)} max_abs={diff.max().item():.3e} "
               f"mean_abs={diff.mean().item():.3e} ref_absmax={expected.abs().max().item():.3e}")
    if clamp_bound is not None:
        # Clamped pixels compare trivially; report how many there are.
        message += f" clamped={(expected.abs() >= clamp_bound).float().mean().item():.2%}"
    print(message)


def _nontrivial_stats() -> tuple[list[float], list[float]]:
    mean = [round(0.05 * (index - 24), 4) for index in range(48)]
    std = [round(0.4 + 0.03 * index, 4) for index in range(48)]
    return mean, std


@pytest.fixture
def stats_vae_dir(vae_dir: Path, tmp_path: Path) -> Path:
    """Checkpoint copy with non-trivial latent statistics (weights symlinked)."""
    config = json.loads((vae_dir / "config.json").read_text())
    config["latents_mean"], config["latents_std"] = _nontrivial_stats()
    target = tmp_path / "vae"
    target.mkdir()
    (target / "config.json").write_text(json.dumps(config))
    (target / "diffusion_pytorch_model.safetensors").symlink_to(
        (vae_dir / "diffusion_pytorch_model.safetensors").resolve())
    return target


def _diffusers_stats(model: DiffusersAutoencoderKLWan, like: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    mean = torch.tensor(model.config.latents_mean).view(1, -1, 1, 1, 1).to(like)
    std = torch.tensor(model.config.latents_std).view(1, -1, 1, 1, 1).to(like)
    return mean, std


def test_pan2_vae_production_loader_matches_checkpoint(vae_dir: Path, cpu_runtime: None) -> None:
    """The loader fills the tiny Wan2.2 arch from config.json and covers every weight.

    ``VAELoader`` loads ``AutoencoderKLWan`` with ``strict=False``, so key
    coverage is checked here instead of trusting the loader.
    """
    raw = json.loads((vae_dir / "config.json").read_text())
    vae = _load_fastvideo(vae_dir)
    arch = vae.config.arch_config
    for key in ("base_dim", "decoder_base_dim", "z_dim", "num_res_blocks", "in_channels", "out_channels",
                "is_residual", "patch_size", "scale_factor_spatial", "scale_factor_temporal", "dropout"):
        assert getattr(arch, key) == raw[key], key
    for key in ("dim_mult", "attn_scales", "temperal_downsample", "latents_mean", "latents_std"):
        assert list(getattr(arch, key)) == list(raw[key]), key
    assert vae.config.use_feature_cache and not vae.config.use_tiling
    assert arch.spatial_compression_ratio == 16 and arch.temporal_compression_ratio == 4
    assert tuple(vae.scaling_factor.shape) == tuple(vae.shift_factor.shape) == (1, 48, 1, 1, 1)
    assert_close(vae.scaling_factor.flatten(), 1.0 / torch.tensor(raw["latents_std"]))
    assert_close(vae.shift_factor.flatten(), torch.tensor(raw["latents_mean"]))

    checkpoint = load_file(str(vae_dir / "diffusion_pytorch_model.safetensors"))
    state = vae.state_dict()
    assert set(state) == set(checkpoint), (sorted(set(checkpoint) - set(state)), sorted(set(state) - set(checkpoint)))
    for key, tensor in checkpoint.items():
        assert torch.equal(state[key], tensor.float()), key


@pytest.mark.parametrize("variant", ["wan22_ti2v_5b", "tiny_pan2"])
def test_pan2_vae_config_shapes_match_diffusers(variant: str, request: pytest.FixtureRequest) -> None:
    """``WanVAEConfig`` expresses both the real Wan2.2 and the tiny PAN2 VAE configs.

    Both models are built on the meta device, so the full-size config costs no
    memory; parameter names and shapes must match one-to-one.
    """
    if variant == "tiny_pan2":
        raw = json.loads((request.getfixturevalue("vae_dir") / "config.json").read_text())
        config = {key: value for key, value in raw.items() if not key.startswith("_")}
    else:
        config = WAN22_TI2V_VAE_CONFIG
    fastvideo_config = _pan2_vae_config()
    fastvideo_config.update_model_arch(config)
    with torch.device("meta"):
        fastvideo = FastVideoAutoencoderKLWan(fastvideo_config)
        reference = DiffusersAutoencoderKLWan.from_config(config)
    fastvideo_shapes = {key: tuple(value.shape) for key, value in fastvideo.state_dict().items()}
    reference_shapes = {key: tuple(value.shape) for key, value in reference.state_dict().items()}
    assert fastvideo_shapes == reference_shapes


def test_pan2_vae_wan22_topology_random_weight_parity() -> None:
    """Numerically cover the Wan2.2 paths the tiny config skips.

    The tiny checkpoint uses ``dim_mult`` [1, 1, 1, 1] and no separate decoder
    width, so channel-changing shortcuts and resamplers never run. This narrow
    random-weight model keeps the real topology (``dim_mult`` [1, 2, 4, 4], two
    res blocks, ``decoder_base_dim`` > ``base_dim``).
    """
    config = dict(WAN22_TI2V_VAE_CONFIG, base_dim=8, decoder_base_dim=16)
    torch.manual_seed(0)
    reference = DiffusersAutoencoderKLWan.from_config(config).eval()
    fastvideo_config = _pan2_vae_config()
    fastvideo_config.update_model_arch(config)
    fastvideo = FastVideoAutoencoderKLWan(fastvideo_config).eval()
    fastvideo.load_state_dict(reference.state_dict(), strict=True)

    video = _pixels(VIDEO_FRAMES, seed=11)
    latents = _latents(seed=12)
    with torch.inference_mode():
        expected_mode = reference.encode(video).latent_dist.mode()
        actual_mode = fastvideo.encode(video).mode()
        expected_video = reference.decode(latents).sample
        actual_video = fastvideo.decode(latents)
    _report("wan22-topology encode mode", actual_mode, expected_mode)
    _report("wan22-topology decode", actual_video, expected_video, clamp_bound=1.0)
    assert_close(actual_mode, expected_mode, atol=ATOL, rtol=RTOL)
    assert_close(actual_video, expected_video, atol=ATOL, rtol=RTOL)


@pytest.mark.parametrize("num_frames", [1, VIDEO_FRAMES], ids=["first_frame", "video_9f"])
def test_pan2_vae_encode_matches_diffusers(num_frames: int, vae_dir: Path, cpu_runtime: None) -> None:
    """Posterior mode (PAN2 uses ``latent_dist.mode()``) and logvar match Diffusers.

    FastVideo's ``encode`` returns the ``DiagonalGaussianDistribution`` itself,
    so the port calls ``vae.encode(x).mode()`` instead of ``.latent_dist.mode()``.
    """
    reference = _load_diffusers(vae_dir)
    fastvideo = _load_fastvideo(vae_dir)
    pixels = _pixels(num_frames, seed=100 + num_frames)
    with torch.inference_mode():
        expected = reference.encode(pixels).latent_dist
        actual = fastvideo.encode(pixels)
    assert tuple(actual.mode().shape) == (1, 48, (num_frames - 1) // 4 + 1, HEIGHT // 16, WIDTH // 16)
    _report(f"encode mode ({num_frames}f)", actual.mode(), expected.mode())
    _report(f"encode logvar ({num_frames}f)", actual.logvar, expected.logvar)
    assert_close(actual.mode(), expected.mode(), atol=ATOL, rtol=RTOL)
    assert_close(actual.logvar, expected.logvar, atol=ATOL, rtol=RTOL)


def test_pan2_vae_decode_matches_diffusers(vae_dir: Path, cpu_runtime: None) -> None:
    """Decode three latent frames so first-chunk and temporal cache paths run."""
    reference = _load_diffusers(vae_dir)
    fastvideo = _load_fastvideo(vae_dir)
    latents = _latents(seed=7)
    with torch.inference_mode():
        expected = reference.decode(latents).sample
        actual = fastvideo.decode(latents)
    assert tuple(actual.shape) == (1, 3, VIDEO_FRAMES, HEIGHT, WIDTH)
    _report("decode", actual, expected, clamp_bound=1.0)
    assert_close(actual, expected, atol=ATOL, rtol=RTOL)


def test_pan2_conditioning_stage_matches_reference(stats_vae_dir: Path, cpu_runtime: None) -> None:
    """``PAN2ConditioningStage`` builds the reference I2V condition: cover-resize and center crop the first frame,
    encode it, apply ``(z - mean) / std`` once, and set the mask channel on the first latent frame only.

    The raw ``encode`` is unnormalized, so the stage is the only place the statistics apply.
    """
    import PIL.Image

    from fastvideo.pipelines.basic.pan2.stages import PAN2ConditioningStage
    from tests.local_tests.pan2.reference import reference_condition_latents

    reference = _load_diffusers(stats_vae_dir)
    fastvideo = _load_fastvideo(stats_vae_dir)
    # Not the output aspect ratio, so the resize and crop are exercised.
    image = PIL.Image.radial_gradient("L").resize((120, 64)).convert("RGB")
    latents = torch.zeros(1, 48, LATENT_FRAMES, HEIGHT // 16, WIDTH // 16)
    with torch.inference_mode():
        pixels = PAN2ConditioningStage.preprocess_image(image, HEIGHT, WIDTH)
        raw = reference.encode(pixels).latent_dist.mode()
        assert_close(fastvideo.encode(pixels).mode(), raw, atol=ATOL, rtol=RTOL)
        expected = reference_condition_latents(reference, latents, image, HEIGHT, WIDTH)

        batch = SimpleNamespace(latents=latents, pil_image=image, height=HEIGHT, width=WIDTH, image_latent=None)
        stage_args = SimpleNamespace(pipeline_config=SimpleNamespace(vae_precision="fp32"), vae_cpu_offload=False)
        actual = PAN2ConditioningStage(fastvideo).forward(batch, stage_args).image_latent
    _report("PAN2 I2V condition", actual, expected)
    assert actual.shape == (1, 49, LATENT_FRAMES, HEIGHT // 16, WIDTH // 16)
    assert_close(actual, expected, atol=ATOL, rtol=RTOL)
    assert torch.equal(actual[:, 48, 0], torch.ones_like(actual[:, 48, 0]))
    assert not actual[:, :, 1:].any()
    assert not torch.allclose(raw, expected[:, :48, :1], atol=1e-2), "statistics too close to identity"


def test_pan2_decoding_stage_denormalizes_once(stats_vae_dir: Path, cpu_runtime: None) -> None:
    """``DecodingStage`` performs PAN2's ``z * std + mean`` itself.

    The PAN2 denoising loop must hand normalized latents to ``DecodingStage``.
    Denormalizing beforehand (as the upstream pipeline does inline) would apply
    the statistics twice, which the negative check below detects.
    """
    from fastvideo.pipelines.stages.decoding import DecodingStage

    reference = _load_diffusers(stats_vae_dir)
    fastvideo = _load_fastvideo(stats_vae_dir)
    latents = _latents(seed=31)
    # vae_tiling=True is the PipelineConfig default; the feature-cache decode ignores tiling, so it must not matter.
    decode_args = SimpleNamespace(
        pipeline_config=SimpleNamespace(vae_precision="fp32", vae_decode_precision="fp32", vae_tiling=True),
        disable_autocast=False,
    )
    stage = DecodingStage(fastvideo)
    with torch.inference_mode():
        mean, std = _diffusers_stats(reference, latents)
        expected = reference.decode(latents * std + mean).sample
        expected = (expected * 0.5 + 0.5).clamp(0, 1)
        actual = stage.decode(latents, decode_args)
        doubled = stage.decode(latents * std + mean, decode_args)
    _report("DecodingStage output", actual, expected)
    assert_close(actual, expected, atol=ATOL, rtol=RTOL)
    assert (doubled - expected).abs().max().item() > 1e-3, "double denormalization went undetected"
