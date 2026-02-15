# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import os
from pathlib import Path
import gc

import pytest
import torch
import torch.distributed as dist

from safetensors.torch import load_file as safetensors_load_file
from torch.testing import assert_close
from transformers import AutoTokenizer, CLIPTextModelWithProjection
from transformers.utils import logging as hf_logging
from diffusers import SD3Transformer2DModel
from diffusers import AutoencoderKL
from transformers import T5EncoderModel as RefT5EncoderModel
from diffusers import FlowMatchEulerDiscreteScheduler as RefScheduler
from safetensors.torch import safe_open

pytestmark = [
    pytest.mark.skipif(
        not torch.cuda.is_available(),
        reason="SD3.5 component parity tests require CUDA",
    ),
    pytest.mark.filterwarnings(
        "ignore:.*torch.jit.script_method.*:DeprecationWarning",
    ),
]

os.environ.setdefault("FASTVIDEO_ATTENTION_BACKEND", "TORCH_SDPA")
hf_logging.set_verbosity_error()

MODEL_DIR = Path(
    os.getenv(
        "SD35_MODEL_DIR",
        "/FastVideo/official_weights/stabilityai__stable-diffusion-3.5-medium",
    )
)


def _load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _pick_device_and_dtype() -> tuple[torch.device, torch.dtype]:
    if torch.cuda.is_available():
        if torch.cuda.is_bf16_supported():
            return torch.device("cuda"), torch.bfloat16
        return torch.device("cuda"), torch.float32
    return torch.device("cpu"), torch.float32


@pytest.fixture(scope="module", autouse=True)
def _init_dist_and_tp_groups():
    if not torch.cuda.is_available():
        yield
        return

    import fastvideo.distributed.parallel_state as ps
    from fastvideo.distributed.parallel_state import (
        destroy_distributed_environment,
        destroy_model_parallel,
        get_tp_group,
        init_distributed_environment,
        initialize_model_parallel,
    )

    created_dist = False
    created_tp = False

    try:
        _ = get_tp_group()
        yield
        return
    except Exception:
        pass

    stubbed = False
    old_tp = old_sp = old_dp = old_world = None

    try:
        if not dist.is_initialized():
            os.environ.setdefault("MASTER_ADDR", "localhost")
            os.environ.setdefault("MASTER_PORT", "29500")
            os.environ.setdefault("RANK", "0")
            os.environ.setdefault("WORLD_SIZE", "1")
            os.environ.setdefault("LOCAL_RANK", "0")
            os.environ.setdefault("GLOO_SOCKET_IFNAME", "lo")

            if torch.cuda.is_available():
                torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", "0")))

            backend = "nccl" if torch.cuda.is_available() else "gloo"
            store_path = f"/tmp/fastvideo_sd35_pg_{os.getpid()}.store"
            dist.init_process_group(
                backend=backend,
                init_method=f"file://{store_path}",
                rank=int(os.environ["RANK"]),
                world_size=int(os.environ["WORLD_SIZE"]),
            )
            created_dist = True

            init_distributed_environment(
                world_size=int(os.environ["WORLD_SIZE"]),
                rank=int(os.environ["RANK"]),
                local_rank=int(os.environ["LOCAL_RANK"]),
                distributed_init_method="env://",
            )
        else:
            init_distributed_environment(
                world_size=dist.get_world_size(),
                rank=dist.get_rank(),
                local_rank=int(os.environ.get("LOCAL_RANK", "0")),
                distributed_init_method="env://",
            )

        try:
            _ = get_tp_group()
        except Exception:
            initialize_model_parallel(
                tensor_model_parallel_size=1,
                sequence_model_parallel_size=1,
                data_parallel_size=dist.get_world_size() if dist.is_initialized() else 1,
            )
            created_tp = True
    except Exception:
        old_tp = getattr(ps, "_TP", None)
        old_sp = getattr(ps, "_SP", None)
        old_dp = getattr(ps, "_DP", None)
        old_world = getattr(ps, "_WORLD", None)

        class _NoOpGroup:
            world_size = 1
            rank_in_group = 0
            local_rank = 0
            device_group = None

            def all_reduce(self, x: torch.Tensor) -> torch.Tensor:
                return x

            def all_gather(self, x: torch.Tensor, dim: int = -1) -> torch.Tensor:
                return x

            def all_to_all_4D(self, x: torch.Tensor, *_args, **_kwargs) -> torch.Tensor:
                return x

            def barrier(self) -> None:
                return None

            def destroy(self) -> None:
                return None

        ps._WORLD = _NoOpGroup()
        ps._TP = _NoOpGroup()
        ps._SP = _NoOpGroup()
        ps._DP = _NoOpGroup()
        stubbed = True

    yield

    if stubbed:
        ps._TP = old_tp
        ps._SP = old_sp
        ps._DP = old_dp
        ps._WORLD = old_world
    else:
        if created_tp:
            destroy_model_parallel()
        if created_dist:
            destroy_distributed_environment()


def _sd3_transformer_cfg() -> dict:
    cfg = _load_json(MODEL_DIR / "transformer" / "config.json")
    cfg.pop("_class_name", None)
    cfg.pop("_diffusers_version", None)
    return cfg


def _vae_cfg() -> dict:
    cfg = _load_json(MODEL_DIR / "vae" / "config.json")
    cfg.pop("_class_name", None)
    cfg.pop("_diffusers_version", None)
    return cfg


def test_sd35_sd3_transformer2d_parity():
    if not MODEL_DIR.exists():
        pytest.skip(f"SD3.5 model dir not found: {MODEL_DIR}")

    from fastvideo.configs.models.dits.sd3 import SD3DiTConfig
    from fastvideo.models.registry import ModelRegistry

    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False

    device, dtype = _pick_device_and_dtype()
    torch.manual_seed(0)

    cfg = _sd3_transformer_cfg()
    weight_path = MODEL_DIR / "transformer" / "diffusion_pytorch_model.safetensors"

    ref = SD3Transformer2DModel.from_config(cfg).eval()
    ref_sd = safetensors_load_file(str(weight_path), device="cpu")
    ref.load_state_dict(ref_sd, strict=True)
    ref = ref.to(device=device, dtype=dtype)

    B, H, W = 1, 16, 16
    hidden = torch.randn(B, cfg["in_channels"], H, W, device=device, dtype=dtype)
    enc = torch.randn(B, 16, cfg["joint_attention_dim"], device=device, dtype=dtype)
    pooled = torch.randn(B, cfg["pooled_projection_dim"], device=device, dtype=dtype)
    t = torch.tensor([10], device=device, dtype=torch.long)

    with torch.no_grad():
        ref_out = ref(
            hidden_states=hidden,
            encoder_hidden_states=enc,
            pooled_projections=pooled,
            timestep=t,
            return_dict=True,
        ).sample.detach().float().cpu()

    del ref
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()

    fv_cls, _ = ModelRegistry.resolve_model_cls("SD3Transformer2DModel")
    dit_cfg = SD3DiTConfig()
    dit_cfg.update_model_arch(cfg)
    fv = fv_cls(config=dit_cfg, hf_config=dict(cfg)).eval()
    fv_sd = safetensors_load_file(str(weight_path), device="cpu")
    fv.load_state_dict(fv_sd, strict=True)
    fv = fv.to(device=device, dtype=dtype)

    with torch.no_grad():
        fv_out = fv(
            hidden_states=hidden,
            encoder_hidden_states=enc,
            pooled_projections=pooled,
            timestep=t,
            return_dict=True,
        ).sample.detach().float().cpu()

    assert_close(ref_out, fv_out, atol=1e-4, rtol=1e-4)

    del fv
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()


def test_sd35_autoencoderkl_parity_encode_decode():
    if not MODEL_DIR.exists():
        pytest.skip(f"SD3.5 model dir not found: {MODEL_DIR}")

    from fastvideo.configs.models.vaes.autoencoder_kl import AutoencoderKLVAEConfig
    from fastvideo.models.registry import ModelRegistry

    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False

    device, dtype = _pick_device_and_dtype()
    torch.manual_seed(0)

    cfg = _vae_cfg()
    weight_path = MODEL_DIR / "vae" / "diffusion_pytorch_model.safetensors"

    ref = AutoencoderKL.from_config(cfg).eval()
    ref_sd = safetensors_load_file(str(weight_path), device="cpu")
    ref.load_state_dict(ref_sd, strict=True)
    ref = ref.to(device=device, dtype=dtype)

    x = torch.randn(1, cfg["in_channels"], 64, 64, device=device, dtype=dtype)

    with torch.no_grad():
        ref_latents = ref.encode(x).latent_dist.mean.detach().float().cpu()
        ref_dec = ref.decode(ref_latents.to(device=device, dtype=dtype)).sample.detach().float().cpu()

    del ref
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()

    fv_cls, _ = ModelRegistry.resolve_model_cls("AutoencoderKL")
    vae_cfg = AutoencoderKLVAEConfig()
    vae_cfg.update_model_arch(cfg)
    fv = fv_cls(vae_cfg).eval()
    fv_sd = safetensors_load_file(str(weight_path), device="cpu")
    fv.load_state_dict(fv_sd, strict=True)
    fv = fv.to(device=device, dtype=dtype)

    with torch.no_grad():
        fv_latents = fv.encode(x).latent_dist.mean.detach().float().cpu()
        fv_dec = fv.decode(fv_latents.to(device=device, dtype=dtype)).sample.detach().float().cpu()

    assert_close(ref_latents, fv_latents, atol=1e-4, rtol=1e-4)
    assert_close(ref_dec, fv_dec, atol=1e-4, rtol=1e-4)

    del fv
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()


@pytest.mark.parametrize("encoder_subdir", ["text_encoder", "text_encoder_2"])
def test_sd35_clip_text_with_projection_parity(encoder_subdir: str):
    if not MODEL_DIR.exists():
        pytest.skip(f"SD3.5 model dir not found: {MODEL_DIR}")

    from fastvideo.configs.models.encoders.clip import CLIPTextConfig
    from fastvideo.forward_context import set_forward_context
    from fastvideo.models.registry import ModelRegistry

    if not torch.cuda.is_available():
        pytest.skip("FastVideo CLIPTextModelWithProjection parity requires CUDA")

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    dtype = torch.float32
    torch.manual_seed(0)

    text_encoder_dir = MODEL_DIR / encoder_subdir
    tokenizer_dir = MODEL_DIR / ("tokenizer" if encoder_subdir == "text_encoder" else "tokenizer_2")
    weight_path = text_encoder_dir / "model.safetensors"

    if not text_encoder_dir.exists():
        pytest.skip(f"Missing {encoder_subdir} dir: {text_encoder_dir}")
    if not tokenizer_dir.exists():
        pytest.skip(f"Missing tokenizer dir: {tokenizer_dir}")

    tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_dir), local_files_only=True)
    prompt = "a photo of a cat"
    toks = tokenizer(
        [prompt],
        padding="max_length",
        truncation=True,
        max_length=77,
        return_tensors="pt",
    )
    input_ids = toks["input_ids"].clone()

    if tokenizer.pad_token_id is not None and tokenizer.eos_token_id is not None:
        input_ids[input_ids == tokenizer.pad_token_id] = tokenizer.eos_token_id
    attention_mask = torch.ones_like(input_ids)

    input_ids = input_ids.to(device=device)
    attention_mask = attention_mask.to(device=device)

    ref = CLIPTextModelWithProjection.from_pretrained(
        str(text_encoder_dir),
        local_files_only=True,
    ).eval()
    ref = ref.to(device=device, dtype=dtype)

    with torch.no_grad():
        ref_out = ref(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=False,
        )
        ref_last = ref_out.last_hidden_state.detach().float().cpu()
        ref_pooled = ref_out.text_embeds.detach().float().cpu()

    del ref
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()

    cfg_raw = _load_json(text_encoder_dir / "config.json")
    for k in (
        "_name_or_path",
        "transformers_version",
        "model_type",
        "tokenizer_class",
        "torch_dtype",
    ):
        cfg_raw.pop(k, None)

    fv_cfg = CLIPTextConfig()
    fv_cfg.update_model_arch(cfg_raw)
    fv_cls, _ = ModelRegistry.resolve_model_cls("CLIPTextModelWithProjection")
    fv = fv_cls(fv_cfg).eval()
    weights_to_load = {n for n, _ in fv.named_parameters()}
    loaded = fv.load_weights(_iter_safetensors(str(weight_path)))
    assert "text_projection.weight" in loaded
    missing = weights_to_load - loaded
    assert not missing, f"Missing FastVideo CLIP weights: {sorted(list(missing))[:50]}"
    fv = fv.to(device=device, dtype=dtype)

    with torch.no_grad():

        with set_forward_context(current_timestep=0, attn_metadata=None):
            fv_out = fv(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=False,
            )
        fv_last = fv_out.last_hidden_state.detach().float().cpu()
        fv_pooled = fv_out.pooler_output.detach().float().cpu()

    assert_close(ref_last, fv_last, atol=1e-3, rtol=1e-3)
    assert_close(ref_pooled, fv_pooled, atol=1e-3, rtol=1e-3)

    del fv
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()


def _iter_safetensors(path: str):

    with safe_open(path, framework="pt", device="cpu") as f:
        for k in f.keys():
            yield k, f.get_tensor(k)


def _iter_pretrained_safetensors(model_dir: Path):
    single = model_dir / "model.safetensors"
    if single.exists():
        yield from _iter_safetensors(str(single))
        return

    index = model_dir / "model.safetensors.index.json"
    if index.exists():
        idx = _load_json(index)
        shard_names = sorted(set(idx["weight_map"].values()))
        for shard in shard_names:
            yield from _iter_safetensors(str(model_dir / shard))
        return

    raise FileNotFoundError(
        f"Missing safetensors checkpoint in {model_dir} (expected model.safetensors or model.safetensors.index.json)"
    )


def _scheduler_cfg() -> dict:
    cfg = _load_json(MODEL_DIR / "scheduler" / "scheduler_config.json")
    cfg.pop("_class_name", None)
    cfg.pop("_diffusers_version", None)
    return cfg

def test_sd35_t5_encoder_model_parity():
    if not MODEL_DIR.exists():
        pytest.skip(f"SD3.5 model dir not found: {MODEL_DIR}")

    if not torch.cuda.is_available():
        pytest.skip("FastVideo T5EncoderModel parity requires CUDA")

    from fastvideo.configs.models.encoders.t5 import T5Config
    from fastvideo.models.encoders.t5_hf import T5EncoderModel as FVT5HFModel

    device = torch.device("cuda")
    dtype = torch.bfloat16
    if not torch.cuda.is_bf16_supported():
        pytest.skip("CUDA bf16 not supported on this device/driver")
    torch.manual_seed(0)

    def _assert_all_finite(name: str, t: torch.Tensor) -> None:
        finite = torch.isfinite(t)
        if bool(finite.all()):
            return
        nan_count = int(torch.isnan(t).sum().item())
        inf_count = int(torch.isinf(t).sum().item())
        pct_finite = float(finite.float().mean().item()) * 100.0
        if bool(finite.any()):
            max_abs = float(t[finite].abs().max().item())
        else:
            max_abs = float("nan")
        print(
            f"[NONFINITE] {name}: shape={tuple(t.shape)} dtype={t.dtype} device={t.device} "
            f"finite={pct_finite:.2f}% nan={nan_count} inf={inf_count} max_abs_finite={max_abs}"
        )
        pytest.fail(f"Non-finite values detected in {name}")

    text_encoder_dir = MODEL_DIR / "text_encoder_3"
    tokenizer_dir = MODEL_DIR / "tokenizer_3"

    if not text_encoder_dir.exists():
        pytest.skip(f"Missing text_encoder_3 dir: {text_encoder_dir}")
    if not tokenizer_dir.exists():
        pytest.skip(f"Missing tokenizer_3 dir: {tokenizer_dir}")

    tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_dir), local_files_only=True)
    prompt = "a photo of a cat"
    toks = tokenizer(
        [prompt],
        padding="max_length",
        truncation=True,
        max_length=77,
        return_tensors="pt",
    )
    input_ids = toks["input_ids"].to(device=device)
    attention_mask = toks["attention_mask"].to(device=device)

    try:
        ref = RefT5EncoderModel.from_pretrained(
            str(text_encoder_dir),
            local_files_only=True,
            dtype=dtype,
            low_cpu_mem_usage=True,
        ).eval()
    except TypeError:
        ref = RefT5EncoderModel.from_pretrained(
            str(text_encoder_dir),
            local_files_only=True,
            torch_dtype=dtype,
            low_cpu_mem_usage=True,
        ).eval()
    ref = ref.to(device=device, dtype=dtype)

    with torch.no_grad():
        ref_out = ref(input_ids=input_ids, attention_mask=attention_mask)
        ref_last_dev = ref_out.last_hidden_state.detach()
        _assert_all_finite("ref_t5.last_hidden_state", ref_last_dev)
        ref_last = ref_last_dev.float().cpu()

    del ref
    gc.collect()
    torch.cuda.empty_cache()

    cfg_raw = _load_json(text_encoder_dir / "config.json")
    for k in ("_name_or_path", "transformers_version", "model_type", "torch_dtype"):
        cfg_raw.pop(k, None)
    fv_cfg = T5Config()
    fv_cfg.update_model_arch(cfg_raw)

    fv = FVT5HFModel.from_pretrained_local(
        str(text_encoder_dir),
        fv_cfg,
        dtype=dtype,
        device=device,
    )

    with torch.no_grad():
        fv_out = fv(input_ids=input_ids, attention_mask=attention_mask)
        fv_last_dev = fv_out.last_hidden_state.detach()
        _assert_all_finite("fv_t5.last_hidden_state", fv_last_dev)
        fv_last = fv_last_dev.float().cpu()

    assert_close(ref_last, fv_last, atol=1e-4, rtol=1e-4)

    del fv
    gc.collect()
    torch.cuda.empty_cache()


def test_sd35_flowmatch_euler_discrete_scheduler_parity():
    if not MODEL_DIR.exists():
        pytest.skip(f"SD3.5 model dir not found: {MODEL_DIR}")

    from fastvideo.models.registry import ModelRegistry

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    cfg = _scheduler_cfg()
    num_inference_steps = 5

    ref = RefScheduler.from_config(cfg)
    fv_cls, _ = ModelRegistry.resolve_model_cls("FlowMatchEulerDiscreteScheduler")
    fv = fv_cls.from_config(cfg)

    ref.set_timesteps(num_inference_steps=num_inference_steps, device=device)
    fv.set_timesteps(num_inference_steps=num_inference_steps, device=device)

    assert_close(ref.timesteps.cpu(), fv.timesteps.cpu(), atol=0.0, rtol=0.0)

    if hasattr(ref, "sigmas") and hasattr(fv, "sigmas"):
        assert_close(ref.sigmas.cpu(), fv.sigmas.cpu(), atol=0.0, rtol=0.0)
