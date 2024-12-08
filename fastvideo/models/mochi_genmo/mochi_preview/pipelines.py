import json
import os
import random
from abc import ABC, abstractmethod
from contextlib import contextmanager
from functools import partial
from typing import Any, Dict, List, Literal, Optional, Union, cast

import numpy as np
import ray
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from einops import repeat
from safetensors import safe_open
from safetensors.torch import load_file
from torch import nn
from torch.distributed.fsdp import (
    BackwardPrefetch,
    MixedPrecision,
    ShardingStrategy,
)
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.wrap import (
    lambda_auto_wrap_policy,
    transformer_auto_wrap_policy,
)
from transformers import T5EncoderModel, T5Tokenizer
from transformers.models.t5.modeling_t5 import T5Block

import genmo.mochi_preview.dit.joint_model.context_parallel as cp
from genmo.lib.progress import get_new_progress_bar, progress_bar
from genmo.lib.utils import Timer
from genmo.mochi_preview.vae.models import (
    Decoder,
    Encoder,
    decode_latents,
    decode_latents_tiled_full,
    decode_latents_tiled_spatial,
)
from genmo.mochi_preview.vae.vae_stats import dit_latents_to_vae_latents


def load_to_cpu(p, weights_only=True):
    if p.endswith(".safetensors"):
        return load_file(p)
    else:
        assert p.endswith(".pt")
        return torch.load(p, map_location="cpu", weights_only=weights_only)


def linear_quadratic_schedule(num_steps, threshold_noise, linear_steps=None):
    if linear_steps is None:
        linear_steps = num_steps // 2
    linear_sigma_schedule = [i * threshold_noise / linear_steps for i in range(linear_steps)]
    threshold_noise_step_diff = linear_steps - threshold_noise * num_steps
    quadratic_steps = num_steps - linear_steps
    quadratic_coef = threshold_noise_step_diff / (linear_steps * quadratic_steps**2)
    linear_coef = threshold_noise / linear_steps - 2 * threshold_noise_step_diff / (quadratic_steps**2)
    const = quadratic_coef * (linear_steps**2)
    quadratic_sigma_schedule = [
        quadratic_coef * (i**2) + linear_coef * i + const for i in range(linear_steps, num_steps)
    ]
    sigma_schedule = linear_sigma_schedule + quadratic_sigma_schedule + [1.0]
    sigma_schedule = [1.0 - x for x in sigma_schedule]
    return sigma_schedule


T5_MODEL = "google/t5-v1_1-xxl"
MAX_T5_TOKEN_LENGTH = 256


def setup_fsdp_sync(model, device_id, *, param_dtype, auto_wrap_policy) -> FSDP:
    model = FSDP(
        model,
        sharding_strategy=ShardingStrategy.FULL_SHARD,
        mixed_precision=MixedPrecision(
            param_dtype=param_dtype,
            reduce_dtype=torch.float32,
            buffer_dtype=torch.float32,
        ),
        auto_wrap_policy=auto_wrap_policy,
        backward_prefetch=BackwardPrefetch.BACKWARD_PRE,
        limit_all_gathers=True,
        device_id=device_id,
        sync_module_states=True,
        use_orig_params=True,
    )
    torch.cuda.synchronize()
    return model


class ModelFactory(ABC):
    def __init__(self, **kwargs):
        self.kwargs = kwargs

    @abstractmethod
    def get_model(self, *, local_rank: int, device_id: Union[int, Literal["cpu"]], world_size: int) -> Any:
        assert isinstance(device_id, int) or device_id == "cpu", "device_id must be an integer or 'cpu'"
        # FSDP does not work when the model is on the CPU
        if device_id == "cpu":
            assert world_size == 1, "CPU offload only supports single-GPU inference"


class T5ModelFactory(ModelFactory):
    def __init__(self, model_dir=None):
        super().__init__()
        self.model_dir = model_dir or T5_MODEL

    def get_model(self, *, local_rank, device_id, world_size):
        super().get_model(local_rank=local_rank, device_id=device_id, world_size=world_size)
        model = T5EncoderModel.from_pretrained(self.model_dir)
        if world_size > 1:
            model = setup_fsdp_sync(
                model,
                device_id=device_id,
                param_dtype=torch.float32,
                auto_wrap_policy=partial(
                    transformer_auto_wrap_policy,
                    transformer_layer_cls={
                        T5Block,
                    },
                ),
            )
        elif isinstance(device_id, int):
            model = model.to(torch.device(f"cuda:{device_id}"))  # type: ignore
        return model.eval()


class DitModelFactory(ModelFactory):
    def __init__(
        self, *,
        model_path: str,
        model_dtype: str,
        lora_path: Optional[str] = None,
        attention_mode: Optional[str] = None
    ):
        # Infer attention mode if not specified
        if attention_mode is None:
            from genmo.lib.attn_imports import flash_varlen_attn  # type: ignore
            attention_mode = "sdpa" if flash_varlen_attn is None else "flash"
        print(f"Attention mode: {attention_mode}")

        super().__init__(
            model_path=model_path,
            lora_path=lora_path,
            model_dtype=model_dtype,
            attention_mode=attention_mode
        )

    def get_model(
        self,
        *,
        local_rank,
        device_id,
        world_size,
        model_kwargs=None,
        patch_model_fns=None,
        strict_load=True,
        load_checkpoint=True,
        fast_init=True,
    ):
        from genmo.mochi_preview.dit.joint_model.asymm_models_joint import AsymmDiTJoint

        if not model_kwargs:
            model_kwargs = {}

        lora_sd = None
        lora_path = self.kwargs["lora_path"]
        if lora_path is not None:
            if lora_path.endswith(".safetensors"):
                lora_sd = {}
                with safe_open(lora_path, framework="pt") as f:
                    for k in f.keys():
                        lora_sd[k] = f.get_tensor(k)
                    lora_kwargs = json.loads(f.metadata()["kwargs"])
                    print(f"Loaded LoRA kwargs: {lora_kwargs}")
            else:
                lora = load_to_cpu(lora_path, weights_only=False)
                lora_sd, lora_kwargs = lora["state_dict"], lora["kwargs"]

            model_kwargs.update(cast(dict, lora_kwargs))

        model_args = dict(
            depth=48,
            patch_size=2,
            num_heads=24,
            hidden_size_x=3072,
            hidden_size_y=1536,
            mlp_ratio_x=4.0,
            mlp_ratio_y=4.0,
            in_channels=12,
            qk_norm=True,
            qkv_bias=False,
            out_bias=True,
            patch_embed_bias=True,
            timestep_mlp_bias=True,
            timestep_scale=1000.0,
            t5_feat_dim=4096,
            t5_token_length=256,
            rope_theta=10000.0,
            attention_mode=self.kwargs["attention_mode"],
            **model_kwargs,
        )

        if fast_init:
            model: nn.Module = torch.nn.utils.skip_init(AsymmDiTJoint, **model_args)
        else:
            model: nn.Module = AsymmDiTJoint(**model_args)

        for fn in patch_model_fns or []:
            model = fn(model)

        # FSDP syncs weights from rank 0 to all other ranks
        if local_rank == 0 and load_checkpoint:
            model_path = self.kwargs["model_path"]
            sd = load_to_cpu(model_path)

            # Load the state dictionary and capture the return value
            load_result = model.load_state_dict(sd, strict=strict_load)
            if not strict_load:
                # Print mismatched keys
                missing_keys = [k for k in load_result.missing_keys if ".lora_" not in k]
                if missing_keys:
                    print(f"Missing keys from {model_path}: {missing_keys}")
                if load_result.unexpected_keys:
                    print(f"Unexpected keys from {model_path}: {load_result.unexpected_keys}")

            if lora_sd:
                model.load_state_dict(lora_sd, strict=strict_load) # type: ignore

        if world_size > 1:
            assert self.kwargs["model_dtype"] == "bf16", "FP8 is not supported for multi-GPU inference"

            model = setup_fsdp_sync(
                model,
                device_id=device_id,
                param_dtype=torch.float32,
                auto_wrap_policy=partial(
                    lambda_auto_wrap_policy,
                    lambda_fn=lambda m: m in model.blocks,
                ),
            )
        elif isinstance(device_id, int):
            model = model.to(torch.device(f"cuda:{device_id}"))
        return model.eval()


class DecoderModelFactory(ModelFactory):
    def __init__(self, *, model_path: str):
        super().__init__(model_path=model_path)

    def get_model(self, *, local_rank=0, device_id=0, world_size=1):
        # TODO(ved): Set flag for torch.compile
        # TODO(ved): Use skip_init

        decoder = Decoder(
            out_channels=3,
            base_channels=128,
            channel_multipliers=[1, 2, 4, 6],
            temporal_expansions=[1, 2, 3],
            spatial_expansions=[2, 2, 2],
            num_res_blocks=[3, 3, 4, 6, 3],
            latent_dim=12,
            has_attention=[False, False, False, False, False],
            output_norm=False,
            nonlinearity="silu",
            output_nonlinearity="silu",
            causal=True,
        )
        # VAE is not FSDP-wrapped
        state_dict = load_file(self.kwargs["model_path"])
        decoder.load_state_dict(state_dict, strict=True)
        device = torch.device(f"cuda:{device_id}") if isinstance(device_id, int) else "cpu"
        decoder.eval().to(device)
        return decoder


class EncoderModelFactory(ModelFactory):
    def __init__(self, *, model_path: str):
        super().__init__(model_path=model_path)

    def get_model(self, *, local_rank=0, device_id=0, world_size=1):
        # TODO(ved): Set flag for torch.compile
        # TODO(ved): Use skip_init

        # We don't FSDP the encoder b/c it is small
        encoder = Encoder(
            in_channels=15,
            base_channels=64,
            channel_multipliers=[1, 2, 4, 6],
            num_res_blocks=[3, 3, 4, 6, 3],
            latent_dim=12,
            temporal_reductions=[1, 2, 3],
            spatial_reductions=[2, 2, 2],
            prune_bottlenecks=[False, False, False, False, False],
            has_attentions=[False, True, True, True, True],
            affine=True,
            bias=True,
            input_is_conv_1x1=True,
            padding_mode="replicate",
        )
        state_dict = load_file(self.kwargs["model_path"])
        encoder.load_state_dict(state_dict, strict=True)
        device = torch.device(f"cuda:{device_id}") if isinstance(device_id, int) else "cpu"
        encoder.eval().to(device)
        return encoder


def get_conditioning(
    tokenizer: T5Tokenizer,
    encoder: Encoder,
    device: torch.device,
    batch_inputs: bool,
    *,
    prompt: str,
    negative_prompt: str,
):
    if batch_inputs:
        return dict(
            batched=get_conditioning_for_prompts(
                tokenizer, encoder, device, [prompt, negative_prompt]
            )
        )
    else:
        cond_input = get_conditioning_for_prompts(tokenizer, encoder, device, [prompt])
        null_input = get_conditioning_for_prompts(tokenizer, encoder, device, [negative_prompt])
        return dict(cond=cond_input, null=null_input)


def get_conditioning_for_prompts(tokenizer, encoder, device, prompts: List[str]):
    assert len(prompts) in [1, 2]  # [neg] or [pos] or [pos, neg]
    B = len(prompts)
    t5_toks = tokenizer(
        prompts,
        padding="max_length",
        truncation=True,
        max_length=MAX_T5_TOKEN_LENGTH,
        return_tensors="pt",
        return_attention_mask=True,
    )
    caption_input_ids_t5 = t5_toks["input_ids"]
    caption_attention_mask_t5 = t5_toks["attention_mask"].bool()
    del t5_toks

    assert caption_input_ids_t5.shape == (B, MAX_T5_TOKEN_LENGTH)
    assert caption_attention_mask_t5.shape == (B, MAX_T5_TOKEN_LENGTH)

    # Special-case empty negative prompt by zero-ing it
    if prompts[-1] == "":
        caption_input_ids_t5[-1] = 0
        caption_attention_mask_t5[-1] = False

    caption_input_ids_t5 = caption_input_ids_t5.to(device, non_blocking=True)
    caption_attention_mask_t5 = caption_attention_mask_t5.to(device, non_blocking=True)

    y_mask = [caption_attention_mask_t5]
    y_feat = [encoder(caption_input_ids_t5, caption_attention_mask_t5).last_hidden_state.detach()]
    # Sometimes returns a tensor, othertimes a tuple, not sure why
    # See: https://huggingface.co/genmo/mochi-1-preview/discussions/3
    assert tuple(y_feat[-1].shape) == (B, MAX_T5_TOKEN_LENGTH, 4096)
    assert y_feat[-1].dtype == torch.float32

    return dict(y_mask=y_mask, y_feat=y_feat)


def compute_packed_indices(
    device: torch.device, text_mask: torch.Tensor, num_latents: int
) -> Dict[str, Union[torch.Tensor, int]]:
    """
    Based on https://github.com/Dao-AILab/flash-attention/blob/765741c1eeb86c96ee71a3291ad6968cfbf4e4a1/flash_attn/bert_padding.py#L60-L80

    Args:
        num_latents: Number of latent tokens
        text_mask: (B, L) List of boolean tensor indicating which text tokens are not padding.

    Returns:
        packed_indices: Dict with keys for Flash Attention:
            - valid_token_indices_kv: up to (B * (N + L),) tensor of valid token indices (non-padding)
                                   in the packed sequence.
            - cu_seqlens_kv: (B + 1,) tensor of cumulative sequence lengths in the packed sequence.
            - max_seqlen_in_batch_kv: int of the maximum sequence length in the batch.
    """
    # Create an expanded token mask saying which tokens are valid across both visual and text tokens.
    PATCH_SIZE = 2
    num_visual_tokens = num_latents // (PATCH_SIZE**2)
    assert num_visual_tokens > 0

    mask = F.pad(text_mask, (num_visual_tokens, 0), value=True)  # (B, N + L)
    seqlens_in_batch = mask.sum(dim=-1, dtype=torch.int32)  # (B,)
    valid_token_indices = torch.nonzero(mask.flatten(), as_tuple=False).flatten()  # up to (B * (N + L),)
    assert valid_token_indices.size(0) >= text_mask.size(0) * num_visual_tokens  # At least (B * N,)
    cu_seqlens = F.pad(torch.cumsum(seqlens_in_batch, dim=0, dtype=torch.int32), (1, 0))
    max_seqlen_in_batch = seqlens_in_batch.max().item()

    return {
        "cu_seqlens_kv": cu_seqlens.to(device, non_blocking=True),
        "max_seqlen_in_batch_kv": cast(int, max_seqlen_in_batch),
        "valid_token_indices_kv": valid_token_indices.to(device, non_blocking=True),
    }


def assert_eq(x, y, msg=None):
    assert x == y, f"{msg or 'Assertion failed'}: {x} != {y}"


def sample_model(device, dit, conditioning, **args):
    random.seed(args["seed"])
    np.random.seed(args["seed"])
    torch.manual_seed(args["seed"])

    generator = torch.Generator(device=device)
    generator.manual_seed(args["seed"])

    w, h, t = args["width"], args["height"], args["num_frames"]
    sample_steps = args["num_inference_steps"]
    cfg_schedule = args["cfg_schedule"]
    sigma_schedule = args["sigma_schedule"]

    assert_eq(len(cfg_schedule), sample_steps, "cfg_schedule must have length sample_steps")
    assert_eq((t - 1) % 6, 0, "t - 1 must be divisible by 6")
    assert_eq(
        len(sigma_schedule),
        sample_steps + 1,
        "sigma_schedule must have length sample_steps + 1",
    )

    B = 1
    SPATIAL_DOWNSAMPLE = 8
    TEMPORAL_DOWNSAMPLE = 6
    IN_CHANNELS = 12
    latent_t = ((t - 1) // TEMPORAL_DOWNSAMPLE) + 1
    latent_w, latent_h = w // SPATIAL_DOWNSAMPLE, h // SPATIAL_DOWNSAMPLE

    z = torch.randn(
        (B, IN_CHANNELS, latent_t, latent_h, latent_w),
        device=device,
        dtype=torch.float32,
    )

    num_latents = latent_t * latent_h * latent_w
    cond_batched = cond_text = cond_null = None
    if "cond" in conditioning:
        cond_text = conditioning["cond"]
        cond_null = conditioning["null"]
        cond_text["packed_indices"] = compute_packed_indices(device, cond_text["y_mask"][0], num_latents)
        cond_null["packed_indices"] = compute_packed_indices(device, cond_null["y_mask"][0], num_latents)
    else:
        cond_batched = conditioning["batched"]
        cond_batched["packed_indices"] = compute_packed_indices(device, cond_batched["y_mask"][0], num_latents)
        z = repeat(z, "b ... -> (repeat b) ...", repeat=2)

    def model_fn(*, z, sigma, cfg_scale):
        if cond_batched:
            with torch.autocast("cuda", dtype=torch.bfloat16):
                out = dit(z, sigma, **cond_batched)
            out_cond, out_uncond = torch.chunk(out, chunks=2, dim=0)
        else:
            nonlocal cond_text, cond_null
            with torch.autocast("cuda", dtype=torch.bfloat16):
                out_cond = dit(z, sigma, **cond_text)
                out_uncond = dit(z, sigma, **cond_null)
        assert out_cond.shape == out_uncond.shape
        out_uncond = out_uncond.to(z)
        out_cond = out_cond.to(z)
        return out_uncond + cfg_scale * (out_cond - out_uncond)

    # Euler sampler w/ customizable sigma schedule & cfg scale
    for i in get_new_progress_bar(range(0, sample_steps), desc="Sampling"):
        sigma = sigma_schedule[i]
        dsigma = sigma - sigma_schedule[i + 1]

        # `pred` estimates `z_0 - eps`.
        pred = model_fn(
            z=z,
            sigma=torch.full([B] if cond_text else [B * 2], sigma, device=z.device),
            cfg_scale=cfg_schedule[i],
        )
        assert pred.dtype == torch.float32
        z = z + dsigma * pred

    z = z[:B] if cond_batched else z
    return dit_latents_to_vae_latents(z)


@contextmanager
def move_to_device(model: nn.Module, target_device, *, enabled=True):
    if not enabled:
        yield
        return

    og_device = next(model.parameters()).device
    if og_device == target_device:
        print(f"move_to_device is a no-op model is already on {target_device}")
    else:
        print(f"moving model from {og_device} -> {target_device}")

    model.to(target_device)
    yield
    if og_device != target_device:
        print(f"moving model from {target_device} -> {og_device}")
    model.to(og_device)


def t5_tokenizer(model_dir=None):
    return T5Tokenizer.from_pretrained(model_dir or T5_MODEL, legacy=False)


class MochiSingleGPUPipeline:
    def __init__(
        self,
        *,
        text_encoder_factory: ModelFactory,
        dit_factory: ModelFactory,
        decoder_factory: ModelFactory,
        cpu_offload: Optional[bool] = False,
        decode_type: str = "full",
        decode_args: Optional[Dict[str, Any]] = None,
        fast_init=True,
        strict_load=True
    ):
        self.device = torch.device("cuda:0")
        self.tokenizer = t5_tokenizer(text_encoder_factory.model_dir)
        t = Timer()
        self.cpu_offload = cpu_offload
        self.decode_args = decode_args or {}
        self.decode_type = decode_type
        init_id = "cpu" if cpu_offload else 0
        with t("load_text_encoder"):
            self.text_encoder = text_encoder_factory.get_model(
                local_rank=0,
                device_id=init_id,
                world_size=1,
            )
        with t("load_dit"):
            self.dit = dit_factory.get_model(local_rank=0, device_id=init_id, world_size=1, fast_init=fast_init, strict_load=strict_load) # type: ignore
        with t("load_vae"):
            self.decoder = decoder_factory.get_model(local_rank=0, device_id=init_id, world_size=1)
        t.print_stats()

    def __call__(self, batch_cfg, prompt, negative_prompt, **kwargs):
        with torch.inference_mode():
            print_max_memory = lambda: print(
                f"Max memory reserved: {torch.cuda.max_memory_reserved() / 1024**3:.2f} GB"
            )
            print_max_memory()

            with move_to_device(self.text_encoder, self.device):
                conditioning = get_conditioning(
                    tokenizer=self.tokenizer,
                    encoder=self.text_encoder,
                    device=self.device,
                    batch_inputs=batch_cfg,
                    prompt=prompt,
                    negative_prompt=negative_prompt,
                )
            print_max_memory()

            with move_to_device(self.dit, self.device):
                latents = sample_model(self.device, self.dit, conditioning, **kwargs)
            print_max_memory()

            with move_to_device(self.decoder, self.device):
                if self.decode_type == "tiled_full":
                    frames = decode_latents_tiled_full(
                        self.decoder, latents, **self.decode_args)
                elif self.decode_type == "tiled_spatial":
                    frames = decode_latents_tiled_spatial(
                        self.decoder, latents, **self.decode_args,
                        num_tiles_w=4, num_tiles_h=2)
                else:
                    frames = decode_latents(self.decoder, latents)
            print_max_memory()
            return frames.cpu().numpy()


def cast_dit(model, dtype):
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            assert any(
                n in name for n in ["mlp", "t5", "mod_", "attn.qkv_", "attn.proj_", "final_layer"]
            ), f"Unexpected linear layer: {name}"
            module.to(dtype=dtype)
        elif isinstance(module, nn.Conv2d):
            assert "x_embedder.proj" in name, f"Unexpected conv2d layer: {name}"
            module.to(dtype=dtype)
    return model


### ALL CODE BELOW HERE IS FOR MULTI-GPU MODE ###


# In multi-gpu mode, all models must belong to a device which has a predefined context parallel group
# So it doesn't make sense to work with models individually
class MultiGPUContext:
    def __init__(
        self,
        *,
        text_encoder_factory,
        dit_factory,
        decoder_factory,
        device_id,
        local_rank,
        world_size,
    ):
        t = Timer()
        self.device = torch.device(f"cuda:{device_id}")
        print(f"Initializing rank {local_rank+1}/{world_size}")
        assert world_size > 1, f"Multi-GPU mode requires world_size > 1, got {world_size}"
        os.environ["MASTER_ADDR"] = "127.0.0.1"
        os.environ["MASTER_PORT"] = "29500"
        with t("init_process_group"):
            dist.init_process_group(
                "nccl",
                rank=local_rank,
                world_size=world_size,
                device_id=self.device,  # force non-lazy init
            )
        pg = dist.group.WORLD
        cp.set_cp_group(pg, list(range(world_size)), local_rank)
        distributed_kwargs = dict(local_rank=local_rank, device_id=device_id, world_size=world_size)
        self.world_size = world_size
        self.tokenizer = t5_tokenizer(text_encoder_factory.model_dir)
        with t("load_text_encoder"):
            self.text_encoder = text_encoder_factory.get_model(**distributed_kwargs)
        with t("load_dit"):
            self.dit = dit_factory.get_model(**distributed_kwargs)
        with t("load_vae"):
            self.decoder = decoder_factory.get_model(**distributed_kwargs)
        self.local_rank = local_rank
        t.print_stats()

    def run(self, *, fn, **kwargs):
        return fn(self, **kwargs)


class MochiMultiGPUPipeline:
    def __init__(
        self,
        *,
        text_encoder_factory: ModelFactory,
        dit_factory: ModelFactory,
        decoder_factory: ModelFactory,
        world_size: int,
    ):
        ray.init()
        RemoteClass = ray.remote(MultiGPUContext)
        self.ctxs = [
            RemoteClass.options(num_gpus=1).remote(
                text_encoder_factory=text_encoder_factory,
                dit_factory=dit_factory,
                decoder_factory=decoder_factory,
                world_size=world_size,
                device_id=0,
                local_rank=i,
            )
            for i in range(world_size)
        ]
        for ctx in self.ctxs:
            ray.get(ctx.__ray_ready__.remote())

    def __call__(self, **kwargs):
        def sample(ctx, *, batch_cfg, prompt, negative_prompt, **kwargs):
            with progress_bar(type="ray_tqdm", enabled=ctx.local_rank == 0), torch.inference_mode():
                conditioning = get_conditioning(
                    ctx.tokenizer,
                    ctx.text_encoder,
                    ctx.device,
                    batch_cfg,
                    prompt=prompt,
                    negative_prompt=negative_prompt,
                )
                latents = sample_model(ctx.device, ctx.dit, conditioning=conditioning, **kwargs)
                if ctx.local_rank == 0:
                    torch.save(latents, "latents.pt")
                frames = decode_latents(ctx.decoder, latents)
            return frames.cpu().numpy()

        return ray.get([ctx.run.remote(fn=sample, **kwargs, show_progress=i == 0) for i, ctx in enumerate(self.ctxs)])[
            0
        ]
