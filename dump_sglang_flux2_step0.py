#!/usr/bin/env python3
"""
Dump step-0 inputs and SGLang transformer output for Flux2 Klein.

Requires SGLang on PYTHONPATH (or SGLANG_PATH env). Uses same seed (0), prompt,
and settings as dump_flux2_step0.py so you can compare FastVideo's DiT to SGLang.

Run once (requires SGLang with Flux2 Klein support):
  python dump_sglang_flux2_step0.py

Saves: flux2_step0_sglang_dump.pt (latent, timestep, prompt_embeds, text_ids,
       latent_ids, noise_pred_sglang, etc.)
"""
import os
import sys

import torch

# Add SGLang to path (try SGLANG_PATH env, else ../sglang/python, else /sglang/python)
_sglang_candidates = [
    os.environ.get("SGLANG_PATH"),
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "sglang", "python"),
    "/sglang/python",
]
SGLANG_PATH = None
for p in _sglang_candidates:
    if p and os.path.isdir(p):
        SGLANG_PATH = os.path.abspath(p)
        break
if SGLANG_PATH and SGLANG_PATH not in sys.path:
    sys.path.insert(0, SGLANG_PATH)

DUMP_PATH = "flux2_step0_sglang_dump.pt"
PROMPT = "a red apple on a table"
HEIGHT, WIDTH = 1024, 1024
NUM_STEPS = 4
SEED = 0
MODEL_ID = "black-forest-labs/FLUX.2-klein-4B"


def _get_model_path(model_id: str) -> str:
    """Resolve model path (local or HF cache)."""
    try:
        from huggingface_hub import snapshot_download

        return snapshot_download(repo_id=model_id)
    except Exception:
        pass
    if os.path.isdir(model_id):
        return model_id
    raise FileNotFoundError(f"Could not find model {model_id}")


def main():
    device = "cuda"
    dtype = torch.bfloat16
    generator = torch.Generator(device=device).manual_seed(SEED)

    # Initialize SGLang distributed + TP so custom Flux2Transformer2DModel can load (ColumnParallelLinear)
    os.environ.setdefault("LOCAL_RANK", "0")
    os.environ.setdefault("RANK", "0")
    os.environ.setdefault("WORLD_SIZE", "1")
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29500")
    from sglang.multimodal_gen.runtime.distributed import (
        maybe_init_distributed_environment_and_model_parallel,
    )
    maybe_init_distributed_environment_and_model_parallel(
        tp_size=1, sp_size=1, enable_cfg_parallel=False
    )

    model_path = _get_model_path(MODEL_ID)

    # Minimal ServerArgs for loading components
    from sglang.multimodal_gen.configs.pipeline_configs.flux import (
        Flux2KleinPipelineConfig,
        _prepare_latent_ids,
        _prepare_text_ids,
        flux2_pack_latents,
    )
    from sglang.multimodal_gen.runtime.loader.component_loader import ComponentLoader
    from sglang.multimodal_gen.runtime.server_args import ServerArgs, set_global_server_args
    from diffusers.utils.torch_utils import randn_tensor

    pipeline_config = Flux2KleinPipelineConfig()
    pipeline_config.dit_precision = "bf16"
    pipeline_config.vae_config.post_init()
    server_args = ServerArgs(
        model_path=model_path,
        pipeline_config=pipeline_config,
        hsdp_shard_dim=1,
        hsdp_replicate_dim=1,
    )
    server_args.model_paths = {
        "text_encoder": os.path.join(model_path, "text_encoder"),
        "tokenizer": os.path.join(model_path, "tokenizer"),
        "transformer": os.path.join(model_path, "transformer"),
        "scheduler": os.path.join(model_path, "scheduler"),
    }
    set_global_server_args(server_args)

    # 1. Encode prompt (SGLang text encoder)
    print("Loading SGLang text encoder and encoding prompt ...")
    tokenizer, _ = ComponentLoader.for_component_type("tokenizer", "transformers").load(
        server_args.model_paths["tokenizer"],
        server_args,
        "tokenizer",
        "transformers",
    )
    text_encoder, _ = ComponentLoader.for_component_type("text_encoder", "transformers").load(
        server_args.model_paths["text_encoder"],
        server_args,
        "text_encoder",
        "transformers",
    )
    # Don't call .to(device).to(dtype) on FSDP/TP models - loader handles placement
    if "FSDP" not in type(text_encoder).__name__:
        text_encoder = text_encoder.to(device).to(dtype).eval()

    tok_out = pipeline_config.tokenize_prompt([[PROMPT]], tokenizer, {})
    input_ids = tok_out["input_ids"].to(device)
    attention_mask = tok_out.get("attention_mask")
    if attention_mask is not None:
        attention_mask = attention_mask.to(device)

    from sglang.multimodal_gen.runtime.managers.forward_context import set_forward_context

    with torch.no_grad(), set_forward_context(current_timestep=0, attn_metadata=None):
        enc_out = text_encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )
    prompt_embeds = pipeline_config.postprocess_text_funcs[0](enc_out, None)
    prompt_embeds = prompt_embeds.to(device).to(dtype)

    # 2. Prepare latents (packed: B, H*W, C) - match Flux2PipelineConfig.prepare_latent_shape
    print("Preparing latents ...")
    batch_for_shape = type("Batch", (), {"height": HEIGHT, "width": WIDTH})()
    shape = pipeline_config.prepare_latent_shape(batch_for_shape, 1, 1)
    latents_4d = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
    latent_ids = _prepare_latent_ids(latents_4d).to(device)
    latents = flux2_pack_latents(latents_4d)

    # 3. Timesteps (same mu as diffusers)
    print("Loading scheduler and preparing timesteps ...")
    from sglang.multimodal_gen.runtime.pipelines.flux_2 import (
        compute_empirical_mu,
    )

    scheduler, _ = ComponentLoader.for_component_type("scheduler", "diffusers").load(
        server_args.model_paths["scheduler"],
        server_args,
        "scheduler",
        "diffusers",
    )

    image_seq_len = latents.shape[1]
    batch_for_mu = type("Batch", (), {"num_inference_steps": NUM_STEPS, "raw_latent_shape": latents.shape})()
    mu_key, mu_val = compute_empirical_mu(batch_for_mu, server_args)
    scheduler.set_timesteps(NUM_STEPS, device=device, **{mu_key: mu_val})
    scheduler.set_begin_index(0)
    timesteps = scheduler.timesteps
    t = timesteps[0]
    timestep = t.expand(1).to(latents.dtype)

    latent_model_input = latents.to(dtype)
    latent_model_input = scheduler.scale_model_input(latent_model_input, t)

    # 4. Text ids for freqs_cis
    txt_ids = _prepare_text_ids(prompt_embeds).to(device)
    img_ids = latent_ids
    if img_ids.ndim == 3:
        img_ids = img_ids[0]
    if txt_ids.ndim == 3:
        txt_ids = txt_ids[0]

    # 5. Load SGLang transformer and run step 0
    print("Loading SGLang transformer ...")
    transformer, _ = ComponentLoader.for_component_type("transformer", "diffusers").load(
        server_args.model_paths["transformer"],
        server_args,
        "transformer",
        "diffusers",
    )
    transformer = transformer.to(device).eval()

    # Compute freqs_cis
    rotary_emb = transformer.rotary_emb
    img_cos, img_sin = rotary_emb.forward(img_ids)
    txt_cos, txt_sin = rotary_emb.forward(txt_ids)
    cos = torch.cat([txt_cos, img_cos], dim=0).to(device)
    sin = torch.cat([txt_sin, img_sin], dim=0).to(device)
    freqs_cis = (cos, sin)

    print("Running SGLang transformer for step 0 ...")
    with torch.no_grad():
        noise_pred = transformer(
            hidden_states=latent_model_input,
            encoder_hidden_states=prompt_embeds,
            timestep=timestep / 1000.0,
            guidance=None,
            freqs_cis=freqs_cis,
            joint_attention_kwargs={},
        )
    noise_pred = noise_pred[:, : latents.size(1)]

    # 6. Save
    to_save = {
        "latent_model_input": latent_model_input.detach().cpu().float(),
        "timestep_scaled": (timestep / 1000.0).detach().cpu().float(),
        "timestep_raw": timestep.detach().cpu(),
        "prompt_embeds": prompt_embeds.detach().cpu().float(),
        "text_ids": txt_ids.detach().cpu(),
        "latent_ids": latent_ids.detach().cpu(),
        "noise_pred_sglang": noise_pred.detach().cpu().float(),
        "height": HEIGHT,
        "width": WIDTH,
        "num_steps": NUM_STEPS,
        "seed": SEED,
        "prompt": PROMPT,
    }
    torch.save(to_save, DUMP_PATH)
    print(f"Saved {DUMP_PATH}")
    print(f"  latent_model_input: {to_save['latent_model_input'].shape}")
    print(f"  timestep_scaled: {to_save['timestep_scaled'].shape}")
    print(f"  prompt_embeds: {to_save['prompt_embeds'].shape}")
    print(f"  noise_pred_sglang: {to_save['noise_pred_sglang'].shape}")


if __name__ == "__main__":
    main()
