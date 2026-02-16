#!/usr/bin/env python3
"""
Debug text encoder: find where FastVideo vs SGLang diverge.

1. Verifies input_ids and attention_mask are identical
2. Compares embedding output (before first layer)
3. Registers hooks on each transformer layer, finds first divergent layer
4. Optionally hooks attention vs MLP in the first divergent layer

Usage:
  python debug_text_encoder_sglang.py
"""
import argparse
import os
import sys

import torch

SGLANG_PATH = os.environ.get(
    "SGLANG_PATH",
    os.path.join(os.path.dirname(__file__), "..", "sglang", "python"),
)
if os.path.isdir(SGLANG_PATH) and SGLANG_PATH not in sys.path:
    sys.path.insert(0, SGLANG_PATH)

PROMPT = "a red apple on a table"
MODEL_ID = "black-forest-labs/FLUX.2-klein-4B"


def _get_text_encoder_path(model_id: str) -> str:
    try:
        from huggingface_hub import snapshot_download
        root = snapshot_download(repo_id=model_id)
        path = os.path.join(root, "text_encoder")
        if os.path.isdir(path):
            return path
    except Exception:
        pass
    if os.path.isdir(model_id):
        te = os.path.join(model_id, "text_encoder")
        if os.path.isdir(te):
            return te
        if os.path.exists(os.path.join(model_id, "config.json")):
            return model_id
    raise FileNotFoundError(f"Could not find text_encoder for {model_id}")


def _get_model_layers(model):
    """Get transformer layers from Qwen3ForCausalLM-style model."""
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers
    if hasattr(model, "layers"):
        return model.layers
    raise AttributeError("Could not find model.model.layers or model.layers")


def _get_embed_tokens(model):
    """Get embedding module from Qwen3ForCausalLM-style model."""
    if hasattr(model, "model") and hasattr(model.model, "embed_tokens"):
        return model.model.embed_tokens
    if hasattr(model, "embed_tokens"):
        return model.embed_tokens
    raise AttributeError("Could not find embed_tokens")


def main():
    parser = argparse.ArgumentParser(
        description="Debug text encoder: find where FastVideo vs SGLang diverge."
    )
    parser.add_argument("--model-path", default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--prompt", default=PROMPT)
    args = parser.parse_args()

    os.environ.setdefault("LOCAL_RANK", "0")
    os.environ.setdefault("RANK", "0")
    os.environ.setdefault("WORLD_SIZE", "1")
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29500")

    model_path = args.model_path or _get_text_encoder_path(MODEL_ID)
    root = (
        os.path.dirname(model_path)
        if os.path.basename(model_path) == "text_encoder"
        else model_path
    )
    text_encoder_path = (
        model_path
        if os.path.basename(model_path) == "text_encoder"
        else os.path.join(root, "text_encoder")
    )
    tokenizer_dir = os.path.join(root, "tokenizer")
    tokenizer_path = tokenizer_dir if os.path.isdir(tokenizer_dir) else root
    device = args.device
    dtype = torch.bfloat16

    # Tokenize (same as compare script with --use-sglang)
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
    messages = [{"role": "user", "content": args.prompt}]
    try:
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
        )
    except TypeError:
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    inputs = tokenizer(
        text,
        padding="max_length",
        truncation=True,
        max_length=512,
        return_tensors="pt",
    )
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)

    # 1. Init distributed for BOTH: FastVideo and SGLang use separate TP state
    from fastvideo.distributed import maybe_init_distributed_environment_and_model_parallel as fv_init
    fv_init(tp_size=1, sp_size=1)

    from sglang.multimodal_gen.runtime.distributed import (
        maybe_init_distributed_environment_and_model_parallel as sgl_init,
    )
    sgl_init(tp_size=1, sp_size=1, enable_cfg_parallel=False)

    # 2. Load both encoders
    from fastvideo.configs.pipelines.flux_2 import Flux2KleinPipelineConfig
    from fastvideo.forward_context import set_forward_context
    from fastvideo.fastvideo_args import FastVideoArgs
    from fastvideo.models.loader.component_loader import TextEncoderLoader

    print("Loading FastVideo text encoder ...")
    fv_args = FastVideoArgs.from_kwargs(
        model_path=root,
        pipeline_config=Flux2KleinPipelineConfig(),
        text_encoder_cpu_offload=False,
    )
    fv_args.pipeline_config.text_encoder_precisions = ("bf16",)
    loader = TextEncoderLoader()
    fv_encoder = loader.load(text_encoder_path, fv_args)
    fv_encoder = fv_encoder.to(device).to(dtype).eval()

    from sglang.multimodal_gen.runtime.loader.component_loaders.component_loader import (
        ComponentLoader,
    )
    from sglang.multimodal_gen.runtime.managers.forward_context import (
        set_forward_context as sgl_set_forward_context,
    )
    from sglang.multimodal_gen.runtime.server_args import ServerArgs, set_global_server_args
    from sglang.multimodal_gen.configs.pipeline_configs.flux import (
        Flux2KleinPipelineConfig as SGLangFlux2KleinConfig,
    )

    print("Loading SGLang text encoder ...")
    sgl_args = ServerArgs(
        model_path=root,
        pipeline_config=SGLangFlux2KleinConfig(),
        hsdp_shard_dim=1,
        hsdp_replicate_dim=1,
        attention_backend="torch_sdpa",
    )
    sgl_args.model_paths = {"text_encoder": text_encoder_path}
    set_global_server_args(sgl_args)
    sgl_loader = ComponentLoader.for_component_type("text_encoder", "transformers")
    sgl_encoder, _ = sgl_loader.load(
        text_encoder_path, sgl_args, "text_encoder", "transformers"
    )
    if "FSDP" not in type(sgl_encoder).__name__:
        sgl_encoder = sgl_encoder.to(device).to(dtype)
    sgl_encoder = sgl_encoder.eval()

    # 2. Verify inputs
    print("\n--- 1. Input verification ---")
    print("  input_ids shape:", input_ids.shape)
    print("  attention_mask shape:", attention_mask.shape)
    print("  (Both encoders receive identical inputs from same tokenizer)")

    # 3. Layer-by-layer hooks (run full forward only - avoid direct submodule calls for FSDP)
    print("\n--- 2. Layer-by-layer comparison ---")
    fv_layers = _get_model_layers(fv_encoder)
    ref_layers = _get_model_layers(sgl_encoder)
    assert len(fv_layers) == len(ref_layers), "Layer count mismatch"

    fv_hiddens = []
    ref_hiddens = []
    fv_emb, ref_emb = [], []

    def make_hook(storage):
        def hook(module, inp, out):
            h = out[0] if isinstance(out, tuple) else out
            storage.append(h.detach())
        return hook

    def make_pre_hook(emb_storage):
        def pre_hook(module, inp):
            h = inp[0] if isinstance(inp, tuple) else inp
            emb_storage.append(h.detach())
        return pre_hook

    fv_layers[0].register_forward_pre_hook(make_pre_hook(fv_emb))
    ref_layers[0].register_forward_pre_hook(make_pre_hook(ref_emb))
    for layer in fv_layers:
        layer.register_forward_hook(make_hook(fv_hiddens))
    for layer in ref_layers:
        layer.register_forward_hook(make_hook(ref_hiddens))

    enc_kwargs = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "output_hidden_states": True,
    }

    with torch.no_grad(), set_forward_context(current_timestep=0, attn_metadata=None):
        _ = fv_encoder(**enc_kwargs)
    with torch.no_grad(), sgl_set_forward_context(current_timestep=0, attn_metadata=None):
        _ = sgl_encoder(**enc_kwargs)

    if fv_emb and ref_emb:
        print("\n--- Embedding (layer 0 input) ---")
        diff_emb = (fv_emb[0].float() - ref_emb[0].float()).abs()
        print(f"  max abs diff: {diff_emb.max().item():.6f}  mean: {diff_emb.mean().item():.6f}")
        print("  -> Embeddings match." if diff_emb.max().item() < 1e-4 else "  -> Embeddings DIFFER.")

    first_divergent = None
    for i, (fv_h, ref_h) in enumerate(zip(fv_hiddens, ref_hiddens)):
        diff = (fv_h.float() - ref_h.float()).abs()
        max_d = diff.max().item()
        mean_d = diff.mean().item()
        status = "OK" if max_d < 1e-3 else "DIVERGE"
        if first_divergent is None and max_d >= 1e-3:
            first_divergent = i
        print(f"  Layer {i:2d}: max_diff={max_d:.6f} mean_diff={mean_d:.6f}  {status}")

    if first_divergent is not None:
        print(f"\n  -> First divergent layer: {first_divergent}")

        # 5. Drill into first divergent layer: attention vs MLP
        print("\n--- 3. First divergent layer: attention vs MLP ---")
        fv_layer = fv_layers[first_divergent]
        ref_layer = ref_layers[first_divergent]

        fv_attn_out = []
        fv_mlp_out = []
        ref_attn_out = []
        ref_mlp_out = []

        def attn_hook(module, inp, out):
            h = out[0] if isinstance(out, tuple) else out
            fv_attn_out.append(h.detach()) if "fv_" in str(module) else ref_attn_out.append(h.detach())

        # Use separate lists via closure
        def make_attn_hook(storage):
            def h(module, inp, out):
                storage.append((out[0] if isinstance(out, tuple) else out).detach())
            return h

        def make_mlp_hook(storage):
            def h(module, inp, out):
                storage.append(out.detach())
            return h

        fv_layer.self_attn.register_forward_hook(make_attn_hook(fv_attn_out))
        fv_layer.mlp.register_forward_hook(make_mlp_hook(fv_mlp_out))
        ref_layer.self_attn.register_forward_hook(make_attn_hook(ref_attn_out))
        ref_layer.mlp.register_forward_hook(make_mlp_hook(ref_mlp_out))

        with torch.no_grad(), set_forward_context(current_timestep=0, attn_metadata=None):
            _ = fv_encoder(**enc_kwargs)
        with torch.no_grad(), sgl_set_forward_context(current_timestep=0, attn_metadata=None):
            _ = sgl_encoder(**enc_kwargs)

        if fv_attn_out and ref_attn_out:
            d_attn = (fv_attn_out[0].float() - ref_attn_out[0].float()).abs()
            print(f"  Attention output: max_diff={d_attn.max().item():.6f} mean_diff={d_attn.mean().item():.6f}")
        if fv_mlp_out and ref_mlp_out:
            d_mlp = (fv_mlp_out[0].float() - ref_mlp_out[0].float()).abs()
            print(f"  MLP output:       max_diff={d_mlp.max().item():.6f} mean_diff={d_mlp.mean().item():.6f}")

        # 6. Layer 0 granular: input_norm, attn, post_attn_norm, mlp
        print("\n--- 4. Layer 0 granular (input_norm -> attn -> post_attn_norm -> mlp) ---")
        fv_layer = fv_layers[0]
        ref_layer = ref_layers[0]

        def make_out_hook(storage):
            def h(module, inp, out):
                x = out[0] if isinstance(out, tuple) else out
                if isinstance(x, tuple):
                    x = x[0]
                storage.append(x.detach())
            return h

        fv_in, ref_in = [], []
        fv_attn, ref_attn = [], []
        fv_post, ref_post = [], []
        fv_mlp2, ref_mlp2 = [], []

        fv_layer.input_layernorm.register_forward_hook(make_out_hook(fv_in))
        ref_layer.input_layernorm.register_forward_hook(make_out_hook(ref_in))
        fv_layer.self_attn.register_forward_hook(make_out_hook(fv_attn))
        ref_layer.self_attn.register_forward_hook(make_out_hook(ref_attn))
        fv_layer.post_attention_layernorm.register_forward_hook(make_out_hook(fv_post))
        ref_layer.post_attention_layernorm.register_forward_hook(make_out_hook(ref_post))
        fv_layer.mlp.register_forward_hook(make_out_hook(fv_mlp2))
        ref_layer.mlp.register_forward_hook(make_out_hook(ref_mlp2))

        with torch.no_grad(), set_forward_context(current_timestep=0, attn_metadata=None):
            _ = fv_encoder(**enc_kwargs)
        with torch.no_grad(), sgl_set_forward_context(current_timestep=0, attn_metadata=None):
            _ = sgl_encoder(**enc_kwargs)

        for name, fa, ra in [
            ("input_layernorm (-> attn)", fv_in, ref_in),
            ("self_attn output", fv_attn, ref_attn),
            ("post_attention_layernorm (-> mlp)", fv_post, ref_post),
            ("mlp output", fv_mlp2, ref_mlp2),
        ]:
            if fa and ra:
                d = (fa[0].float() - ra[0].float()).abs()
                status = "OK" if d.max().item() < 1e-3 else "DIVERGE"
                print(f"  {name}: max={d.max().item():.6f} mean={d.mean().item():.6f}  {status}")

        # 7. Attention-internal: qkv_proj, q_norm, k_norm, rope, attn, o_proj
        print("\n--- 5. Layer 0 attention internal ---")
        fv_attn_mod = fv_layers[0].self_attn
        ref_attn_mod = ref_layers[0].self_attn

        fv_qkv, ref_qkv = [], []
        fv_qn, ref_qn = [], []
        fv_kn, ref_kn = [], []
        fv_qr, ref_qr = [], []
        fv_kr, ref_kr = [], []
        fv_attn_out, ref_attn_out = [], []
        fv_o, ref_o = [], []

        def _out(storage):
            return lambda m, i, o: storage.append(
                (o[0] if isinstance(o, tuple) else o).detach()
            )

        def _rope_hook(q_stor, k_stor):
            def h(m, i, o):
                q_stor.append(o[0].detach())
                k_stor.append(o[1].detach())
            return h

        fv_attn_mod.qkv_proj.register_forward_hook(_out(fv_qkv))
        ref_attn_mod.qkv_proj.register_forward_hook(_out(ref_qkv))
        fv_attn_mod.q_norm.register_forward_hook(_out(fv_qn))
        ref_attn_mod.q_norm.register_forward_hook(_out(ref_qn))
        fv_attn_mod.k_norm.register_forward_hook(_out(fv_kn))
        ref_attn_mod.k_norm.register_forward_hook(_out(ref_kn))
        fv_attn_mod.rotary_emb.register_forward_hook(_rope_hook(fv_qr, fv_kr))
        ref_attn_mod.rotary_emb.register_forward_hook(_rope_hook(ref_qr, ref_kr))
        fv_attn_mod.attn.register_forward_hook(_out(fv_attn_out))
        ref_attn_mod.attn.register_forward_hook(_out(ref_attn_out))
        fv_attn_mod.o_proj.register_forward_hook(_out(fv_o))
        ref_attn_mod.o_proj.register_forward_hook(_out(ref_o))

        with torch.no_grad(), set_forward_context(current_timestep=0, attn_metadata=None):
            _ = fv_encoder(**enc_kwargs)
        with torch.no_grad(), sgl_set_forward_context(current_timestep=0, attn_metadata=None):
            _ = sgl_encoder(**enc_kwargs)

        for name, fa, ra in [
            ("qkv_proj", fv_qkv, ref_qkv),
            ("q_norm", fv_qn, ref_qn),
            ("k_norm", fv_kn, ref_kn),
            ("q after RoPE", fv_qr, ref_qr),
            ("k after RoPE", fv_kr, ref_kr),
            ("attn (before o_proj)", fv_attn_out, ref_attn_out),
            ("o_proj", fv_o, ref_o),
        ]:
            if fa and ra:
                d = (fa[0].float() - ra[0].float()).abs()
                status = "OK" if d.max().item() < 1e-3 else "DIVERGE"
                print(f"  {name}: max={d.max().item():.6f} mean={d.mean().item():.6f}  {status}")
    else:
        print("\n  -> All layers match within tolerance.")


if __name__ == "__main__":
    main()
