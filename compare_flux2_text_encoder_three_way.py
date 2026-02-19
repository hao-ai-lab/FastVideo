#!/usr/bin/env python3
"""
Compare FastVideo, SGLang, and Diffusers Flux2 Klein text encoder outputs in one run.

Uses the same tokenized input (diffusers pipeline tokenizer) for all three.
Reports pairwise max diff: FV vs Diffusers, SGLang vs Diffusers, FV vs SGLang.
Diffusers is the reference implementation.

  python compare_flux2_text_encoder_three_way.py
  python compare_flux2_text_encoder_three_way.py --no-sglang   # skip SGLang (no SGLANG_PATH)
"""
import argparse
import inspect
import os
import sys

import torch

# Add SGLang to path if available
SGLANG_PATH = os.environ.get(
    "SGLANG_PATH",
    os.path.join(os.path.dirname(__file__), "..", "sglang", "python"),
)
if os.path.isdir(SGLANG_PATH) and SGLANG_PATH not in sys.path:
    sys.path.insert(0, SGLANG_PATH)

PROMPT = "a red apple on a table"
MODEL_ID = "black-forest-labs/FLUX.2-klein-4B"


def _get_text_encoder_path(model_id: str) -> str:
    """Resolve text encoder path from model ID."""
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


def flux2_klein_postprocess(outputs, hidden_states_layers=(9, 18, 27)):
    """Stack hidden states from layers 9, 18, 27 -> prompt_embeds."""
    if outputs.hidden_states is None:
        raise ValueError("output_hidden_states must be True")
    out = torch.stack([outputs.hidden_states[k] for k in hidden_states_layers], dim=1)
    batch_size, num_channels, seq_len, hidden_dim = out.shape
    prompt_embeds = out.permute(0, 2, 1, 3).reshape(
        batch_size, seq_len, num_channels * hidden_dim
    )
    return prompt_embeds


def _tokenize_with_pipeline(pipe, messages, device, max_length=512):
    """Tokenize with pipeline tokenizer; return input_ids, attention_mask on device."""
    messages_batch = [messages]
    try:
        inputs = pipe.tokenizer.apply_chat_template(
            messages_batch,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=max_length,
        )
    except TypeError:
        inputs = pipe.tokenizer.apply_chat_template(
            messages_batch,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=max_length,
        )
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)
    return input_ids, attention_mask


def main():
    parser = argparse.ArgumentParser(
        description="Compare FastVideo, SGLang, and Diffusers Flux2 Klein text encoders (same input, pairwise diffs)."
    )
    parser.add_argument("--model-path", default=None, help="Path to FLUX.2-klein-4B or text_encoder dir")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--prompt", default=PROMPT)
    parser.add_argument("--no-sglang", action="store_true", help="Skip loading SGLang encoder")
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
    load_path = root if os.path.isdir(root) else MODEL_ID
    device = args.device
    dtype = torch.bfloat16
    messages = [{"role": "user", "content": args.prompt}]
    max_length = 512

    # 1. Load diffusers pipeline (tokenizer + reference encoder)
    try:
        from diffusers import Flux2KleinPipeline
    except ImportError:
        from diffusers.pipelines.flux2 import Flux2KleinPipeline
    print("Loading Diffusers Flux2KleinPipeline (tokenizer + text encoder) ...")
    pipe = Flux2KleinPipeline.from_pretrained(load_path, torch_dtype=dtype)
    pipe = pipe.to(device)
    input_ids, attention_mask = _tokenize_with_pipeline(pipe, messages, device, max_length)
    print(f"  Tokenized prompt: {args.prompt!r} -> input_ids shape {input_ids.shape}")

    # 2. Run Diffusers encoder
    enc = pipe.text_encoder
    enc_kwargs = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "output_hidden_states": True,
    }
    if "use_cache" in inspect.signature(enc.forward).parameters:
        enc_kwargs["use_cache"] = False
    with torch.no_grad():
        diffusers_outputs = enc(**enc_kwargs)
    diffusers_prompt_embeds = flux2_klein_postprocess(diffusers_outputs)
    diffusers_prompt_embeds = diffusers_prompt_embeds.detach().cpu().float()
    print(f"  Diffusers prompt_embeds shape: {diffusers_prompt_embeds.shape}")

    # 3. FastVideo encoder (same input)
    from fastvideo.distributed import maybe_init_distributed_environment_and_model_parallel
    maybe_init_distributed_environment_and_model_parallel(tp_size=1, sp_size=1)

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
    with torch.no_grad(), set_forward_context(current_timestep=0, attn_metadata=None):
        fv_outputs = fv_encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )
    fv_prompt_embeds = flux2_klein_postprocess(fv_outputs)
    fv_prompt_embeds = fv_prompt_embeds.cpu().float()
    print(f"  FastVideo prompt_embeds shape: {fv_prompt_embeds.shape}")

    # 4. SGLang encoder (same input), optional
    sglang_prompt_embeds = None
    if not args.no_sglang:
        try:
            from sglang.multimodal_gen.runtime.distributed import (
                maybe_init_distributed_environment_and_model_parallel as sgl_init,
            )
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

            sgl_init(tp_size=1, sp_size=1, enable_cfg_parallel=False)
            print("Loading SGLang text encoder ...")
            sgl_args = ServerArgs(
                model_path=root,
                pipeline_config=SGLangFlux2KleinConfig(),
                attention_backend="torch_sdpa",
                text_encoder_cpu_offload=False,
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
            with torch.no_grad(), sgl_set_forward_context(current_timestep=0, attn_metadata=None):
                sgl_outputs = sgl_encoder(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    output_hidden_states=True,
                )
            sglang_prompt_embeds = flux2_klein_postprocess(sgl_outputs)
            sglang_prompt_embeds = sglang_prompt_embeds.cpu().float()
            print(f"  SGLang prompt_embeds shape: {sglang_prompt_embeds.shape}")
        except ImportError as e:
            print(f"  SGLang skipped (not available): {e}")

    # 5. Pairwise comparison
    def max_diff(a: torch.Tensor, b: torch.Tensor) -> float:
        if a.shape != b.shape:
            return float("nan")
        return (a - b).abs().max().item()

    print("\n--- Three-way text encoder comparison (same input, prompt_embeds) ---")
    print(f"  Shape: {diffusers_prompt_embeds.shape}")
    print()

    # FV vs Diffusers (reference)
    fv_vs_diff = max_diff(fv_prompt_embeds, diffusers_prompt_embeds)
    print(f"  FastVideo vs Diffusers:  max |diff| = {fv_vs_diff:.6f}")
    print(f"    allclose(rtol=1e-2, atol=1e-2): {torch.allclose(fv_prompt_embeds, diffusers_prompt_embeds, rtol=1e-2, atol=1e-2)}")

    if sglang_prompt_embeds is not None:
        # SGLang vs Diffusers
        sgl_vs_diff = max_diff(sglang_prompt_embeds, diffusers_prompt_embeds)
        print(f"  SGLang vs Diffusers:     max |diff| = {sgl_vs_diff:.6f}")
        print(f"    allclose(rtol=1e-2, atol=1e-2): {torch.allclose(sglang_prompt_embeds, diffusers_prompt_embeds, rtol=1e-2, atol=1e-2)}")
        # FV vs SGLang
        fv_vs_sgl = max_diff(fv_prompt_embeds, sglang_prompt_embeds)
        print(f"  FastVideo vs SGLang:     max |diff| = {fv_vs_sgl:.6f}")
        print(f"    allclose(rtol=1e-2, atol=1e-2): {torch.allclose(fv_prompt_embeds, sglang_prompt_embeds, rtol=1e-2, atol=1e-2)}")

    print()
    if sglang_prompt_embeds is None:
        print("  (Run without --no-sglang and SGLANG_PATH set to include SGLang in comparison.)")
        if fv_vs_diff <= 1e-2:
            print("  FastVideo aligns with Diffusers (reference).")
        else:
            print("  FastVideo does not match Diffusers within tolerance.")
    else:
        ref_match = "Diffusers" if fv_vs_diff <= 1e-2 else ("SGLang" if fv_vs_sgl <= 1e-2 else "neither")
        print(f"  FastVideo aligns with: {ref_match}")


if __name__ == "__main__":
    main()
