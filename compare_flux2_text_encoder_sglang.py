#!/usr/bin/env python3
"""
Compare FastVideo vs reference (diffusers or SGLang) Flux2 Klein text encoder outputs.

Uses same prompt; compares prompt_embeds (layers 9, 18, 27 stacked).
Reference: diffusers Flux2KleinPipeline.encode_prompt (official).
Optionally uses SGLang if PYTHONPATH includes sglang and --use-sglang is passed.

  python compare_flux2_text_encoder_sglang.py
  python compare_flux2_text_encoder_sglang.py --use-sglang  # requires SGLang
"""
import argparse
import os
import sys

import torch

# Add SGLang to path if available
SGLANG_PATH = os.environ.get("SGLANG_PATH", os.path.join(os.path.dirname(__file__), "..", "sglang", "python"))
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


def main():
    parser = argparse.ArgumentParser(description="Compare FastVideo vs reference Flux2 Klein text encoder.")
    parser.add_argument("--model-path", default=None, help="Path to FLUX.2-klein-4B or text_encoder dir")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--prompt", default=PROMPT)
    parser.add_argument("--use-sglang", action="store_true", help="Use SGLang as reference (default: diffusers)")
    args = parser.parse_args()

    os.environ.setdefault("LOCAL_RANK", "0")
    os.environ.setdefault("RANK", "0")
    os.environ.setdefault("WORLD_SIZE", "1")
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29500")

    model_path = args.model_path or _get_text_encoder_path(MODEL_ID)
    # Repo root: FastVideoArgs expects model_index.json; tokenizer is at root or tokenizer/
    root = os.path.dirname(model_path) if os.path.basename(model_path) == "text_encoder" else model_path
    text_encoder_path = model_path if os.path.basename(model_path) == "text_encoder" else os.path.join(root, "text_encoder")
    tokenizer_dir = os.path.join(root, "tokenizer")
    tokenizer_path = tokenizer_dir if os.path.isdir(tokenizer_dir) else root
    device = args.device
    prompt = args.prompt
    dtype = torch.bfloat16

    # Tokenize once (identical input for both)
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
    # Use chat template to match typical Flux2 Klein usage
    messages = [{"role": "user", "content": prompt}]
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

    # 1. FastVideo text encoder
    print("Loading FastVideo text encoder ...")
    from fastvideo.distributed import maybe_init_distributed_environment_and_model_parallel
    maybe_init_distributed_environment_and_model_parallel(tp_size=1, sp_size=1)

    from fastvideo.configs.pipelines.flux_2 import Flux2KleinPipelineConfig
    from fastvideo.fastvideo_args import FastVideoArgs
    from fastvideo.models.loader.component_loader import TextEncoderLoader

    fv_args = FastVideoArgs.from_kwargs(
        model_path=root,
        pipeline_config=Flux2KleinPipelineConfig(),
    )
    fv_args.pipeline_config.text_encoder_precisions = ("bf16",)
    loader = TextEncoderLoader()
    fv_encoder = loader.load(text_encoder_path, fv_args)
    fv_encoder = fv_encoder.eval()

    with torch.no_grad():
        fv_outputs = fv_encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )
    fv_prompt_embeds = flux2_klein_postprocess(fv_outputs)
    fv_prompt_embeds = fv_prompt_embeds.cpu().float()

    # 2. Reference text encoder (diffusers or SGLang)
    ref_prompt_embeds = None
    ref_name = "diffusers"

    if args.use_sglang:
        try:
            from sglang.multimodal_gen.runtime.loader.component_loader import TextEncoderLoader as SGLangTextEncoderLoader
            from sglang.multimodal_gen.runtime.server_args import ServerArgs
            from sglang.multimodal_gen.configs.pipeline_configs.flux import Flux2KleinPipelineConfig as SGLangFlux2KleinConfig

            print("Loading SGLang text encoder ...")
            sgl_args = ServerArgs(
                model_path=model_path,
                pipeline_config=SGLangFlux2KleinConfig(),
            )
            sgl_loader = SGLangTextEncoderLoader()
            sgl_encoder = sgl_loader.load(model_path, sgl_args)
            sgl_encoder = sgl_encoder.to(device).to(dtype).eval()

            with torch.no_grad():
                sgl_outputs = sgl_encoder(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    output_hidden_states=True,
                )
            ref_prompt_embeds = flux2_klein_postprocess(sgl_outputs)
            ref_prompt_embeds = ref_prompt_embeds.cpu().float()
            ref_name = "SGLang"
        except ImportError as e:
            print(f"SGLang not available ({e}), falling back to diffusers.")
            args.use_sglang = False

    if ref_prompt_embeds is None:
        # Use diffusers Flux2KleinPipeline.encode_prompt
        try:
            from diffusers import Flux2KleinPipeline
        except ImportError:
            from diffusers.pipelines.flux2 import Flux2KleinPipeline
        load_path = root if os.path.isdir(root) else MODEL_ID
        print("Loading diffusers Flux2KleinPipeline for reference ...")
        pipe = Flux2KleinPipeline.from_pretrained(
            load_path,
            torch_dtype=dtype,
        )
        pipe = pipe.to(device)
        ref_prompt_embeds, _ = pipe.encode_prompt(
            prompt=prompt,
            device=device,
            num_images_per_prompt=1,
            max_sequence_length=512,
            text_encoder_out_layers=(9, 18, 27),
        )
        ref_prompt_embeds = ref_prompt_embeds.detach().cpu().float()
        ref_name = "diffusers"

    # Compare
    print(f"\n--- Text encoder comparison (FastVideo vs {ref_name}) ---")
    print(f"  FastVideo prompt_embeds shape: {fv_prompt_embeds.shape}")
    print(f"  {ref_name:10} prompt_embeds shape: {ref_prompt_embeds.shape}")
    if fv_prompt_embeds.shape != ref_prompt_embeds.shape:
        print("  SHAPE MISMATCH")
    else:
        diff = (fv_prompt_embeds - ref_prompt_embeds).abs()
        print(f"  max abs diff:  {diff.max().item():.6f}")
        print(f"  mean abs diff: {diff.mean().item():.6f}")
        allclose = torch.allclose(fv_prompt_embeds, ref_prompt_embeds, rtol=1e-2, atol=1e-2)
        print(f"  allclose(rtol=1e-2, atol=1e-2): {allclose}")
        if allclose:
            print("  -> Text encoder outputs match.")
        else:
            print("  -> Text encoder outputs differ.")


if __name__ == "__main__":
    main()
