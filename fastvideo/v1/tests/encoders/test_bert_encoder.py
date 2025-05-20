import sys
sys.argv = [sys.argv[0]] 
import os
from fastvideo.v1.models.loader.component_loader import VAELoader, TokenizerLoader, TextEncoderLoader
from fastvideo.v1.fastvideo_args import FastVideoArgs
import torch
from fastvideo.v1.configs.models.vaes import StepVideoVAEConfig
from fastvideo.models.stepvideo.text_encoder.clip import HunyuanClip
from fastvideo.v1.configs.models.encoders.bert import BertConfig
import torch.distributed as dist
from fastvideo.v1.distributed.parallel_state import initialize_model_parallel
import pytest
from fastvideo.v1.forward_context import set_forward_context
from types import SimpleNamespace
os.environ.setdefault("MASTER_ADDR", "localhost")
os.environ.setdefault("MASTER_PORT", "29500")
# if not dist.is_initialized():
#     dist.init_process_group("gloo", rank=0, world_size=1)
# initialize_model_parallel(tensor_model_parallel_size=1)

model_path = 'cache/hub/models--FastVideo--stepvideo-t2v-diffusers/snapshots/572e0ce299de9fe2f8b843afe5afce5facb23c13'
VAE_PATH = model_path + '/vae'
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
precision = torch.bfloat16
precision_str = "bf16"
args = FastVideoArgs(model_path=model_path,
                        text_encoder_precisions=("fp16",),
                        text_encoder_configs=(BertConfig(),))
args.device_str = "cuda:0"
# args = FastVideoArgs(model_path=model_path, vae_precision=precision_str)
args.vae_config = StepVideoVAEConfig()
PROMPTS = [
    "A beautiful sunset over the mountains",
    "A futuristic city skyline at night",
    "A serene beach with palm trees and clear water",
    "A bustling market street in a vibrant city",
    "A cozy cabin in the woods during winter"
]

@pytest.mark.usefixtures("distributed_setup")
def build_fastvideo_encoder(old_model_path):
    """Instantiate the FV tokenizer + text encoder the same way your loader will
    inside the pipeline, but *outside* the full pipeline for easier unit test."""

    tokenizer = TokenizerLoader().load(
        model_path=os.path.join(old_model_path, "tokenizer"),
        architecture="bert",
        fastvideo_args=args,
    )

    encoder = TextEncoderLoader().load(
        model_path=os.path.join(old_model_path, "text_encoder"),
        architecture="bert",
        fastvideo_args=args,
    ).to(dtype=precision, device=device)

    return tokenizer, encoder  

@pytest.mark.usefixtures("distributed_setup")
def test_hidden_and_pooled_equivalence():
    # ----- original HF path
    old_model_path = os.path.join(model_path, "hunyuan_clip")
    legacy = HunyuanClip(old_model_path).to(dtype=precision, device=device).eval()
    with torch.no_grad():
        h_legacy, text_inputs = legacy(PROMPTS)

    # ----- fastvideo path
    tok, fv_enc = build_fastvideo_encoder(old_model_path)
    tok_out = tok(
        PROMPTS,
        padding="max_length",
        max_length=legacy.max_length,
        truncation=True,
        return_attention_mask=True,
        return_tensors="pt",
    )
    # 2) Get the “fastvideo” tokenizer outputs
    fv_tok = tok_out  # already a BatchEncoding from your TokenizerLoader

    # 3) Make sure they have the same keys
    assert set(text_inputs.keys()) == set(fv_tok.keys()), \
        f"Mismatch in keys: {text_inputs.keys()} vs {fv_tok.keys()}"

    # 4) For each tensor, compare shape and contents
    for key in text_inputs.keys():
        a = text_inputs[key]
        b = fv_tok[key]
        # Move both to CPU so the comparison is unambiguous
        a_cpu = a.cpu()
        b_cpu = b.cpu()
        # Check shapes
        assert a_cpu.shape == b_cpu.shape, f"Shape mismatch for `{key}`: {a_cpu.shape} vs {b_cpu.shape}"
        # Check values exactly
        if a_cpu.dtype.is_floating_point:
            # floats: allow exact equality since token IDs/masks are integers
            assert torch.equal(a_cpu, b_cpu), f"Floating values differ for `{key}`"
            print(key, torch.equal(a_cpu, b_cpu))
        else:
            # ints (input_ids, attention_mask): must be identical
            assert torch.equal(a_cpu, b_cpu), f"Integer tensors differ for `{key}`"
            print(key, torch.equal(a_cpu, b_cpu))
        
        
    seq_len      = tok_out.input_ids.size(1)               # 77
    position_ids = torch.arange(seq_len, device=device).unsqueeze(0)
    position_ids = position_ids.expand(tok_out.input_ids.size(0), -1)


    with set_forward_context(current_timestep=0, attn_metadata=None):
        fv_out = fv_enc(
            input_ids=tok_out.input_ids.to(device),
            position_ids=position_ids.to(device),
            attention_mask=tok_out.attention_mask.to(device),
        )
    h_fv= fv_out
    
    # assume h_legacy, h_fv are Float/BFloat tensors on the same device
    diff = (h_legacy - h_fv).abs()

    # overall stats
    max_diff  = diff.max().item()
    mean_diff = diff.mean().item()
    print(f"[DEBUG] overall max abs diff   = {max_diff:.6f}")
    print(f"[DEBUG] overall mean abs diff  = {mean_diff:.6f}")

    # per‐position mean diff (averaged over hidden dim)
    per_token = diff.mean(dim=-1)      # shape [batch, seq_len]
    print("[DEBUG] per-token mean diff:", per_token[0].tolist())

    # if you only want to compare *valid* tokens (mask==1)
    mask = tok_out.attention_mask.to(device).unsqueeze(-1)   # [B, L, 1]
    valid_diff = (h_legacy - h_fv).abs()[mask.bool().expand_as(h_legacy)]
    print(f"[DEBUG] valid-token max diff = {valid_diff.max().item():.6f}")
    print(f"[DEBUG] valid-token mean diff= {valid_diff.mean().item():.6f}")

    # after you have `legacy` and `fv_enc` on the same device
    legacy_state = {n: p.detach().cpu() for n, p in legacy.text_encoder.named_parameters()}
    fv_state     = {n: p.detach().cpu() for n, p in fv_enc.named_parameters()}
    print(len(legacy_state.keys()), len(fv_state.keys()))
    print([k for k in fv_state.keys() if not k.startswith('layer.')])
    for name in ["embeddings.word_embeddings.weight",
                "embeddings.position_embeddings.weight"]:
        a = legacy_state[name]
        b = fv_state[name][:47020,:]
        print(f"for {name}: {a.shape}, {b.shape}")
        print(f"{name}  max abs diff = {(a - b).abs().max():.6f}")

    for k in legacy_state.keys():
        if k not in fv_state:
            print(f"missing in fv_state: {k}")
    # ----- 5.  assertions
    assert h_legacy.shape == h_fv.shape
    # assert p_legacy.shape == p_fv.shape
    assert torch.allclose(h_legacy, h_fv, atol=1e-4, rtol=1e-3)
    # assert torch.allclose(p_legacy, p_fv)

if __name__ == "__main__":
    test_hidden_and_pooled_equivalence()