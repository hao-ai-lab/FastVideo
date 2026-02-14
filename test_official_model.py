"""Run the OFFICIAL Overworld WorldModel directly (no FastVideo)
to verify whether the checkpoint + VAE produce a correct image.

Usage: python test_official_model.py
"""
# ruff: noqa: E402  # imports are mid-file for dynamic loading
import torch
import os
import glob

# ── Load the VAE (same as FastVideo does - dynamic import) ──
snapshot_dir = glob.glob(
    os.path.expanduser(
        "~/.cache/huggingface/hub/models--FastVideo--Waypoint-1-Small-Diffusers"
        "/snapshots/*"))[0]
vae_dir = os.path.join(snapshot_dir, "vae")
transformer_dir = os.path.join(snapshot_dir, "transformer")

# Load VAE via diffusers dynamic loading
import json
import importlib
import sys
import types

vae_config_path = os.path.join(vae_dir, "config.json")
with open(vae_config_path) as f:
    vae_config = json.load(f)

auto_map = vae_config.get("auto_map", {})
target = auto_map.get("AutoModel") or auto_map.get(
    "AutoencoderKL") or auto_map.get("Autoencoder")
module_name, cls_name = target.rsplit(".", 1)

# Dynamically load the VAE module
py_path = os.path.join(vae_dir, f"{module_name}.py")
pkg_name = "test_vae_pkg"
pkg = types.ModuleType(pkg_name)
pkg.__path__ = [vae_dir]
pkg.__package__ = pkg_name
sys.modules[pkg_name] = pkg

# Pre-load sibling .py files
for pyf in sorted(glob.glob(os.path.join(vae_dir, "*.py"))):
    mod_name = os.path.splitext(os.path.basename(pyf))[0]
    full_name = f"{pkg_name}.{mod_name}"
    spec = importlib.util.spec_from_file_location(full_name,
                                                  pyf,
                                                  submodule_search_locations=[])
    if spec is None:
        continue
    assert spec is not None  # narrow for mypy after continue
    mod = importlib.util.module_from_spec(spec)
    mod.__package__ = pkg_name
    sys.modules[full_name] = mod
    spec.loader.exec_module(mod)

vae_mod = sys.modules[f"{pkg_name}.{module_name}"]
VAEClass = getattr(vae_mod, cls_name)

cfg = dict(vae_config)
for k in ("_class_name", "_diffusers_version", "auto_map"):
    cfg.pop(k, None)

# Filter to only params the class __init__ actually accepts
import inspect

sig = inspect.signature(VAEClass.__init__)
valid_params = set(sig.parameters.keys()) - {"self"}
if any(p.kind == inspect.Parameter.VAR_KEYWORD
       for p in sig.parameters.values()):
    filtered_cfg = cfg  # accepts **kwargs
else:
    filtered_cfg = {k: v for k, v in cfg.items() if k in valid_params}
    dropped = set(cfg) - set(filtered_cfg)
    if dropped:
        print(f"  Dropped config keys not in VAE __init__: {dropped}")
vae = VAEClass(**filtered_cfg)

# Load VAE weights
from safetensors.torch import load_file

vae_weights = {}
for sf in glob.glob(os.path.join(vae_dir, "*.safetensors")):
    vae_weights.update(load_file(sf))
vae.load_state_dict(vae_weights, strict=False)
vae = vae.to("cuda").eval()
print(f"VAE loaded: {type(vae).__name__}, dtype={next(vae.parameters()).dtype}")

# ── Load the official transformer ──
# Use the HF model code from Overworld/Waypoint-1-Small directly
from huggingface_hub import snapshot_download

official_dir = snapshot_download("Overworld/Waypoint-1-Small",
                                 allow_patterns=["transformer/*"])
print(f"Official model dir: {official_dir}")

# Load transformer dynamically
t_dir = os.path.join(official_dir, "transformer")
t_config_path = os.path.join(t_dir, "config.json")
with open(t_config_path) as f:
    t_config = json.load(f)

pkg2_name = "test_transformer_pkg"
pkg2 = types.ModuleType(pkg2_name)
pkg2.__path__ = [t_dir]
pkg2.__package__ = pkg2_name
sys.modules[pkg2_name] = pkg2

for pyf in sorted(glob.glob(os.path.join(t_dir, "*.py"))):
    mod_name = os.path.splitext(os.path.basename(pyf))[0]
    full_name = f"{pkg2_name}.{mod_name}"
    spec = importlib.util.spec_from_file_location(full_name,
                                                  pyf,
                                                  submodule_search_locations=[])
    if spec is None:
        continue
    assert spec is not None  # narrow for mypy after continue
    mod = importlib.util.module_from_spec(spec)
    mod.__package__ = pkg2_name
    sys.modules[full_name] = mod
    try:
        spec.loader.exec_module(mod)
    except Exception as e:
        print(f"  Warning loading {mod_name}: {e}")

model_mod = sys.modules.get(f"{pkg2_name}.model")
if model_mod is None:
    print("ERROR: could not load transformer model.py")
    sys.exit(1)

WorldModel = model_mod.WorldModel

# Create model from config
model_cfg = {
    k: v
    for k, v in t_config.items()
    if k not in ("_class_name", "_diffusers_version", "auto_map")
}

sig2 = inspect.signature(WorldModel.__init__)
valid2 = set(sig2.parameters.keys()) - {"self"}
if any(p.kind == inspect.Parameter.VAR_KEYWORD
       for p in sig2.parameters.values()):
    filtered_model_cfg = model_cfg
else:
    filtered_model_cfg = {k: v for k, v in model_cfg.items() if k in valid2}
    dropped2 = set(model_cfg) - set(filtered_model_cfg)
    if dropped2:
        print(f"  Dropped config keys not in WorldModel __init__: {dropped2}")
transformer = WorldModel(**filtered_model_cfg)

# Load weights
t_weights = {}
for sf in glob.glob(os.path.join(t_dir, "*.safetensors")):
    t_weights.update(load_file(sf))
missing, unexpected = transformer.load_state_dict(t_weights, strict=False)
print(
    f"Transformer loaded: missing={len(missing)} unexpected={len(unexpected)}")
if missing:
    print(f"  Missing: {missing[:5]}")
if unexpected:
    print(f"  Unexpected: {unexpected[:5]}")

transformer = transformer.to(device="cuda", dtype=torch.bfloat16).eval()

# ── Load text encoder + tokenizer ──
from transformers import AutoTokenizer

te_dir = os.path.join(snapshot_dir, "text_encoder")
tok_dir = os.path.join(snapshot_dir, "tokenizer")

tokenizer = AutoTokenizer.from_pretrained(tok_dir)
# The text encoder is UMT5ForConditionalGeneration, we need just the encoder
from transformers import T5EncoderModel

text_encoder = T5EncoderModel.from_pretrained(te_dir).to("cuda").eval()
print(f"Text encoder loaded: {type(text_encoder).__name__}")


# ── Build KV cache (using official cache.py or dummy) ──
class DummyKVCache:
    """Minimal cache interface so official Attn.forward() can call kv_cache.upsert()."""

    def set_frozen(self, frozen):
        pass

    def upsert(self, k, v, pos_ids, layer_idx):
        return k, v, None  # no block mask = full attention within current frame


cache_mod = sys.modules.get(f"{pkg2_name}.cache")
kv_cache = None
if cache_mod is not None:
    StaticKVCache = getattr(cache_mod, "StaticKVCache", None)
    if StaticKVCache:
        try:
            kv_cache = StaticKVCache(transformer.config,
                                     max_frames=64,
                                     device="cuda",
                                     dtype=torch.bfloat16)
            print("KV cache created (StaticKVCache)")
        except Exception as e:
            print(f"StaticKVCache init failed: {e}, using DummyKVCache")
            kv_cache = DummyKVCache()
    else:
        kv_cache = DummyKVCache()
        print("No StaticKVCache found, using DummyKVCache (single-frame only)")
else:
    kv_cache = DummyKVCache()
    print("Cache module not loaded, using DummyKVCache")

# ── Run inference ──
# Use a fixed seed for reproducibility when comparing with run_fastvideo_single_frame.py
SEED = 42
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

prompt = "first person view of a grassy field with blue sky"
inputs = tokenizer(prompt,
                   padding="max_length",
                   max_length=512,
                   truncation=True,
                   return_tensors="pt")
input_ids = inputs.input_ids.to("cuda")
attention_mask = inputs.attention_mask.to("cuda")

with torch.no_grad():
    enc_out = text_encoder(input_ids=input_ids, attention_mask=attention_mask)
    prompt_emb = enc_out.last_hidden_state  # [1, 512, 2048]
    prompt_emb = prompt_emb * attention_mask.unsqueeze(-1).float()
    prompt_pad_mask = attention_mask.eq(0)

print(f"Prompt emb: shape={list(prompt_emb.shape)} dtype={prompt_emb.dtype}")

# Sigma schedule (from config)
sigmas = torch.tensor(t_config.get("scheduler_sigmas",
                                   [1.0, 0.9483, 0.8380, 0.0]),
                      device="cuda",
                      dtype=torch.bfloat16)
print(f"Sigmas: {sigmas.tolist()}")

# Initial noise
x = torch.randn(1, 1, 16, 32, 32, device="cuda", dtype=torch.bfloat16)
frame_ts = torch.zeros(1, 1, device="cuda", dtype=torch.long)
mouse = torch.zeros(1, 1, 2, device="cuda", dtype=torch.bfloat16)
button = torch.zeros(1, 1, 256, device="cuda", dtype=torch.bfloat16)
scroll = torch.zeros(1, 1, 1, device="cuda", dtype=torch.bfloat16)

sigma = x.new_empty((1, 1))

# Denoise
with torch.no_grad():
    if kv_cache is not None:
        kv_cache.set_frozen(True)

    for step_sig, step_dsig in zip(sigmas, sigmas.diff(), strict=False):
        v = transformer(
            x=x,
            sigma=sigma.fill_(step_sig),
            frame_timestamp=frame_ts,
            prompt_emb=prompt_emb.to(torch.bfloat16),
            prompt_pad_mask=prompt_pad_mask,
            mouse=mouse,
            button=button,
            scroll=scroll,
            kv_cache=kv_cache,
        )
        x = x + step_dsig * v
        xf = x.float()
        print(
            f"Step sigma={step_sig.item():.4f}: v mean={v.float().mean():.4f} std={v.float().std():.4f}  "
            f"x mean={xf.mean():.4f} std={xf.std():.4f}")

    print(
        f"Denoised: mean={xf.mean():.4f} std={xf.std():.4f} min={xf.min():.4f} max={xf.max():.4f}"
    )

    # Decode
    latent = x.squeeze(1).float()  # [1, 16, 32, 32]
    print(f"VAE input: shape={list(latent.shape)} dtype={latent.dtype}")
    decoded = vae.decode(latent)
    print(
        f"VAE output: type={type(decoded).__name__} shape={list(decoded.shape)} dtype={decoded.dtype}"
    )

    if decoded.dtype == torch.uint8:
        print(
            f"Pixel stats: mean={decoded.float().mean():.1f} min={decoded.min()} max={decoded.max()}"
        )
        # Per channel
        if decoded.dim() == 3 and decoded.shape[-1] == 3:
            for c, name in enumerate(["R", "G", "B"]):
                ch = decoded[..., c].float()
                print(f"  {name}: mean={ch.mean():.1f}")

    # Save as image
    import numpy as np
    from PIL import Image
    if decoded.dim() == 3 and decoded.shape[-1] == 3:
        img = decoded.cpu().numpy()
    elif decoded.dim() == 4:
        img = decoded[0].cpu().numpy()
        if img.shape[0] == 3:
            img = img.transpose(1, 2, 0)
    img = np.clip(img, 0, 255).astype(np.uint8)
    Image.fromarray(img).save("waypoint_official_frame.png")
    print(f"Saved waypoint_official_frame.png ({img.shape[1]}x{img.shape[0]})")
