#!/usr/bin/env python3
"""Reconnaissance script for FastVideo model ports.

Fetches a GitHub repo (no clone needed) and extracts a structured JSON
summary of the model architecture to guide porting decisions.

Usage:
    python recon.py https://github.com/GAIR-NLP/daVinci-MagiHuman
    python recon.py https://github.com/NVIDIA/Cosmos --output recon_cosmos.json

Output JSON shape:
{
  "repo": "owner/name",
  "model_files": ["path/to/model.py", ...],
  "config_files": ["path/to/config.py", ...],
  "architecture": {
    "class_name": "ModelClass",
    "num_layers": N,
    "hidden_dim": D,
    "num_heads": H,
    "attention_type": "self_attn | cross_attn | self+cross",
    "conditioning": ["text", "image", "audio", ...],
    "input_channels": N,
    "patch_size": N
  },
  "components": {
    "text_encoders": [{"name": "...", "hf_id": "..."}],
    "vae": {"name": "...", "hf_id": "...", "latent_channels": N},
    "scheduler": {"type": "...", "shift": N}
  },
  "hf_repo": "org/model-name or null",
  "diffusers_format": true | false,
  "tasks": ["t2v", "i2v", ...],
  "notes": ["...free-form observations..."]
}
"""
import argparse
import json
import os
import re
import sys
from urllib.request import urlopen, Request

GITHUB_API = "https://api.github.com"

MODEL_FILE_PATTERNS = [
    r"model[s]?\.py$",
    r"dit\.py$",
    r"transformer\.py$",
    r"unet\.py$",
    r"diffusion_model\.py$",
    r"network\.py$",
    r"architecture\.py$",
]

CONFIG_FILE_PATTERNS = [
    r"config\.py$",
    r"configs?\.py$",
    r"configuration\.py$",
    r"args\.py$",
    r"params\.py$",
    r"settings\.py$",
    r"config\.json$",
]

# Ordered from most to least specific — first match wins per key
ARCH_PATTERNS: list[tuple[str, str]] = [
    # Pydantic Field(default=N, ...) — most reliable source
    (r"num_layers\s*:\s*int\s*=\s*Field\s*\(\s*default\s*=\s*(\d+)", "num_layers"),
    (r"num_hidden_layers\s*:\s*int\s*=\s*Field\s*\(\s*default\s*=\s*(\d+)", "num_layers"),
    (r"depth\s*:\s*int\s*=\s*Field\s*\(\s*default\s*=\s*(\d+)", "num_layers"),
    (r"hidden_size\s*:\s*int\s*=\s*Field\s*\(\s*default\s*=\s*(\d+)", "hidden_dim"),
    (r"hidden_dim\s*:\s*int\s*=\s*Field\s*\(\s*default\s*=\s*(\d+)", "hidden_dim"),
    (r"d_model\s*:\s*int\s*=\s*Field\s*\(\s*default\s*=\s*(\d+)", "hidden_dim"),
    (r"embed_dim\s*:\s*int\s*=\s*Field\s*\(\s*default\s*=\s*(\d+)", "hidden_dim"),
    (r"num_attention_heads\s*:\s*int\s*=\s*Field\s*\(\s*default\s*=\s*(\d+)", "num_heads"),
    (r"num_heads\s*:\s*int\s*=\s*Field\s*\(\s*default\s*=\s*(\d+)", "num_heads"),
    (r"head_dim\s*:\s*int\s*=\s*Field\s*\(\s*default\s*=\s*(\d+)", "head_dim"),
    (r"num_query_groups\s*:\s*int\s*=\s*Field\s*\(\s*default\s*=\s*(\d+)", "num_query_groups"),
    (r"patch_size\s*:\s*int\s*=\s*Field\s*\(\s*default\s*=\s*(\d+)", "patch_size"),
    (r"in_channels\s*:\s*int\s*=\s*Field\s*\(\s*default\s*=\s*(\d+)", "input_channels"),
    # Dataclass / plain assignment fallbacks
    (r"num_layers\s*[=:]\s*(\d+)", "num_layers"),
    (r"num_hidden_layers\s*[=:]\s*(\d+)", "num_layers"),
    (r"depth\s*[=:]\s*(\d+)", "num_layers"),
    (r"hidden_size\s*[=:]\s*(\d+)", "hidden_dim"),
    (r"hidden_dim\s*[=:]\s*(\d+)", "hidden_dim"),
    (r"d_model\s*[=:]\s*(\d+)", "hidden_dim"),
    (r"embed_dim\s*[=:]\s*(\d+)", "hidden_dim"),
    (r"num_attention_heads\s*[=:]\s*(\d+)", "num_heads"),
    (r"num_heads_q\s*[=:]\s*(\d+)", "num_heads"),
    (r"in_channels\s*[=:]\s*(\d+)", "input_channels"),
]

# Scheduler shift: look for Field(default=N) form first
SCHEDULER_SHIFT_PATTERNS = [
    r"shift\s*:\s*float\s*=\s*Field\s*\(\s*default\s*=\s*([\d.]+)",
    r"flow_shift\s*[=:]\s*([\d.]+)",
    r"\bshift\s*=\s*([\d.]+)",
]

# Text encoder: ordered most-specific to least; first key that appears wins
TEXT_ENCODER_HINTS: list[tuple[str, str]] = [
    ("t5gemma", "google/t5gemma-9b"),
    ("t5_gemma", "google/t5gemma-9b"),
    ("t5-gemma", "google/t5gemma-9b"),
    ("qwen2_5_vl", "Qwen/Qwen2.5-VL-7B-Instruct"),
    ("qwen2_vl", "Qwen/Qwen2-VL-7B-Instruct"),
    ("qwen2", "Qwen/Qwen2.5-VL-7B-Instruct"),
    ("llama", "meta-llama/Llama-3.1-8B"),
    ("gemma", "google/gemma-2-9b"),
    ("clip", "openai/clip-vit-large-patch14"),
    ("t5", "google/t5-v1_1-xxl"),
]

VAE_HINTS = {
    "wan": "Wan-AI/Wan2.1-T2V-1.3B-Diffusers",
    "causal_vae": None,
    "ae": None,
    "vae": None,
}


def gh_get(path: str, token: str | None = None) -> dict | list:
    url = f"{GITHUB_API}/{path}"
    headers = {"Accept": "application/vnd.github.v3+json"}
    if token:
        headers["Authorization"] = f"token {token}"
    req = Request(url, headers=headers)
    with urlopen(req, timeout=15) as r:
        return json.loads(r.read())


def get_default_branch(repo: str, token: str | None = None) -> str:
    try:
        info = gh_get(f"repos/{repo}", token)
        return info.get("default_branch", "main")
    except Exception:
        return "main"


def gh_raw(repo: str, path: str, ref: str = "main", token: str | None = None) -> str | None:
    url = f"https://raw.githubusercontent.com/{repo}/{ref}/{path}"
    try:
        with urlopen(url, timeout=15) as r:
            return r.read().decode("utf-8", errors="ignore")
    except Exception:
        if ref == "main":
            try:
                default = get_default_branch(repo, token)
                if default != ref:
                    url2 = f"https://raw.githubusercontent.com/{repo}/{default}/{path}"
                    with urlopen(url2, timeout=10) as r:
                        return r.read().decode("utf-8", errors="ignore")
            except Exception:
                pass
        return None


def parse_repo_url(url: str) -> str:
    m = re.search(r"github\.com[:/]([^/\s?#]+/[^/\s?#]+)", url)
    if not m:
        raise ValueError(f"Cannot parse GitHub URL: {url}")
    return m.group(1).removesuffix(".git").rstrip("/")


def list_all_files(repo: str, token: str | None = None, max_files: int = 400) -> list[str]:
    try:
        tree = gh_get(f"repos/{repo}/git/trees/HEAD?recursive=1", token)
        return [item["path"] for item in tree.get("tree", []) if item["type"] == "blob"][:max_files]
    except Exception as e:
        print(f"[warn] Could not list files: {e}", file=sys.stderr)
        return []


def find_model_files(all_files: list[str]) -> list[str]:
    hits = []
    for f in all_files:
        fname = f.split("/")[-1].lower()
        if any(re.search(p, fname) for p in MODEL_FILE_PATTERNS):
            hits.append(f)
    for f in all_files:
        if ("/models/" in f or "/model/" in f) and f.endswith(".py"):
            if f not in hits:
                hits.append(f)
    return hits[:10]


def find_config_files(all_files: list[str]) -> list[str]:
    """Find Python config files and JSON example configs."""
    hits = []
    for f in all_files:
        fname = f.split("/")[-1].lower()
        if any(re.search(p, fname) for p in CONFIG_FILE_PATTERNS):
            hits.append(f)
    # Also grab any JSON under example/ or examples/ directories
    for f in all_files:
        if re.search(r"^example[s]?/.*\.json$", f):
            if f not in hits:
                hits.append(f)
    return hits[:10]


def extract_architecture(source: str, from_config: bool = False) -> dict:
    """Extract architecture params. Config files get priority (from_config=True)."""
    result: dict = {}
    for pattern, key in ARCH_PATTERNS:
        if key not in result:
            m = re.search(pattern, source)
            if m:
                result[key] = int(m.group(1))

    # Attention type heuristic (only from model source, not config)
    if not from_config:
        source_lc = source.lower()
        if "cross_attn" in source_lc or "cross_attention" in source_lc:
            if "self_attn" in source_lc or "self_attention" in source_lc:
                result["attention_type"] = "self+cross"
            else:
                result["attention_type"] = "cross_attn"
        elif "self_attn" in source_lc or "selfattention" in source_lc:
            result["attention_type"] = "self_attn"

        # Conditioning signals
        cond = []
        if re.search(r"text_embed|encoder_hidden|text_enc|text_in_channels", source):
            cond.append("text")
        if re.search(r"image_embed|image_cond|reference_image", source):
            cond.append("image")
        if re.search(r"audio_embed|audio_cond|audio_enc|audio_in_channels", source):
            cond.append("audio")
        if cond:
            result["conditioning"] = cond

        # Top-level DiT class: prefer names that suggest a top-level model,
        # skip known sub-module helpers (Attention, MLP, Norm, Embed, Layer, Block)
        HELPER_WORDS = {"attention", "mlp", "ffn", "norm", "embed", "layer",
                        "block", "head", "proj", "router", "dispatcher",
                        "adapter", "linear", "function", "config", "enum"}
        preferred: str | None = None
        fallback: str | None = None
        for m in re.finditer(r"^class\s+(\w+)\([^)]*(?:(?:nn\.|torch\.nn\.)?Module|Model|Transformer|DiT)[^)]*\)", source, re.MULTILINE):
            cls_name = m.group(1)
            cls_lower = cls_name.lower()
            if any(w in cls_lower for w in HELPER_WORDS):
                continue
            cls_idx = m.start()
            snippet = source[cls_idx:cls_idx + 4000]
            if not re.search(r"def forward\s*\(\s*self\s*,\s*\w+\s*:", snippet):
                continue
            # Strongly prefer names that suggest a top-level model
            if re.search(r"(?:DiT|Model|Pipeline|Net)$", cls_name):
                preferred = cls_name
                break
            if fallback is None:
                fallback = cls_name
        top_cls = preferred or fallback
        if top_cls:
            result["class_name"] = top_cls

    return result


def extract_arch_from_json_config(source: str) -> dict:
    """Extract architecture fields from JSON example configs."""
    result: dict = {}
    try:
        data = json.loads(source)
    except Exception:
        return result

    def walk(obj: object) -> None:
        if isinstance(obj, dict):
            for k, v in obj.items():
                if isinstance(v, int):
                    if k in ("num_layers", "depth", "num_hidden_layers"):
                        result.setdefault("num_layers", v)
                    elif k in ("hidden_size", "hidden_dim", "d_model", "embed_dim"):
                        result.setdefault("hidden_dim", v)
                    elif k in ("num_heads", "num_attention_heads", "num_heads_q"):
                        result.setdefault("num_heads", v)
                    elif k == "head_dim":
                        result.setdefault("head_dim", v)
                    elif k == "num_query_groups":
                        result.setdefault("num_query_groups", v)
                elif isinstance(v, (dict, list)):
                    walk(v)
        elif isinstance(obj, list):
            for item in obj:
                walk(item)

    walk(data)
    return result


def detect_text_encoder(all_source: str, readme: str) -> list[dict]:
    """Return at most one text encoder entry — most specific match wins."""
    combined = all_source + readme
    combined_lc = combined.lower()
    for key, hf_id in TEXT_ENCODER_HINTS:
        if key.lower() in combined_lc:
            return [{"name": key, "hf_id": hf_id}]
    return []


def detect_components(readme: str, all_source: str, config_source: str) -> dict:
    text_encoders = detect_text_encoder(all_source + config_source, readme)

    vae = None
    combined_lc = (readme + all_source + config_source).lower()
    for key in VAE_HINTS:
        if key in combined_lc:
            vae = {"name": key, "hf_id": VAE_HINTS[key]}
            m = re.search(r"latent_channels\s*[=:]\s*(\d+)", all_source + config_source)
            if m:
                vae["latent_channels"] = int(m.group(1))
            break

    # Scheduler: prefer flow_matching (check config source first)
    scheduler: dict = {}
    all_text = config_source + all_source + readme
    all_lc = all_text.lower()

    shift_val = None
    for pat in SCHEDULER_SHIFT_PATTERNS:
        m = re.search(pat, all_text)
        if m:
            shift_val = float(m.group(1))
            break

    if "flow_match" in all_lc or "flow match" in all_lc or "flowmatch" in all_lc:
        scheduler["type"] = "flow_matching"
    elif shift_val is not None and shift_val > 0:
        # A non-zero shift parameter is a strong signal for flow matching
        scheduler["type"] = "flow_matching"
    elif "ddim" in all_lc:
        scheduler["type"] = "ddim"
    elif "ddpm" in all_lc:
        scheduler["type"] = "ddpm"

    if shift_val is not None:
        scheduler["shift"] = shift_val

    return {
        "text_encoders": text_encoders,
        "vae": vae,
        "scheduler": scheduler or None,
    }


def detect_hf_repo(readme: str) -> tuple[str | None, bool]:
    # Exclude HuggingFace Spaces — those are demos, not weight repos
    for m in re.finditer(r"huggingface\.co/([A-Za-z0-9_\-]+/[A-Za-z0-9_\-\.]+)", readme):
        candidate = m.group(1)
        if not candidate.startswith("spaces/"):
            return candidate, bool(re.search(r"diffusers|from_pretrained|DiffusionPipeline", readme))
    return None, bool(re.search(r"diffusers|from_pretrained|DiffusionPipeline", readme))


def detect_tasks(readme: str) -> list[str]:
    tasks = []
    lower = readme.lower()
    if "text-to-video" in lower or "t2v" in lower:
        tasks.append("t2v")
    if "image-to-video" in lower or "i2v" in lower:
        tasks.append("i2v")
    if "video-to-video" in lower or "v2v" in lower:
        tasks.append("v2v")
    if "text-to-world" in lower or "t2w" in lower:
        tasks.append("t2w")
    if "audio" in lower:
        tasks.append("audio-driven")
    return tasks or ["t2v"]


def main():
    parser = argparse.ArgumentParser(description="Recon a GitHub repo for FastVideo porting")
    parser.add_argument("repo_url", help="GitHub URL, e.g. https://github.com/org/repo")
    parser.add_argument("--output", "-o", default=None, help="JSON output file (default: stdout)")
    parser.add_argument("--token", default=None,
                        help="GitHub personal access token (overrides GITHUB_TOKEN env var)")
    args = parser.parse_args()

    token = args.token or os.environ.get("GITHUB_TOKEN")
    repo = parse_repo_url(args.repo_url)
    print(f"[recon] Analyzing {repo} ...", file=sys.stderr)

    readme = gh_raw(repo, "README.md", token=token) or gh_raw(repo, "readme.md", token=token) or ""
    print(f"[recon] README: {len(readme)} chars", file=sys.stderr)

    all_files = list_all_files(repo, token)
    py_files = [f for f in all_files if f.endswith(".py")]
    print(f"[recon] Found {len(py_files)} Python files, {len(all_files)} total", file=sys.stderr)

    model_files = find_model_files(all_files)
    config_files = find_config_files(all_files)
    print(f"[recon] Model files: {model_files}", file=sys.stderr)
    print(f"[recon] Config files: {config_files}", file=sys.stderr)

    # Fetch and extract from config files first (higher priority for numeric values)
    arch: dict = {}
    all_config_source = ""
    for cf in config_files:
        src = gh_raw(repo, cf, token=token)
        if not src:
            continue
        all_config_source += src
        if cf.endswith(".json"):
            extracted = extract_arch_from_json_config(src)
        else:
            extracted = extract_architecture(src, from_config=True)
        for k, v in extracted.items():
            if k not in arch:
                arch[k] = v

    # Fetch model source (lower priority for numeric values, primary for attention/class/conditioning)
    all_model_source = ""
    for mf in model_files:
        src = gh_raw(repo, mf, token=token)
        if not src:
            continue
        all_model_source += src
        extracted = extract_architecture(src, from_config=False)
        for k, v in extracted.items():
            if k not in arch:
                arch[k] = v

    components = detect_components(readme, all_model_source, all_config_source)
    hf_repo, diffusers_fmt = detect_hf_repo(readme)
    tasks = detect_tasks(readme)

    result = {
        "repo": repo,
        "model_files": model_files,
        "config_files": config_files,
        "architecture": arch,
        "components": components,
        "hf_repo": hf_repo,
        "diffusers_format": diffusers_fmt,
        "tasks": tasks,
        "notes": [
            "Auto-generated by recon.py — verify all fields manually.",
            "Numeric fields extracted from config files first (Pydantic Field defaults), then model source.",
            f"README: {len(readme)} chars, model source: {len(all_model_source)} chars, config source: {len(all_config_source)} chars.",
        ]
    }

    output = json.dumps(result, indent=2)
    if args.output:
        with open(args.output, "w") as f:
            f.write(output)
        print(f"[recon] Written to {args.output}", file=sys.stderr)
    else:
        print(output)


if __name__ == "__main__":
    main()