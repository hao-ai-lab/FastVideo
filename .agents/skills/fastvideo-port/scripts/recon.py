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
  "architecture": {
    "class_name": "ModelClass",
    "num_layers": N,
    "hidden_dim": D,
    "num_heads": H,
    "attention_type": "self_attn | cross_attn | unified",
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
import re
import sys
from urllib.request import urlopen, Request
from urllib.error import HTTPError

GITHUB_API = "https://api.github.com"

# Patterns that suggest model definition files
MODEL_FILE_PATTERNS = [
    r"model[s]?\.py$",
    r"dit\.py$",
    r"transformer\.py$",
    r"unet\.py$",
    r"diffusion_model\.py$",
    r"network\.py$",
    r"architecture\.py$",
]

# Patterns to extract architecture info from Python source
LAYER_PATTERNS = [
    (r"num_layers\s*[=:]\s*(\d+)", "num_layers"),
    (r"num_hidden_layers\s*[=:]\s*(\d+)", "num_layers"),
    (r"depth\s*[=:]\s*(\d+)", "num_layers"),
    (r"hidden_(?:size|dim)\s*[=:]\s*(\d+)", "hidden_dim"),
    (r"d_model\s*[=:]\s*(\d+)", "hidden_dim"),
    (r"embed_dim\s*[=:]\s*(\d+)", "hidden_dim"),
    (r"num_(?:attention_)?heads\s*[=:]\s*(\d+)", "num_heads"),
    (r"in_channels\s*[=:]\s*(\d+)", "input_channels"),
    (r"patch_size\s*[=:]\s*(\d+)", "patch_size"),
]

TEXT_ENCODER_HINTS = {
    "t5": "google/t5-v1_1-xxl",
    "t5gemma": "google/t5gemma-9b",
    "clip": "openai/clip-vit-large-patch14",
    "qwen2": "Qwen/Qwen2.5-VL-7B-Instruct",
    "llama": "meta-llama/Llama-3.1-8B",
    "gemma": "google/gemma-2-9b",
}

VAE_HINTS = {
    "wan": "Wan-AI/Wan2.1-T2V-1.3B-Diffusers",
    "ae": None,
    "vae": None,
    "causal_vae": None,
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
            # Try the repo's actual default branch (may be "master" or something else)
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
    """Extract 'owner/repo' from a GitHub URL."""
    m = re.search(r"github\.com[:/]([^/\s?#]+/[^/\s?#]+)", url)
    if not m:
        raise ValueError(f"Cannot parse GitHub URL: {url}")
    return m.group(1).removesuffix(".git").rstrip("/")


def list_python_files(repo: str, token: str | None = None, max_files: int = 200) -> list[str]:
    """Return all .py file paths in the repo (up to max_files)."""
    try:
        tree = gh_get(f"repos/{repo}/git/trees/HEAD?recursive=1", token)
        files = [
            item["path"] for item in tree.get("tree", [])
            if item["type"] == "blob" and item["path"].endswith(".py")
        ]
        return files[:max_files]
    except Exception as e:
        print(f"[warn] Could not list files: {e}", file=sys.stderr)
        return []


def find_model_files(all_files: list[str]) -> list[str]:
    """Return files likely to contain model definitions."""
    hits = []
    for f in all_files:
        fname = f.split("/")[-1].lower()
        if any(re.search(p, fname) for p in MODEL_FILE_PATTERNS):
            hits.append(f)
    # Also grab any file with "model" in the path under common dirs
    for f in all_files:
        if "/models/" in f or "/model/" in f:
            if f not in hits:
                hits.append(f)
    return hits[:10]  # cap at 10 to avoid huge context


def extract_architecture(source: str) -> dict:
    """Pull numeric architecture params out of Python source."""
    result = {}
    for pattern, key in LAYER_PATTERNS:
        if key not in result:
            m = re.search(pattern, source)
            if m:
                result[key] = int(m.group(1))

    # Attention type heuristic
    source_lc = source.lower()
    if "cross_attn" in source_lc or "cross_attention" in source_lc:
        if "self_attn" in source_lc or "self_attention" in source_lc:
            result["attention_type"] = "self+cross"
        else:
            result["attention_type"] = "cross_attn"
    elif "self_attn" in source_lc or "selfattention" in source_lc:
        result["attention_type"] = "self_attn"

    # Conditioning
    cond = []
    if re.search(r"text_embed|encoder_hidden|text_enc", source):
        cond.append("text")
    if re.search(r"image_embed|image_cond|reference_image", source):
        cond.append("image")
    if re.search(r"audio_embed|audio_cond|audio_enc", source):
        cond.append("audio")
    if cond:
        result["conditioning"] = cond

    # Top-level class name
    m = re.search(r"^class\s+(\w+)\(.*(?:Module|Model|Transformer|DiT)", source, re.MULTILINE)
    if m:
        result["class_name"] = m.group(1)

    return result


def detect_components(readme: str, all_source: str) -> dict:
    """Detect text encoders, VAE, and scheduler from README + source."""
    text_encoders = []
    lower = (readme + all_source).lower()

    for key, hf_id in TEXT_ENCODER_HINTS.items():
        if key in lower:
            text_encoders.append({"name": key, "hf_id": hf_id})

    vae = None
    for key in VAE_HINTS:
        if key in lower:
            vae = {"name": key, "hf_id": VAE_HINTS[key]}
            # Try to find latent channels
            m = re.search(r"latent_channels\s*[=:]\s*(\d+)", all_source)
            if m:
                vae["latent_channels"] = int(m.group(1))
            break

    scheduler = {}
    if "flow" in lower or "flow_match" in lower:
        scheduler["type"] = "flow_matching"
    elif "ddim" in lower:
        scheduler["type"] = "ddim"
    elif "ddpm" in lower:
        scheduler["type"] = "ddpm"
    m = re.search(r"shift\s*[=:]\s*([\d.]+)", all_source)
    if m:
        scheduler["shift"] = float(m.group(1))

    return {
        "text_encoders": text_encoders,
        "vae": vae,
        "scheduler": scheduler or None,
    }


def detect_hf_repo(readme: str, repo: str) -> tuple[str | None, bool]:
    """Try to find the HuggingFace repo ID and whether it's Diffusers format."""
    m = re.search(r"huggingface\.co/([A-Za-z0-9_\-]+/[A-Za-z0-9_\-\.]+)", readme)
    hf_id = m.group(1) if m else None
    diffusers = bool(re.search(r"diffusers|from_pretrained|DiffusionPipeline", readme))
    return hf_id, diffusers


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
    return tasks or ["t2v"]  # default assumption


def main():
    parser = argparse.ArgumentParser(description="Recon a GitHub repo for FastVideo porting")
    parser.add_argument("repo_url", help="GitHub URL, e.g. https://github.com/org/repo")
    parser.add_argument("--output", "-o", default=None, help="JSON output file (default: stdout)")
    parser.add_argument("--token", default=None, help="GitHub personal access token (optional)")
    args = parser.parse_args()

    repo = parse_repo_url(args.repo_url)
    print(f"[recon] Analyzing {repo} ...", file=sys.stderr)

    # 1. Fetch README
    readme = gh_raw(repo, "README.md") or gh_raw(repo, "readme.md") or ""
    print(f"[recon] README: {len(readme)} chars", file=sys.stderr)

    # 2. List Python files and find model files
    all_files = list_python_files(repo, args.token)
    print(f"[recon] Found {len(all_files)} Python files", file=sys.stderr)
    model_files = find_model_files(all_files)
    print(f"[recon] Model files: {model_files}", file=sys.stderr)

    # 3. Fetch and analyze model source
    arch = {}
    all_model_source = ""
    for mf in model_files:
        src = gh_raw(repo, mf)
        if src:
            all_model_source += src
            extracted = extract_architecture(src)
            # Take first non-None value for each key
            for k, v in extracted.items():
                if k not in arch:
                    arch[k] = v

    # 4. Detect components
    components = detect_components(readme, all_model_source)

    # 5. HF repo + tasks
    hf_repo, diffusers_fmt = detect_hf_repo(readme, repo)
    tasks = detect_tasks(readme)

    result = {
        "repo": repo,
        "model_files": model_files,
        "architecture": arch,
        "components": components,
        "hf_repo": hf_repo,
        "diffusers_format": diffusers_fmt,
        "tasks": tasks,
        "notes": [
            "Auto-generated by recon.py — verify all fields manually.",
            "num_layers/hidden_dim extracted via regex; may miss dynamic configs.",
            f"README length: {len(readme)} chars, model source: {len(all_model_source)} chars scanned.",
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
