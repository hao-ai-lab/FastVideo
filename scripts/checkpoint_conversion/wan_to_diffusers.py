# pyright: reportMissingImports=false
from safetensors.torch import save_file, load_file as safe_load_file
import argparse
import os
import re
import sys
from collections import OrderedDict
from typing import Any, Dict, Mapping, Tuple
import torch

try:
    from huggingface_hub import save_torch_state_dict, load_state_dict_from_file  # type: ignore
except Exception:
    save_torch_state_dict = None  # type: ignore[assignment]
    load_state_dict_from_file = None  # type: ignore[assignment]

_param_names_mapping: dict = {
    r"^text_embedding\.0\.(.*)$":
    r"condition_embedder.text_embedder.linear_1.\1",
    r"^text_embedding\.2\.(.*)$":
    r"condition_embedder.text_embedder.linear_2.\1",
    r"^time_embedding\.0\.(.*)$":
    r"condition_embedder.time_embedder.linear_1.\1",
    r"^time_embedding\.2\.(.*)$":
    r"condition_embedder.time_embedder.linear_2.\1",
    r"^time_projection\.1\.(.*)$":
    r"condition_embedder.time_proj.\1",
    r"^img_emb\.proj\.0\.(.*)$":
    r"condition_embedder.image_embedder.norm1.\1",
    r"^img_emb\.proj\.1\.(.*)$":
    r"condition_embedder.image_embedder.ff.net.0.proj.\1",
    r"^img_emb\.proj\.3\.(.*)$":
    r"condition_embedder.image_embedder.ff.net.2.\1",
    r"^img_emb\.proj\.4\.(.*)$":
    r"condition_embedder.image_embedder.norm2.\1",
    r"^head\.modulation":
    r"scale_shift_table",
    r"^head\.head\.(.*)$":
    r"proj_out.\1",
    r"^blocks\.(\d+)\.self_attn\.q\.(.*)$":
    r"blocks.\1.attn1.to_q.\2",
    r"^blocks\.(\d+)\.self_attn\.k\.(.*)$":
    r"blocks.\1.attn1.to_k.\2",
    r"^blocks\.(\d+)\.self_attn\.v\.(.*)$":
    r"blocks.\1.attn1.to_v.\2",
    r"^blocks\.(\d+)\.self_attn\.o\.(.*)$":
    r"blocks.\1.attn1.to_out.0.\2",
    r"^blocks\.(\d+)\.self_attn\.norm_q\.(.*)$":
    r"blocks.\1.attn1.norm_q.\2",
    r"^blocks\.(\d+)\.self_attn\.norm_k\.(.*)$":
    r"blocks.\1.attn1.norm_k.\2",
    r"^blocks\.(\d+)\.cross_attn\.q\.(.*)$":
    r"blocks.\1.attn2.to_q.\2",
    r"^blocks\.(\d+)\.cross_attn\.k\.(.*)$":
    r"blocks.\1.attn2.to_k.\2",
    r"^blocks\.(\d+)\.cross_attn\.k_img\.(.*)$":
    r"blocks.\1.attn2.add_k_proj.\2",
    r"^blocks\.(\d+)\.cross_attn\.v\.(.*)$":
    r"blocks.\1.attn2.to_v.\2",
    r"^blocks\.(\d+)\.cross_attn\.v_img\.(.*)$":
    r"blocks.\1.attn2.add_v_proj.\2",
    r"^blocks\.(\d+)\.cross_attn\.o\.(.*)$":
    r"blocks.\1.attn2.to_out.0.\2",
    r"^blocks\.(\d+)\.cross_attn\.norm_q\.(.*)$":
    r"blocks.\1.attn2.norm_q.\2",
    r"^blocks\.(\d+)\.cross_attn\.norm_k\.(.*)$":
    r"blocks.\1.attn2.norm_k.\2",
    r"^blocks\.(\d+)\.cross_attn\.norm_k_img\.(.*)$":
    r"blocks.\1.attn2.norm_added_k.\2",
    r"^blocks\.(\d+)\.ffn\.0\.(.*)$":
    r"blocks.\1.ffn.net.0.proj.\2",
    r"^blocks\.(\d+)\.ffn\.2\.(.*)$":
    r"blocks.\1.ffn.net.2.\2",
    r"^blocks\.(\d+)\.modulation":
    r"blocks.\1.scale_shift_table",
    r"^blocks\.(\d+)\.norm3\.(.*)$":
    r"blocks.\1.norm2.\2",
}

# The following mapping has an extra 'patch_embedding' field and also contains
# the 'model' prefixes
_self_forcing_to_diffusers_param_names_mapping: dict = {
    r"^model.patch_embedding\.(.*)$":
    r"patch_embedding.\1",
    r"^model.text_embedding\.0\.(.*)$":
    r"condition_embedder.text_embedder.linear_1.\1",
    r"^model.text_embedding\.2\.(.*)$":
    r"condition_embedder.text_embedder.linear_2.\1",
    r"^model.time_embedding\.0\.(.*)$":
    r"condition_embedder.time_embedder.linear_1.\1",
    r"^model.time_embedding\.2\.(.*)$":
    r"condition_embedder.time_embedder.linear_2.\1",
    r"^model.time_projection\.1\.(.*)$":
    r"condition_embedder.time_proj.\1",
    r"^model.img_emb\.proj\.0\.(.*)$":
    r"condition_embedder.image_embedder.norm1.\1",
    r"^model.img_emb\.proj\.1\.(.*)$":
    r"condition_embedder.image_embedder.ff.net.0.proj.\1",
    r"^model.img_emb\.proj\.3\.(.*)$":
    r"condition_embedder.image_embedder.ff.net.2.\1",
    r"^model.img_emb\.proj\.4\.(.*)$":
    r"condition_embedder.image_embedder.norm2.\1",
    r"^model.head\.modulation":
    r"scale_shift_table",
    r"^model.head\.head\.(.*)$":
    r"proj_out.\1",
    r"^model.blocks\.(\d+)\.self_attn\.q\.(.*)$":
    r"blocks.\1.attn1.to_q.\2",
    r"^model.blocks\.(\d+)\.self_attn\.k\.(.*)$":
    r"blocks.\1.attn1.to_k.\2",
    r"^model.blocks\.(\d+)\.self_attn\.v\.(.*)$":
    r"blocks.\1.attn1.to_v.\2",
    r"^model.blocks\.(\d+)\.self_attn\.o\.(.*)$":
    r"blocks.\1.attn1.to_out.0.\2",
    r"^model.blocks\.(\d+)\.self_attn\.norm_q\.(.*)$":
    r"blocks.\1.attn1.norm_q.\2",
    r"^model.blocks\.(\d+)\.self_attn\.norm_k\.(.*)$":
    r"blocks.\1.attn1.norm_k.\2",
    r"^model.blocks\.(\d+)\.cross_attn\.q\.(.*)$":
    r"blocks.\1.attn2.to_q.\2",
    r"^model.blocks\.(\d+)\.cross_attn\.k\.(.*)$":
    r"blocks.\1.attn2.to_k.\2",
    r"^model.blocks\.(\d+)\.cross_attn\.k_img\.(.*)$":
    r"blocks.\1.attn2.add_k_proj.\2",
    r"^model.blocks\.(\d+)\.cross_attn\.v\.(.*)$":
    r"blocks.\1.attn2.to_v.\2",
    r"^model.blocks\.(\d+)\.cross_attn\.v_img\.(.*)$":
    r"blocks.\1.attn2.add_v_proj.\2",
    r"^model.blocks\.(\d+)\.cross_attn\.o\.(.*)$":
    r"blocks.\1.attn2.to_out.0.\2",
    r"^model.blocks\.(\d+)\.cross_attn\.norm_q\.(.*)$":
    r"blocks.\1.attn2.norm_q.\2",
    r"^model.blocks\.(\d+)\.cross_attn\.norm_k\.(.*)$":
    r"blocks.\1.attn2.norm_k.\2",
    r"^model.blocks\.(\d+)\.cross_attn\.norm_k_img\.(.*)$":
    r"blocks.\1.attn2.norm_added_k.\2",
    r"^model.blocks\.(\d+)\.ffn\.0\.(.*)$":
    r"blocks.\1.ffn.net.0.proj.\2",
    r"^model.blocks\.(\d+)\.ffn\.2\.(.*)$":
    r"blocks.\1.ffn.net.2.\2",
    r"^model.blocks\.(\d+)\.modulation":
    r"blocks.\1.scale_shift_table",
    r"^model.blocks\.(\d+)\.norm3\.(.*)$":
    r"blocks.\1.norm2.\2",
}

def _replacement_to_regex_template(replacement: str) -> Tuple[str, int]:
    r"""
    Convert a replacement template like "blocks.\1.attn2.to_q.\2" into a regex pattern
    that can be used to match in the reverse direction: "^blocks\.(.*)\.attn2\.to_q\.(.*)$".

    Returns the regex template and the number of capture groups.
    """
    # First, protect placeholders \1..\9
    placeholder_tokens: Dict[str, str] = {}
    group_count = 0
    def _token_for(idx: int) -> str:
        return f"__CAP_{idx}__"

    out = replacement
    for i in range(1, 10):
        token = _token_for(i)
        if f"\\{i}" in out:
            out = out.replace(f"\\{i}", token)
            placeholder_tokens[token] = f"\\{i}"
            group_count = max(group_count, i)

    # Escape all regex meta in the literal parts
    out = re.escape(out)
    # Restore placeholders as (.*)
    for token in placeholder_tokens.keys():
        out = out.replace(re.escape(token), "(.*)")
    return out, group_count


def invert_mapping(forward_mapping: Mapping[str, str]) -> OrderedDict:
    """Create a reverse regex mapping by inverting pattern→replacement pairs.

    - Maintains order from the forward mapping
    - If the forward pattern is anchored with '$', the reverse is anchored as well
    - If forward pattern is prefix-only (no '$'), reverse is also prefix-only
    """
    reversed_mapping: "OrderedDict[str, str]" = OrderedDict()
    for pattern, replacement in forward_mapping.items():
        # Build reverse pattern from replacement template
        reverse_pat_core, _ = _replacement_to_regex_template(replacement)
        # Respect anchoring: keep '^' always; add '$' only if original had it
        anchored_end = pattern.endswith('$')
        reverse_pattern = f"^{reverse_pat_core}" + ("$" if anchored_end else "")
        # Reverse replacement must be a literal template with backrefs (\1, \2, ...)
        reverse_replacement = _pattern_to_replacement_template(pattern)
        reversed_mapping[reverse_pattern] = reverse_replacement
    return reversed_mapping


def _pattern_to_replacement_template(pattern: str) -> str:
    r"""
    Convert a regex pattern like "^model.blocks\.(\d+)\.self_attn\.q\.(.*)$" into a replacement
    template suitable for re.sub, e.g., "model.blocks.\1.self_attn.q.\2".
    Only supports simple capturing groups of the form (.*) or (\d+), which
    matches the patterns used in the forward mapping.
    """
    # strip anchors
    core = pattern
    if core.startswith('^'):
        core = core[1:]
    if core.endswith('$'):
        core = core[:-1]

    # replace groups (.*) or (\d+) with backref tokens in increasing order
    group_index = 0
    def repl(_m: "re.Match[str]") -> str:
        nonlocal group_index
        group_index += 1
        return f"\\{group_index}"

    core = re.sub(r"\((?:\.\*|\\d\+)\)", repl, core)

    # unescape literal dots
    core = core.replace(r"\.", ".")
    return core


def select_inner_state_dict(loaded: Mapping[str, Any], key: str = "") -> Tuple[Mapping[str, Any], str]:
    if key:
        if key not in loaded:
            raise KeyError(f"Key '{key}' not found in loaded object. Available keys: {list(loaded.keys())[:20]}")
        return loaded[key], key

    # If looks like a state dict (all tensors)
    if len(loaded) > 0 and all(torch.is_tensor(v) for v in loaded.values()):
        return loaded, "<root>"

    # Common containers
    for candidate in ("state_dict", "generator_ema", "model", "ema", "module"):
        if candidate in loaded and isinstance(loaded[candidate], Mapping):
            inner = loaded[candidate]
            if len(inner) > 0 and all(torch.is_tensor(v) for v in inner.values()):
                return inner, candidate

    # Fallback: first tensor-dict value
    for v in loaded.values():
        if isinstance(v, Mapping) and len(v) > 0 and all(torch.is_tensor(t) for t in v.values()):
            return v, "<auto>"

    raise ValueError("Could not locate a state_dict (mapping of tensor parameters) in the loaded file.")


def convert_state_dict(state_dict: Mapping[str, torch.Tensor],
                       mapping: Mapping[str, str],
                       *,
                       strict: bool = True,
                       add_norm_added_q_dummy: bool = False) -> Tuple[OrderedDict, Dict[str, int]]:
    new_state_dict: "OrderedDict[str, torch.Tensor]" = OrderedDict()
    matched_count = 0
    unmatched_count = 0
    dummy_added = 0
    examples = []  # type: ignore[var-annotated]
    for k, v in state_dict.items():
        new_key = None
        for pattern, replacement in mapping.items():
            if re.match(pattern, k):
                new_key = re.sub(pattern, replacement, k)
                break
        if new_key is None:
            if strict:
                raise ValueError(f"No mapping rule matched for key: {k}")
            else:
                new_key = k  # keep original
                unmatched_count += 1
        else:
            matched_count += 1
        new_state_dict[new_key] = v

        if len(examples) < 5:
            examples.append((k, new_key))

        if add_norm_added_q_dummy and "norm_added_k" in new_key:
            dummy_key = new_key.replace("norm_added_k", "norm_added_q")
            dummy_value = torch.zeros_like(v)
            new_state_dict[dummy_key] = dummy_value
            dummy_added += 1
    stats = {"matched": matched_count, "unmatched": unmatched_count, "dummy_added": dummy_added}
    # store examples count-wise in stats by encoding as counts in print time (examples returned separately not typed)
    new_state_dict.__dict__["_examples"] = examples  # lightweight attach for printing
    return new_state_dict, stats


def save_output(new_state_dict: Mapping[str, torch.Tensor],
                output: str,
                *,
                shard: bool = True,
                max_shard_size: str = "10GB",
                wrapper_key: str = "",
                force_pt: bool = False) -> None:
    if force_pt:
        out_path = coerce_pt_output_path(output)
        obj: Dict[str, Any]
        if wrapper_key:
            obj = {wrapper_key: OrderedDict(new_state_dict)}
        else:
            # Save raw state_dict mapping
            obj = OrderedDict(new_state_dict)  # type: ignore[assignment]
        torch.save(obj, out_path)
        return

    if shard or output.endswith('/') or os.path.isdir(output):
        if save_torch_state_dict is None:
            raise RuntimeError("Saving shards requires 'huggingface_hub'. Install it or use --single-file.")
        out_dir = output if output.endswith('/') else output + '/'
        os.makedirs(out_dir, exist_ok=True)
        save_torch_state_dict(OrderedDict(new_state_dict), out_dir, max_shard_size=max_shard_size)
    else:
        # Save a single safetensors file
        save_file(OrderedDict(new_state_dict), output)


def coerce_pt_output_path(output: str) -> str:
    """Ensure output path is a .pt/.pth/.bin file. If a directory or unknown ext, coerce to .pt."""
    if output.endswith('/') or os.path.isdir(output):
        os.makedirs(output, exist_ok=True)
        return os.path.join(output, 'converted_wan.pt')
    lower = output.lower()
    if lower.endswith('.pt') or lower.endswith('.pth') or lower.endswith('.bin'):
        return output
    return output + '.pt'


def load_checkpoint(input_path: str) -> Mapping[str, Any]:
    if load_state_dict_from_file is not None:
        return load_state_dict_from_file(input_path)
    # Fallbacks by extension
    lower = input_path.lower()
    if lower.endswith('.safetensors'):
        return safe_load_file(input_path)
    # torch serialized
    obj = torch.load(input_path, map_location='cpu')
    if isinstance(obj, Mapping):
        return obj
    raise TypeError("Unsupported checkpoint format without huggingface_hub. Provide a mapping-like object.")


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Convert WAN <-> Diffusers state_dict key names.")
    p.add_argument("--input", "-i", required=True, help="Path to input checkpoint file (.pt/.bin/.safetensors)")
    p.add_argument("--output", "-o", required=True, help="Output directory (for shards) or .safetensors file")
    p.add_argument("--direction", "-d", choices=["wan-to-diffusers", "diffusers-to-wan"], default="wan-to-diffusers",
                   help="Conversion direction")
    p.add_argument("--inner-key", "-k", default="", help="WAN->Diffusers: unwrap this key. Diffusers->WAN: wrap output under this key.")
    p.add_argument("--max-shard-size", default="10GB", help="Shard size when saving to a directory")
    p.add_argument("--keep-unmatched", action="store_true", help="Keep keys with no mapping instead of failing")
    p.add_argument("--single-file", action="store_true", help="Save a single .safetensors file instead of shards")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)

    print(f"[conversion] Direction: {args.direction}")
    print(f"[conversion] Input: {args.input}")
    save_mode = "torch .pt (forced)" if args.direction == "diffusers-to-wan" else ("single safetensors" if args.single_file else f"sharded (max_shard_size={args.max_shard_size})")
    print(f"[conversion] Output: {args.output} [{save_mode}]")

    loaded = load_checkpoint(args.input)
    if not isinstance(loaded, Mapping):
        raise TypeError("Loaded checkpoint is not a mapping.")

    # Behavior of --inner-key differs by direction
    if args.direction == "wan-to-diffusers":
        inner, inner_source = select_inner_state_dict(loaded, key=args.inner_key)
        print(f"[conversion] Using inner state_dict: {inner_source}")
    else:
        inner, inner_source = select_inner_state_dict(loaded, key="")  # do not unwrap; wrap later if -k is provided
        print(f"[conversion] Using inner state_dict: {inner_source} (ignoring --inner-key for unwrap; will wrap on save)")
    print(f"[conversion] Parameters found: {len(inner)}")

    if args.direction == "wan-to-diffusers":
        mapping = _self_forcing_to_diffusers_param_names_mapping
        add_dummy = True
    else:
        mapping = invert_mapping(_self_forcing_to_diffusers_param_names_mapping)
        add_dummy = False
    print(f"[conversion] Mapping rules: {len(mapping)}")

    new_state, stats = convert_state_dict(inner, mapping, strict=not args.keep_unmatched, add_norm_added_q_dummy=add_dummy)
    examples = getattr(new_state, "_examples", [])
    if examples:
        print("[conversion] Sample key mappings:")
        for old_k, new_k in examples[:5]:
            print(f"  - {old_k} -> {new_k}")
    print(f"[conversion] Converted parameters: {len(new_state)} (matched={stats['matched']}, unmatched_kept={stats['unmatched']})")
    if add_dummy:
        print(f"[conversion] Added dummy norm_added_q tensors: {stats['dummy_added']}")

    print("[conversion] Saving...")
    wrapper_key = args.inner_key if (args.direction == "diffusers-to-wan" and args.inner_key) else ""
    if args.direction == "diffusers-to-wan":
        if wrapper_key:
            print(f"[conversion] Wrapping output under key: {wrapper_key}")
        out_path = coerce_pt_output_path(args.output)
        print(f"[conversion] Final output path: {out_path}")
        save_output(new_state, out_path, shard=False, max_shard_size=args.max_shard_size, wrapper_key=wrapper_key, force_pt=True)
    else:
        save_output(new_state, args.output, shard=not args.single_file, max_shard_size=args.max_shard_size, wrapper_key=wrapper_key)
    print("[conversion] Done.")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"[conversion] Error: {e}", file=sys.stderr)
        sys.exit(1)
