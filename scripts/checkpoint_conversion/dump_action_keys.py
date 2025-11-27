import json
from pathlib import Path
import re
from collections import defaultdict
import torch
from safetensors.torch import load_file
from fastvideo.models.dits.matrix_game.causal_model import CausalMatrixGameWanModel
from fastvideo.configs.models.dits.matrixgame import MatrixGameWanVideoConfig

block_pat = re.compile(r"(blocks\.)(\d+)(\..*)")

def group_block_keys(keys):
    groups = defaultdict(lambda: {"idxs": [], "raw": []})
    for k in keys:
        m = block_pat.match(k)
        if m:
            prefix, idx, suffix = m.groups()
            template = prefix + "{i}" + suffix
            groups[template]["idxs"].append(int(idx))
            groups[template]["raw"].append(k)
        else:
            groups[k]["idxs"].append(None)
            groups[k]["raw"].append(k)
    return groups

def load_hf_state(p):
    ckpt = p / "transformer" / "diffusion_pytorch_model.safetensors"
    return load_file(str(ckpt))

def build_model(p):
    cfg_path = p / "transformer" / "config.json"
    with open(cfg_path) as f:
        hf_cfg = json.load(f)
    cfg = MatrixGameWanVideoConfig()
    model = CausalMatrixGameWanModel(cfg, hf_cfg)
    return model

def main():
    model_path = Path("/workspace/Matrix-Game-2.0-Diffusers/base_distilled_model")
    hf_state = load_hf_state(model_path)
    model = build_model(model_path)
    model_sd = model.state_dict()

    hf_keys = sorted(hf_state.keys())
    model_keys = sorted(model_sd.keys())

    print("\n=== hf keys (grouped) ===")
    hf_groups = group_block_keys(hf_keys)
    for template, info in sorted(hf_groups.items()):
        idxs = info["idxs"]
        if idxs == [None]:
            print(template)
        else:
            idxs = sorted(idxs)
            print(template.replace("{i}", f"[{idxs[0]}..{idxs[-1]}]"))

    print("\n=== model keys (grouped) ===")
    model_groups = group_block_keys(model_keys)
    for template, info in sorted(model_groups.items()):
        idxs = info["idxs"]
        if idxs == [None]:
            print(template)
        else:
            idxs = sorted(idxs)
            print(template.replace("{i}", f"[{idxs[0]}..{idxs[-1]}]"))

    print("\n=== only in hf (grouped) ===")
    only_hf = sorted(set(hf_keys) - set(model_keys))
    only_hf_groups = group_block_keys(only_hf)
    for template, info in sorted(only_hf_groups.items()):
        idxs = info["idxs"]
        if idxs == [None]:
            print(template)
        else:
            idxs = sorted(idxs)
            print(template.replace("{i}", f"[{idxs[0]}..{idxs[-1]}]"))

    print("\n=== only in model (grouped) ===")
    only_model = sorted(set(model_keys) - set(hf_keys))
    only_model_groups = group_block_keys(only_model)
    for template, info in sorted(only_model_groups.items()):
        idxs = info["idxs"]
        if idxs == [None]:
            print(template)
        else:
            idxs = sorted(idxs)
            print(template.replace("{i}", f"[{idxs[0]}..{idxs[-1]}]"))

    print("\n=== mapping patch (templated) ===")
    for template, info in sorted(hf_groups.items()):
        idxs = info["idxs"]
        if idxs == [None]:
            suffix = template.replace("blocks.", "")
            model_key = "model." + suffix
            print(f'mapping["{model_key}"] = "{template}"')
        else:
            hf_tpl = template
            model_tpl = "model." + template.replace("blocks.", "")
            print(f'mapping["{model_tpl}"] = "{hf_tpl}"')

if __name__ == "__main__":
    main()
