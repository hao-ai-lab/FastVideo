import json
from pathlib import Path

import torch
from safetensors.torch import load_file

from fastvideo.models.dits.matrix_game.causal_model import CausalMatrixGameWanModel
from fastvideo.configs.models.dits.matrixgame import MatrixGameWanVideoConfig


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

    hf_keys = sorted(k for k in hf_state.keys() if "action_model" in k)
    model_keys = sorted(k for k in model_sd.keys() if "action_model" in k)

    print("\n=== hf action keys ===")
    for k in hf_keys:
        print(k)

    print("\n=== model action keys ===")
    for k in model_keys:
        print(k)

    only_in_hf = [k for k in hf_keys if k not in model_keys]
    only_in_model = [k for k in model_keys if k not in hf_keys]

    print("\n=== only in hf ===")
    for k in only_in_hf:
        print(k)

    print("\n=== only in model ===")
    for k in only_in_model:
        print(k)

    print("\n=== mapping patch ===")
    for k in hf_keys:
        suffix = k.replace("blocks.", "")
        model_key = "model." + suffix
        print(f'mapping["{model_key}"] = "{k}"')


if __name__ == "__main__":
    main()
