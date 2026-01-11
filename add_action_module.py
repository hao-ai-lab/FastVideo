import argparse
import json
import os
from typing import Any

import torch
from safetensors.torch import load_file, save_file

from fastvideo.models.dits.matrix_game.action_module import ActionModule


def build_default_action_config(*, keyboard_dim_in: int = 3) -> dict[str, Any]:
    return {
        "blocks": list(range(30)),
        "enable_keyboard": True,
        "enable_mouse": False,
        "heads_num": 16,
        "hidden_size": 128,
        "img_hidden_size": 1536,
        "keyboard_dim_in": keyboard_dim_in,
        "keyboard_hidden_dim": 1024,
        "mouse_dim_in": 0,
        "mouse_hidden_dim": 1024,
        "mouse_qk_dim_list": [8, 28, 28],
        "patch_size": [1, 2, 2],
        "qk_norm": True,
        "qkv_bias": False,
        "rope_dim_list": [8, 28, 28],
        "rope_theta": 256,
        "vae_time_compression_ratio": 4,
        "windows_size": 3,
    }


def add_action_module(
    sd: dict[str, torch.Tensor],
    *,
    num_layers: int,
    action_config: dict[str, Any],
    dtype: torch.dtype = torch.float32,
    force_overwrite: bool = False,
    linear_std: float = 0.02,
) -> dict[str, torch.Tensor]:
    am = ActionModule(
        mouse_dim_in=action_config.get("mouse_dim_in", 0),
        keyboard_dim_in=action_config["keyboard_dim_in"],
        hidden_size=action_config["hidden_size"],
        img_hidden_size=action_config["img_hidden_size"],
        keyboard_hidden_dim=action_config["keyboard_hidden_dim"],
        mouse_hidden_dim=action_config.get("mouse_hidden_dim", 1024),
        vae_time_compression_ratio=action_config["vae_time_compression_ratio"],
        windows_size=action_config["windows_size"],
        heads_num=action_config["heads_num"],
        patch_size=action_config["patch_size"],
        qk_norm=action_config["qk_norm"],
        qkv_bias=action_config["qkv_bias"],
        rope_dim_list=action_config["rope_dim_list"],
        rope_theta=action_config["rope_theta"],
        mouse_qk_dim_list=action_config.get("mouse_qk_dim_list", [8, 28, 28]),
        enable_mouse=action_config["enable_mouse"],
        enable_keyboard=action_config["enable_keyboard"],
        local_attn_size=action_config.get("local_attn_size", 6),
    )
    template = {k: v.detach().to(dtype=dtype).cpu() for k, v in am.state_dict().items()}

    g = torch.Generator(device="cpu").manual_seed(0)

    out_sd = dict(sd)
    blocks = action_config.get("blocks", list(range(num_layers)))
    for i in range(num_layers):
        if i not in blocks:
            continue
        for k, t in template.items():
            full_key = f"blocks.{i}.action_model.{k}"
            if (not force_overwrite) and (full_key in out_sd):
                continue

            out = t.clone()
            if k.endswith(".weight"):
                if out.ndim == 1:
                    out.fill_(1)
                else:
                    out.normal_(mean=0.0, std=linear_std, generator=g)
            elif k.endswith(".bias"):
                out.zero_()
            
            out_sd[full_key] = out
    return out_sd


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--input_transformer_dir", type=str, required=True)
    p.add_argument("--output_transformer_dir", type=str, required=True)
    p.add_argument("--keyboard_dim_in", type=int, default=3)
    p.add_argument("--force", action="store_true")
    p.add_argument("--linear-std", type=float, default=0.02)
    args = p.parse_args(argv)

    in_dir = args.input_transformer_dir
    out_dir = args.output_transformer_dir
    os.makedirs(out_dir, exist_ok=True)

    in_weights = os.path.join(in_dir, "diffusion_pytorch_model.safetensors")
    in_config = os.path.join(in_dir, "config.json")
    if not os.path.exists(in_weights):
        raise FileNotFoundError(in_weights)
    if not os.path.exists(in_config):
        raise FileNotFoundError(in_config)

    cfg = json.load(open(in_config))
    num_layers = int(cfg.get("num_layers", 30))
    action_config = cfg.get("action_config") or build_default_action_config(
        keyboard_dim_in=args.keyboard_dim_in)
    if "blocks" not in action_config:
        action_config["blocks"] = list(range(num_layers))
    cfg["action_config"] = action_config

    sd = load_file(in_weights)
    sd = {k: v for k, v in sd.items()}

    new_sd = add_action_module(
        sd,
        num_layers=num_layers,
        action_config=action_config,
        dtype=torch.float32,
        force_overwrite=args.force,
        linear_std=args.linear_std,
    )

    out_weights = os.path.join(out_dir, "diffusion_pytorch_model.safetensors")
    out_config = os.path.join(out_dir, "config.json")
    save_file(new_sd, out_weights)
    json.dump(cfg, open(out_config, "w"), indent=2)

    print("Saved:", out_weights)
    print("Saved:", out_config)


if __name__ == "__main__":
    main()

