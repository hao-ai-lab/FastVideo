import argparse
import os

import torch
from safetensors.torch import save_file

parser = argparse.ArgumentParser()
parser.add_argument("--diffusers_path", required=True, type=str)
parser.add_argument("--transformer_path", type=str, default=None, help="Path to save transformer model")
parser.add_argument("--vae_encoder_path", type=str, default=None, help="Path to save VAE encoder model")
parser.add_argument("--vae_decoder_path", type=str, default=None, help="Path to save VAE decoder model")

args = parser.parse_args()


def reverse_scale_shift(weight, dim):
    scale, shift = weight.chunk(2, dim=0)
    new_weight = torch.cat([shift, scale], dim=0)
    return new_weight


def reverse_proj_gate(weight):
    gate, proj = weight.chunk(2, dim=0)
    new_weight = torch.cat([proj, gate], dim=0)
    return new_weight


def convert_diffusers_transformer_to_mochi(state_dict):
    original_state_dict = state_dict.copy()
    new_state_dict = {}

    # Convert patch_embed
    new_state_dict["x_embedder.proj.weight"] = original_state_dict.pop("patch_embed.proj.weight")
    new_state_dict["x_embedder.proj.bias"] = original_state_dict.pop("patch_embed.proj.bias")

    # Convert time_embed
    new_state_dict["t_embedder.mlp.0.weight"] = original_state_dict.pop("time_embed.timestep_embedder.linear_1.weight")
    new_state_dict["t_embedder.mlp.0.bias"] = original_state_dict.pop("time_embed.timestep_embedder.linear_1.bias")
    new_state_dict["t_embedder.mlp.2.weight"] = original_state_dict.pop("time_embed.timestep_embedder.linear_2.weight")
    new_state_dict["t_embedder.mlp.2.bias"] = original_state_dict.pop("time_embed.timestep_embedder.linear_2.bias")
    new_state_dict["t5_y_embedder.to_kv.weight"] = original_state_dict.pop("time_embed.pooler.to_kv.weight")
    new_state_dict["t5_y_embedder.to_kv.bias"] = original_state_dict.pop("time_embed.pooler.to_kv.bias")
    new_state_dict["t5_y_embedder.to_q.weight"] = original_state_dict.pop("time_embed.pooler.to_q.weight")
    new_state_dict["t5_y_embedder.to_q.bias"] = original_state_dict.pop("time_embed.pooler.to_q.bias")
    new_state_dict["t5_y_embedder.to_out.weight"] = original_state_dict.pop("time_embed.pooler.to_out.weight")
    new_state_dict["t5_y_embedder.to_out.bias"] = original_state_dict.pop("time_embed.pooler.to_out.bias")
    new_state_dict["t5_yproj.weight"] = original_state_dict.pop("time_embed.caption_proj.weight")
    new_state_dict["t5_yproj.bias"] = original_state_dict.pop("time_embed.caption_proj.bias")

    # Convert transformer blocks
    num_layers = 48
    for i in range(num_layers):
        block_prefix = f"transformer_blocks.{i}."
        new_prefix = f"blocks.{i}."

        # norm1
        new_state_dict[new_prefix + "mod_x.weight"] = original_state_dict.pop(block_prefix + "norm1.linear.weight")
        new_state_dict[new_prefix + "mod_x.bias"] = original_state_dict.pop(block_prefix + "norm1.linear.bias")

        if i < num_layers - 1:
            new_state_dict[new_prefix + "mod_y.weight"] = original_state_dict.pop(block_prefix +
                                                                                  "norm1_context.linear.weight")
            new_state_dict[new_prefix + "mod_y.bias"] = original_state_dict.pop(block_prefix +
                                                                                "norm1_context.linear.bias")
        else:
            new_state_dict[new_prefix + "mod_y.weight"] = original_state_dict.pop(block_prefix +
                                                                                  "norm1_context.linear_1.weight")
            new_state_dict[new_prefix + "mod_y.bias"] = original_state_dict.pop(block_prefix +
                                                                                "norm1_context.linear_1.bias")

        # Visual attention
        q = original_state_dict.pop(block_prefix + "attn1.to_q.weight")
        k = original_state_dict.pop(block_prefix + "attn1.to_k.weight")
        v = original_state_dict.pop(block_prefix + "attn1.to_v.weight")
        qkv_weight = torch.cat([q, k, v], dim=0)
        new_state_dict[new_prefix + "attn.qkv_x.weight"] = qkv_weight

        new_state_dict[new_prefix + "attn.q_norm_x.weight"] = original_state_dict.pop(block_prefix +
                                                                                      "attn1.norm_q.weight")
        new_state_dict[new_prefix + "attn.k_norm_x.weight"] = original_state_dict.pop(block_prefix +
                                                                                      "attn1.norm_k.weight")
        new_state_dict[new_prefix + "attn.proj_x.weight"] = original_state_dict.pop(block_prefix +
                                                                                    "attn1.to_out.0.weight")
        new_state_dict[new_prefix + "attn.proj_x.bias"] = original_state_dict.pop(block_prefix + "attn1.to_out.0.bias")

        # Context attention
        q = original_state_dict.pop(block_prefix + "attn1.add_q_proj.weight")
        k = original_state_dict.pop(block_prefix + "attn1.add_k_proj.weight")
        v = original_state_dict.pop(block_prefix + "attn1.add_v_proj.weight")
        qkv_weight = torch.cat([q, k, v], dim=0)
        new_state_dict[new_prefix + "attn.qkv_y.weight"] = qkv_weight

        new_state_dict[new_prefix + "attn.q_norm_y.weight"] = original_state_dict.pop(block_prefix +
                                                                                      "attn1.norm_added_q.weight")
        new_state_dict[new_prefix + "attn.k_norm_y.weight"] = original_state_dict.pop(block_prefix +
                                                                                      "attn1.norm_added_k.weight")
        if i < num_layers - 1:
            new_state_dict[new_prefix + "attn.proj_y.weight"] = original_state_dict.pop(block_prefix +
                                                                                        "attn1.to_add_out.weight")
            new_state_dict[new_prefix + "attn.proj_y.bias"] = original_state_dict.pop(block_prefix +
                                                                                      "attn1.to_add_out.bias")

        # MLP
        new_state_dict[new_prefix + "mlp_x.w1.weight"] = reverse_proj_gate(
            original_state_dict.pop(block_prefix + "ff.net.0.proj.weight"))
        new_state_dict[new_prefix + "mlp_x.w2.weight"] = original_state_dict.pop(block_prefix + "ff.net.2.weight")
        if i < num_layers - 1:
            new_state_dict[new_prefix + "mlp_y.w1.weight"] = reverse_proj_gate(
                original_state_dict.pop(block_prefix + "ff_context.net.0.proj.weight"))
            new_state_dict[new_prefix + "mlp_y.w2.weight"] = original_state_dict.pop(block_prefix +
                                                                                     "ff_context.net.2.weight")

    # Output layers
    new_state_dict["final_layer.mod.weight"] = reverse_scale_shift(original_state_dict.pop("norm_out.linear.weight"),
                                                                   dim=0)
    new_state_dict["final_layer.mod.bias"] = reverse_scale_shift(original_state_dict.pop("norm_out.linear.bias"), dim=0)
    new_state_dict["final_layer.linear.weight"] = original_state_dict.pop("proj_out.weight")
    new_state_dict["final_layer.linear.bias"] = original_state_dict.pop("proj_out.bias")

    new_state_dict["pos_frequencies"] = original_state_dict.pop("pos_frequencies")

    print("Remaining Keys:", original_state_dict.keys())

    return new_state_dict


def convert_diffusers_vae_to_mochi(state_dict):
    original_state_dict = state_dict.copy()
    encoder_state_dict = {}
    decoder_state_dict = {}

    # Convert encoder
    prefix = "encoder."

    encoder_state_dict["layers.0.weight"] = original_state_dict.pop(f"{prefix}proj_in.weight")
    encoder_state_dict["layers.0.bias"] = original_state_dict.pop(f"{prefix}proj_in.bias")

    # Convert block_in
    for i in range(3):
        encoder_state_dict[f"layers.{i+1}.stack.0.weight"] = original_state_dict.pop(
            f"{prefix}block_in.resnets.{i}.norm1.norm_layer.weight")
        encoder_state_dict[f"layers.{i+1}.stack.0.bias"] = original_state_dict.pop(
            f"{prefix}block_in.resnets.{i}.norm1.norm_layer.bias")
        encoder_state_dict[f"layers.{i+1}.stack.2.weight"] = original_state_dict.pop(
            f"{prefix}block_in.resnets.{i}.conv1.conv.weight")
        encoder_state_dict[f"layers.{i+1}.stack.2.bias"] = original_state_dict.pop(
            f"{prefix}block_in.resnets.{i}.conv1.conv.bias")
        encoder_state_dict[f"layers.{i+1}.stack.3.weight"] = original_state_dict.pop(
            f"{prefix}block_in.resnets.{i}.norm2.norm_layer.weight")
        encoder_state_dict[f"layers.{i+1}.stack.3.bias"] = original_state_dict.pop(
            f"{prefix}block_in.resnets.{i}.norm2.norm_layer.bias")
        encoder_state_dict[f"layers.{i+1}.stack.5.weight"] = original_state_dict.pop(
            f"{prefix}block_in.resnets.{i}.conv2.conv.weight")
        encoder_state_dict[f"layers.{i+1}.stack.5.bias"] = original_state_dict.pop(
            f"{prefix}block_in.resnets.{i}.conv2.conv.bias")

    # Convert down_blocks
    down_block_layers = [3, 4, 6]
    for block in range(3):
        encoder_state_dict[f"layers.{block+4}.layers.0.weight"] = original_state_dict.pop(
            f"{prefix}down_blocks.{block}.conv_in.conv.weight")
        encoder_state_dict[f"layers.{block+4}.layers.0.bias"] = original_state_dict.pop(
            f"{prefix}down_blocks.{block}.conv_in.conv.bias")

        for i in range(down_block_layers[block]):
            # Convert resnets
            encoder_state_dict[f"layers.{block+4}.layers.{i+1}.stack.0.weight"] = original_state_dict.pop(
                f"{prefix}down_blocks.{block}.resnets.{i}.norm1.norm_layer.weight")
            encoder_state_dict[f"layers.{block+4}.layers.{i+1}.stack.0.bias"] = original_state_dict.pop(
                f"{prefix}down_blocks.{block}.resnets.{i}.norm1.norm_layer.bias")
            encoder_state_dict[f"layers.{block+4}.layers.{i+1}.stack.2.weight"] = original_state_dict.pop(
                f"{prefix}down_blocks.{block}.resnets.{i}.conv1.conv.weight")
            encoder_state_dict[f"layers.{block+4}.layers.{i+1}.stack.2.bias"] = original_state_dict.pop(
                f"{prefix}down_blocks.{block}.resnets.{i}.conv1.conv.bias")
            encoder_state_dict[f"layers.{block+4}.layers.{i+1}.stack.3.weight"] = original_state_dict.pop(
                f"{prefix}down_blocks.{block}.resnets.{i}.norm2.norm_layer.weight")
            encoder_state_dict[f"layers.{block+4}.layers.{i+1}.stack.3.bias"] = original_state_dict.pop(
                f"{prefix}down_blocks.{block}.resnets.{i}.norm2.norm_layer.bias")
            encoder_state_dict[f"layers.{block+4}.layers.{i+1}.stack.5.weight"] = original_state_dict.pop(
                f"{prefix}down_blocks.{block}.resnets.{i}.conv2.conv.weight")
            encoder_state_dict[f"layers.{block+4}.layers.{i+1}.stack.5.bias"] = original_state_dict.pop(
                f"{prefix}down_blocks.{block}.resnets.{i}.conv2.conv.bias")

            # Convert attentions
            q = original_state_dict.pop(f"{prefix}down_blocks.{block}.attentions.{i}.to_q.weight")
            k = original_state_dict.pop(f"{prefix}down_blocks.{block}.attentions.{i}.to_k.weight")
            v = original_state_dict.pop(f"{prefix}down_blocks.{block}.attentions.{i}.to_v.weight")
            qkv_weight = torch.cat([q, k, v], dim=0)
            encoder_state_dict[f"layers.{block+4}.layers.{i+1}.attn_block.attn.qkv.weight"] = qkv_weight

            encoder_state_dict[f"layers.{block+4}.layers.{i+1}.attn_block.attn.out.weight"] = original_state_dict.pop(
                f"{prefix}down_blocks.{block}.attentions.{i}.to_out.0.weight")
            encoder_state_dict[f"layers.{block+4}.layers.{i+1}.attn_block.attn.out.bias"] = original_state_dict.pop(
                f"{prefix}down_blocks.{block}.attentions.{i}.to_out.0.bias")
            encoder_state_dict[f"layers.{block+4}.layers.{i+1}.attn_block.norm.weight"] = original_state_dict.pop(
                f"{prefix}down_blocks.{block}.norms.{i}.norm_layer.weight")
            encoder_state_dict[f"layers.{block+4}.layers.{i+1}.attn_block.norm.bias"] = original_state_dict.pop(
                f"{prefix}down_blocks.{block}.norms.{i}.norm_layer.bias")

    # Convert block_out
    for i in range(3):
        encoder_state_dict[f"layers.{i+7}.stack.0.weight"] = original_state_dict.pop(
            f"{prefix}block_out.resnets.{i}.norm1.norm_layer.weight")
        encoder_state_dict[f"layers.{i+7}.stack.0.bias"] = original_state_dict.pop(
            f"{prefix}block_out.resnets.{i}.norm1.norm_layer.bias")
        encoder_state_dict[f"layers.{i+7}.stack.2.weight"] = original_state_dict.pop(
            f"{prefix}block_out.resnets.{i}.conv1.conv.weight")
        encoder_state_dict[f"layers.{i+7}.stack.2.bias"] = original_state_dict.pop(
            f"{prefix}block_out.resnets.{i}.conv1.conv.bias")
        encoder_state_dict[f"layers.{i+7}.stack.3.weight"] = original_state_dict.pop(
            f"{prefix}block_out.resnets.{i}.norm2.norm_layer.weight")
        encoder_state_dict[f"layers.{i+7}.stack.3.bias"] = original_state_dict.pop(
            f"{prefix}block_out.resnets.{i}.norm2.norm_layer.bias")
        encoder_state_dict[f"layers.{i+7}.stack.5.weight"] = original_state_dict.pop(
            f"{prefix}block_out.resnets.{i}.conv2.conv.weight")
        encoder_state_dict[f"layers.{i+7}.stack.5.bias"] = original_state_dict.pop(
            f"{prefix}block_out.resnets.{i}.conv2.conv.bias")

        q = original_state_dict.pop(f"{prefix}block_out.attentions.{i}.to_q.weight")
        k = original_state_dict.pop(f"{prefix}block_out.attentions.{i}.to_k.weight")
        v = original_state_dict.pop(f"{prefix}block_out.attentions.{i}.to_v.weight")
        qkv_weight = torch.cat([q, k, v], dim=0)
        encoder_state_dict[f"layers.{i+7}.attn_block.attn.qkv.weight"] = qkv_weight

        encoder_state_dict[f"layers.{i+7}.attn_block.attn.out.weight"] = original_state_dict.pop(
            f"{prefix}block_out.attentions.{i}.to_out.0.weight")
        encoder_state_dict[f"layers.{i+7}.attn_block.attn.out.bias"] = original_state_dict.pop(
            f"{prefix}block_out.attentions.{i}.to_out.0.bias")
        encoder_state_dict[f"layers.{i+7}.attn_block.norm.weight"] = original_state_dict.pop(
            f"{prefix}block_out.norms.{i}.norm_layer.weight")
        encoder_state_dict[f"layers.{i+7}.attn_block.norm.bias"] = original_state_dict.pop(
            f"{prefix}block_out.norms.{i}.norm_layer.bias")

    # Convert output layers
    encoder_state_dict["output_norm.weight"] = original_state_dict.pop(f"{prefix}norm_out.norm_layer.weight")
    encoder_state_dict["output_norm.bias"] = original_state_dict.pop(f"{prefix}norm_out.norm_layer.bias")
    encoder_state_dict["output_proj.weight"] = original_state_dict.pop(f"{prefix}proj_out.weight")

    # Convert decoder
    prefix = "decoder."

    decoder_state_dict["blocks.0.0.weight"] = original_state_dict.pop(f"{prefix}conv_in.weight")
    decoder_state_dict["blocks.0.0.bias"] = original_state_dict.pop(f"{prefix}conv_in.bias")

    # Convert block_in
    for i in range(3):
        decoder_state_dict[f"blocks.0.{i+1}.stack.0.weight"] = original_state_dict.pop(
            f"{prefix}block_in.resnets.{i}.norm1.norm_layer.weight")
        decoder_state_dict[f"blocks.0.{i+1}.stack.0.bias"] = original_state_dict.pop(
            f"{prefix}block_in.resnets.{i}.norm1.norm_layer.bias")
        decoder_state_dict[f"blocks.0.{i+1}.stack.2.weight"] = original_state_dict.pop(
            f"{prefix}block_in.resnets.{i}.conv1.conv.weight")
        decoder_state_dict[f"blocks.0.{i+1}.stack.2.bias"] = original_state_dict.pop(
            f"{prefix}block_in.resnets.{i}.conv1.conv.bias")
        decoder_state_dict[f"blocks.0.{i+1}.stack.3.weight"] = original_state_dict.pop(
            f"{prefix}block_in.resnets.{i}.norm2.norm_layer.weight")
        decoder_state_dict[f"blocks.0.{i+1}.stack.3.bias"] = original_state_dict.pop(
            f"{prefix}block_in.resnets.{i}.norm2.norm_layer.bias")
        decoder_state_dict[f"blocks.0.{i+1}.stack.5.weight"] = original_state_dict.pop(
            f"{prefix}block_in.resnets.{i}.conv2.conv.weight")
        decoder_state_dict[f"blocks.0.{i+1}.stack.5.bias"] = original_state_dict.pop(
            f"{prefix}block_in.resnets.{i}.conv2.conv.bias")

    # Convert up_blocks
    up_block_layers = [6, 4, 3]
    for block in range(3):
        for i in range(up_block_layers[block]):
            decoder_state_dict[f"blocks.{block+1}.blocks.{i}.stack.0.weight"] = original_state_dict.pop(
                f"{prefix}up_blocks.{block}.resnets.{i}.norm1.norm_layer.weight")
            decoder_state_dict[f"blocks.{block+1}.blocks.{i}.stack.0.bias"] = original_state_dict.pop(
                f"{prefix}up_blocks.{block}.resnets.{i}.norm1.norm_layer.bias")
            decoder_state_dict[f"blocks.{block+1}.blocks.{i}.stack.2.weight"] = original_state_dict.pop(
                f"{prefix}up_blocks.{block}.resnets.{i}.conv1.conv.weight")
            decoder_state_dict[f"blocks.{block+1}.blocks.{i}.stack.2.bias"] = original_state_dict.pop(
                f"{prefix}up_blocks.{block}.resnets.{i}.conv1.conv.bias")
            decoder_state_dict[f"blocks.{block+1}.blocks.{i}.stack.3.weight"] = original_state_dict.pop(
                f"{prefix}up_blocks.{block}.resnets.{i}.norm2.norm_layer.weight")
            decoder_state_dict[f"blocks.{block+1}.blocks.{i}.stack.3.bias"] = original_state_dict.pop(
                f"{prefix}up_blocks.{block}.resnets.{i}.norm2.norm_layer.bias")
            decoder_state_dict[f"blocks.{block+1}.blocks.{i}.stack.5.weight"] = original_state_dict.pop(
                f"{prefix}up_blocks.{block}.resnets.{i}.conv2.conv.weight")
            decoder_state_dict[f"blocks.{block+1}.blocks.{i}.stack.5.bias"] = original_state_dict.pop(
                f"{prefix}up_blocks.{block}.resnets.{i}.conv2.conv.bias")
        decoder_state_dict[f"blocks.{block+1}.proj.weight"] = original_state_dict.pop(
            f"{prefix}up_blocks.{block}.proj.weight")
        decoder_state_dict[f"blocks.{block+1}.proj.bias"] = original_state_dict.pop(
            f"{prefix}up_blocks.{block}.proj.bias")

    # Convert block_out
    for i in range(3):
        decoder_state_dict[f"blocks.4.{i}.stack.0.weight"] = original_state_dict.pop(
            f"{prefix}block_out.resnets.{i}.norm1.norm_layer.weight")
        decoder_state_dict[f"blocks.4.{i}.stack.0.bias"] = original_state_dict.pop(
            f"{prefix}block_out.resnets.{i}.norm1.norm_layer.bias")
        decoder_state_dict[f"blocks.4.{i}.stack.2.weight"] = original_state_dict.pop(
            f"{prefix}block_out.resnets.{i}.conv1.conv.weight")
        decoder_state_dict[f"blocks.4.{i}.stack.2.bias"] = original_state_dict.pop(
            f"{prefix}block_out.resnets.{i}.conv1.conv.bias")
        decoder_state_dict[f"blocks.4.{i}.stack.3.weight"] = original_state_dict.pop(
            f"{prefix}block_out.resnets.{i}.norm2.norm_layer.weight")
        decoder_state_dict[f"blocks.4.{i}.stack.3.bias"] = original_state_dict.pop(
            f"{prefix}block_out.resnets.{i}.norm2.norm_layer.bias")
        decoder_state_dict[f"blocks.4.{i}.stack.5.weight"] = original_state_dict.pop(
            f"{prefix}block_out.resnets.{i}.conv2.conv.weight")
        decoder_state_dict[f"blocks.4.{i}.stack.5.bias"] = original_state_dict.pop(
            f"{prefix}block_out.resnets.{i}.conv2.conv.bias")

    # Convert output layers
    decoder_state_dict["output_proj.weight"] = original_state_dict.pop(f"{prefix}proj_out.weight")
    decoder_state_dict["output_proj.bias"] = original_state_dict.pop(f"{prefix}proj_out.bias")

    return encoder_state_dict, decoder_state_dict


def ensure_safetensors_extension(path):
    if not path.endswith(".safetensors"):
        path = path + ".safetensors"
    return path


def ensure_directory_exists(path):
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)


def main(args):
    from diffusers import MochiPipeline

    pipe = MochiPipeline.from_pretrained(args.diffusers_path)

    if args.transformer_path:
        transformer_path = ensure_safetensors_extension(args.transformer_path)
        ensure_directory_exists(transformer_path)

        print("Converting transformer model...")
        transformer_state_dict = convert_diffusers_transformer_to_mochi(pipe.transformer.state_dict())
        save_file(transformer_state_dict, transformer_path)
        print(f"Saved transformer to {transformer_path}")

    if args.vae_encoder_path and args.vae_decoder_path:
        encoder_path = ensure_safetensors_extension(args.vae_encoder_path)
        decoder_path = ensure_safetensors_extension(args.vae_decoder_path)

        ensure_directory_exists(encoder_path)
        ensure_directory_exists(decoder_path)

        print("Converting VAE models...")
        encoder_state_dict, decoder_state_dict = convert_diffusers_vae_to_mochi(pipe.vae.state_dict())

        save_file(encoder_state_dict, encoder_path)
        print(f"Saved VAE encoder to {encoder_path}")

        save_file(decoder_state_dict, decoder_path)
        print(f"Saved VAE decoder to {decoder_path}")
    elif args.vae_encoder_path or args.vae_decoder_path:
        print("Warning: Both VAE encoder and decoder paths must be specified to convert VAE models.")


if __name__ == "__main__":
    main(args)
