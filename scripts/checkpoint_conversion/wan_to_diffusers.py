from huggingface_hub import save_torch_state_dict, load_state_dict_from_file
# from safetensors import safetensors
from safetensors.torch import save_file
import torch
import re
from collections import OrderedDict

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

state_dict = load_state_dict_from_file("checkpoints/self_forcing_dmd.pt")
state_dict = state_dict["generator_ema"]
new_state_dict = OrderedDict()
for k, v in state_dict.items():
    new_key = k
    for pattern, replacement in _self_forcing_to_diffusers_param_names_mapping.items():
        if re.match(pattern, k):
            new_key = re.sub(pattern, replacement, k)
            break  # Stop at the first match
    else:
        # print(f"No match found for {k}")
        raise ValueError(f"No match found for {k}")
    new_state_dict[new_key] = v
    if "norm_added_k" in new_key:
        dummy_key = new_key.replace("norm_added_k", "norm_added_q")
        dummy_value = torch.zeros_like(v)
        new_state_dict[dummy_key] = dummy_value
del state_dict

save_torch_state_dict(
    new_state_dict,
    "new2/",
    max_shard_size="10GB"
)
