import torch
from fastvideo.models.mochi_hf.modeling_mochi import MochiTransformer3DModel, MochiTransformerBlock
from fastvideo.models.hunyuan.modules.models import  HYVideoDiffusionTransformer, MMDoubleStreamBlock, MMSingleStreamBlock

hunyuan_config =  {
    "mm_double_blocks_depth": 20,
    "mm_single_blocks_depth": 40,
    "rope_dim_list": [16, 56, 56],
    "hidden_size": 3072,
    "heads_num": 24,
    "mlp_width_ratio": 4,
    "guidance_embed": True,
}


def load_hunyuan_state_dict(model, dit_model_name_or_path):
    load_key = "module"
    model_path = dit_model_name_or_path
    bare_model = "unknown"
        
    state_dict = torch.load(model_path, map_location=lambda storage, loc: storage)

    if bare_model == "unknown" and ("ema" in state_dict or "module" in state_dict):
        bare_model = False
    if bare_model is False:
        if load_key in state_dict:
            state_dict = state_dict[load_key]
        else:
            raise KeyError(
                f"Missing key: `{load_key}` in the checkpoint: {model_path}. The keys in the checkpoint "
                f"are: {list(state_dict.keys())}."
            )
    model.load_state_dict(state_dict, strict=True)
    return model
        
def load_transformer(model_type,dit_model_name_or_path, pretrained_model_name_or_path ):
    if model_type == "mochi":
        if dit_model_name_or_path:
            transformer = MochiTransformer3DModel.from_pretrained(
                dit_model_name_or_path,
                torch_dtype=torch.float32,
                # torch_dtype=torch.bfloat16 if args.use_lora else torch.float32,
            )
        else:
            transformer = MochiTransformer3DModel.from_pretrained(
                pretrained_model_name_or_path,
                subfolder="transformer",
                torch_dtype=torch.float32,
                # torch_dtype=torch.bfloat16 if args.use_lora else torch.float32,
            )
    elif model_type == "hunyuan":
        transformer = HYVideoDiffusionTransformer(
            in_channels=16,
            out_channels=16,
            **hunyuan_config,
            dtype=torch.bfloat16,
        )
        transformer = load_hunyuan_state_dict(transformer, dit_model_name_or_path)
    else:
        raise ValueError(f"Unsupported model type: {model_type}")
    return transformer

def get_no_split_modules(transformer):
    # if of type MochiTransformer3DModel
    if isinstance(transformer, MochiTransformer3DModel):
        return [MochiTransformerBlock]
    elif isinstance(transformer, HYVideoDiffusionTransformer):
        return [MMDoubleStreamBlock, MMSingleStreamBlock]
    else:
        raise ValueError(f"Unsupported transformer type: {type(transformer)}")