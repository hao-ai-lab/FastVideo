import torch
from fastvideo.models.mochi_hf.modeling_mochi import MochiTransformer3DModel

def load_transformer(args):
    if args.model_type == "mochi":
        if args.dit_model_name_or_path:
            transformer = MochiTransformer3DModel.from_pretrained(
                args.dit_model_name_or_path,
                torch_dtype=torch.float32,
                # torch_dtype=torch.bfloat16 if args.use_lora else torch.float32,
            )
        else:
            transformer = MochiTransformer3DModel.from_pretrained(
                args.pretrained_model_name_or_path,
                subfolder="transformer",
                torch_dtype=torch.float32,
                # torch_dtype=torch.bfloat16 if args.use_lora else torch.float32,
            )
    elif args.model_type == "hunyuan"
        pass

    return transformer

def get_no_split_modules(args):
    if args.model_type == "mochi":
        return ["MochiTransformerBlock"]
    elif args.model_type == "hunyuan":
        pass