import torch
from fastvideo.models.mochi_hf.modeling_mochi import MochiTransformer3DModel, MochiTransformerBlock
from fastvideo.models.hunyuan.modules.models import  HYVideoDiffusionTransformer, MMDoubleStreamBlock, MMSingleStreamBlock
from fastvideo.models.hunyuan.vae.autoencoder_kl_causal_3d import AutoencoderKLCausal3D
from diffusers import AutoencoderKLMochi
from transformers import T5EncoderModel, AutoTokenizer
import os 
from fastvideo.utils.logging import main_print
from torch import nn
# Path
from pathlib import Path
hunyuan_config =  {
    "mm_double_blocks_depth": 20,
    "mm_single_blocks_depth": 40,
    "rope_dim_list": [16, 56, 56],
    "hidden_size": 3072,
    "heads_num": 24,
    "mlp_width_ratio": 4,
    "guidance_embed": True,
}

class HunyuanTextEncoderWrapper(nn.Module):
    def __init__(self, pretrained_model_name_or_path):
        super().__init__()


    def encode_prompt(self, text):
        return self.text_encoder(text)
    
class MochiTextEncoderWrapper(nn.Module):
    def __init__(self, pretrained_model_name_or_path):
        super().__init__()
        self.text_encoder = T5EncoderModel.from_pretrained(os.path.join(pretrained_model_name_or_path, "text_encoder"))
        self.tokenizer = AutoTokenizer.from_pretrained(os.path.join(pretrained_model_name_or_path, "text_encoder"))
        self.max_sequence_length = 256
    def encode_prompt(self, prompt):
        device = self.device 
        dtype = self.dtype

        prompt = [prompt] if isinstance(prompt, str) else prompt
        batch_size = len(prompt)

        text_inputs = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=self.max_sequence_length,
            truncation=True,
            add_special_tokens=True,
            return_tensors="pt",
        )
        text_input_ids = text_inputs.input_ids
        prompt_attention_mask = text_inputs.attention_mask
        prompt_attention_mask = prompt_attention_mask.bool().to(device)

        untruncated_ids = self.tokenizer(
            prompt, padding="longest", return_tensors="pt"
        ).input_ids

        if untruncated_ids.shape[-1] >= text_input_ids.shape[-1] and not torch.equal(
            text_input_ids, untruncated_ids
        ):
            removed_text = self.tokenizer.batch_decode(
                untruncated_ids[:, self.max_sequence_length - 1 : -1]
            )
            main_print(
                f"Truncated text input: {prompt} to: {removed_text} for model input."
            )
        prompt_embeds = self.text_encoder(
            text_input_ids.to(device), attention_mask=prompt_attention_mask
        )[0]
        prompt_embeds = prompt_embeds.to(dtype=dtype, device=device)

        # duplicate text embeddings for each generation per prompt, using mps friendly method
        _, seq_len, _ = prompt_embeds.shape
        prompt_embeds = prompt_embeds.view(
            batch_size , seq_len, -1
        )
        prompt_attention_mask = prompt_attention_mask.view(batch_size, -1)

        return prompt_embeds, prompt_attention_mask

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

def load_vae(model_type, pretrained_model_name_or_path):
    weight_dtype = torch.float32
    if model_type == "mochi":
        vae = AutoencoderKLMochi.from_pretrained(
            pretrained_model_name_or_path, subfolder="vae", torch_dtype=weight_dtype
        ).to("cuda")
        autocast_type = torch.bfloat16
        fps = 30
    elif model_type == "hunyuan":
        vae_precision = torch.float32
        vae_path = os.path.join(pretrained_model_name_or_path, "hunyuan-video-t2v-720p/vae")
    
        config = AutoencoderKLCausal3D.load_config(vae_path)
        vae = AutoencoderKLCausal3D.from_config(config)
        
        vae_ckpt = Path(vae_path) / "pytorch_model.pt"
        assert vae_ckpt.exists(), f"VAE checkpoint not found: {vae_ckpt}"
        
        ckpt = torch.load(vae_ckpt, map_location=vae.device)
        if "state_dict" in ckpt:
            ckpt = ckpt["state_dict"]
        if any(k.startswith("vae.") for k in ckpt.keys()):
            ckpt = {k.replace("vae.", ""): v for k, v in ckpt.items() if k.startswith("vae.")}
        vae.load_state_dict(ckpt)
        vae = vae.to(dtype=vae_precision)
        vae.requires_grad_(False)
        vae = vae.to("cuda")
        vae.eval()
        autocast_type = torch.float32
        fps = 24
    return vae, autocast_type, fps
        


def load_text_encoder(model_type, pretrained_model_name_or_path ):
    if model_type == "mochi":
        text_encoder = MochiTextEncoderWrapper(pretrained_model_name_or_path)
    elif model_type == "hunyuan":
        text_encoder = HunyuanTextEncoderWrapper(pretrained_model_name_or_path)
    else:
        raise ValueError(f"Unsupported model type: {model_type}")
    return text_encoder

def get_no_split_modules(transformer):
    # if of type MochiTransformer3DModel
    if isinstance(transformer, MochiTransformer3DModel):
        return [MochiTransformerBlock]
    elif isinstance(transformer, HYVideoDiffusionTransformer):
        return [MMDoubleStreamBlock, MMSingleStreamBlock]
    else:
        raise ValueError(f"Unsupported transformer type: {type(transformer)}")