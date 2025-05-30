from fastvideo.utils.load import load_transformer
from fastvideo.models.mochi_hf.modeling_mochi import MochiTransformer3DModel
from fastvideo.models.mochi_hf.pipeline_mochi import MochiPipeline  
import torch
from peft import LoraConfig

model_id = "genmo/mochi-1-preview"
transformer = MochiTransformer3DModel.from_pretrained(
    model_id, subfolder="transformer", torch_dtype=torch.bfloat16
)
lora_config = LoraConfig(r=16, lora_alpha=32, target_modules=["to_q", "to_k", "to_v", "to_out.0"])
transformer.add_adapter(lora_config)
breakpoint()
pipe = MochiPipeline.from_pretrained(model_id, transformer=transformer, torch_dtype=torch.float16)
pipe.vae.enable_tiling()
pipe.to("cuda")
