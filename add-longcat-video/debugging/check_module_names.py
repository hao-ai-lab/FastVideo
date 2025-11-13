import os
import numpy as np
os.environ['CUDA_VISIBLE_DEVICES'] = '7'
os.environ['MASTER_ADDR'] = 'localhost'
os.environ['MASTER_PORT'] = str(29500 + np.random.randint(0, 1000))
os.environ['RANK'] = '0'
os.environ['WORLD_SIZE'] = '1'
os.environ['LOCAL_RANK'] = '0'

from fastvideo.models.dits.longcat import LongCatTransformer3DModel
from fastvideo.configs.models.dits.longcat import LongCatVideoConfig
from fastvideo.distributed.parallel_state import init_distributed_environment, initialize_model_parallel
from safetensors.torch import load_file
import json

init_distributed_environment()
initialize_model_parallel()

with open('weights/longcat-native/transformer/config.json') as f:
    config_dict = json.load(f)
model_config = LongCatVideoConfig()
model = LongCatTransformer3DModel(config=model_config, hf_config=config_dict)

print('All modules in blocks.0.cross_attn:')
for name, module in model.named_modules():
    if name.startswith('blocks.0.cross_attn'):
        print(f'  {name}: {type(module).__name__}')

print('\nLoRA keys for blocks.0.cross_attn:')
lora = load_file('weights/longcat-native/lora/distilled/cfg_step_lora.safetensors')
for key in list(lora.keys())[:10]:
    if 'blocks.0.cross_attn' in key:
        print(f'  {key}')

