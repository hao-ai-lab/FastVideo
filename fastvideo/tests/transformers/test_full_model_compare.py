#!/usr/bin/env python3
"""Compare full model forward pass between FastVideo and Official to find divergence points."""
import os
os.environ["MASTER_ADDR"] = "localhost"
os.environ["MASTER_PORT"] = "29685"
os.environ["TORCHDYNAMO_DISABLE"] = "1"

import sys
sys.path.insert(0, '/FastVideo/Matrix-Game/Matrix-Game-2')

import pytest
import torch
from safetensors.torch import load_file

from fastvideo.configs.models.dits.matrixgame import MatrixGameWanVideoConfig
from fastvideo.configs.pipelines import PipelineConfig
from fastvideo.fastvideo_args import FastVideoArgs
from fastvideo.forward_context import set_forward_context
from fastvideo.models.loader.component_loader import TransformerLoader
from fastvideo.pipelines.pipeline_batch_info import ForwardBatch

from wan.modules.causal_model import CausalWanModel

SEED = 42
PRECISION = torch.bfloat16


@pytest.mark.usefixtures("distributed_setup")
def test_full_model_comparison():
    device = torch.device("cuda:0")
    
    # Load FastVideo model (30 layers)
    transformer_path = '/workspace/Matrix-Game-2.0-Diffusers/base_distilled_model/transformer'
    args = FastVideoArgs(
        model_path=transformer_path, dit_cpu_offload=False, use_fsdp_inference=False,
        pipeline_config=PipelineConfig(dit_config=MatrixGameWanVideoConfig(), dit_precision='bf16'),
    )
    args.device = device
    loader = TransformerLoader()
    fv_model = loader.load(transformer_path, args).to(device, dtype=PRECISION)
    fv_model.eval()
    fv_model.num_frame_per_block = 15
    fv_model.block_mask = None
    fv_model.block_mask_mouse = None
    fv_model.block_mask_keyboard = None
    
    # Load Official model (30 layers)
    action_config = {
        'blocks': list(range(15)),
        'heads_num': 16, 'hidden_size': 128, 'img_hidden_size': 1536,
        'mouse_hidden_dim': 1024, 'keyboard_hidden_dim': 1024,
        'mouse_dim_in': 2, 'keyboard_dim_in': 4,
        'enable_mouse': True, 'enable_keyboard': True, 'windows_size': 3,
        'rope_theta': 256, 'rope_dim_list': [8, 28, 28],
        'mouse_qk_dim_list': [8, 28, 28], 'patch_size': [1, 2, 2],
        'qk_norm': True, 'qkv_bias': False, 'vae_time_compression_ratio': 4,
    }
    off_config = dict(
        model_type='i2v', patch_size=(1, 2, 2), text_len=512, in_dim=36,
        dim=1536, ffn_dim=8960, freq_dim=256, out_dim=16, num_heads=12,
        num_layers=30,  # Full 30 layers!
        local_attn_size=-1, sink_size=0, qk_norm=True,
        cross_attn_norm=True, eps=1e-6, action_config=action_config,
    )
    
    off_model = CausalWanModel(**off_config).to(device, dtype=PRECISION)
    state_dict = load_file('/workspace/Matrix-Game-2.0/base_distilled_model/base_distill.safetensors')
    state_dict = {k.replace('model.', '', 1): v for k, v in state_dict.items()}
    off_model.load_state_dict(state_dict, strict=False)
    off_model.eval()
    off_model.num_frame_per_block = 15
    
    # Add hooks to track each block output
    fv_block_outputs = {}
    off_block_outputs = {}
    
    def make_fv_hook(block_idx):
        def hook(module, input, output):
            fv_block_outputs[block_idx] = output.double().sum().item()
        return hook
    
    def make_off_hook(block_idx):
        def hook(module, input, output):
            off_block_outputs[block_idx] = output.double().sum().item()
        return hook
    
    for i, block in enumerate(fv_model.blocks):
        block.register_forward_hook(make_fv_hook(i))
    
    for i, block in enumerate(off_model.blocks):
        block.register_forward_hook(make_off_hook(i))
    
    # Generate inputs
    torch.manual_seed(SEED)
    batch_size, latent_frames = 1, 15
    latent_height, latent_width = 44, 80
    
    x = torch.randn(batch_size, 16, latent_frames, latent_height, latent_width, device=device, dtype=PRECISION)
    cond = torch.randn(batch_size, 20, latent_frames, latent_height, latent_width, device=device, dtype=PRECISION)
    hidden_states = torch.cat([x, cond], dim=1)
    encoder_hidden_states_image = [torch.randn(batch_size, 257, 1280, device=device, dtype=PRECISION)]
    timestep = torch.full((batch_size, latent_frames), 500, device=device, dtype=PRECISION)
    num_pixel_frames = (latent_frames - 1) * 4 + 1
    mouse_cond = torch.randn(batch_size, num_pixel_frames, 2, device=device, dtype=PRECISION)
    keyboard_cond = torch.randn(batch_size, num_pixel_frames, 4, device=device, dtype=PRECISION)
    encoder_hidden_states = torch.zeros(batch_size, 0, 1536, device=device, dtype=PRECISION)
    
    visual = torch.randn(batch_size, 257, 1280, device=device, dtype=PRECISION)
    t_off = torch.full((batch_size, latent_frames), 500, device=device, dtype=torch.long)
    
    forward_batch = ForwardBatch(data_type="dummy")
    
    # Run FastVideo
    with torch.no_grad():
        with torch.amp.autocast('cuda', dtype=PRECISION):
            with set_forward_context(current_timestep=0, attn_metadata=None, forward_batch=forward_batch):
                fv_output = fv_model(
                    hidden_states=hidden_states,
                    encoder_hidden_states=encoder_hidden_states,
                    encoder_hidden_states_image=encoder_hidden_states_image,
                    timestep=timestep,
                    mouse_cond=mouse_cond,
                    keyboard_cond=keyboard_cond,
                )
    
    # Regenerate inputs
    torch.manual_seed(SEED)
    x = torch.randn(batch_size, 16, latent_frames, latent_height, latent_width, device=device, dtype=PRECISION)
    cond = torch.randn(batch_size, 20, latent_frames, latent_height, latent_width, device=device, dtype=PRECISION)
    visual = torch.randn(batch_size, 257, 1280, device=device, dtype=PRECISION)
    mouse_cond = torch.randn(batch_size, num_pixel_frames, 2, device=device, dtype=PRECISION)
    keyboard_cond = torch.randn(batch_size, num_pixel_frames, 4, device=device, dtype=PRECISION)
    
    # Run Official
    with torch.no_grad():
        with torch.amp.autocast('cuda', dtype=PRECISION):
            off_output = off_model._forward_train(x, t_off, visual, cond, mouse_cond, keyboard_cond)
    
    # Compare block outputs
    print("\n" + "=" * 80)
    print("BLOCK-BY-BLOCK COMPARISON")
    print("=" * 80)
    
    for i in range(min(len(fv_block_outputs), len(off_block_outputs))):
        fv_val = fv_block_outputs.get(i, 0)
        off_val = off_block_outputs.get(i, 0)
        diff = fv_val - off_val
        
        has_action = i < 15
        marker = "⚠️" if abs(diff) > 10 else "✅"
        action_str = "[ACTION]" if has_action else ""
        print(f"Block {i:2d} {action_str:8s}: FV={fv_val:15.2f}, OFF={off_val:15.2f}, diff={diff:12.2f} {marker}")
    
    print("\n" + "=" * 80)
    print("FINAL OUTPUT")
    print("=" * 80)
    
    fv_final = fv_output.double().sum().item()
    off_final = off_output.double().sum().item()
    print(f"FastVideo: {fv_final:.4f}")
    print(f"Official:  {off_final:.4f}")
    print(f"Diff:      {fv_final - off_final:.4f}")
