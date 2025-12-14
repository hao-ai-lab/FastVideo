import sys
sys.path.insert(0, '/FastVideo/Matrix-Game/Matrix-Game-2')

import torch
from safetensors.torch import load_file
from wan.modules.causal_model import CausalWanModel

SEED = 42
PRECISION = torch.bfloat16
DEVICE = 'cuda'

def main():
    config = dict(
        model_type='i2v',
        patch_size=(1, 2, 2),
        text_len=512,
        in_dim=36,
        dim=1536,
        ffn_dim=8960,
        freq_dim=256,
        out_dim=16,
        num_heads=12,
        num_layers=30,
        local_attn_size=-1,
        sink_size=0,
        qk_norm=True,
        cross_attn_norm=True,
        eps=1e-6,
        action_config={
            'blocks': list(range(15)),
            'heads_num': 16,
            'hidden_size': 128,
            'img_hidden_size': 1536,
            'mouse_hidden_dim': 1024,
            'keyboard_hidden_dim': 1024,
            'mouse_dim_in': 2,
            'keyboard_dim_in': 4,
            'enable_mouse': True,
            'enable_keyboard': True,
            'windows_size': 3,
            'rope_theta': 256,
            'rope_dim_list': [8, 28, 28],
            'mouse_qk_dim_list': [8, 28, 28],
            'patch_size': [1, 2, 2],
            'qk_norm': True,
            'qkv_bias': False,
            'vae_time_compression_ratio': 4,
        },
    )

    print("Loading official CausalWanModel...")
    model = CausalWanModel(**config).to(DEVICE, dtype=PRECISION)

    checkpoint_path = '/workspace/Matrix-Game-2.0/base_distilled_model/base_distill.safetensors'
    state_dict = load_file(checkpoint_path)
    # keys are like "model.blocks.0.action_model.xxx", just remove "model." prefix
    state_dict = {k.replace('model.', '', 1): v for k, v in state_dict.items()}
    
    result = model.load_state_dict(state_dict, strict=False)
    print(f"Missing keys: {len(result.missing_keys)}")
    print(f"Unexpected keys: {len(result.unexpected_keys)}")
    
    model.eval()
    model.num_frame_per_block = 15  # Match latent_frames

    torch.manual_seed(SEED)
    
    batch_size = 1
    latent_frames = 15
    latent_height = 44
    latent_width = 80

    x = torch.randn(batch_size, 16, latent_frames, latent_height, latent_width,
                    device=DEVICE, dtype=PRECISION)
    cond = torch.randn(batch_size, 20, latent_frames, latent_height, latent_width,
                       device=DEVICE, dtype=PRECISION)
    visual = torch.randn(batch_size, 257, 1280, device=DEVICE, dtype=PRECISION)
    t = torch.full((batch_size, latent_frames), 500, device=DEVICE, dtype=torch.long)
    
    # Action conditions
    num_pixel_frames = (latent_frames - 1) * 4 + 1  # 57
    mouse = torch.randn(batch_size, num_pixel_frames, 2, device=DEVICE, dtype=PRECISION)
    keyboard = torch.randn(batch_size, num_pixel_frames, 4, device=DEVICE, dtype=PRECISION)

    captured = {}
    block0 = model.blocks[0]
    orig_action = block0.action_model.forward
    def action_hook(x, t, h, w, mouse_cond, keyboard_cond, block_mask_mouse, block_mask_keyboard,
                   is_causal=True, kv_cache_mouse=None, kv_cache_keyboard=None, start_frame=0,
                   use_rope_keyboard=False, num_frame_per_block=3):
        captured['action_mouse_sum'] = mouse_cond.double().sum().item() if mouse_cond is not None else 0
        captured['action_keyboard_sum'] = keyboard_cond.double().sum().item() if keyboard_cond is not None else 0
        captured['action_t'] = t
        captured['action_h'] = h
        captured['action_w'] = w
        captured['action_is_causal'] = is_causal
        captured['action_start_frame'] = start_frame
        captured['action_use_rope_keyboard'] = use_rope_keyboard
        captured['action_num_frame_per_block'] = num_frame_per_block
        return orig_action(x, t, h, w, mouse_cond, keyboard_cond, block_mask_mouse, block_mask_keyboard,
                          is_causal=is_causal, kv_cache_mouse=kv_cache_mouse, kv_cache_keyboard=kv_cache_keyboard,
                          start_frame=start_frame, use_rope_keyboard=use_rope_keyboard,
                          num_frame_per_block=num_frame_per_block)
    block0.action_model.forward = action_hook
    
    print("\nRunning forward pass...")
    with torch.no_grad():
        with torch.amp.autocast('cuda', dtype=PRECISION):
            output = model._forward_train(x, t, visual, cond, mouse, keyboard)
    
    # Print captured action_model inputs
    print(f"\nOfficial action_mouse_cond:    {captured.get('action_mouse_sum', 0):.6f}")
    print(f"Official action_keyboard_cond: {captured.get('action_keyboard_sum', 0):.6f}")
    for k in ['action_t', 'action_h', 'action_w', 'action_is_causal', 'action_start_frame',
              'action_use_rope_keyboard', 'action_num_frame_per_block']:
        if k in captured:
            print(f"Official {k}: {captured[k]}")
    
    latent_sum = output.double().sum().item()
    
    print("\n" + "=" * 60)
    print("REFERENCE_LATENT for test_matrixgame.py:")
    print(f"REFERENCE_LATENT = {latent_sum}")
    print("=" * 60)
    
    return latent_sum

if __name__ == "__main__":
    main()
