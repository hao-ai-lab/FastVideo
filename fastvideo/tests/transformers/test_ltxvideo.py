import os

import numpy as np
import pytest
import torch
from diffusers import LTXVideoTransformer3DModel

from fastvideo.configs.pipelines import PipelineConfig
from fastvideo.forward_context import set_forward_context
from fastvideo.fastvideo_args import FastVideoArgs
from fastvideo.logger import init_logger
from fastvideo.models.loader.component_loader import TransformerLoader
from fastvideo.utils import maybe_download_model
from fastvideo.configs.models.dits import LTXVideoConfig
from huggingface_hub import snapshot_download

from fastvideo.pipelines.pipeline_batch_info import ForwardBatch


logger = init_logger(__name__)

os.environ["MASTER_ADDR"] = "localhost"
os.environ["MASTER_PORT"] = "29503"


snapshot_download(
    "Lightricks/LTX-Video",
    local_dir="data/Lightricks/LTX-Video",
    allow_patterns=[
        "vae/*.json",
        "vae/*.safetensors",        # Explicitly allow safetensors in vae
        "transformer/*.json",
        "transformer/*.safetensors", # Explicitly allow safetensors in transformer
        "tokenizer/*",
        "scheduler/*",
        "*.json",
        "README.md"
    ]
)
TRANSFORMER_PATH = "data/Lightricks/LTX-Video/transformer"


def add_debug_hooks(model, model_name):
    hooks = []
    for i, block in enumerate(model.transformer_blocks):
        def make_hook(block_idx, name):
            def hook(module, input, output):
                if isinstance(output, tuple):
                    tensor = output[0]
                else:
                    tensor = output
                print(f"{name} Block {block_idx}: max={tensor.max():.6f}")
            return hook
        
        hook = block.register_forward_hook(make_hook(i, model_name))
        hooks.append(hook)
    return hooks



@pytest.mark.usefixtures("distributed_setup")
def test_ltx_transformer():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    precision = torch.bfloat16
    precision_str = "bf16"
    args = FastVideoArgs(
        model_path=TRANSFORMER_PATH,
        dit_cpu_offload=True,
        pipeline_config=PipelineConfig(
            dit_config=LTXVideoConfig(), 
            dit_precision=precision_str
        )
    )
    args.device = device

    loader = TransformerLoader()
    model2 = loader.load(TRANSFORMER_PATH, args).to(dtype=precision)

    model1 = LTXVideoTransformer3DModel.from_pretrained(
        TRANSFORMER_PATH, device=device,
        torch_dtype=precision).to(device, dtype=precision).requires_grad_(False)

    total_params = sum(p.numel() for p in model1.parameters())
    # Calculate weight sum for model1 (converting to float64 to avoid overflow)
    weight_sum_model1 = sum(
        p.to(torch.float64).sum().item() for p in model1.parameters())
    # Also calculate mean for more stable comparison
    weight_mean_model1 = weight_sum_model1 / total_params
    logger.info("Model 1 weight sum: %s", weight_sum_model1)
    logger.info("Model 1 weight mean: %s", weight_mean_model1)

    # Calculate weight sum for model2 (converting to float64 to avoid overflow)
    total_params_model2 = sum(p.numel() for p in model2.parameters())
    weight_sum_model2 = sum(
        p.to(torch.float64).sum().item() for p in model2.parameters())
    # Also calculate mean for more stable comparison
    weight_mean_model2 = weight_sum_model2 / total_params_model2
    logger.info("Model 2 weight sum: %s", weight_sum_model2)
    logger.info("Model 2 weight mean: %s", weight_mean_model2)

    weight_sum_diff = abs(weight_sum_model1 - weight_sum_model2)
    logger.info("Weight sum difference: %s", weight_sum_diff)
    weight_mean_diff = abs(weight_mean_model1 - weight_mean_model2)
    logger.info("Weight mean difference: %s", weight_mean_diff)

    # Set both models to eval mode
    model1 = model1.eval()
    model2 = model2.eval()

    # Create identical inputs for both models
    batch_size = 1
    seq_len = 128 

    # Video latents [B, C, T, H, W]
    # LTX uses 128 latent channels
    hidden_states = torch.randn(batch_size,
                                128,  # LTX latent channels
                                5,    # temporal dimension (17 frames -> 5 latent frames)
                                64,   # height (256 -> 64 in latent space)
                                64,   # width (256 -> 64 in latent space)
                                device=device,
                                dtype=precision)

    # Text embeddings [B, L, D]
    # LTX uses 4096 dimensional embeddings
    encoder_hidden_states = torch.randn(batch_size,
                                        seq_len,
                                        4096,  # LTX embedding dimension
                                        device=device,
                                        dtype=precision)

    # Timestep
    timestep = torch.tensor([500], device=device, dtype=torch.long)

    # Create attention mask for LTX
    encoder_attention_mask = torch.ones(batch_size, seq_len, device=device, dtype=precision)

    forward_batch = ForwardBatch(
        data_type="dummy",
    )


    # TODO: clean up
    hidden_states_5d = torch.randn(batch_size, 128, 5, 64, 64, device=device, dtype=precision)
    # LTX uses different shape version as other tests
    hidden_states_3d = hidden_states_5d.permute(0, 2, 3, 4, 1).reshape(batch_size, -1, 128)

    # Add hooks to both models
    hooks1 = add_debug_hooks(model1, "Model1")
    hooks2 = add_debug_hooks(model2, "Model2")


    with torch.amp.autocast('cuda', dtype=precision):
        output1 = model1(
            hidden_states=hidden_states_3d,
            encoder_hidden_states=encoder_hidden_states,
            timestep=timestep,
            encoder_attention_mask=encoder_attention_mask,
            num_frames=5,
            height=64,
            width=64,
            return_dict=False,
        )[0]
            
        with set_forward_context(
                current_timestep=0,
                attn_metadata=None,
                forward_batch=forward_batch,
        ):
            output2 = model2(
                hidden_states=hidden_states_3d,
                encoder_hidden_states=encoder_hidden_states,
                timestep=timestep,
                encoder_attention_mask=encoder_attention_mask,
                num_frames=5,      # Same actual dimensions
                height=64,
                width=64,
                return_dict=False,

            )[0]



    # Remove hooks when done
    for hook in hooks1 + hooks2:
        hook.remove()


    def compare_models_layer_by_layer(model1, model2, hidden_states, encoder_hidden_states, timestep, encoder_attention_mask, num_frames, height, width):
        """Compare outputs of each transformer block between two models."""
        
        batch_size = hidden_states.size(0)
        
        with torch.no_grad():
            # Initial processing comparison
            h1 = model1.proj_in(hidden_states)
            h2 = model2.proj_in(hidden_states)
            print(f"After proj_in: {(h1 - h2).abs().max():.6f}")
            
            # Time embedding comparison
            temb1, embedded_timestep1 = model1.time_embed(timestep.flatten(), batch_size=batch_size, hidden_dtype=h1.dtype)
            temb2, embedded_timestep2 = model2.time_embed(timestep.flatten(), batch_size=batch_size, hidden_dtype=h2.dtype)
            print(f"Time embed diff: {(temb1 - temb2).abs().max():.6f}")
            print(f"Embedded timestep diff: {(embedded_timestep1 - embedded_timestep2).abs().max():.6f}")
            
            temb1 = temb1.view(batch_size, -1, temb1.size(-1))
            temb2 = temb2.view(batch_size, -1, temb2.size(-1))
            
            # Rotary embedding comparison
            image_rotary_emb1 = model1.rope(hidden_states, num_frames, height, width)
            image_rotary_emb2 = model2.rope(hidden_states, num_frames, height, width)
            print(f"Rotary cos diff: {(image_rotary_emb1[0] - image_rotary_emb2[0]).abs().max():.6f}")
            print(f"Rotary sin diff: {(image_rotary_emb1[1] - image_rotary_emb2[1]).abs().max():.6f}")
            
            # Caption projection comparison
            enc_h1 = model1.caption_projection(encoder_hidden_states)
            enc_h2 = model2.caption_projection(encoder_hidden_states)
            enc_h1 = enc_h1.view(batch_size, -1, h1.size(-1))
            enc_h2 = enc_h2.view(batch_size, -1, h2.size(-1))
            print(f"Caption projection diff: {(enc_h1 - enc_h2).abs().max():.6f}")
            
            print("\n" + "="*80)
            layer_diffs = []
            
            for i, (block1, block2) in enumerate(zip(model1.transformer_blocks, model2.transformer_blocks)):
                print(f"\nBlock {i}:")
                print("-"*40)
                
                # Compare scale_shift_table
                if i == 0:  # Only check first block to avoid spam
                    table_diff = (block1.scale_shift_table - block2.scale_shift_table).abs().max()
                    print(f"  Scale-shift table diff: {table_diff:.6f}")
                
                # Compute ada values before block
                num_ada_params = block1.scale_shift_table.shape[0]
                ada_values1 = block1.scale_shift_table[None, None] + temb1.reshape(batch_size, temb1.size(1), num_ada_params, -1)
                ada_values2 = block2.scale_shift_table[None, None] + temb2.reshape(batch_size, temb2.size(1), num_ada_params, -1)
                print(f"  Ada values diff: {(ada_values1 - ada_values2).abs().max():.6f}")
                
                # Extract individual ada components
                shift_msa1, scale_msa1, gate_msa1, shift_mlp1, scale_mlp1, gate_mlp1 = ada_values1.unbind(dim=2)
                shift_msa2, scale_msa2, gate_msa2, shift_mlp2, scale_mlp2, gate_mlp2 = ada_values2.unbind(dim=2)
                
                print(f"  MSA scale diff: {(scale_msa1 - scale_msa2).abs().max():.6f}")
                print(f"  MSA shift diff: {(shift_msa1 - shift_msa2).abs().max():.6f}")
                print(f"  MSA gate diff: {(gate_msa1 - gate_msa2).abs().max():.6f}")
                
                # Pre-block hidden states
                print(f"  Hidden states before block: {(h1 - h2).abs().max():.6f}")
                
                # Norm1
                norm_h1 = block1.norm1(h1)
                norm_h2 = block2.norm1(h2)
                print(f"  After norm1: {(norm_h1 - norm_h2).abs().max():.6f}")
                
                # Apply ada modulation to norm
                norm_h1_ada = norm_h1 * (1 + scale_msa1) + shift_msa1
                norm_h2_ada = norm_h2 * (1 + scale_msa2) + shift_msa2
                print(f"  After ada modulation: {(norm_h1_ada - norm_h2_ada).abs().max():.6f}")
                
                # Self-attention
                attn_out1 = block1.attn1(norm_h1_ada, encoder_hidden_states=None, image_rotary_emb=image_rotary_emb1)
                attn_out2 = block2.attn1(norm_h2_ada, encoder_hidden_states=None, image_rotary_emb=image_rotary_emb2)
                print(f"  After self-attention: {(attn_out1 - attn_out2).abs().max():.6f}")
                
                # After gated residual
                h1_after_sa = h1 + attn_out1 * gate_msa1
                h2_after_sa = h2 + attn_out2 * gate_msa2
                print(f"  After self-attn residual: {(h1_after_sa - h2_after_sa).abs().max():.6f}")
                
                # Cross-attention
                cross_attn1 = block1.attn2(h1_after_sa, encoder_hidden_states=enc_h1, attention_mask=encoder_attention_mask)
                cross_attn2 = block2.attn2(h2_after_sa, encoder_hidden_states=enc_h2, attention_mask=encoder_attention_mask)
                print(f"  After cross-attention: {(cross_attn1 - cross_attn2).abs().max():.6f}")
                
                # After cross-attention residual
                h1_after_ca = h1_after_sa + cross_attn1
                h2_after_ca = h2_after_sa + cross_attn2
                print(f"  After cross-attn residual: {(h1_after_ca - h2_after_ca).abs().max():.6f}")
                
                # Norm2 and FF
                norm2_h1 = block1.norm2(h1_after_ca) * (1 + scale_mlp1) + shift_mlp1
                norm2_h2 = block2.norm2(h2_after_ca) * (1 + scale_mlp2) + shift_mlp2
                print(f"  After norm2 + ada: {(norm2_h1 - norm2_h2).abs().max():.6f}")
                
                ff_out1 = block1.ff(norm2_h1)
                ff_out2 = block2.ff(norm2_h2)
                print(f"  After feedforward: {(ff_out1 - ff_out2).abs().max():.6f}")
                
                # Final output
                h1 = h1_after_ca + ff_out1 * gate_mlp1
                h2 = h2_after_ca + ff_out2 * gate_mlp2
                print(f"  Final block output: {(h1 - h2).abs().max():.6f}")
                
                layer_diffs.append((h1 - h2).abs().max().item())
            print("\n" + "="*80)
            print(f"Maximum difference across all blocks: {max(layer_diffs):.6f}")
            print(f"Average per-layer difference: {sum(layer_diffs)/len(layer_diffs):.6f}")
                        
        return layer_diffs
    layer_diffs = compare_models_layer_by_layer(
        model1, model2, hidden_states_3d, encoder_hidden_states, 
        timestep, encoder_attention_mask, 5, 64, 64
    )
   
    print("Block 0 scale_shift_table comparison:")
    print(f"Model1: {model1.transformer_blocks[0].scale_shift_table}")
    print(f"Model2: {model2.transformer_blocks[0].scale_shift_table}")
    # Check if outputs have the same shape
    assert output1.shape == output2.shape, f"Output shapes don't match: {output1.shape} vs {output2.shape}"
    assert output1.dtype == output2.dtype, f"Output dtype don't match: {output1.dtype} vs {output2.dtype}"

    # Check if outputs are similar (allowing for small numerical differences)
    max_diff = torch.max(torch.abs(output1 - output2))
    mean_diff = torch.mean(torch.abs(output1 - output2))
    logger.info(f"Output1 shape: {output1.shape}, dtype {output1.dtype}, range: [{output1.min().item():.4f}, {output1.max().item():.4f}]")
    logger.info(f"Output2 shape: {output2.shape}, dtype {output2.dtype}, range: [{output2.min().item():.4f}, {output2.max().item():.4f}]")

    logger.info("Max Diff: %s", max_diff.item())
    logger.info("Mean Diff: %s", mean_diff.item())
    assert max_diff < 1e-1, f"Maximum difference between outputs: {max_diff.item()}"
    # mean diff
    assert mean_diff < 1e-2, f"Mean difference between outputs: {mean_diff.item()}"


# TODO: try testing this
# @pytest.mark.usefixtures("distributed_setup")
# def test_ltx_transformer_rope_interpolation():
#     """Test LTX transformer with rope interpolation for different resolutions."""
#     device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#     precision = torch.bfloat16
#     precision_str = "bf16"
#     args = FastVideoArgs(
#         model_path=TRANSFORMER_PATH,
#         dit_cpu_offload=False,
#         pipeline_config=PipelineConfig(
#             dit_config=LTXVideoConfig(), 
#             dit_precision=precision_str
#         )
#     )
#     args.device = device

#     loader = TransformerLoader()
#     model2 = loader.load(TRANSFORMER_PATH, args).to(dtype=precision)

#     model1 = LTXVideoTransformer3DModel.from_pretrained(
#         TRANSFORMER_PATH, device=device,
#         torch_dtype=precision).to(device, dtype=precision).requires_grad_(False)

#     model1 = model1.eval()
#     model2 = model2.eval()

#     batch_size = 1
#     seq_len = 128

#     # Test with different resolutions (rope interpolation)
#     # Higher resolution than training
#     hidden_states = torch.randn(batch_size,
#                                 128,
#                                 5,
#                                 96,   # larger height
#                                 96,   # larger width
#                                 device=device,
#                                 dtype=precision)

#     encoder_hidden_states = torch.randn(batch_size,
#                                         seq_len,
#                                         4096,
#                                         device=device,
#                                         dtype=precision)

#     timestep = torch.tensor([500], device=device, dtype=torch.long)
#     encoder_attention_mask = torch.ones(batch_size, seq_len, device=device, dtype=precision)

#     # Test with rope interpolation scale
#     rope_interpolation_scale = (1.0, 1.5, 1.5)  # temporal, height, width scales

#     forward_batch = ForwardBatch(
#         data_type="dummy",
#     )

#     with torch.amp.autocast('cuda', dtype=precision):
#         output1 = model1(
#             hidden_states=hidden_states,
#             encoder_hidden_states=encoder_hidden_states,
#             timestep=timestep,
#             encoder_attention_mask=encoder_attention_mask,
#             num_frames=5,
#             height=96,
#             width=96,
#             rope_interpolation_scale=rope_interpolation_scale,
#             return_dict=False,
#         )[0]
        
#         with set_forward_context(
#                 current_timestep=0,
#                 attn_metadata=None,
#                 forward_batch=forward_batch,
#         ):
#             output2 = model2(
#                 hidden_states=hidden_states,
#                 encoder_hidden_states=encoder_hidden_states,
#                 timestep=timestep,
#                 encoder_attention_mask=encoder_attention_mask,
#                 num_frames=5,
#                 height=96,
#                 width=96,
#                 rope_interpolation_scale=rope_interpolation_scale,
#             )

#     # Check outputs
#     assert output1.shape == output2.shape, f"Output shapes don't match: {output1.shape} vs {output2.shape}"
    
#     max_diff = torch.max(torch.abs(output1 - output2))
#     logger.info("Max diff with rope interpolation: %s", max_diff.item())
#     assert max_diff < 1e-1, f"Outputs differ with rope interpolation: {max_diff.item()}"