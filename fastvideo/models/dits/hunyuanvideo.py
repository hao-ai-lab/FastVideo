import torch
import torch.nn as nn
from typing import Optional, Tuple, List

from fastvideo.attention.distributed_attn import DistributedAttention
from fastvideo.layers.linear import ReplicatedLinear
from fastvideo.layers.layernorm import RMSNorm, LayerNormScaleShift, ScaleResidual, ScaleResidualLayerNormScaleShift
from fastvideo.layers.activation import SiluAndMul, GeluAndMul
from fastvideo.layers.visual_embed import PatchEmbed, TimestepEmbedder, MLPEmbedder, ModulateProjection


class DistributedMLP(nn.Module):
    """MLP with distributed linear layers."""
    
    def __init__(
        self,
        hidden_size: int,
        mlp_hidden_dim: int,
        act_layer_type: str = "gelu_tanh",
        bias: bool = True,
        params_dtype: Optional[torch.dtype] = None,
    ):
        super().__init__()
        self.fc1 = ReplicatedLinear(
            hidden_size,
            mlp_hidden_dim * 2,  # For activation functions like SiLU that need 2x width
            bias=bias,
            params_dtype=params_dtype
        )
        
        self.act = GeluAndMul(approximate="tanh") if act_layer_type == "gelu_tanh" else SiluAndMul()
        
        self.fc2 = ReplicatedLinear(
            mlp_hidden_dim,
            hidden_size,
            bias=bias,
            params_dtype=params_dtype
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x, _ = self.fc1(x)
        x = self.act(x)
        x, _ = self.fc2(x)
        return x






class MMDoubleStreamBlock(nn.Module):
    """
    A multimodal DiT block with separate modulation for text and image/video,
    using distributed attention and linear layers.
    """

    def __init__(
        self,
        hidden_size: int,
        heads_num: int,
        mlp_width_ratio: float,
        mlp_act_type: str = "gelu_tanh",
        qk_norm: bool = True,
        qk_norm_type: str = "rms",
        qkv_bias: bool = False,
        dtype: Optional[torch.dtype] = None,
    ):
        super().__init__()

        self.deterministic = False
        self.heads_num = heads_num
        head_dim = hidden_size // heads_num
        mlp_hidden_dim = int(hidden_size * mlp_width_ratio)

        # Image modulation components
        self.img_mod = ModulateProjection(
            hidden_size,
            factor=6,
            act_layer=nn.SiLU(),
            dtype=dtype,
        )
        
        # Fused operations for image stream
        self.img_attn_norm = LayerNormScaleShift(hidden_size, norm_type=qk_norm_type)
        self.img_attn_residual_mlp_norm = ScaleResidualLayerNormScaleShift(hidden_size, norm_type=qk_norm_type)
        self.img_mlp_residual = ScaleResidual()

        # Image attention components
        self.img_attn_qkv = ReplicatedLinear(
            hidden_size,
            hidden_size * 3,
            bias=qkv_bias,
            params_dtype=dtype
        )
        
        # QK norm layers
        if qk_norm:
            self.img_attn_q_norm = RMSNorm(head_dim, eps=1e-6) if qk_norm_type == "rms" else nn.LayerNorm(
                head_dim, elementwise_affine=True, eps=1e-6
            )
            self.img_attn_k_norm = RMSNorm(head_dim, eps=1e-6) if qk_norm_type == "rms" else nn.LayerNorm(
                head_dim, elementwise_affine=True, eps=1e-6
            )
        else:
            self.img_attn_q_norm = nn.Identity()
            self.img_attn_k_norm = nn.Identity()
            
        self.img_attn_proj = ReplicatedLinear(
            hidden_size,
            hidden_size,
            bias=qkv_bias,
            params_dtype=dtype
        )
        
        self.img_mlp = DistributedMLP(
            hidden_size,
            mlp_hidden_dim,
            act_layer_type=mlp_act_type,
            bias=True,
            params_dtype=dtype
        )

        # Text modulation components
        self.txt_mod = ModulateProjection(
            hidden_size,
            factor=6,
            act_layer=nn.SiLU(),
            dtype=dtype,
        )
        
        # Fused operations for text stream
        self.txt_attn_norm = LayerNormScaleShift(hidden_size, norm_type=qk_norm_type)
        self.txt_attn_residual_mlp_norm = ScaleResidualLayerNormScaleShift(hidden_size, norm_type=qk_norm_type)
        self.txt_mlp_residual = ScaleResidual()

        # Text attention components
        self.txt_attn_qkv = ReplicatedLinear(
            hidden_size,
            hidden_size * 3,
            bias=qkv_bias,
            params_dtype=dtype
        )
        
        # QK norm layers for text
        if qk_norm:
            self.txt_attn_q_norm = RMSNorm(head_dim, eps=1e-6) if qk_norm_type == "rms" else nn.LayerNorm(
                head_dim, elementwise_affine=True, eps=1e-6
            )
            self.txt_attn_k_norm = RMSNorm(head_dim, eps=1e-6) if qk_norm_type == "rms" else nn.LayerNorm(
                head_dim, elementwise_affine=True, eps=1e-6
            )
        else:
            self.txt_attn_q_norm = nn.Identity()
            self.txt_attn_k_norm = nn.Identity()
            
        self.txt_attn_proj = ReplicatedLinear(
            hidden_size,
            hidden_size,
            bias=qkv_bias,
            params_dtype=dtype
        )
        
        self.txt_mlp = DistributedMLP(
            hidden_size,
            mlp_hidden_dim,
            act_layer_type=mlp_act_type,
            bias=True,
            params_dtype=dtype
        )
        
        # Distributed attention
        self.attn = DistributedAttention(
            dropout_rate=0.0,
            causal=False
        )


    def forward(
        self,
        img: torch.Tensor,
        txt: torch.Tensor,
        vec: torch.Tensor,
        freqs_cis: tuple = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Process modulation vectors
        img_mod_outputs = self.img_mod(vec)
        (
            img_attn_shift,
            img_attn_scale,
            img_attn_gate,
            img_mlp_shift,
            img_mlp_scale,
            img_mlp_gate,
        ) = torch.chunk(img_mod_outputs, 6, dim=-1)
        
        txt_mod_outputs = self.txt_mod(vec)
        (
            txt_attn_shift,
            txt_attn_scale,
            txt_attn_gate,
            txt_mlp_shift,
            txt_mlp_scale,
            txt_mlp_gate,
        ) = torch.chunk(txt_mod_outputs, 6, dim=-1)

        # Prepare image for attention using fused operation
        img_attn_input = self.img_attn_norm(img, img_attn_shift, img_attn_scale)
        
        # Get QKV for image
        img_qkv, _ = self.img_attn_qkv(img_attn_input)
        batch_size, seq_len = img_qkv.shape[0], img_qkv.shape[1]
        
        # Split QKV
        img_qkv = img_qkv.view(batch_size, seq_len, 3, self.heads_num, -1)
        img_q, img_k, img_v = img_qkv[:, :, 0], img_qkv[:, :, 1], img_qkv[:, :, 2]
        
        # Apply QK-Norm if needed
        img_q = self.img_attn_q_norm(img_q)
        img_k = self.img_attn_k_norm(img_k)


        # Apply rotary embeddings
        cos, sin = freqs_cis
        img_q, img_k = apply_rotary_pos_emb(img_q, img_k, cos, sin)

        # Prepare text for attention using fused operation
        txt_attn_input = self.txt_attn_norm(txt, txt_attn_shift, txt_attn_scale)
        
        # Get QKV for text
        txt_qkv, _ = self.txt_attn_qkv(txt_attn_input)
        batch_size, seq_len = txt_qkv.shape[0], txt_qkv.shape[1]
        
        # Split QKV
        txt_qkv = txt_qkv.view(batch_size, seq_len, 3, self.heads_num, -1)
        txt_q, txt_k, txt_v = txt_qkv[:, :, 0], txt_qkv[:, :, 1], txt_qkv[:, :, 2]
        
        # Apply QK-Norm if needed
        txt_q = self.txt_attn_q_norm(txt_q)
        txt_k = self.txt_attn_k_norm(txt_k)

        # Combine image and text tensors
        q = torch.cat((img_q, txt_q), dim=1)
        k = torch.cat((img_k, txt_k), dim=1)
        v = torch.cat((img_v, txt_v), dim=1)
        
        # Run distributed attention
        attn_output, _ = self.attn(q, k, v)
        
        # Split attention output back to image and text
        img_attn, txt_attn = torch.split(attn_output, [img_q.shape[1], txt_q.shape[1]], dim=1)
        
        # Process image attention output
        img_attn_out, _ = self.img_attn_proj(img_attn)
        
        # Use fused operation for residual connection, normalization, and modulation
        img_mlp_input, img_residual = self.img_attn_residual_mlp_norm(
            img, img_attn_out, img_attn_gate, img_mlp_shift, img_mlp_scale
        )
        
        # Process image MLP
        img_mlp_out = self.img_mlp(img_mlp_input)
        img = self.img_mlp_residual(img_residual, img_mlp_out, img_mlp_gate)

        # Process text attention output
        txt_attn_out, _ = self.txt_attn_proj(txt_attn)
        
        # Use fused operation for residual connection, normalization, and modulation
        txt_mlp_input, txt_residual = self.txt_attn_residual_mlp_norm(
            txt, txt_attn_out, txt_attn_gate, txt_mlp_shift, txt_mlp_scale
        )
        
        # Process text MLP
        txt_mlp_out = self.txt_mlp(txt_mlp_input)
        txt = self.txt_mlp_residual(txt_residual, txt_mlp_out, txt_mlp_gate)

        return img, txt


class MMSingleStreamBlock(nn.Module):
    """
    A DiT block with parallel linear layers using distributed attention
    and tensor parallelism.
    """

    def __init__(
        self,
        hidden_size: int,
        heads_num: int,
        mlp_width_ratio: float = 4.0,
        mlp_act_type: str = "gelu_tanh",
        qk_norm: bool = True,
        qk_norm_type: str = "rms",
        qk_scale: float = None,
        dtype: Optional[torch.dtype] = None,
    ):
        super().__init__()

        self.deterministic = False
        self.hidden_size = hidden_size
        self.heads_num = heads_num
        head_dim = hidden_size // heads_num
        mlp_hidden_dim = int(hidden_size * mlp_width_ratio)
        self.mlp_hidden_dim = mlp_hidden_dim
        self.scale = qk_scale or head_dim ** -0.5

        # Combined QKV and MLP input projection
        self.linear1 = ReplicatedLinear(
            hidden_size, 
            hidden_size * 3 + mlp_hidden_dim,
            bias=True,
            params_dtype=dtype
        )
        
        # Combined projection and MLP output
        self.linear2 = ReplicatedLinear(
            hidden_size + mlp_hidden_dim,
            hidden_size,
            bias=True,
            params_dtype=dtype
        )

        # QK norm layers
        if qk_norm:
            self.q_norm = RMSNorm(head_dim, eps=1e-6) if qk_norm_type == "rms" else nn.LayerNorm(
                head_dim, elementwise_affine=True, eps=1e-6
            )
            self.k_norm = RMSNorm(head_dim, eps=1e-6) if qk_norm_type == "rms" else nn.LayerNorm(
                head_dim, elementwise_affine=True, eps=1e-6
            )
        else:
            self.q_norm = nn.Identity()
            self.k_norm = nn.Identity()

        # Fused operations with better naming
        self.input_norm_scale_shift = LayerNormScaleShift(hidden_size, norm_type=qk_norm_type)
        self.output_residual = ScaleResidual()

        # Activation function
        if mlp_act_type == "gelu_tanh":
            self.mlp_act = GeluAndMul(approximate="tanh")
        else:
            self.mlp_act = SiluAndMul()
            
        # Modulation
        self.modulation = ModulateProjection(
            hidden_size,
            factor=3,
            act_layer=nn.SiLU(),
            dtype=dtype
        )
        
        # Distributed attention
        self.attn = DistributedAttention(
            dropout_rate=0.0,
            causal=False
        )

    def forward(
        self,
        x: torch.Tensor,
        vec: torch.Tensor,
        txt_len: int,
        freqs_cis: Tuple[torch.Tensor, torch.Tensor] = None,
    ) -> torch.Tensor:
        # Process modulation
        mod_outputs = self.modulation(vec)
        mod_shift, mod_scale, mod_gate = torch.chunk(mod_outputs, 3, dim=-1)
        
        # Apply pre-norm and modulation using fused operation
        x_mod = self.input_norm_scale_shift(x, mod_shift, mod_scale)
        
        # Get combined projections
        linear1_out, _ = self.linear1(x_mod)
        
        # Split into QKV and MLP parts
        qkv, mlp = torch.split(
            linear1_out, [3 * self.hidden_size, self.mlp_hidden_dim], dim=-1
        )
        
        # Process QKV
        batch_size, seq_len = qkv.shape[0], qkv.shape[1]
        qkv = qkv.view(batch_size, seq_len, 3, self.heads_num, -1)
        q, k, v = qkv[:, :, 0], qkv[:, :, 1], qkv[:, :, 2]
        
        # Apply QK-Norm
        q = self.q_norm(q)
        k = self.k_norm(k)


        # Split into image and text parts
        img_q, txt_q = q[:, :-txt_len], q[:, -txt_len:]
        img_k, txt_k = k[:, :-txt_len], k[:, -txt_len:]
        
        # Apply rotary embeddings to image parts
        cos, sin = freqs_cis
        img_q, img_k = apply_rotary_pos_emb(img_q, img_k, cos, sin)
        
        # Recombine
        q = torch.cat((img_q, txt_q), dim=1)
        k = torch.cat((img_k, txt_k), dim=1)

        # Run distributed attention
        attn_output, _ = self.attn(q, k, v)
        
        # Process MLP activation
        mlp_output = self.mlp_act(mlp)
        
        # Combine attention and MLP outputs
        combined = torch.cat((attn_output, mlp_output), dim=-1)
        
        # Final projection
        output, _ = self.linear2(combined)
        
        # Apply residual connection with gating using fused operation
        return self.output_residual(x, output, mod_gate)


def apply_rotary_pos_emb(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Apply rotary positional embeddings to query and key tensors.
    
    Args:
        q: Query tensor of shape (batch_size, seq_len, heads_num, head_dim)
        k: Key tensor of shape (batch_size, seq_len, heads_num, head_dim)
        cos: Cosine part of rotary embeddings
        sin: Sine part of rotary embeddings
        
    Returns:
        Tuple containing rotated query and key tensors
    """
    # Reshape cos and sin for broadcasting
    cos = cos.unsqueeze(1)  # (seq_len, 1, dim)
    sin = sin.unsqueeze(1)  # (seq_len, 1, dim)
    
    # Apply rotary embeddings
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    
    return q_embed, k_embed


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """
    Rotate half of the hidden dimensions of the input.
    
    Args:
        x: Input tensor of shape (..., dim)
        
    Returns:
        Tensor with half of its hidden dimensions rotated
    """
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


class HunyuanDiT(nn.Module):
    """
    HunyuanVideo Transformer backbone adapted for distributed training.
    
    This implementation uses distributed attention and linear layers for efficient
    parallel processing across multiple GPUs.
    
    Based on the architecture from:
    - Flux.1: https://github.com/black-forest-labs/flux
    - MMDiT: http://arxiv.org/abs/2403.03206
    """

    def __init__(
        self,
        patch_size: list = [1, 2, 2],
        in_channels: int = 4,
        out_channels: int = None,
        hidden_size: int = 3072,
        heads_num: int = 24,
        mlp_width_ratio: float = 4.0,
        mlp_act_type: str = "gelu_tanh",
        mm_double_blocks_depth: int = 20,
        mm_single_blocks_depth: int = 40,
        rope_dim_list: List[int] = [16, 56, 56],
        qkv_bias: bool = True,
        qk_norm: bool = True,
        qk_norm_type: str = "rms",
        guidance_embed: bool = False,
        text_projection: str = "single_refiner",
        use_attention_mask: bool = True,
        dtype: Optional[torch.dtype] = None,
        text_states_dim: int = 4096,
        text_states_dim_2: int = 768,
        rope_theta: int = 256,
    ):
        super().__init__()

        self.patch_size = patch_size
        self.in_channels = in_channels
        self.out_channels = in_channels if out_channels is None else out_channels
        self.unpatchify_channels = self.out_channels
        self.guidance_embed = guidance_embed
        self.rope_dim_list = rope_dim_list
        self.rope_theta = rope_theta
        self.use_attention_mask = use_attention_mask
        self.text_projection = text_projection
        self.text_states_dim = text_states_dim
        self.text_states_dim_2 = text_states_dim_2

        if hidden_size % heads_num != 0:
            raise ValueError(f"Hidden size {hidden_size} must be divisible by heads_num {heads_num}")
        
        pe_dim = hidden_size // heads_num
        if sum(rope_dim_list) != pe_dim:
            raise ValueError(f"Got {rope_dim_list} but expected positional dim {pe_dim}")
        
        self.hidden_size = hidden_size
        self.heads_num = heads_num

        # Image projection
        self.img_in = PatchEmbed(
            patch_size=self.patch_size,
            in_channels=self.in_channels,
            hidden_size=self.hidden_size,
            dtype=dtype
        )

        # Text projection
        if self.text_projection == "linear":
            self.txt_in = TextProjection(
                self.text_states_dim,
                self.hidden_size,
                dtype=dtype
            )
        elif self.text_projection == "single_refiner":
            self.txt_in = SingleTokenRefiner(
                self.text_states_dim,
                hidden_size,
                heads_num,
                depth=2,
                dtype=dtype
            )
        else:
            raise NotImplementedError(f"Unsupported text_projection: {self.text_projection}")

        # Time modulation
        self.time_in = TimestepEmbedder(
            self.hidden_size,
            act_layer=SiluAndMul(),
            dtype=dtype
        )

        # Text modulation
        self.vector_in = MLPEmbedder(
            self.text_states_dim_2,
            self.hidden_size,
            dtype=dtype
        )

        # Guidance modulation
        self.guidance_in = (
            TimestepEmbedder(self.hidden_size, act_layer=SiluAndMul(), dtype=dtype)
            if guidance_embed else None
        )

        # Double blocks
        self.double_blocks = nn.ModuleList([
            MMDoubleStreamBlock(
                hidden_size,
                heads_num,
                mlp_width_ratio=mlp_width_ratio,
                mlp_act_type=mlp_act_type,
                qk_norm=qk_norm,
                qk_norm_type=qk_norm_type,
                qkv_bias=qkv_bias,
                dtype=dtype,
            ) for _ in range(mm_double_blocks_depth)
        ])

        # Single blocks
        self.single_blocks = nn.ModuleList([
            MMSingleStreamBlock(
                hidden_size,
                heads_num,
                mlp_width_ratio=mlp_width_ratio,
                mlp_act_type=mlp_act_type,
                qk_norm=qk_norm,
                qk_norm_type=qk_norm_type,
                dtype=dtype,
            ) for _ in range(mm_single_blocks_depth)
        ])

        self.final_layer = FinalLayer(
            hidden_size,
            self.patch_size,
            self.out_channels,
            dtype=dtype
        )


    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        timestep: torch.LongTensor,
        encoder_attention_mask: torch.Tensor,
        mask_strategy=None,
        output_features=False,
        output_features_stride=8,
        guidance=None,
    ):
        """
        Forward pass of the HunyuanDiT model.
        
        Args:
            hidden_states: Input image/video latents [B, C, T, H, W]
            encoder_hidden_states: Text embeddings [B, L, D]
            timestep: Diffusion timestep
            encoder_attention_mask: Attention mask for text
            mask_strategy: Strategy for masking attention
            output_features: Whether to output intermediate features
            output_features_stride: Stride for feature output
            guidance: Guidance scale for CFG
            
        Returns:
            Tuple of (output, features_list)
        """
        if guidance is None:
            guidance = torch.tensor([6016.0], device=hidden_states.device, dtype=hidden_states.dtype)
            
        if mask_strategy is None:
            mask_strategy = [[None] * self.heads_num for _ in range(len(self.double_blocks) + len(self.single_blocks))]
        img = x = hidden_states
        text_mask = encoder_attention_mask
        t = timestep
        
        # Split text embeddings - first token is global, rest are per-token
        txt = encoder_hidden_states[:, 1:]
        text_states_2 = encoder_hidden_states[:, 0, :self.text_states_dim_2]
        
        # Get spatial dimensions
        _, _, ot, oh, ow = x.shape
        tt, th, tw = (
            ot // self.patch_size[0],
            oh // self.patch_size[1],
            ow // self.patch_size[2],
        )
        
        # Get rotary embeddings
        freqs_cos, freqs_sin = self.get_rotary_pos_embed((tt, th, tw))
        
        # Prepare modulation vectors
        vec = self.time_in(t)
        
        # Add text modulation
        vec = vec + self.vector_in(text_states_2)
        
        # Add guidance modulation if needed
        if self.guidance_embed and guidance is not None:
            vec = vec + self.guidance_in(guidance)
            
        # Embed image and text
        img = self.img_in(img)
        
        if self.text_projection == "linear":
            txt = self.txt_in(txt)
        elif self.text_projection == "single_refiner":
            txt = self.txt_in(txt, t, text_mask if self.use_attention_mask else None)
            
        txt_seq_len = txt.shape[1]
        img_seq_len = img.shape[1]
        
        freqs_cis = (freqs_cos, freqs_sin) if freqs_cos is not None else None
        
        # Process through double stream blocks
        for index, block in enumerate(self.double_blocks):
            double_block_args = [img, txt, vec, freqs_cis]
            img, txt = block(*double_block_args)
            
        # Merge txt and img to pass through single stream blocks
        x = torch.cat((img, txt), 1)
        
        if output_features:
            features_list = []
            
        # Process through single stream blocks
        if len(self.single_blocks) > 0:
            for index, block in enumerate(self.single_blocks):
                single_block_args = [
                    x,
                    vec,
                    txt_seq_len,
                    freqs_cis,
                ]
                x = block(*single_block_args)
                
                if output_features and index % output_features_stride == 0:
                    features_list.append(x[:, :img_seq_len, ...])
                    
        # Extract image features
        img = x[:, :img_seq_len, ...]
        
        # Final layer processing
        img = self.final_layer(img, vec)
        
        # Unpatchify to get original shape
        img = self.unpatchify(img, tt, th, tw)
        
        if output_features:
            features_list = torch.stack(features_list, dim=0)
        else:
            features_list = None
            
        return img, features_list
        
    def unpatchify(self, x, t, h, w):
        """
        Convert patched representation back to image space.
        
        Args:
            x: Tensor of shape [B, T*H*W, C*P_t*P_h*P_w]
            t, h, w: Temporal and spatial dimensions
            
        Returns:
            Unpatchified tensor of shape [B, C, T*P_t, H*P_h, W*P_w]
        """
        c = self.unpatchify_channels
        pt, ph, pw = self.patch_size
        assert t * h * w == x.shape[1]
        
        x = x.reshape(shape=(x.shape[0], t, h, w, c, pt, ph, pw))
        x = torch.einsum("nthwcopq->nctohpwq", x)
        imgs = x.reshape(shape=(x.shape[0], c, t * pt, h * ph, w * pw))
        
        return imgs




class TextProjection(nn.Module):
    """
    Projects text embeddings from text encoder dimension to model hidden dimension.
    """
    
    def __init__(self, in_channels, hidden_size, dtype=None):
        super().__init__()
        self.linear_1 = ReplicatedLinear(
            in_channels,
            hidden_size,
            bias=True,
            params_dtype=dtype
        )
        self.act_1 = SiluAndMul()
        self.linear_2 = ReplicatedLinear(
            hidden_size,
            hidden_size,
            bias=True,
            params_dtype=dtype
        )
        
    def forward(self, caption):
        hidden_states, _ = self.linear_1(caption)
        hidden_states = self.act_1(hidden_states)
        hidden_states, _ = self.linear_2(hidden_states)
        return hidden_states


class SingleTokenRefiner(nn.Module):
    """
    A token refiner that processes text embeddings with attention to improve
    their representation for cross-attention with image features.
    """
    
    def __init__(
        self,
        in_channels,
        hidden_size,
        heads_num,
        depth=2,
        mlp_width_ratio=4.0,
        mlp_drop_rate=0.0,
        act_type="silu",
        qk_norm=False,
        qk_norm_type="rms",
        qkv_bias=True,
        dtype=None,
    ):
        super().__init__()
        
        # Input projection
        self.input_embedder = ReplicatedLinear(
            in_channels,
            hidden_size,
            bias=True,
            params_dtype=dtype
        )
        
        # Timestep embedding
        self.t_embedder = TimestepEmbedder(
            hidden_size,
            act_layer=SiluAndMul(),
            dtype=dtype
        )
        
        # Context embedding
        self.c_embedder = TextProjection(
            in_channels,
            hidden_size,
            dtype=dtype
        )
        
        # Refiner blocks
        self.refiner_blocks = nn.ModuleList([
            IndividualTokenRefinerBlock(
                hidden_size,
                heads_num,
                mlp_width_ratio=mlp_width_ratio,
                mlp_drop_rate=mlp_drop_rate,
                act_type=act_type,
                qk_norm=qk_norm,
                qk_norm_type=qk_norm_type,
                qkv_bias=qkv_bias,
                dtype=dtype,
            ) for _ in range(depth)
        ])
        
    def forward(self, x, t, mask=None):
        # Get timestep embeddings
        timestep_aware_representations = self.t_embedder(t)
        
        # Get context-aware representations
        if mask is None:
            context_aware_representations = x.mean(dim=1)
        else:
            mask_float = mask.float().unsqueeze(-1)  # [b, s1, 1]
            context_aware_representations = (x * mask_float).sum(dim=1) / mask_float.sum(dim=1)
            
        context_aware_representations = self.c_embedder(context_aware_representations)
        c = timestep_aware_representations + context_aware_representations
        
        # Project input
        x, _ = self.input_embedder(x)
        
        # Process through refiner blocks
        for block in self.refiner_blocks:
            x = block(x, c, mask)
            
        return x


class IndividualTokenRefinerBlock(nn.Module):
    """
    A transformer block for refining individual tokens with self-attention.
    """
    
    def __init__(
        self,
        hidden_size,
        heads_num,
        mlp_width_ratio=4.0,
        mlp_drop_rate=0.0,
        act_type="silu",
        qk_norm=False,
        qk_norm_type="rms",
        qkv_bias=True,
        dtype=None,
    ):
        super().__init__()
        self.heads_num = heads_num
        head_dim = hidden_size // heads_num
        mlp_hidden_dim = int(hidden_size * mlp_width_ratio)
        
        # Normalization and attention
        self.norm1 = RMSNorm(hidden_size, eps=1e-6)
        
        self.self_attn_qkv = ReplicatedLinear(
            hidden_size,
            hidden_size * 3,
            bias=qkv_bias,
            params_dtype=dtype
        )
        
        # QK norm layers
        if qk_norm:
            self.self_attn_q_norm = RMSNorm(head_dim, eps=1e-6) if qk_norm_type == "rms" else nn.LayerNorm(
                head_dim, elementwise_affine=True, eps=1e-6
            )
            self.self_attn_k_norm = RMSNorm(head_dim, eps=1e-6) if qk_norm_type == "rms" else nn.LayerNorm(
                head_dim, elementwise_affine=True, eps=1e-6
            )
        else:
            self.self_attn_q_norm = nn.Identity()
            self.self_attn_k_norm = nn.Identity()
            
        self.self_attn_proj = ReplicatedLinear(
            hidden_size,
            hidden_size,
            bias=qkv_bias,
            params_dtype=dtype
        )
        
        # MLP
        self.norm2 = RMSNorm(hidden_size, eps=1e-6)
        self.mlp = DistributedMLP(
            hidden_size,
            mlp_hidden_dim,
            act_layer_type=act_type,
            bias=True,
            params_dtype=dtype
        )
        
        # Modulation
        self.adaLN_modulation = ModulateProjection(
            hidden_size,
            factor=2,
            act_layer=SiluAndMul(),
            dtype=dtype
        )
        
        # Distributed attention
        self.attn = DistributedAttention(
            dropout_rate=0.0,
            causal=False
        )
        
    def forward(self, x, c, attn_mask=None):
        # Get modulation parameters
        gate_msa, gate_mlp = self.adaLN_modulation(c).chunk(2, dim=-1)
        
        # Self-attention
        norm_x = self.norm1(x)
        qkv, _ = self.self_attn_qkv(norm_x)
        
        batch_size, seq_len = qkv.shape[0], qkv.shape[1]
        qkv = qkv.view(batch_size, seq_len, 3, self.heads_num, -1)
        q, k, v = qkv[:, :, 0], qkv[:, :, 1], qkv[:, :, 2]
        
        # Apply QK-Norm
        q = self.self_attn_q_norm(q)
        k = self.self_attn_k_norm(k)
        
        # Run attention
        attn_output, _ = self.attn(q, k, v, attn_mask=attn_mask)
        
        # Project and apply residual connection with gating
        attn_out, _ = self.self_attn_proj(attn_output)
        x = x + attn_out * gate_msa
        
        # MLP
        mlp_out = self.mlp(self.norm2(x))
        x = x + mlp_out * gate_mlp
        
        return x




class FinalLayer(nn.Module):
    """
    The final layer of DiT that projects features to pixel space.
    """
    
    def __init__(self, hidden_size, patch_size, out_channels, dtype=None):
        super().__init__()
        
        # Normalization
        self.norm_final = RMSNorm(hidden_size, eps=1e-6)
        
        # Output projection
        if isinstance(patch_size, int):
            output_dim = patch_size * patch_size * out_channels
        else:
            output_dim = patch_size[0] * patch_size[1] * patch_size[2] * out_channels
            
        self.linear = ReplicatedLinear(
            hidden_size,
            output_dim,
            bias=True,
            params_dtype=dtype
        )
        
        
        # Modulation
        self.adaLN_modulation = ModulateProjection(
            hidden_size,
            factor=2,
            act_layer=SiluAndMul(),
            dtype=dtype
        )
        
    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=-1)
        x = self.norm_final(x) * (1.0 + scale) + shift
        x, _ = self.linear(x)
        return x

