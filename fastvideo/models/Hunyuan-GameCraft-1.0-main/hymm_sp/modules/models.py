from typing import List, Tuple, Optional, Union, Dict
from einops import rearrange

import torch, os
import torch.nn as nn
import torch.nn.functional as F
from diffusers.models import ModelMixin
from diffusers.configuration_utils import ConfigMixin, register_to_config
from flash_attn.flash_attn_interface import flash_attn_varlen_func

from .activation_layers import get_activation_layer
from .norm_layers import get_norm_layer
from .embed_layers import TimestepEmbedder, PatchEmbed, TextProjection
from .attn_layers import apply_rotary_emb
from .mlp_layers import MLP, MLPEmbedder, FinalLayer
from .modulate_layers import ModulateDiT, modulate, apply_gate
from .token_refiner import SingleTokenRefiner
from .cameranet import CameraNet

from .parallel_states import (
    nccl_info,
    get_cu_seqlens,
    get_sequence_parallel_state,
    parallel_attention,
    all_gather,
)

CPU_OFFLOAD = int(os.environ.get("CPU_OFFLOAD", 0))
DISABLE_SP = int(os.environ.get("DISABLE_SP", 0))
print(f'models: cpu_offload={CPU_OFFLOAD}, DISABLE_SP={DISABLE_SP}')

class DoubleStreamBlock(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        mlp_width_ratio: float,
        mlp_act_type: str = 'gelu_tanh',
        qk_norm: bool = True,
        qk_norm_type: str = 'rms',
        qkv_bias: bool = False,
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None,
    ):
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()

        self.deterministic = False
        self.num_heads = num_heads
        head_dim = hidden_size // num_heads
        mlp_hidden_dim = int(hidden_size * mlp_width_ratio)

        self.img_mod = ModulateDiT(hidden_size, factor=6, act_layer=get_activation_layer("silu"), **factory_kwargs)
        self.img_norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6, **factory_kwargs)

        self.img_attn_qkv = nn.Linear(hidden_size, hidden_size * 3, bias=qkv_bias, **factory_kwargs)
        qk_norm_layer = get_norm_layer(qk_norm_type)
        self.img_attn_q_norm = (
            qk_norm_layer(head_dim, elementwise_affine=True, eps=1e-6, **factory_kwargs)
            if qk_norm
            else nn.Identity()
        )
        self.img_attn_k_norm = (
            qk_norm_layer(head_dim, elementwise_affine=True, eps=1e-6, **factory_kwargs)
            if qk_norm
            else nn.Identity()
        )
        self.img_attn_proj = nn.Linear(hidden_size, hidden_size, bias=qkv_bias, **factory_kwargs)

        self.img_norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6, **factory_kwargs)
        self.img_mlp = MLP(
            hidden_size,
            mlp_hidden_dim,
            act_layer=get_activation_layer(mlp_act_type),
            bias=True,
            **factory_kwargs
        )

        self.txt_mod = ModulateDiT(hidden_size, factor=6, act_layer=get_activation_layer("silu"), **factory_kwargs)
        self.txt_norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6, **factory_kwargs)

        self.txt_attn_qkv = nn.Linear(hidden_size, hidden_size * 3, bias=qkv_bias, **factory_kwargs)
        qk_norm_layer = get_norm_layer(qk_norm_type)
        self.txt_attn_q_norm = (
            qk_norm_layer(head_dim, elementwise_affine=True, eps=1e-6, **factory_kwargs)
            if qk_norm
            else nn.Identity()
        )
        self.txt_attn_k_norm = (
            qk_norm_layer(head_dim, elementwise_affine=True, eps=1e-6, **factory_kwargs)
            if qk_norm
            else nn.Identity()
        )
        self.txt_attn_proj = nn.Linear(hidden_size, hidden_size, bias=qkv_bias, **factory_kwargs)

        self.txt_norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6, **factory_kwargs)
        self.txt_mlp = MLP(
            hidden_size,
            mlp_hidden_dim,
            act_layer=get_activation_layer(mlp_act_type),
            bias=True,
            **factory_kwargs
        )

    def enable_deterministic(self):
        self.deterministic = True

    def disable_deterministic(self):
        self.deterministic = False

    def forward(
        self,
        img: torch.Tensor,
        txt: torch.Tensor,
        vec: torch.Tensor,
        cu_seqlens_q: Optional[torch.Tensor] = None,
        cu_seqlens_kv: Optional[torch.Tensor] = None,
        max_seqlen_q: Optional[int] = None,
        max_seqlen_kv: Optional[int] = None,
        freqs_cis: tuple = None,
        use_sage: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        img_mod1_shift, img_mod1_scale, img_mod1_gate, img_mod2_shift, img_mod2_scale, img_mod2_gate = (
            self.img_mod(vec).chunk(6, dim=-1)
        )
        txt_mod1_shift, txt_mod1_scale, txt_mod1_gate, txt_mod2_shift, txt_mod2_scale, txt_mod2_gate = (
            self.txt_mod(vec).chunk(6, dim=-1)
        )
        if CPU_OFFLOAD: torch.cuda.empty_cache()

        # Prepare image for attention.
        img_modulated = self.img_norm1(img)
        img_modulated = modulate(img_modulated, shift=img_mod1_shift, scale=img_mod1_scale)
        img_qkv = self.img_attn_qkv(img_modulated)
        if CPU_OFFLOAD: torch.cuda.empty_cache()
        img_q, img_k, img_v = rearrange(img_qkv, "B L (K H D) -> K B L H D", K=3, H=self.num_heads)
        # Apply QK-Norm if needed
        img_q = self.img_attn_q_norm(img_q).to(img_v)
        img_k = self.img_attn_k_norm(img_k).to(img_v)
        if CPU_OFFLOAD: torch.cuda.empty_cache()

        # Apply RoPE if needed.
        if freqs_cis is not None:
            img_qq, img_kk = apply_rotary_emb(img_q, img_k, freqs_cis, head_first=False)
            assert img_qq.shape == img_q.shape and img_kk.shape == img_k.shape, \
                f'img_kk: {img_qq.shape}, img_q: {img_q.shape}, img_kk: {img_kk.shape}, img_k: {img_k.shape}'
            img_q, img_k = img_qq, img_kk

        # Prepare txt for attention.
        txt_modulated = self.txt_norm1(txt)
        txt_modulated = modulate(txt_modulated, shift=txt_mod1_shift, scale=txt_mod1_scale)
        if CPU_OFFLOAD: torch.cuda.empty_cache()
        txt_qkv = self.txt_attn_qkv(txt_modulated)
        txt_q, txt_k, txt_v = rearrange(txt_qkv, "B L (K H D) -> K B L H D", K=3, H=self.num_heads)
        # Apply QK-Norm if needed.
        txt_q = self.txt_attn_q_norm(txt_q).to(txt_v)
        txt_k = self.txt_attn_k_norm(txt_k).to(txt_v)
        if CPU_OFFLOAD: torch.cuda.empty_cache()

        # Run actual attention.
        q = torch.cat((img_q, txt_q), dim=1)
        k = torch.cat((img_k, txt_k), dim=1)
        v = torch.cat((img_v, txt_v), dim=1)

        # Compute attention.
        if DISABLE_SP:
            assert cu_seqlens_q.shape[0] == 2 * img.shape[0] + 1

            q, k, v = [
                x.view(x.shape[0] * x.shape[1], *x.shape[2:])
                for x in [q, k, v]
            ]
            attn = flash_attn_varlen_func(
                q,
                k,
                v,
                cu_seqlens_q,
                cu_seqlens_kv,
                max_seqlen_q,
                max_seqlen_kv,
            )
            attn = attn.view(img_k.shape[0], max_seqlen_q, -1).contiguous()
        else:
            attn, _ = parallel_attention(
                (img_q, txt_q),
                (img_k, txt_k),
                (img_v, txt_v),
                img_q_len=img_q.shape[1],
                img_kv_len=img_k.shape[1],
                cu_seqlens_q=cu_seqlens_q,
                cu_seqlens_kv=cu_seqlens_kv,
                max_seqlen_q=max_seqlen_q,
                max_seqlen_kv=max_seqlen_kv,
                use_sage=use_sage,
            )
        img_attn, txt_attn = attn[:, :img.shape[1]], attn[:, img.shape[1]:]

        if CPU_OFFLOAD: torch.cuda.empty_cache()

        # Calculate the img bloks.
        img = img + apply_gate(self.img_attn_proj(img_attn), gate=img_mod1_gate)
        img = img + apply_gate(self.img_mlp(modulate(
                                            self.img_norm2(img), 
                                            shift=img_mod2_shift, 
                                            scale=img_mod2_scale)), gate=img_mod2_gate)
        if CPU_OFFLOAD: torch.cuda.empty_cache()
        # Calculate the txt bloks.
        txt = txt + apply_gate(self.txt_attn_proj(txt_attn), gate=txt_mod1_gate)
        txt = txt + apply_gate(self.txt_mlp(modulate(self.txt_norm2(txt), 
                                                     shift=txt_mod2_shift, 
                                                     scale=txt_mod2_scale)), gate=txt_mod2_gate)
        if CPU_OFFLOAD: torch.cuda.empty_cache()
        return img, txt


class SingleStreamBlock(nn.Module):
    """
    A DiT block with parallel linear layers as described in
    https://arxiv.org/abs/2302.05442 and adapted modulation interface.
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        mlp_width_ratio: float = 4.0,
        mlp_act_type: str = 'gelu_tanh',
        qk_norm: bool = True,
        qk_norm_type: str = 'rms',
        qk_scale: float = None,
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None,
    ):
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()

        self.deterministic = False
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        head_dim = hidden_size // num_heads
        mlp_hidden_dim = int(hidden_size * mlp_width_ratio)
        self.mlp_hidden_dim = mlp_hidden_dim
        self.scale = qk_scale or head_dim**-0.5

        # qkv and mlp_in
        self.linear1 = nn.Linear(hidden_size, hidden_size * 3 + mlp_hidden_dim, **factory_kwargs)
        # proj and mlp_out
        self.linear2 = nn.Linear(hidden_size + mlp_hidden_dim, hidden_size, **factory_kwargs)

        qk_norm_layer = get_norm_layer(qk_norm_type)
        self.q_norm = (
            qk_norm_layer(head_dim, elementwise_affine=True, eps=1e-6, **factory_kwargs)
            if qk_norm
            else nn.Identity()
        )
        self.k_norm = (
            qk_norm_layer(head_dim, elementwise_affine=True, eps=1e-6, **factory_kwargs)
            if qk_norm
            else nn.Identity()
        )

        self.pre_norm = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6, **factory_kwargs)

        self.mlp_act = get_activation_layer(mlp_act_type)()
        self.modulation = ModulateDiT(hidden_size, factor=3, act_layer=get_activation_layer("silu"), **factory_kwargs)

    def enable_deterministic(self):
        self.deterministic = True

    def disable_deterministic(self):
        self.deterministic = False

    def forward(
        self,
        x: torch.Tensor,
        vec: torch.Tensor,
        txt_len: int,
        cu_seqlens_q: Optional[torch.Tensor] = None,
        cu_seqlens_kv: Optional[torch.Tensor] = None,
        max_seqlen_q: Optional[int] = None,
        max_seqlen_kv: Optional[int] = None,
        freqs_cis: Tuple[torch.Tensor, torch.Tensor] = None,
        use_sage: bool = True,
    ) -> torch.Tensor:
        mod_shift, mod_scale, mod_gate = (
            self.modulation(vec).chunk(3, dim=-1)
        )
        x_mod = modulate(self.pre_norm(x), shift=mod_shift, scale=mod_scale)
        if CPU_OFFLOAD: torch.cuda.empty_cache()
        qkv, mlp = torch.split(self.linear1(x_mod), [3 * self.hidden_size, self.mlp_hidden_dim], dim=-1)

        q, k, v = rearrange(qkv, "B L (K H D) -> K B L H D", K=3, H=self.num_heads)
        if CPU_OFFLOAD: torch.cuda.empty_cache()
        
        # Apply QK-Norm if needed.
        q = self.q_norm(q).to(v)
        k = self.k_norm(k).to(v)
        if CPU_OFFLOAD: torch.cuda.empty_cache()

        # Apply RoPE if needed.
        if freqs_cis is not None:
            img_q, txt_q = q[:, :-txt_len, :, :], q[:, -txt_len:, :, :]
            img_k, txt_k = k[:, :-txt_len, :, :], k[:, -txt_len:, :, :]
            img_qq, img_kk = apply_rotary_emb(img_q, img_k, freqs_cis, head_first=False)
            assert img_qq.shape == img_q.shape and img_kk.shape == img_k.shape, \
                f'img_kk: {img_qq.shape}, img_q: {img_q.shape}, img_kk: {img_kk.shape}, img_k: {img_k.shape}'
            img_q, img_k = img_qq, img_kk
            q = torch.cat((img_q, txt_q), dim=1)
            k = torch.cat((img_k, txt_k), dim=1)

        if CPU_OFFLOAD: torch.cuda.empty_cache()

        # Compute attention.
        if DISABLE_SP:
            assert cu_seqlens_q.shape[0] == 2 * x.shape[0] + 1, \
                f"cu_seqlens_q.shape:{cu_seqlens_q.shape}, x.shape[0]:{x.shape[0]}"
            # [b, s+l, a, d] -> [s+l, b, a, d]
            q, k, v = [
                x.view(x.shape[0] * x.shape[1], *x.shape[2:])
                for x in [q, k, v]
            ]

            attn = flash_attn_varlen_func(
                q,
                k,
                v,
                cu_seqlens_q,
                cu_seqlens_kv,
                max_seqlen_q,
                max_seqlen_kv,
            )
            attn = attn.view(x.shape[0], max_seqlen_q, -1).contiguous()
        else:
            img_v, txt_v = v[:, :-txt_len, :, :], v[:, -txt_len:, :, :]
            attn, _ = parallel_attention(
                (img_q, txt_q),
                (img_k, txt_k),
                (img_v, txt_v),
                img_q_len=img_q.shape[1],
                img_kv_len=img_k.shape[1],
                cu_seqlens_q=cu_seqlens_q,
                cu_seqlens_kv=cu_seqlens_kv,
                max_seqlen_q=max_seqlen_q,
                max_seqlen_kv=max_seqlen_kv,
                use_sage=use_sage,
            )
        if CPU_OFFLOAD:
            torch.cuda.empty_cache()
            tmp = torch.cat((attn, self.mlp_act(mlp)), 2)
            torch.cuda.empty_cache()
            output = self.linear2(tmp)
        else:
            output = self.linear2(torch.cat((attn, self.mlp_act(mlp)), 2))
        return x + apply_gate(output, gate=mod_gate)


class HYVideoDiffusionTransformer(ModelMixin, ConfigMixin):
    """
    HunyuanVideo Transformer backbone

    Inherited from ModelMixin and ConfigMixin for compatibility with diffusers' sampler StableDiffusionPipeline.
    
    Reference:
    [1] Flux.1: https://github.com/black-forest-labs/flux
    [2] MMDiT: http://arxiv.org/abs/2403.03206,
               https://github.com/huggingface/diffusers/blob/main/src/diffusers/pipelines/stable_diffusion_3/pipeline_stable_diffusion_3.py

    """
    @register_to_config
    def __init__(
        self,
        args,
        patch_size: list = [1,2,2],
        in_channels: int = 4, # Should be VAE.config.latent_channels.
        out_channels: int = None,
        hidden_size: int = 3072,
        mlp_width_ratio: float = 4.0,
        mlp_act_type: str = 'gelu_tanh',
        num_heads: int = 24,
        depth_double_blocks: int = 19,
        depth_single_blocks: int = 38,
        rope_dim_list: List[int] = [16, 56, 56],
        qkv_bias: bool = True,
        qk_norm: bool = True,
        qk_norm_type: str = 'rms',
        guidance_embed: bool = False, # For modulation.
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None,
        multitask_mask_training_type: Optional[str] = None,
        camera_in_channels: int = 6,
        camera_down_coef: int = 8,
    ):
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()

        # Text projection. Default to linear projection.
        # Alternative: TokenRefiner. See more details (LI-DiT): http://arxiv.org/abs/2406.11831
        self.text_projection = args.text_projection
        self.text_states_dim = args.text_states_dim
        self.use_attention_mask = args.use_attention_mask
        self.text_states_dim_2 = args.text_states_dim_2

        # Now we only use above configs from args.
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.out_channels = in_channels if out_channels is None else out_channels
        self.unpatchify_channels = self.out_channels
        self.guidance_embed = guidance_embed
        self.rope_dim_list = rope_dim_list
        self.multitask_mask_training_type = multitask_mask_training_type

        if hidden_size % num_heads != 0:
            raise ValueError(
                f"Hidden size {hidden_size} must be divisible by num_heads {num_heads}"
            )
        pe_dim = hidden_size // num_heads
        if sum(rope_dim_list) != pe_dim:
            raise ValueError(f"Got {rope_dim_list} but expected positional dim {pe_dim}")
        self.hidden_size = hidden_size
        self.num_heads = num_heads
    
        # image projection
        self.img_in = PatchEmbed(
            self.patch_size, self.in_channels, self.hidden_size, self.multitask_mask_training_type, **factory_kwargs
        )

        # text projection
        if self.text_projection == "linear":
            self.txt_in = TextProjection(
                self.text_states_dim,
                self.hidden_size,
                get_activation_layer("silu"),
                **factory_kwargs
            )
        elif self.text_projection == "single_refiner":
            self.txt_in = SingleTokenRefiner(
                self.text_states_dim, hidden_size, num_heads, depth=2, **factory_kwargs
            )
        else:
            raise NotImplementedError(f"Unsupported text_projection: {self.text_projection}")

        # time modulation
        self.time_in = TimestepEmbedder(
            self.hidden_size, get_activation_layer("silu"), **factory_kwargs
        )

        # text modulation
        self.vector_in = MLPEmbedder(
            self.text_states_dim_2, self.hidden_size, **factory_kwargs
        )

        # guidance modulation
        self.guidance_in = TimestepEmbedder(
            self.hidden_size, get_activation_layer("silu"), **factory_kwargs
        ) if guidance_embed else None

        # double blocks
        self.double_blocks = nn.ModuleList(
            [
                DoubleStreamBlock(
                    self.hidden_size,
                    self.num_heads,
                    mlp_width_ratio=mlp_width_ratio,
                    mlp_act_type=mlp_act_type,
                    qk_norm=qk_norm,
                    qk_norm_type=qk_norm_type,
                    qkv_bias=qkv_bias,
                    **factory_kwargs
                )
                for _ in range(depth_double_blocks)
            ]
        )

        # single blocks
        self.single_blocks = nn.ModuleList(
            [
                SingleStreamBlock(
                    self.hidden_size,
                    self.num_heads,
                    mlp_width_ratio=mlp_width_ratio,
                    mlp_act_type=mlp_act_type,
                    qk_norm=qk_norm,
                    qk_norm_type=qk_norm_type,
                    **factory_kwargs
                )
                for _ in range(depth_single_blocks)
            ]
        )

        self.final_layer = FinalLayer(
            self.hidden_size,
            self.patch_size,
            self.out_channels,
            get_activation_layer("silu"),
            **factory_kwargs
        )
        
        self.camera_net = CameraNet(in_channels=camera_in_channels, 
                                    out_channels=in_channels, 
                                    downscale_coef=camera_down_coef,
                                    patch_size=self.patch_size,
                                    hidden_size=self.hidden_size,
                                    )

    def enable_deterministic(self):
        for block in self.double_blocks:
            block.enable_deterministic()
        for block in self.single_blocks:
            block.enable_deterministic()

    def disable_deterministic(self):
        for block in self.double_blocks:
            block.disable_deterministic()
        for block in self.single_blocks:
            block.disable_deterministic()

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor, # Should be in range(0, 1000).
        text_states: torch.Tensor = None,
        text_mask: torch.Tensor = None, # Now we don't use it.
        text_states_2: Optional[torch.Tensor] = None, # Text embedding for modulation.
        freqs_cos: Optional[torch.Tensor] = None,
        freqs_sin: Optional[torch.Tensor] = None,
        guidance: torch.Tensor = None, # Guidance for modulation, should be cfg_scale x 1000.
        return_dict: bool = True,
        is_cache: bool = False,
        cam_latents = None,
        use_sage: bool = False,
    ) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        out = {}
        img = x
        txt = text_states
        _, _, ot, oh, ow = x.shape
        tt, th, tw = ot // self.patch_size[0], oh // self.patch_size[1], ow // self.patch_size[2]

        # Prepare modulation vectors.
        vec = self.time_in(t)

        # text modulation
        vec = vec + self.vector_in(text_states_2)

        # guidance modulation
        if self.guidance_embed:
            if guidance is None:
                raise ValueError("Didn't get guidance strength for guidance distilled model.")
            else:
                # our timestep_embedding is merged into guidance_in(TimestepEmbedder)
                vec = vec + self.guidance_in(guidance)

        if CPU_OFFLOAD: torch.cuda.empty_cache()
        
        camera_condition = cam_latents
        assert camera_condition is not None, print("plucker_embedding is not provided")
        latent_len = img.shape[2]
        
        
        # Embed image and text.
        img = self.img_in(img)
        # ref_latents = self.img_in(ref_latents) # off in latent concat
        if self.text_projection == "linear":
            txt = self.txt_in(txt)
        elif self.text_projection == "single_refiner":
            txt = self.txt_in(txt, t, text_mask if self.use_attention_mask else None)
        else:
            raise NotImplementedError(f"Unsupported text_projection: {self.text_projection}")
        
        if camera_condition is not None:

            if latent_len == 18:
                camera_latents = torch.cat([self.camera_net(torch.zeros_like(camera_condition)), \
                                            self.camera_net(camera_condition)], dim=1)
            elif latent_len == 9:
                camera_latents = self.camera_net(camera_condition)
            elif latent_len == 10:
                camera_latents = torch.cat([self.camera_net(torch.zeros_like(camera_condition)[:,0:4,:,:,:]), \
                                            self.camera_net(camera_condition)], dim=1)
            img = img + camera_latents
        
        if CPU_OFFLOAD: torch.cuda.empty_cache()
        # ref_length = ref_latents.shape[-2]
        # img = torch.cat([ref_latents, img], dim=-2) # t c
        txt_seq_len = txt.shape[1]
        img_seq_len = img.shape[1]
        # Compute 'self-attention mask'.

        cu_seqlens_q = get_cu_seqlens(text_mask, img_seq_len)
        cu_seqlens_kv = cu_seqlens_q
        max_seqlen_q = img_seq_len + txt_seq_len
        max_seqlen_kv = max_seqlen_q

        if get_sequence_parallel_state():
            sp_size = nccl_info.sp_size
            sp_rank = nccl_info.rank_within_group
            assert img.shape[1] % sp_size == 0, f"Cannot split video sequence into ulysses SP ({sp_size}) parts evenly"
            img = torch.chunk(img, sp_size, dim=1)[sp_rank]
            freqs_cos = torch.chunk(freqs_cos, sp_size, dim=0)[sp_rank]
            freqs_sin = torch.chunk(freqs_sin, sp_size, dim=0)[sp_rank]

        if CPU_OFFLOAD: torch.cuda.empty_cache()
        freqs_cis = (freqs_cos, freqs_sin) if freqs_cos is not None else None
        # --------------------- Pass through DiT blocks ------------------------
        if not is_cache:
            for layer_num, block in enumerate(self.double_blocks):
                double_block_args = [img, txt, vec, cu_seqlens_q, cu_seqlens_kv, \
                                     max_seqlen_q, max_seqlen_kv, freqs_cis, use_sage]
                img, txt = block(*double_block_args)
                if CPU_OFFLOAD: torch.cuda.empty_cache()

            # Merge txt and img to pass through single stream blocks.
            x = torch.cat((img, txt), 1)
            # Compatible with MMDiT.
            if len(self.single_blocks) > 0:
                for layer_num, block in enumerate(self.single_blocks):
                    if layer_num == (len(self.single_blocks) - 1):
                        self.cache_out = x
                    single_block_args = [x, vec, txt_seq_len, cu_seqlens_q, \
                        cu_seqlens_kv, max_seqlen_q, max_seqlen_kv, (freqs_cos, freqs_sin), use_sage]
                    x = block(*single_block_args)
                    if CPU_OFFLOAD: torch.cuda.empty_cache()
        else:
            x = self.cache_out
            if len(self.single_blocks) > 0:
                for layer_num, block in enumerate(self.single_blocks):
                    if layer_num < (len(self.single_blocks) - 1):
                        continue
                    single_block_args = [x, vec, txt_seq_len, cu_seqlens_q, \
                        cu_seqlens_kv, max_seqlen_q, max_seqlen_kv, (freqs_cos, freqs_sin), use_sage]
                    x = block(*single_block_args)
                    if CPU_OFFLOAD: torch.cuda.empty_cache()

        img = x[:, :-txt_seq_len, ...]

        if get_sequence_parallel_state():
            img = all_gather(img, dim=1) 

        # img = img[:, ref_length:]
        # ---------------------------- Final layer ------------------------------
        img = self.final_layer(img, vec)  # (N, T, patch_size ** 2 * out_channels)
        img = self.unpatchify(img, tt, th, tw)
        
        if return_dict:
            out['x'] = img
            return out
        return img

    def unpatchify(self, x, t, h, w):
        """
        x: (N, T, patch_size**2 * C)
        imgs: (N, H, W, C)
        """
        c = self.unpatchify_channels
        pt, ph, pw = self.patch_size
        assert t * h * w == x.shape[1]

        x = x.reshape(shape=(x.shape[0], t, h, w, c, pt, ph, pw))
        x = torch.einsum('nthwcopq->nctohpwq', x)
        imgs = x.reshape(shape=(x.shape[0], c, t * pt, h * ph, w * pw))

        return imgs
    
    def params_count(self):
        counts = {
            "double": sum([
                sum(p.numel() for p in block.img_attn_qkv.parameters()) +
                sum(p.numel() for p in block.img_attn_proj.parameters()) +
                sum(p.numel() for p in block.img_mlp.parameters()) +
                sum(p.numel() for p in block.txt_attn_qkv.parameters()) +
                sum(p.numel() for p in block.txt_attn_proj.parameters()) +
                sum(p.numel() for p in block.txt_mlp.parameters())
                for block in self.double_blocks
            ]),
            "single": sum([
                sum(p.numel() for p in block.linear1.parameters()) +
                sum(p.numel() for p in block.linear2.parameters())
                for block in self.single_blocks
            ]),
            "total": sum(p.numel() for p in self.parameters()),
        }
        counts["attn+mlp"] = counts["double"] + counts["single"]
        return counts

#################################################################################
#                             HunyuanVideo Configs                              #
#################################################################################

HUNYUAN_VIDEO_CONFIG = {                                                                   # Attn+MLP / Total
    'HYVideo-T/2': {                                                                       #   9.0B   / 12.5B
        'depth_double_blocks': 20,
        'depth_single_blocks': 40,
        'rope_dim_list': [16, 56, 56],
        'hidden_size': 3072,
        'num_heads': 24,
        'mlp_width_ratio': 4,
    },
}
