import math

import torch
import torch.nn as nn

from fastvideo.layers.visual_embedding import TimestepEmbedder, ModulateProjection, timestep_embedding
from fastvideo.platforms import AttentionBackendEnum
from fastvideo.attention import DistributedAttention
from fastvideo.forward_context import set_forward_context
from fastvideo.models.dits.wanvideo import WanImageEmbedding

from fastvideo.models.dits.hyworld.camera_rope import prope_qkv
from fastvideo.layers.rotary_embedding import _apply_rotary_emb
from fastvideo.layers.mlp import MLP

class WanGameActionTimeTextImageEmbedding(nn.Module):
    """
    Embedding module that incorporates action signals in addition to timestep, text, and image embeddings.
    Action embeddings are combined with timestep embeddings before projection.
    """

    def __init__(
        self,
        dim: int,
        time_freq_dim: int,
        text_embed_dim: int,
        image_embed_dim: int | None = None,
    ):
        super().__init__()
        self.dim = dim
        self.time_freq_dim = time_freq_dim

        self.time_embedder = TimestepEmbedder(
            dim, frequency_embedding_size=time_freq_dim, act_layer="silu")
        self.time_modulation = ModulateProjection(dim,
                                                  factor=6,
                                                  act_layer="silu")
        self.text_embedder = MLP(text_embed_dim,
                                 dim,
                                 dim,
                                 bias=True,
                                 act_type="gelu_pytorch_tanh") if text_embed_dim > 0 else None
        
        self.image_embedder = None
        if image_embed_dim is not None:
            self.image_embedder = WanImageEmbedding(image_embed_dim, dim)

        self.action_embedder = MLP(
            time_freq_dim,
            dim,
            dim,
            bias=True,
            act_type="silu"
        )
        # Initialize fc_in with kaiming_uniform (same as nn.Linear default)
        nn.init.kaiming_uniform_(self.action_embedder.fc_in.weight, a=math.sqrt(5))
        # Initialize fc_out with zeros for residual-like behavior
        nn.init.zeros_(self.action_embedder.fc_out.weight)
        if self.action_embedder.fc_out.bias is not None:
            nn.init.zeros_(self.action_embedder.fc_out.bias)

    def forward(
        self,
        timestep: torch.Tensor,
        action: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        encoder_hidden_states_image: torch.Tensor | None = None,
        timestep_seq_len: int | None = None,
    ):
        """
        Args:
            timestep: [B] diffusion timesteps (one per batch sample)
            action: [B, T] action labels (one per frame per batch sample)
            encoder_hidden_states: [B, L, D] text embeddings
        
        Returns:
            temb: [B*T, dim] combined timestep + action embedding
            timestep_proj: [B*T, 6*dim] modulation projection
            encoder_hidden_states: [B, L, dim] processed text embeddings
            encoder_hidden_states_image: [B, L_img, dim] processed image embeddings
        """
        # timestep: [B] -> temb: [B, dim]
        temb = self.time_embedder(timestep, timestep_seq_len)
        
        # Handle action embedding for batch > 1
        # action shape: [B, T] where B=batch_size, T=num_frames
        batch_size = action.shape[0]
        num_frames = action.shape[1]
        
        # Compute action embeddings: [B, T] -> [B*T] -> [B*T, dim]
        action_flat = action.flatten()  # [B*T]
        action_emb = timestep_embedding(action_flat, self.time_freq_dim)
        action_embedder_dtype = next(iter(self.action_embedder.parameters())).dtype
        if (
            action_emb.dtype != action_embedder_dtype
            and action_embedder_dtype != torch.int8
        ):
            action_emb = action_emb.to(action_embedder_dtype)
        action_emb = self.action_embedder(action_emb).type_as(temb)  # [B*T, dim]
        
        # Expand temb to match action_emb: [B, dim] -> [B, T, dim] -> [B*T, dim]
        # Each batch's temb is repeated for all its frames
        temb_expanded = temb.unsqueeze(1).expand(-1, num_frames, -1)  # [B, T, dim]
        temb_expanded = temb_expanded.reshape(batch_size * num_frames, -1)  # [B*T, dim]
        
        # Add action embedding to expanded temb
        temb = temb_expanded + action_emb  # [B*T, dim]

        timestep_proj = self.time_modulation(temb)  # [B*T, 6*dim]

        # Process text embeddings through text_embedder if available
        if self.text_embedder is not None and encoder_hidden_states is not None:
            encoder_hidden_states = self.text_embedder(encoder_hidden_states)
        else:
            encoder_hidden_states = torch.zeros((batch_size, 0, temb.shape[-1]),
                                                device=temb.device,
                                                dtype=temb.dtype)
        
            assert self.image_embedder is not None
            encoder_hidden_states_image = self.image_embedder(
                encoder_hidden_states_image)

        return temb, timestep_proj, encoder_hidden_states, encoder_hidden_states_image

class WanGameActionSelfAttention(nn.Module):
    """
    Self-attention module with support for:
    - Standard RoPE-based attention
    - Camera PRoPE-based attention (when viewmats and Ks are provided)
    - KV caching for autoregressive generation
    """

    def __init__(self,
                 dim: int,
                 num_heads: int,
                 local_attn_size: int = -1,
                 sink_size: int = 0,
                 qk_norm=True,
                 eps=1e-6) -> None:
        assert dim % num_heads == 0
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.local_attn_size = local_attn_size
        self.sink_size = sink_size
        self.qk_norm = qk_norm
        self.eps = eps
        self.max_attention_size = 32760 if local_attn_size == -1 else local_attn_size * 1560

        # Scaled dot product attention (using DistributedAttention for SP support)
        self.attn = DistributedAttention(
            num_heads=num_heads,
            head_size=self.head_dim,
            softmax_scale=None,
            causal=False,
            supported_attention_backends=(AttentionBackendEnum.FLASH_ATTN,
                                          AttentionBackendEnum.TORCH_SDPA))

    def forward(self,
                q: torch.Tensor,
                k: torch.Tensor,
                v: torch.Tensor,
                freqs_cis: tuple[torch.Tensor, torch.Tensor],
                kv_cache: dict | None = None,
                current_start: int = 0,
                cache_start: int | None = None,
                viewmats: torch.Tensor | None = None,
                Ks: torch.Tensor | None = None,
                is_cache: bool = False,
                attention_mask: torch.Tensor | None = None):
        """
        Forward pass with camera PRoPE attention combining standard RoPE and projective positional encoding.
        
        Args:
            q, k, v: Query, key, value tensors [B, L, num_heads, head_dim]
            freqs_cis: RoPE frequency cos/sin tensors
            kv_cache: KV cache dict (may have None values for training)
            current_start: Current position for KV cache
            cache_start: Cache start position
            viewmats: Camera view matrices for PRoPE [B, cameras, 4, 4]
            Ks: Camera intrinsics for PRoPE [B, cameras, 3, 3]
            is_cache: Whether to store to KV cache (for inference)
            attention_mask: Attention mask [B, L] (1 = attend, 0 = mask)
        """
        if cache_start is None:
            cache_start = current_start

        # Apply RoPE manually
        cos, sin = freqs_cis
        query_rope = _apply_rotary_emb(q, cos, sin, is_neox_style=False).type_as(v)
        key_rope = _apply_rotary_emb(k, cos, sin, is_neox_style=False).type_as(v)
        value_rope = v

        # # DEBUG: Check camera matrices
        # if self.training and torch.distributed.get_rank() == 0:
        #     vm_info = f"viewmats={viewmats.shape if viewmats is not None else None}"
        #     ks_info = f"Ks={Ks.shape if Ks is not None else None}"
        #     vm_nonzero = (viewmats != 0).sum().item() if viewmats is not None else 0
        #     ks_nonzero = (Ks != 0).sum().item() if Ks is not None else 0
        #     print(f"[DEBUG] PRoPE input: {vm_info} nonzero={vm_nonzero}, {ks_info} nonzero={ks_nonzero}", flush=True)
        
        # Get PRoPE transformed q, k, v
        query_prope, key_prope, value_prope, apply_fn_o = prope_qkv(
            q.transpose(1, 2),  # [B, num_heads, L, head_dim]
            k.transpose(1, 2),
            v.transpose(1, 2),
            viewmats=viewmats,
            Ks=Ks,
            patches_x=40,  # hardcoded for now
            patches_y=22,
        )
        # PRoPE returns [B, num_heads, L, head_dim], convert to [B, L, num_heads, head_dim]
        query_prope = query_prope.transpose(1, 2)
        key_prope = key_prope.transpose(1, 2)
        value_prope = value_prope.transpose(1, 2)
        
        # # DEBUG: Check prope_qkv output
        # if self.training and torch.distributed.get_rank() == 0:
        #     q_nz = (query_prope != 0).sum().item()
        #     k_nz = (key_prope != 0).sum().item()
        #     v_nz = (value_prope != 0).sum().item()
        #     print(f"[DEBUG] prope_qkv output: q_nonzero={q_nz}, k_nonzero={k_nz}, v_nonzero={v_nz}", flush=True)

        # KV cache handling
        if kv_cache is not None:
            cache_key = kv_cache.get("k", None)
            cache_value = kv_cache.get("v", None)

            if cache_value is not None and not is_cache:
                cache_key_rope, cache_key_prope = cache_key.chunk(2, dim=-1)
                cache_value_rope, cache_value_prope = cache_value.chunk(2, dim=-1)

                key_rope = torch.cat([cache_key_rope, key_rope], dim=1)
                value_rope = torch.cat([cache_value_rope, value_rope], dim=1)
                key_prope = torch.cat([cache_key_prope, key_prope], dim=1)
                value_prope = torch.cat([cache_value_prope, value_prope], dim=1)

            if is_cache:
                # Store to cache (update input dict directly)
                kv_cache["k"] = torch.cat([key_rope, key_prope], dim=-1)
                kv_cache["v"] = torch.cat([value_rope, value_prope], dim=-1)

        # Concatenate rope and prope paths (matching original)
        query_all = torch.cat([query_rope, query_prope], dim=0)
        key_all = torch.cat([key_rope, key_prope], dim=0)
        value_all = torch.cat([value_rope, value_prope], dim=0)

        # Check if Q and KV have different sequence lengths (KV cache mode)
        # In this case, use LocalAttention (supports different Q/KV lengths)
        if query_all.shape[1] != key_all.shape[1]:
            raise ValueError("Q and KV have different sequence lengths")
            # KV cache mode: Q has new tokens only, KV has cached + new tokens
            # Use LocalAttention which supports different Q/KV lengths
            # LocalAttention will use the appropriate backend (SageAttn, FlashAttn, etc.)
            if not hasattr(self, '_kv_cache_attn'):
                from fastvideo.attention import LocalAttention
                self._kv_cache_attn = LocalAttention(
                    num_heads=self.num_heads,
                    head_size=self.head_dim,
                    causal=False,
                    supported_attention_backends=(AttentionBackendEnum.SAGE_ATTN,
                                                  AttentionBackendEnum.FLASH_ATTN,
                                                  AttentionBackendEnum.TORCH_SDPA)
                )
            hidden_states_all = self._kv_cache_attn(query_all, key_all, value_all)
        else:
            # Same sequence length: use DistributedAttention (supports SP)
            # Create default attention mask if not provided
            # NOTE: query_all has shape [2*B, L, ...] (rope+prope concatenated), so mask needs 2*B
            if attention_mask is None:
                batch_size, seq_len = q.shape[0], q.shape[1]
                attention_mask = torch.ones(batch_size * 2, seq_len, device=q.device, dtype=q.dtype)
        
            if q.dtype == torch.float32:
                from fastvideo.attention.backends.sdpa import SDPAMetadataBuilder
                attn_metadata_builder = SDPAMetadataBuilder
            else:
                from fastvideo.attention.backends.flash_attn import FlashAttnMetadataBuilder
                attn_metadata_builder = FlashAttnMetadataBuilder
            attn_metadata = attn_metadata_builder().build(
                current_timestep=0,
                attn_mask=attention_mask,
            )
            with set_forward_context(current_timestep=0, attn_metadata=attn_metadata):
                hidden_states_all, _ = self.attn(query_all, key_all, value_all, attention_mask=attention_mask)

        hidden_states_rope, hidden_states_prope = hidden_states_all.chunk(2, dim=0)
        
        # # DEBUG: Check attention output and apply_fn_o
        # if self.training and torch.distributed.get_rank() == 0:
        #     attn_all_nz = (hidden_states_all != 0).sum().item()
        #     rope_nz = (hidden_states_rope != 0).sum().item()
        #     prope_before = (hidden_states_prope != 0).sum().item()
        #     print(f"[DEBUG] attn output: all_nonzero={attn_all_nz}, rope_nonzero={rope_nz}, prope_before_apply={prope_before}", flush=True)
        
        hidden_states_prope = apply_fn_o(hidden_states_prope.transpose(1, 2)).transpose(1, 2)
        
        # # DEBUG: Check after apply_fn_o
        # if self.training and torch.distributed.get_rank() == 0:
        #     prope_after = (hidden_states_prope != 0).sum().item()
        #     print(f"[DEBUG] prope_after_apply_fn_o={prope_after}", flush=True)

        return hidden_states_rope, hidden_states_prope