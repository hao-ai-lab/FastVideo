# SPDX-License-Identifier: Apache-2.0
from abc import ABC, abstractmethod
from typing import List, Optional, Tuple, Union

import torch
from torch import nn
import numpy as np

from fastvideo.v1.configs.models import DiTConfig
from fastvideo.v1.platforms import _Backend


# TODO
class BaseDiT(nn.Module, ABC):
    _fsdp_shard_conditions: list = []
    _param_names_mapping: dict
    hidden_size: int
    num_attention_heads: int
    num_channels_latents: int
    # always supports torch_sdpa
    _supported_attention_backends: Tuple[
        _Backend, ...] = DiTConfig()._supported_attention_backends

    def __init_subclass__(cls) -> None:
        required_class_attrs = [
            "_fsdp_shard_conditions", "_param_names_mapping"
        ]
        super().__init_subclass__()
        for attr in required_class_attrs:
            if not hasattr(cls, attr):
                raise AttributeError(
                    f"Subclasses of BaseDiT must define '{attr}' class variable"
                )

    def __init__(self, config: DiTConfig, **kwargs) -> None:
        super().__init__()
        self.config = config
        if not self.supported_attention_backends:
            raise ValueError(
                f"Subclass {self.__class__.__name__} must define _supported_attention_backends"
            )

    @abstractmethod
    def forward(self,
                hidden_states: torch.Tensor,
                encoder_hidden_states: Union[torch.Tensor, List[torch.Tensor]],
                timestep: torch.LongTensor,
                encoder_hidden_states_image: Optional[Union[
                    torch.Tensor, List[torch.Tensor]]] = None,
                guidance=None,
                **kwargs) -> torch.Tensor:
        pass

    def __post_init__(self) -> None:
        required_attrs = [
            "hidden_size", "num_attention_heads", "num_channels_latents"
        ]
        for attr in required_attrs:
            if not hasattr(self, attr):
                raise AttributeError(
                    f"Subclasses of BaseDiT must define '{attr}' instance variable"
                )

    @property
    def supported_attention_backends(self) -> Tuple[_Backend, ...]:
        return self._supported_attention_backends

from fastvideo.v1.layers.rotary_embedding import (_apply_rotary_emb,
                                                  get_rotary_pos_embed)
from fastvideo.v1.distributed.parallel_state import (
    get_sequence_model_parallel_world_size)

class TeaCacheBaseDiT(BaseDiT):
    """
    An intermediate base class that adds TeaCache optimization functionality to DiT models.
    TeaCache accelerates inference by selectively skipping redundant computation when consecutive
    diffusion steps are similar enough.
    """
    # These are required class attributes that should be overridden by concrete implementations
    _fsdp_shard_conditions = []
    _param_names_mapping = {}
    # Ensure these instance attributes are properly defined in subclasses
    hidden_size: int
    num_attention_heads: int
    num_channels_latents: int
    # always supports torch_sdpa
    _supported_attention_backends: Tuple[
        _Backend, ...] = DiTConfig()._supported_attention_backends

    def __init__(self, config: DiTConfig, **kwargs) -> None:
        super().__init__(config, **kwargs)

        self.enable_teacache = False
        if self.config.cache_config and self.config.cache_config.enable_teacache:
            self.enable_teacache = True
            self.cnt = 0
            self.teacache_thresh = self.config.cache_config.teacache_thresh
            self.use_ret_steps = self.config.cache_config.use_ret_steps
            self.ret_steps = self.config.cache_config.ret_steps
            self.cutoff_steps = self.config.cache_config.cutoff_steps
            self.num_steps = self.config.cache_config.num_steps
            self.coefficients = self.config.cache_config.coefficients
            self.is_even = False
            self.previous_e0_even = None
            self.previous_e0_odd = None
            self.previous_residual_even = None
            self.previous_residual_odd = None
            self.accumulated_rel_l1_distance_even = 0
            self.accumulated_rel_l1_distance_odd = 0

    # def enable_teacache_optimization(
    #     self, 
    #     threshold: float = 0.2, 
    #     use_ret_steps: bool = False, 
    #     ret_steps: int = 10, 
    #     cutoff_steps: Optional[int] = None,
    #     num_steps: Optional[int] = None,
    #     coefficients: Optional[List[float]] = None
    # ) -> None:
    #     """
    #     Enable TeaCache optimization with configurable parameters.
        
    #     Args:
    #         threshold: Threshold for relative L1 distance. Higher values give more speedup
    #                   but potentially lower quality. Recommended: 0.1 for 2x speedup, 0.2 for 3x speedup.
    #         use_ret_steps: Whether to use retention steps for better quality.
    #         ret_steps: Number of initial steps to always calculate (x2 for cond/uncond).
    #         cutoff_steps: Number of steps after which to always calculate.
    #         num_steps: Total number of steps (cond + uncond) for the schedule.
    #         coefficients: Polynomial coefficients for scaling the L1 distance.
    #     """
    #     self.enable_teacache = True
    #     self.teacache_thresh = threshold
    #     self.use_ret_steps = use_ret_steps
    #     self.ret_steps = ret_steps
    #     self.cutoff_steps = cutoff_steps
    #     self.num_steps = num_steps
    #     self.coefficients = coefficients or [-114.36346466, 65.26524496, -18.82220707, 4.91518089, -0.23412683]

    def forward(self,
                hidden_states: torch.Tensor,
                encoder_hidden_states: Union[torch.Tensor, List[torch.Tensor]],
                timestep: torch.LongTensor,
                encoder_hidden_states_image: Optional[Union[
                    torch.Tensor, List[torch.Tensor]]] = None,
                guidance=None,
                **kwargs) -> torch.Tensor:
        """
        Forward pass with TeaCache optimization support.
        
        This method wraps the actual implementation (forward_impl) with TeaCache logic
        to selectively skip computations when consecutive timesteps are similar.
        """
        # if not self.enable_teacache:
        #     return self.forward_impl(
        #         hidden_states, 
        #         encoder_hidden_states, 
        #         timestep, 
        #         encoder_hidden_states_image, 
        #         guidance, 
        #         **kwargs
        #     self.config.cache_config is None or not self.config.cache_config.enable_teacache:)
        # print(' input hidden_states.shape', hidden_states.shape)
        orig_dtype = hidden_states.dtype
        if not isinstance(encoder_hidden_states, torch.Tensor):
            encoder_hidden_states = encoder_hidden_states[0]
        if isinstance(encoder_hidden_states_image,
                      list) and len(encoder_hidden_states_image) > 0:
            encoder_hidden_states_image = encoder_hidden_states_image[0]
        else:
            encoder_hidden_states_image = None

        batch_size, num_channels, num_frames, height, width = hidden_states.shape
        p_t, p_h, p_w = self.patch_size
        post_patch_num_frames = num_frames // p_t
        post_patch_height = height // p_h
        post_patch_width = width // p_w

        # Get rotary embeddings
        d = self.hidden_size // self.num_attention_heads
        rope_dim_list = [d - 4 * (d // 6), 2 * (d // 6), 2 * (d // 6)]
        freqs_cos, freqs_sin = get_rotary_pos_embed(
            (post_patch_num_frames * get_sequence_model_parallel_world_size(),
             post_patch_height, post_patch_width),
            self.hidden_size,
            self.num_attention_heads,
            rope_dim_list,
            dtype=torch.float64,
            rope_theta=10000)
        freqs_cos = freqs_cos.to(hidden_states.device)
        freqs_sin = freqs_sin.to(hidden_states.device)
        freqs_cis = (freqs_cos.float(),
                     freqs_sin.float()) if freqs_cos is not None else None

        hidden_states = self.patch_embedding(hidden_states)
        hidden_states = hidden_states.flatten(2).transpose(1, 2)

        temb, timestep_proj, encoder_hidden_states, encoder_hidden_states_image = self.condition_embedder(
            timestep, encoder_hidden_states, encoder_hidden_states_image)
        timestep_proj = timestep_proj.unflatten(1, (6, -1))

        # For TeaCache - store the modulated timestep projection
        # kwargs['timestep_proj'] = timestep_proj
        # kwargs['temb'] = temb

        if encoder_hidden_states_image is not None:
            encoder_hidden_states = torch.concat(
                [encoder_hidden_states_image, encoder_hidden_states], dim=1)
        
        # The main differentiating input between steps is the timestep
        # timestep_proj = kwargs.get('timestep_proj', None)
        # temb = kwargs.get('temb', None)
        # if timestep_proj is None or temb is None:
        #     if not isinstance(encoder_hidden_states, torch.Tensor):
        #         encoder_hidden_states = encoder_hidden_states[0]
        #     if isinstance(encoder_hidden_states_image,
        #                 list) and len(encoder_hidden_states_image) > 0:
        #         encoder_hidden_states_image = encoder_hidden_states_image[0]
        #     else:
        #         encoder_hidden_states_image = None
        #     temb, timestep_proj, encoder_hidden_states, encoder_hidden_states_image = self.condition_embedder(
        #         timestep, encoder_hidden_states, encoder_hidden_states_image)
        #     timestep_proj = timestep_proj.unflatten(1, (6, -1))
        # print(' before any teacahce hidden_states.shape', hidden_states.shape)

        if self.enable_teacache:
            modulated_inp = timestep_proj if self.use_ret_steps else temb
            
            if self.cnt%2==0: # even -> conditon
                self.is_even = True
                if self.cnt < self.ret_steps or self.cnt >= self.cutoff_steps:
                        should_calc_even = True
                        self.accumulated_rel_l1_distance_even = 0
                else:
                    rescale_func = np.poly1d(self.coefficients)
                    self.accumulated_rel_l1_distance_even += rescale_func(((modulated_inp-self.previous_e0_even).abs().mean() / self.previous_e0_even.abs().mean()).cpu().item())
                    if self.accumulated_rel_l1_distance_even < self.teacache_thresh:
                        should_calc_even = False
                    else:
                        should_calc_even = True
                        self.accumulated_rel_l1_distance_even = 0
                self.previous_e0_even = modulated_inp.clone()

            else: # odd -> unconditon
                self.is_even = False
                if self.cnt < self.ret_steps or self.cnt >= self.cutoff_steps:
                        should_calc_odd = True
                        self.accumulated_rel_l1_distance_odd = 0
                else: 
                    rescale_func = np.poly1d(self.coefficients)
                    self.accumulated_rel_l1_distance_odd += rescale_func(((modulated_inp-self.previous_e0_odd).abs().mean() / self.previous_e0_odd.abs().mean()).cpu().item())
                    if self.accumulated_rel_l1_distance_odd < self.teacache_thresh:
                        should_calc_odd = False
                    else:
                        should_calc_odd = True
                        self.accumulated_rel_l1_distance_odd = 0
                self.previous_e0_odd = modulated_inp.clone()



            if self.is_even:
                if not should_calc_even:
                    hidden_states += self.previous_residual_even
                else:
                    ori_x = hidden_states.clone()
                    # for block in self.blocks:
                    #     x = block(x, **kwargs)
                    hidden_states = self.forward_impl(
                        hidden_states, 
                        encoder_hidden_states, 
                        timestep, 
                        encoder_hidden_states_image, 
                        guidance, 
                        orig_dtype=orig_dtype,
                        timestep_proj=timestep_proj,
                        freqs_cis=freqs_cis,
                        temb=temb,
                        batch_size=batch_size,
                        post_patch_num_frames=post_patch_num_frames,
                        post_patch_height=post_patch_height,
                        post_patch_width=post_patch_width,
                        p_t=p_t,
                        p_h=p_h,
                        p_w=p_w,
                        **kwargs)
                    self.previous_residual_even = hidden_states.squeeze(0) - ori_x
            else:
                if not should_calc_odd:
                    hidden_states += self.previous_residual_odd
                else:
                    ori_x = hidden_states.clone()
                    # for block in self.blocks:
                    #     x = block(x, **kwargs)
                    hidden_states = self.forward_impl(
                        hidden_states, 
                        encoder_hidden_states, 
                        timestep, 
                        encoder_hidden_states_image, 
                        guidance, 
                        orig_dtype=orig_dtype,
                        timestep_proj=timestep_proj,
                        freqs_cis=freqs_cis,
                        temb=temb,
                        batch_size=batch_size,
                        post_patch_num_frames=post_patch_num_frames,
                        post_patch_height=post_patch_height,
                        post_patch_width=post_patch_width,
                        p_t=p_t,
                        p_h=p_h,
                        p_w=p_w,
                        **kwargs)
                    self.previous_residual_odd = hidden_states.squeeze(0) - ori_x
        else: # not enable teacache
            # print(' before forward_impl hidden_states.shape', hidden_states.shape)
            hidden_states = self.forward_impl(
                hidden_states, 
                encoder_hidden_states, 
                timestep, 
                encoder_hidden_states_image, 
                guidance, 
                orig_dtype=orig_dtype,
                timestep_proj=timestep_proj,
                freqs_cis=freqs_cis,
                temb=temb,
                batch_size=batch_size,
                post_patch_num_frames=post_patch_num_frames,
                post_patch_height=post_patch_height,
                post_patch_width=post_patch_width,
                p_t=p_t,
                p_h=p_h,
                p_w=p_w,
                **kwargs)
        # print(' before any 5 hidden_states.shape', hidden_states.shape)
        # 5. Output norm, projection & unpatchify
        shift, scale = (self.scale_shift_table + temb.unsqueeze(1)).chunk(2,
                                                                          dim=1)
        hidden_states = self.norm_out(hidden_states.float(), shift, scale)
        hidden_states = self.proj_out(hidden_states)

        hidden_states = hidden_states.reshape(batch_size, post_patch_num_frames,
                                              post_patch_height,
                                              post_patch_width, p_t, p_h, p_w,
                                              -1)
        hidden_states = hidden_states.permute(0, 7, 1, 4, 2, 5, 3, 6)
        hidden_states = hidden_states.flatten(6, 7).flatten(4, 5).flatten(2, 3)


        if self.enable_teacache:
            # Update step counter
            self.cnt += 1
            if self.num_steps is not None and self.cnt >= self.num_steps:
                self.cnt = 0
        # print(' after any 5 hidden_states.shape', hidden_states.shape)
            
        return hidden_states
    
    @abstractmethod
    def forward_impl(self,
                     hidden_states: torch.Tensor,
                     encoder_hidden_states: Union[torch.Tensor, List[torch.Tensor]],
                     timestep: torch.LongTensor,
                     encoder_hidden_states_image: Optional[Union[
                         torch.Tensor, List[torch.Tensor]]] = None,
                     guidance=None,
                     **kwargs) -> torch.Tensor:
        """
        The actual forward implementation to be defined by subclasses.
        This is called by the TeaCache-enabled forward method when needed.
        """
        pass
