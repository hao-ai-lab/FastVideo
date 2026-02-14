# Copyright 2025 Bytedance Ltd. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0

import copy
from turtle import position
from typing import List, Tuple, Optional
import math
import random
import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.attention.flex_attention import create_block_mask
from transformers.configuration_utils import PretrainedConfig
from transformers.modeling_utils import PreTrainedModel
from diffusers.utils.torch_utils import randn_tensor
from flow_grpo.bagel.data.data_utils import (
    create_sparse_mask, 
    get_flattened_position_ids_extrapolate, 
    get_flattened_position_ids_interpolate,
    patchify, 
)
from torch.distributed.fsdp.fully_sharded_data_parallel import FullyShardedDataParallel as fsdp
from contextlib import nullcontext
from torch.profiler import profile, ProfilerActivity, record_function
from .qwen2_navit import NaiveCache
from .modeling_utils import MLPconnector, TimestepEmbedder, PositionEmbedding

from tqdm import tqdm
from functools import partial

tqdm = partial(tqdm, dynamic_ncols=True, leave=False, position=1)

class BagelConfig(PretrainedConfig):
    def __init__(
        self,
        visual_gen=True,
        visual_und=True,
        llm_config=None,
        vit_config=None,
        vae_config=None,
        latent_patch_size=2,
        max_latent_size=32,
        vit_max_num_patch_per_side=70,
        connector_act="gelu_pytorch_tanh",
        interpolate_pos=False,
        timestep_shift=1.0,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.visual_gen = visual_gen
        self.visual_und = visual_und
        self.llm_config = llm_config
        self.vit_config = vit_config
        self.vae_config = vae_config
        self.latent_patch_size = latent_patch_size
        self.max_latent_size = max_latent_size
        self.vit_max_num_patch_per_side = vit_max_num_patch_per_side
        self.connector_act = connector_act
        self.interpolate_pos = interpolate_pos
        self.timestep_shift = timestep_shift


class Bagel(PreTrainedModel):
    config_class = BagelConfig
    base_model_prefix = 'bagel'

    def __init__(self, language_model, vit_model, config: BagelConfig):
        super().__init__(config)    
        self.language_model = language_model
        self.hidden_size = config.llm_config.hidden_size
        self.use_moe = "Mo" in config.llm_config.layer_module
        self.num_heads = config.llm_config.num_attention_heads

        if config.visual_gen:
            self.latent_patch_size = config.latent_patch_size
            self.timestep_shift = config.timestep_shift
            self.latent_downsample = config.vae_config.downsample * config.latent_patch_size
            self.max_latent_size = config.max_latent_size
            self.latent_channel = config.vae_config.z_channels
            self.patch_latent_dim = self.latent_patch_size ** 2 * self.latent_channel
            self.time_embedder = TimestepEmbedder(self.hidden_size)
            self.vae2llm = nn.Linear(self.patch_latent_dim, self.hidden_size)
            self.llm2vae = nn.Linear(self.hidden_size, self.patch_latent_dim)
            self.latent_pos_embed = PositionEmbedding(self.max_latent_size, self.hidden_size)

        if config.visual_und:
            self.vit_model = vit_model
            self.vit_patch_size = config.vit_config.patch_size
            self.vit_max_num_patch_per_side = config.vit_max_num_patch_per_side
            self.vit_hidden_size = config.vit_config.hidden_size
            self.connector = MLPconnector(self.vit_hidden_size, self.hidden_size, config.connector_act)
            self.vit_pos_embed = PositionEmbedding(self.vit_max_num_patch_per_side, self.hidden_size)

        if config.interpolate_pos:
            self.get_flattened_position_ids = get_flattened_position_ids_interpolate
        else:
            self.get_flattened_position_ids = get_flattened_position_ids_extrapolate

        self.config = config
        self._init_weights()

    def _init_weights(self):
        if self.config.visual_gen:
            nn.init.constant_(self.llm2vae.weight, 0)
            nn.init.constant_(self.llm2vae.bias, 0)

    def forward(
        self,
        sequence_length: int,
        packed_text_ids: torch.LongTensor,
        packed_text_indexes: torch.LongTensor,
        sample_lens: List[int],
        packed_position_ids: torch.LongTensor,
        nested_attention_masks: List[torch.Tensor] = None,
        split_lens: List[int] = None,
        attn_modes: List[str] = None,
        # for visual understanding
        ce_loss_indexes: Optional[torch.BoolTensor] = None,
        packed_label_ids: Optional[torch.LongTensor] = None,
        packed_vit_tokens: Optional[torch.Tensor] = None,
        packed_vit_token_indexes: Optional[torch.LongTensor] = None,
        packed_vit_position_ids: Optional[torch.LongTensor] = None,
        vit_token_seqlens: Optional[torch.IntTensor] = None,
        # for visual generation
        padded_latent: Optional[torch.Tensor] = None,
        patchified_vae_latent_shapes: Optional[List[Tuple[int, int]]] = None,
        packed_latent_position_ids: Optional[torch.LongTensor] = None,
        packed_vae_token_indexes: Optional[torch.LongTensor] = None,
        packed_timesteps: Optional[torch.LongTensor] = None,
        mse_loss_indexes: Optional[torch.BoolTensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            sequence_length: length of sequence.
            packed_text_ids: 1-D int tensor, packed text token ids.
            packed_text_indexes: 1-D int tensor, packed text token indexes in sequence.
            sample_lens: A list of N ints, length of each sample in packed_sequence.
            nested_attention_masks: A list of N 2-D float tensor,  where 0.0 means attention and 
                -inf means ignore.
            packed_position_ids: packed 1-D positions, an image has only one global position shared
                by all latent tokens.

            packed_vit_tokens: packed patchified image tokens for vit model.
            packed_vit_position_ids: 1-D int tensor, the position of each token for vit model.
            packed_vit_token_indexes: 1-D int tensor, packed vit token indexes in sequence.
            vit_token_seqlens: 1-D int tensor, the length of each image tokens for vit model.
            packed_label_ids: 1-D int tensor, packed label token ids.
            ce_loss_indexes: 1-D bool tensor, where to compute ce loss.

            padded_latent: padded latent from VAE encoder.
            patchified_vae_latent_shapes: A list of (h, w) tuples, patchfied latent shapes of each image.
            packed_latent_position_ids: 1-D int tensor, the position of each token for latent.
            packed_vae_token_indexes: 1-D int tensor, padded image token indexes in sequence.
            packed_timesteps: 1-D float tensor, flow timesteps. 0 indicates use clean image.
            mse_loss_indexes: 1-D bool tensor, where to compute mse loss.
        """
        packed_text_embedding = self.language_model.forward(mode="get_embeddings", input_ids=packed_text_ids)
        packed_sequence = packed_text_embedding.new_zeros(size=(sequence_length, self.hidden_size))
        packed_sequence[packed_text_indexes] = packed_text_embedding

        if nested_attention_masks is None:
            sparse_mask = create_sparse_mask(sample_lens, split_lens, attn_modes, packed_text_embedding.device)
            seqlen = sum(sample_lens)
            block_mask = create_block_mask(
                sparse_mask, B=1, H=self.num_heads, Q_LEN=seqlen, KV_LEN=seqlen, 
                device=packed_text_embedding.device, BLOCK_SIZE=128, _compile=True
            )
            attention_mask = block_mask
        else:
            attention_mask = nested_attention_masks

        if self.config.visual_und:
            cu_seqlens = torch.nn.functional.pad(torch.cumsum(vit_token_seqlens, dim=0), (1, 0))
            cu_seqlens = cu_seqlens.to(torch.int32)
            max_seqlen = torch.max(vit_token_seqlens).item()
            packed_vit_token_embed = self.vit_model(
                packed_pixel_values=packed_vit_tokens, 
                packed_flattened_position_ids=packed_vit_position_ids,
                cu_seqlens=cu_seqlens,
                max_seqlen=max_seqlen,
            )
            packed_vit_token_embed = self.connector(packed_vit_token_embed)
            vit_token_pos_emb = self.vit_pos_embed(packed_vit_position_ids)
            packed_vit_token_embed = packed_vit_token_embed + vit_token_pos_emb
            packed_sequence = packed_sequence.to(packed_vit_token_embed.dtype)
            packed_sequence[packed_vit_token_indexes] = packed_vit_token_embed

        if self.config.visual_gen:
            p = self.latent_patch_size
            packed_latent = []
            for latent, (h, w) in zip(padded_latent, patchified_vae_latent_shapes):
                latent = latent[:, :h * p, :w * p].reshape(self.latent_channel, h, p, w, p)
                latent = torch.einsum("chpwq->hwpqc", latent).reshape(-1, p * p * self.latent_channel)
                packed_latent.append(latent)
            packed_latent_clean = torch.cat(packed_latent, dim=0)

            noise = torch.randn_like(packed_latent_clean)
            packed_timesteps = torch.sigmoid(packed_timesteps)
            packed_timesteps = self.timestep_shift * packed_timesteps / (1 + (self.timestep_shift - 1) * packed_timesteps)
            packed_latent = (1 - packed_timesteps[:, None]) * packed_latent_clean + packed_timesteps[:, None] * noise
            packed_timestep_embeds = self.time_embedder(packed_timesteps)
            latent_token_pos_emb = self.latent_pos_embed(packed_latent_position_ids)
            packed_latent = self.vae2llm(packed_latent) + packed_timestep_embeds + latent_token_pos_emb
            packed_sequence[packed_vae_token_indexes] = packed_latent

        extra_inputs = {}
        if self.use_moe:
            packed_und_token_indexes = packed_text_indexes
            if packed_vit_token_indexes is not None:
                packed_und_token_indexes=torch.cat([packed_text_indexes, packed_vit_token_indexes], dim=0)
            extra_inputs.update(
                packed_und_token_indexes=packed_und_token_indexes,
                packed_gen_token_indexes=packed_vae_token_indexes,
            )

        last_hidden_state = self.language_model(
            packed_sequence=packed_sequence,
            sample_lens=sample_lens,
            attention_mask=attention_mask,
            packed_position_ids=packed_position_ids,
            **extra_inputs,
        )

        mse = None
        if self.config.visual_gen:
            packed_mse_preds = self.llm2vae(last_hidden_state[mse_loss_indexes])
            target = noise - packed_latent_clean # NOTE: v_t=dx_t/dt=x_1-x_0, pointing from data to noise
            has_mse = packed_timesteps > 0
            mse = (packed_mse_preds - target[has_mse]) ** 2

        ce = None
        if ce_loss_indexes is not None:
            packed_ce_preds = self.language_model.lm_head(last_hidden_state[ce_loss_indexes])
            ce = F.cross_entropy(packed_ce_preds, packed_label_ids, reduction="none")

        return dict(mse=mse, ce=ce)


    def prepare_prompts(self, curr_kvlens, curr_rope, prompts, tokenizer, new_token_ids):
        packed_text_ids = list()
        packed_text_position_ids = list()
        text_token_lens = list()
        packed_text_indexes = list()
        packed_key_value_indexes = list()

        curr = 0
        newlens, new_rope = list(), list()
        for prompt, curr_kvlen, curr_position_id in zip(prompts, curr_kvlens, curr_rope):
            packed_key_value_indexes.extend(range(curr, curr + curr_kvlen))
            curr += curr_kvlen

            text_ids = tokenizer.encode(prompt)
            text_ids = [new_token_ids['bos_token_id']] + text_ids + [new_token_ids['eos_token_id']]
            text_token_lens.append(len(text_ids))
            packed_text_ids.extend(text_ids)
            packed_text_position_ids.extend(range(curr_position_id, curr_position_id + len(text_ids)))
            packed_text_indexes.extend(range(curr, curr + len(text_ids)))
            newlens.append(curr_kvlen + len(text_ids))
            new_rope.append(curr_position_id + len(text_ids))
            curr += len(text_ids)

        generation_input = {
            "text_token_lens": torch.tensor(text_token_lens, dtype=torch.int),
            "packed_text_ids": torch.tensor(packed_text_ids, dtype=torch.long),
            "packed_text_position_ids": torch.tensor(packed_text_position_ids, dtype=torch.long),
            "packed_text_indexes": torch.tensor(packed_text_indexes, dtype=torch.long),
            "packed_key_value_indexes": torch.tensor(packed_key_value_indexes, dtype=torch.long),
            "key_values_lens": torch.tensor(curr_kvlens, dtype=torch.int),
        }

        return generation_input, newlens, new_rope

    @torch.no_grad
    def forward_cache_update_text(
        self,
        past_key_values: NaiveCache,
        packed_text_ids: torch.IntTensor,
        packed_text_position_ids: torch.LongTensor,
        text_token_lens: torch.LongTensor,
        packed_text_indexes: torch.LongTensor,
        packed_key_value_indexes: torch.LongTensor,
        key_values_lens: torch.IntTensor,
    ):
        packed_text_embedding = self.language_model.forward(mode="get_embeddings", input_ids=packed_text_ids)
        extra_inputs = {}
        if self.use_moe:
            extra_inputs = {"mode": "und"}

        output = self.language_model.forward(
            packed_query_sequence=packed_text_embedding,
            query_lens=text_token_lens,
            packed_query_position_ids=packed_text_position_ids,
            packed_query_indexes=packed_text_indexes,
            past_key_values=past_key_values,
            packed_key_value_indexes=packed_key_value_indexes,
            key_values_lens=key_values_lens,
            update_past_key_values=True,
            is_causal=True,
            **extra_inputs,
        )
        past_key_values = output.past_key_values

        return past_key_values

    def prepare_vit_images(self, curr_kvlens, curr_rope, images, transforms, new_token_ids):
        packed_vit_token_indexes = list()
        vit_token_seqlens, packed_vit_tokens, packed_vit_position_ids = list(), list(), list()
        packed_text_ids, packed_text_indexes = list(), list()
        packed_seqlens, packed_position_ids, packed_indexes = list(), list(), list()
        packed_key_value_indexes = list()

        _curr = curr = 0
        newlens, new_rope = list(), list()
        for image, curr_kvlen, curr_position_id in zip(images, curr_kvlens, curr_rope):
            packed_key_value_indexes.extend(range(curr, curr + curr_kvlen))
            curr += curr_kvlen

            packed_text_ids.append(new_token_ids['start_of_image'])
            packed_text_indexes.append(_curr)
            packed_indexes.append(curr)
            curr += 1
            _curr += 1

            image_tensor = transforms(image)
            vit_position_ids = self.get_flattened_position_ids(
                image_tensor.size(1), image_tensor.size(2), 
                self.vit_patch_size, 
                max_num_patches_per_side=self.vit_max_num_patch_per_side
            )
            vit_tokens = patchify(image_tensor, self.vit_patch_size)
            packed_vit_tokens.append(vit_tokens)
            num_img_tokens = vit_tokens.shape[0]
            packed_vit_position_ids.append(vit_position_ids)
            vit_token_seqlens.append(num_img_tokens)
            packed_vit_token_indexes.extend(range(_curr, _curr + num_img_tokens))
            packed_indexes.extend(range(curr, curr + num_img_tokens))
            curr += num_img_tokens
            _curr += num_img_tokens

            packed_text_ids.append(new_token_ids['end_of_image'])
            packed_text_indexes.append(_curr)
            packed_indexes.append(curr)
            curr += 1
            _curr += 1

            packed_position_ids.extend([curr_position_id] * (num_img_tokens + 2))
            packed_seqlens.append(num_img_tokens + 2)
            newlens.append(curr_kvlen + num_img_tokens + 2)
            new_rope.append(curr_position_id + 1)

        generation_input = {
            "packed_text_ids": torch.tensor(packed_text_ids, dtype=torch.long),
            "packed_text_indexes": torch.tensor(packed_text_indexes, dtype=torch.long),
            "vit_token_seqlens": torch.tensor(vit_token_seqlens, dtype=torch.int),
            "packed_vit_tokens": torch.cat(packed_vit_tokens, dim=0),
            "packed_vit_position_ids": torch.cat(packed_vit_position_ids, dim=0),
            "packed_vit_token_indexes": torch.tensor(packed_vit_token_indexes, dtype=torch.long),
            "packed_position_ids": torch.tensor(packed_position_ids, dtype=torch.long),
            "packed_seqlens": torch.tensor(packed_seqlens, dtype=torch.int),
            "packed_indexes": torch.tensor(packed_indexes, dtype=torch.long),
            "packed_key_value_indexes": torch.tensor(packed_key_value_indexes, dtype=torch.long),
            "key_values_lens": torch.tensor(curr_kvlens, dtype=torch.int),
        }

        return generation_input, newlens, new_rope

    @torch.no_grad
    def forward_cache_update_vit(
        self,
        past_key_values: NaiveCache,
        packed_text_ids: torch.LongTensor,
        packed_text_indexes: torch.LongTensor,
        packed_vit_tokens: torch.Tensor,
        packed_vit_token_indexes: torch.LongTensor,
        packed_vit_position_ids: torch.LongTensor,
        vit_token_seqlens: torch.IntTensor,
        packed_position_ids: torch.LongTensor,
        packed_seqlens: torch.IntTensor,
        packed_indexes: torch.LongTensor,
        packed_key_value_indexes: torch.LongTensor,
        key_values_lens: torch.IntTensor,
    ):
        packed_text_embedding = self.language_model.forward(mode="get_embeddings", input_ids=packed_text_ids)
        packed_sequence = packed_text_embedding.new_zeros((sum(packed_seqlens), self.hidden_size))
        packed_sequence[packed_text_indexes] = packed_text_embedding

        cu_seqlens = torch.nn.functional.pad(torch.cumsum(vit_token_seqlens, dim=0), (1, 0))
        cu_seqlens = cu_seqlens.to(torch.int32)
        max_seqlen = torch.max(vit_token_seqlens).item()
        packed_vit_token_embed = self.vit_model(
            packed_pixel_values=packed_vit_tokens, 
            packed_flattened_position_ids=packed_vit_position_ids,
            cu_seqlens=cu_seqlens,
            max_seqlen=max_seqlen,
        )
        packed_vit_token_embed = self.connector(packed_vit_token_embed)
        pos_emb = self.vit_pos_embed(packed_vit_position_ids)
        packed_vit_token_embed = packed_vit_token_embed + pos_emb
        if packed_vit_token_embed.dtype != packed_sequence.dtype:
            packed_vit_token_embed = packed_vit_token_embed.to(packed_sequence.dtype)
        packed_sequence[packed_vit_token_indexes] = packed_vit_token_embed

        extra_inputs = {}
        if self.use_moe:
            extra_inputs = {"mode": "und"}

        output = self.language_model.forward(
            packed_query_sequence=packed_sequence,
            query_lens=packed_seqlens,
            packed_query_position_ids=packed_position_ids,
            packed_query_indexes=packed_indexes,
            past_key_values=past_key_values,
            packed_key_value_indexes=packed_key_value_indexes,
            key_values_lens=key_values_lens,
            update_past_key_values=True,
            is_causal=False,
            **extra_inputs,
        )
        past_key_values = output.past_key_values

        return past_key_values

    def prepare_vae_images(self, curr_kvlens, curr_rope, images, transforms, new_token_ids, timestep=0):
        patchified_vae_latent_shapes, packed_vae_position_ids = list(), list()
        packed_vae_token_indexes = list()
        packed_text_ids, packed_text_indexes = list(), list()
        packed_seqlens, packed_position_ids, packed_indexes = list(), list(), list()
        packed_key_value_indexes = list()

        _curr = curr = 0
        vae_image_tensors = list()
        newlens, new_rope = list(), list()
        for image, curr_kvlen, curr_position_id in zip(images, curr_kvlens, curr_rope):
            packed_key_value_indexes.extend(range(curr, curr + curr_kvlen))
            curr += curr_kvlen

            packed_text_ids.append(new_token_ids['start_of_image'])
            packed_text_indexes.append(_curr)
            packed_indexes.append(curr)
            curr += 1
            _curr += 1

            image_tensor = transforms(image)
            vae_image_tensors.append(image_tensor)
            vae_posiiton_ids = self.get_flattened_position_ids(
                image_tensor.size(1), image_tensor.size(2),
                self.latent_downsample, 
                max_num_patches_per_side=self.max_latent_size
            )
            packed_vae_position_ids.append(vae_posiiton_ids)
            H, W = image_tensor.shape[1:]
            h = H // self.latent_downsample
            w = W // self.latent_downsample
            patchified_vae_latent_shapes.append((h, w))

            num_img_tokens = w * h
            packed_vae_token_indexes.extend(range(_curr, _curr + num_img_tokens))
            packed_indexes.extend(range(curr, curr + num_img_tokens))
            curr += num_img_tokens
            _curr += num_img_tokens

            packed_text_ids.append(new_token_ids['end_of_image'])
            packed_text_indexes.append(_curr)
            packed_indexes.append(curr)
            curr += 1
            _curr += 1

            packed_position_ids.extend([curr_position_id] * (num_img_tokens + 2))
            packed_seqlens.append(num_img_tokens + 2)
            newlens.append(curr_kvlen + num_img_tokens + 2)
            new_rope.append(curr_position_id + 1)

        image_sizes = [item.shape for item in vae_image_tensors]
        max_image_size = [max(item) for item in list(zip(*image_sizes))]
        padded_images = torch.zeros(size=(len(vae_image_tensors), *max_image_size))
        for i, image_tensor in enumerate(vae_image_tensors):
            padded_images[i, :, :image_tensor.shape[1], :image_tensor.shape[2]] = image_tensor

        generation_input = {
            "padded_images": padded_images,
            "patchified_vae_latent_shapes": patchified_vae_latent_shapes,
            "packed_vae_position_ids": torch.cat(packed_vae_position_ids, dim=0),
            "packed_timesteps": torch.tensor([timestep]),
            "packed_vae_token_indexes": torch.tensor(packed_vae_token_indexes, dtype=torch.long),
            "packed_text_ids": torch.tensor(packed_text_ids, dtype=torch.long),
            "packed_text_indexes": torch.tensor(packed_text_indexes, dtype=torch.long),
            "packed_position_ids": torch.tensor(packed_position_ids, dtype=torch.long),
            "packed_seqlens": torch.tensor(packed_seqlens, dtype=torch.int),
            "packed_indexes": torch.tensor(packed_indexes, dtype=torch.long),
            "packed_key_value_indexes": torch.tensor(packed_key_value_indexes, dtype=torch.long),
            "key_values_lens": torch.tensor(curr_kvlens, dtype=torch.int),
        }

        return generation_input, newlens, new_rope

    @torch.no_grad
    def forward_cache_update_vae(
        self,
        vae_model,
        past_key_values: NaiveCache,
        padded_images: torch.Tensor,
        patchified_vae_latent_shapes: List,
        packed_vae_position_ids: torch.LongTensor,
        packed_timesteps: torch.Tensor,
        packed_vae_token_indexes: torch.LongTensor,
        packed_text_ids: torch.LongTensor,
        packed_text_indexes: torch.LongTensor,
        packed_position_ids: torch.LongTensor,
        packed_seqlens: torch.IntTensor,
        packed_indexes: torch.LongTensor,
        key_values_lens: torch.IntTensor,
        packed_key_value_indexes: torch.Tensor,
    ):
        packed_text_embedding = self.language_model.forward(mode="get_embeddings", input_ids=packed_text_ids)
        packed_sequence = packed_text_embedding.new_zeros((sum(packed_seqlens), self.hidden_size))
        packed_sequence[packed_text_indexes] = packed_text_embedding

        padded_latent = vae_model.encode(padded_images)

        p = self.latent_patch_size
        packed_latent = list()
        for latent, (h, w) in zip(padded_latent, patchified_vae_latent_shapes):
            latent = latent[:, :h * p, :w * p].reshape(self.latent_channel, h, p, w, p)
            latent = torch.einsum("chpwq->hwpqc", latent).reshape(-1, p * p * self.latent_channel)
            packed_latent.append(latent)
        packed_latent = torch.cat(packed_latent, dim=0)
        packed_pos_embed = self.latent_pos_embed(packed_vae_position_ids)
        packed_timestep_embeds = self.time_embedder(packed_timesteps)
        packed_latent = self.vae2llm(packed_latent) + packed_timestep_embeds + packed_pos_embed
        if packed_latent.dtype != packed_sequence.dtype:
            packed_latent = packed_latent.to(packed_sequence.dtype)
        packed_sequence[packed_vae_token_indexes] = packed_latent

        extra_inputs = {}
        if self.use_moe:
            extra_inputs = {
                "mode": "gen",
                "packed_vae_token_indexes": packed_vae_token_indexes,
                "packed_text_indexes": packed_text_indexes
            }

        output = self.language_model.forward(
            packed_query_sequence=packed_sequence,
            query_lens=packed_seqlens,
            packed_query_position_ids=packed_position_ids,
            packed_query_indexes=packed_indexes,
            past_key_values=past_key_values,
            key_values_lens=key_values_lens,
            packed_key_value_indexes=packed_key_value_indexes,
            update_past_key_values=True,
            is_causal=False,
            **extra_inputs,
        )
        past_key_values = output.past_key_values

        return past_key_values

    def prepare_vae_latent(self, curr_kvlens, curr_rope, image_sizes, new_token_ids, generators=None):
        packed_text_ids, packed_text_indexes = list(), list()
        packed_vae_position_ids, packed_vae_token_indexes, packed_init_noises = list(), list(), list()
        packed_position_ids, packed_seqlens, packed_indexes = list(), list(), list()
        packed_key_value_indexes = list()

        query_curr = curr = 0
        index=0
        for (H, W), curr_kvlen, curr_position_id in zip(image_sizes, curr_kvlens, curr_rope):
            packed_key_value_indexes.extend(range(curr, curr + curr_kvlen))
            curr += curr_kvlen

            packed_text_ids.append(new_token_ids['start_of_image'])
            packed_text_indexes.append(query_curr)
            packed_indexes.append(curr)
            curr += 1
            query_curr += 1

            vae_posiiton_ids = self.get_flattened_position_ids(
                H, W,
                self.latent_downsample, 
                max_num_patches_per_side=self.max_latent_size
            )
            packed_vae_position_ids.append(vae_posiiton_ids)

            h, w = H // self.latent_downsample, W // self.latent_downsample
            num_image_tokens = h * w
            if generators:
                packed_init_noises.append(
                    torch.randn((num_image_tokens, self.latent_channel * self.latent_patch_size ** 2),generator=generators[index])
                )
            else:
                packed_init_noises.append(
                    torch.randn(num_image_tokens, self.latent_channel * self.latent_patch_size ** 2)
                )
            packed_vae_token_indexes.extend(range(query_curr, query_curr + num_image_tokens))
            packed_indexes.extend(range(curr, curr + num_image_tokens))
            curr += num_image_tokens
            query_curr += num_image_tokens

            packed_text_ids.append(new_token_ids['end_of_image'])
            packed_text_indexes.append(query_curr)
            packed_indexes.append(curr)
            curr += 1
            query_curr += 1
            index+=1

            packed_position_ids.extend([curr_position_id] * (num_image_tokens + 2))
            packed_seqlens.append(num_image_tokens + 2)

        generation_input = {
            "packed_text_ids": torch.tensor(packed_text_ids, dtype=torch.long),
            "packed_text_indexes": torch.tensor(packed_text_indexes, dtype=torch.long),
            "packed_init_noises": torch.cat(packed_init_noises, dim=0),
            "packed_vae_position_ids": torch.cat(packed_vae_position_ids, dim=0),
            "packed_vae_token_indexes": torch.tensor(packed_vae_token_indexes, dtype=torch.long),
            "packed_seqlens": torch.tensor(packed_seqlens, dtype=torch.int),
            "packed_position_ids": torch.tensor(packed_position_ids, dtype=torch.long),
            "key_values_lens": torch.tensor(curr_kvlens, dtype=torch.int),
            "packed_indexes": torch.tensor(packed_indexes, dtype=torch.long),
            "packed_key_value_indexes": torch.tensor(packed_key_value_indexes, dtype=torch.long),
        }

        return generation_input

    def prepare_vae_latent_cfg(self, curr_kvlens, curr_rope, image_sizes):
        packed_position_ids, packed_indexes, packed_key_value_indexes = list(), list(), list()

        query_curr = curr = 0
        for (H, W), curr_kvlen, curr_position_id in zip(image_sizes, curr_kvlens, curr_rope):
            packed_key_value_indexes.extend(range(curr, curr + curr_kvlen))
            curr += curr_kvlen

            packed_indexes.append(curr)
            curr += 1
            query_curr += 1

            h, w = H // self.latent_downsample, W // self.latent_downsample
            num_image_tokens = h * w
            packed_indexes.extend(range(curr, curr + num_image_tokens))
            curr += num_image_tokens
            query_curr += num_image_tokens

            packed_indexes.append(curr)
            curr += 1
            query_curr += 1

            packed_position_ids.extend([curr_position_id] * (num_image_tokens + 2))

        generation_input = {
            "cfg_packed_position_ids": torch.tensor(packed_position_ids, dtype=torch.long),
            "cfg_key_values_lens": torch.tensor(curr_kvlens, dtype=torch.int),
            "cfg_packed_query_indexes": torch.tensor(packed_indexes, dtype=torch.long),
            "cfg_packed_key_value_indexes": torch.tensor(packed_key_value_indexes, dtype=torch.long),
        }

        return generation_input

    # @torch.no_grad
    def generate_image(
        self,
        packed_text_ids: torch.LongTensor,
        packed_text_indexes: torch.LongTensor,
        packed_init_noises: torch.Tensor,
        packed_vae_position_ids: torch.LongTensor,
        packed_vae_token_indexes: torch.LongTensor,
        packed_seqlens: torch.IntTensor,
        packed_position_ids: torch.LongTensor,
        packed_indexes: torch.LongTensor,
        past_key_values: NaiveCache,
        key_values_lens: torch.IntTensor,
        packed_key_value_indexes: torch.LongTensor,
        num_timesteps: int = 24,
        timestep_shift: float = 1.0,
        cfg_renorm_min: float = 0.0,
        cfg_renorm_type: str = "global",
        cfg_interval: Optional[Tuple[float, float]] = [0, 1],
        # cfg_text
        cfg_text_scale: float = 1.0,
        cfg_text_packed_query_indexes: Optional[torch.LongTensor] = None,
        cfg_text_packed_position_ids: Optional[torch.LongTensor] = None,
        cfg_text_past_key_values: Optional[NaiveCache] = None,
        cfg_text_key_values_lens: Optional[torch.IntTensor] = None,
        cfg_text_packed_key_value_indexes: Optional[torch.LongTensor] = None,
        # cfg_img
        cfg_img_scale: float = 1.0,
        cfg_img_packed_query_indexes: Optional[torch.LongTensor] = None,
        cfg_img_packed_position_ids: Optional[torch.LongTensor] = None,
        cfg_img_past_key_values: Optional[NaiveCache] = None,
        cfg_img_key_values_lens: Optional[torch.IntTensor] = None,
        cfg_img_packed_key_value_indexes: Optional[torch.LongTensor] = None,
        cfg_type: str = "parallel",
        noise_level: float = 0.7,
        sample_sde_window_size: int = 1,
        sample_sde_window_range: Tuple[int, int] = (0, 5),
        process_index: int = 0,
        device = "cuda",
    ):
        x_t = packed_init_noises.to(device)
        random.seed(process_index)
        # sde_timestep_begin = random.randint(0, num_timesteps//3)
        sde_timestep_begin = random.randint(sample_sde_window_range[0], sample_sde_window_range[1] - sample_sde_window_size)

        timesteps = torch.linspace(1, 0, num_timesteps, device=x_t.device)
        timesteps = timestep_shift * timesteps / (1 + (timestep_shift - 1) * timesteps)
        dts = timesteps[1:] - timesteps[:-1]
        timesteps = timesteps[:-1]

        all_latents = []
        all_log_probs = []
        all_timesteps = []

        for i, t in tqdm(enumerate(timesteps), total=len(timesteps)):
            if i < sde_timestep_begin:
                cur_noise_level = 0
            elif i == sde_timestep_begin:
                cur_noise_level= noise_level
                all_latents.append(x_t)
            elif i > sde_timestep_begin and i < sde_timestep_begin + sample_sde_window_size:
                cur_noise_level = noise_level
            else:
                cur_noise_level= 0
            timestep = torch.tensor([t] * x_t.shape[0], device=x_t.device)
            if t > cfg_interval[0] and t <= cfg_interval[1]:
                cfg_text_scale_ = cfg_text_scale
                cfg_img_scale_ = cfg_img_scale
            else:
                cfg_text_scale_ = 1.0
                cfg_img_scale_ = 1.0
            v_t = self._forward_flow(
                x_t=x_t,
                timestep=timestep, 
                packed_vae_token_indexes=packed_vae_token_indexes,
                packed_vae_position_ids=packed_vae_position_ids,
                packed_text_ids=packed_text_ids,
                packed_text_indexes=packed_text_indexes,
                packed_position_ids=packed_position_ids,
                packed_indexes=packed_indexes,
                packed_seqlens=packed_seqlens,
                key_values_lens=key_values_lens,
                past_key_values=past_key_values,
                packed_key_value_indexes=packed_key_value_indexes,
                cfg_renorm_min=cfg_renorm_min,
                cfg_renorm_type=cfg_renorm_type,
                # cfg_text
                cfg_text_scale=cfg_text_scale_,
                cfg_text_packed_position_ids=cfg_text_packed_position_ids,
                cfg_text_packed_query_indexes=cfg_text_packed_query_indexes,
                cfg_text_key_values_lens=cfg_text_key_values_lens,
                cfg_text_past_key_values=cfg_text_past_key_values,
                cfg_text_packed_key_value_indexes=cfg_text_packed_key_value_indexes,
                # cfg_img
                cfg_img_scale=cfg_img_scale_,
                cfg_img_packed_position_ids=cfg_img_packed_position_ids,
                cfg_img_packed_query_indexes=cfg_img_packed_query_indexes,
                cfg_img_key_values_lens=cfg_img_key_values_lens,
                cfg_img_past_key_values=cfg_img_past_key_values,
                cfg_img_packed_key_value_indexes=cfg_img_packed_key_value_indexes,
                cfg_type=cfg_type,
            )
            x_t, log_prob, _, _ = self._sde_step_with_logprob(
                v_t, 
                timesteps[i], 
                timesteps[i+1] if i+1 < len(timesteps) else timesteps[i]*0, # 最后一个step, timestep是0
                dts[i], 
                x_t, 
                sigma_max=timesteps[1], 
                noise_level=cur_noise_level
            )
            if i >= sde_timestep_begin and i < sde_timestep_begin + sample_sde_window_size:
                all_latents.append(x_t)
                all_log_probs.append(log_prob)
                all_timesteps.append(t)
        all_timesteps = torch.tensor(all_timesteps, device=log_prob.device)
        unpacked_latent = x_t.split((packed_seqlens - 2).tolist())
        return unpacked_latent, all_latents, all_log_probs, all_timesteps.to(log_prob.device)
    
    def generate_image_learn(
        self,
        sample,
        grpo_config,
        accelerator,
        optimizer,
        transformer,
        packed_text_ids: torch.LongTensor,
        packed_text_indexes: torch.LongTensor,
        packed_init_noises: torch.Tensor,
        packed_vae_position_ids: torch.LongTensor,
        packed_vae_token_indexes: torch.LongTensor,
        packed_seqlens: torch.IntTensor,
        packed_position_ids: torch.LongTensor,
        packed_indexes: torch.LongTensor,
        past_key_values: NaiveCache,
        key_values_lens: torch.IntTensor,
        packed_key_value_indexes: torch.LongTensor,
        num_timesteps: int = 24,
        timestep_shift: float = 1.0,
        cfg_renorm_min: float = 0.0,
        cfg_renorm_type: str = "global",
        cfg_interval: Optional[Tuple[float, float]] = [0, 1],
        # cfg_text
        cfg_text_scale: float = 1.0,
        cfg_text_packed_query_indexes: Optional[torch.LongTensor] = None,
        cfg_text_packed_position_ids: Optional[torch.LongTensor] = None,
        cfg_text_past_key_values: Optional[NaiveCache] = None,
        cfg_text_key_values_lens: Optional[torch.IntTensor] = None,
        cfg_text_packed_key_value_indexes: Optional[torch.LongTensor] = None,
        # cfg_img
        cfg_img_scale: float = 1.0,
        cfg_img_packed_query_indexes: Optional[torch.LongTensor] = None,
        cfg_img_packed_position_ids: Optional[torch.LongTensor] = None,
        cfg_img_past_key_values: Optional[NaiveCache] = None,
        cfg_img_key_values_lens: Optional[torch.IntTensor] = None,
        cfg_img_packed_key_value_indexes: Optional[torch.LongTensor] = None,
        cfg_type: str = "parallel",
        noise_level: float = 0.7,
    ):
        latents = sample["latents"]
        prev_latents = sample["prev_latents"]
        timesteps = sample["timesteps"]

        original_timesteps = torch.linspace(1, 0, num_timesteps, device=latents.device)
        original_timesteps = timestep_shift * original_timesteps / (1 + (timestep_shift - 1) * original_timesteps)
        dtimesteps = original_timesteps[1:] - original_timesteps[:-1]

        advantages = torch.clamp(
            sample["advantages"],
            -grpo_config.train.adv_clip_max,
            grpo_config.train.adv_clip_max,
        )
        clipfrac=[]
        clipfrac_gt_one=[]
        clipfrac_lt_one=[]
        policy_loss_list=[]
        kl_loss_list=[]
        loss_list=[]

        for i, t in tqdm(enumerate(timesteps), total=len(timesteps), disable=not accelerator.is_local_main_process):
            timestep = torch.tensor([t] * latents[0].shape[0], device=latents[0].device)
            if t > cfg_interval[0] and t <= cfg_interval[1]:
                cfg_text_scale_ = cfg_text_scale
                cfg_img_scale_ = cfg_img_scale
            else:
                cfg_text_scale_ = 1.0
                cfg_img_scale_ = 1.0
            with accelerator.accumulate(transformer):
                v_t = self._forward_flow(
                    x_t=latents[i],
                    timestep=timestep, 
                    packed_vae_token_indexes=packed_vae_token_indexes,
                    packed_vae_position_ids=packed_vae_position_ids,
                    packed_text_ids=packed_text_ids,
                    packed_text_indexes=packed_text_indexes,
                    packed_position_ids=packed_position_ids,
                    packed_indexes=packed_indexes,
                    packed_seqlens=packed_seqlens,
                    key_values_lens=key_values_lens,
                    past_key_values=past_key_values,
                    packed_key_value_indexes=packed_key_value_indexes,
                    cfg_renorm_min=cfg_renorm_min,
                    cfg_renorm_type=cfg_renorm_type,
                    # cfg_text
                    cfg_text_scale=cfg_text_scale_,
                    cfg_text_packed_position_ids=cfg_text_packed_position_ids,
                    cfg_text_packed_query_indexes=cfg_text_packed_query_indexes,
                    cfg_text_key_values_lens=cfg_text_key_values_lens,
                    cfg_text_past_key_values=cfg_text_past_key_values,
                    cfg_text_packed_key_value_indexes=cfg_text_packed_key_value_indexes,
                    # cfg_img
                    cfg_img_scale=cfg_img_scale_,
                    cfg_img_packed_position_ids=cfg_img_packed_position_ids,
                    cfg_img_packed_query_indexes=cfg_img_packed_query_indexes,
                    cfg_img_key_values_lens=cfg_img_key_values_lens,
                    cfg_img_past_key_values=cfg_img_past_key_values,
                    cfg_img_packed_key_value_indexes=cfg_img_packed_key_value_indexes,
                    cfg_type=cfg_type,
                )
            t_index = (original_timesteps == timesteps[i]).nonzero(as_tuple=True)[0]
            _, log_prob, prev_sample_mean, std_dev_t = self._sde_step_with_logprob(
                v_t, 
                timesteps[i], 
                timesteps[i+1] if i+1 < len(timesteps) else timesteps[i]*0, # 最后一个step, timestep是0
                dtimesteps[t_index], 
                latents[i], 
                prev_sample=prev_latents[i], 
                sigma_max=original_timesteps[1], 
                noise_level=noise_level
            )
            if grpo_config.train.beta > 0:
                with torch.no_grad():
                    v_t = self._forward_flow(
                        x_t=latents[i],
                        timestep=timestep, 
                        packed_vae_token_indexes=packed_vae_token_indexes,
                        packed_vae_position_ids=packed_vae_position_ids,
                        packed_text_ids=packed_text_ids,
                        packed_text_indexes=packed_text_indexes,
                        packed_position_ids=packed_position_ids,
                        packed_indexes=packed_indexes,
                        packed_seqlens=packed_seqlens,
                        key_values_lens=key_values_lens,
                        past_key_values=past_key_values,
                        packed_key_value_indexes=packed_key_value_indexes,
                        cfg_renorm_min=cfg_renorm_min,
                        cfg_renorm_type=cfg_renorm_type,
                        # cfg_text
                        cfg_text_scale=cfg_text_scale_,
                        cfg_text_packed_position_ids=cfg_text_packed_position_ids,
                        cfg_text_packed_query_indexes=cfg_text_packed_query_indexes,
                        cfg_text_key_values_lens=cfg_text_key_values_lens,
                        cfg_text_past_key_values=cfg_text_past_key_values,
                        cfg_text_packed_key_value_indexes=cfg_text_packed_key_value_indexes,
                        # cfg_img
                        cfg_img_scale=cfg_img_scale_,
                        cfg_img_packed_position_ids=cfg_img_packed_position_ids,
                        cfg_img_packed_query_indexes=cfg_img_packed_query_indexes,
                        cfg_img_key_values_lens=cfg_img_key_values_lens,
                        cfg_img_past_key_values=cfg_img_past_key_values,
                        cfg_img_packed_key_value_indexes=cfg_img_packed_key_value_indexes,
                        cfg_type=cfg_type,
                        ref_model=True,
                    )
                    _, _, prev_sample_mean_ref, _ = self._sde_step_with_logprob(
                        v_t, 
                        timesteps[i], 
                        timesteps[i+1] if i+1 < len(timesteps) else timesteps[i]*0, # 最后一个step, timestep是0
                        dtimesteps[t_index], 
                        latents[i], 
                        prev_sample=prev_latents[i], 
                        sigma_max=original_timesteps[1], 
                        noise_level=noise_level
                    )
            
            # grpo logic
            ratio = torch.exp(log_prob - sample["log_probs"][i])
            # print('ratio', ratio)
            unclipped_loss = -advantages * ratio
            clipped_loss = -advantages * torch.clamp(
                ratio,
                1.0 - grpo_config.train.clip_range_lt,
                1.0 + grpo_config.train.clip_range_gt,
            )
            policy_loss = torch.mean(torch.maximum(unclipped_loss, clipped_loss))
            if grpo_config.train.beta > 0:
                kl_loss = ((prev_sample_mean - prev_sample_mean_ref) ** 2).mean() / (2 * std_dev_t ** 2)
                loss = policy_loss + grpo_config.train.beta * kl_loss
            else:
                loss = policy_loss
            clipfrac.append(
                torch.mean(
                    (
                        ratio - 1.0 > grpo_config.train.clip_range_gt or 1.0 - ratio > grpo_config.train.clip_range_lt
                    ).float()
                )
            )
            clipfrac_gt_one.append(
                torch.mean(
                    (
                        ratio - 1.0 > grpo_config.train.clip_range_gt
                    ).float()
                )
            )
            clipfrac_lt_one.append(
                torch.mean(
                    (
                        1.0 - ratio > grpo_config.train.clip_range_lt
                    ).float()
                )
            )
            policy_loss_list.append(policy_loss)
            if grpo_config.train.beta > 0:
                kl_loss_list.append(kl_loss)
            loss_list.append(loss)
            accelerator.backward(loss)
            optimizer.step()
            optimizer.zero_grad()
        policy_loss = torch.cat([t.unsqueeze(0) for t in policy_loss_list]).mean().detach()
        loss = torch.cat([t.unsqueeze(0) for t in loss_list]).mean().detach()
        clipfrac = torch.cat([t.unsqueeze(0) for t in clipfrac]).mean().detach()
        clipfrac_gt_one = torch.cat([t.unsqueeze(0) for t in clipfrac_gt_one]).mean().detach()
        clipfrac_lt_one = torch.cat([t.unsqueeze(0) for t in clipfrac_lt_one]).mean().detach()

        if grpo_config.train.beta > 0:
            kl_loss = torch.cat([t.unsqueeze(0) for t in kl_loss_list]).mean().detach()
        else:
            kl_loss = policy_loss*0-1
        return clipfrac, clipfrac_gt_one, clipfrac_lt_one, policy_loss, kl_loss, loss

    def _sde_step_with_logprob(
        self,
        model_output,
        timestep,
        prev_timestep,
        d_timestep,
        sample,
        prev_sample = None,
        generator = None,
        sigma_max = None,
        noise_level: float = 0.8,
        sde_type: str = "sde",
    ):
        # bf16 can overflow here when compute prev_sample_mean, we must convert all variable to fp32
        model_output=model_output.float()
        sample=sample.float()
        if prev_sample is not None:
            prev_sample=prev_sample.float()
        
        if sde_type == "sde":
            std_dev_t = torch.sqrt(timestep / (1 - torch.where(timestep == 1, sigma_max, timestep)))*noise_level
            prev_sample_mean = sample*(1+std_dev_t**2/(2*timestep)*d_timestep)+model_output*(1+std_dev_t**2*(1-timestep)/(2*timestep))*d_timestep

            if prev_sample is not None and generator is not None:
                raise ValueError(
                    "Cannot pass both generator and prev_sample. Please make sure that either `generator` or"
                    " `prev_sample` stays `None`."
                )

            if prev_sample is None:
                variance_noise = randn_tensor(
                    model_output.shape,
                    generator=generator,
                    device=model_output.device,
                    dtype=model_output.dtype,
                )
                prev_sample = prev_sample_mean + std_dev_t * torch.sqrt(-1*d_timestep) * variance_noise

            log_prob = -((prev_sample.detach() - prev_sample_mean) ** 2) / (2 * ((std_dev_t * torch.sqrt(-1*d_timestep))**2))

        elif sde_type == "cps":
            std_dev_t = prev_timestep  * math.sin(noise_level * math.pi / 2) # sigma_t in paper
            prev_original_sample = sample - timestep * model_output # predicted x_0 in paper
            noise_estimate = sample + model_output * (1 - timestep) # predicted x_1 in paper
            prev_sample_mean = prev_original_sample * (1 - prev_timestep) + noise_estimate * torch.sqrt(prev_timestep**2 - std_dev_t**2)

            if prev_sample is None:
                variance_noise = randn_tensor(
                    model_output.shape,
                    generator=generator,
                    device=model_output.device,
                    dtype=model_output.dtype,
                )
                prev_sample = prev_sample_mean + std_dev_t * variance_noise

            log_prob = -((prev_sample.detach() - prev_sample_mean)**2)

        # mean along all but batch dimension
        log_prob = log_prob.mean()

        return prev_sample, log_prob, prev_sample_mean, std_dev_t



    # @torch.no_grad
    def _forward_flow(
        self,
        x_t: torch.Tensor,
        timestep: torch.LongTensor,
        packed_vae_token_indexes: torch.LongTensor,
        packed_vae_position_ids: torch.LongTensor,
        packed_text_ids: torch.LongTensor,
        packed_text_indexes: torch.LongTensor,
        packed_indexes: torch.LongTensor,
        packed_position_ids: torch.LongTensor,
        packed_seqlens: torch.IntTensor,
        key_values_lens: torch.IntTensor,
        past_key_values: NaiveCache,
        packed_key_value_indexes: torch.LongTensor,
        cfg_renorm_min: float = 0.0,
        cfg_renorm_type: str = "global",
        # cfg_text
        cfg_text_scale: float = 1.0,
        cfg_text_packed_position_ids: Optional[torch.LongTensor] = None,
        cfg_text_packed_query_indexes: Optional[torch.LongTensor] = None,
        cfg_text_key_values_lens: Optional[torch.Tensor] = None,
        cfg_text_past_key_values: Optional[NaiveCache] = None,
        cfg_text_packed_key_value_indexes: Optional[torch.LongTensor] = None,
        # cfg_img
        cfg_img_scale: float = 1.0,
        cfg_img_packed_position_ids: Optional[torch.LongTensor] = None,
        cfg_img_packed_query_indexes: Optional[torch.LongTensor] = None,
        cfg_img_key_values_lens: Optional[torch.Tensor] = None,
        cfg_img_past_key_values: Optional[NaiveCache] = None,
        cfg_img_packed_key_value_indexes: Optional[torch.LongTensor] = None,
        cfg_type: str = "parallel",
        ref_model: bool = False
    ):  
        if ref_model:
            forward_model = self.language_model_ref
        else:
            forward_model = self.language_model
        packed_text_embedding = forward_model.forward(mode="get_embeddings", input_ids=packed_text_ids)
        packed_sequence = packed_text_embedding.new_zeros((sum(packed_seqlens), self.hidden_size))
        packed_sequence[packed_text_indexes] = packed_text_embedding

        assert timestep.unique().shape[0] == 1
        packed_pos_embed = self.latent_pos_embed(packed_vae_position_ids)
        packed_timestep_embeds = self.time_embedder(timestep)
        x_t = self.vae2llm(x_t) + packed_timestep_embeds + packed_pos_embed
        if x_t.dtype != packed_sequence.dtype:
            x_t = x_t.to(packed_sequence.dtype)
        packed_sequence[packed_vae_token_indexes] = x_t

        extra_inputs = {}
        if self.use_moe:
            extra_inputs = {
                "mode": "gen",
                "packed_vae_token_indexes": packed_vae_token_indexes,
                "packed_text_indexes": packed_text_indexes
            }
        output = forward_model.forward(
            packed_query_sequence=packed_sequence, # packed_sequence.shape=[1026, 3584]
            query_lens=packed_seqlens,   # packed_seqlens=tensor([1026])
            packed_query_position_ids=packed_position_ids,  # packed_position_ids.shape=[1026]
            packed_query_indexes=packed_indexes,    # packed_indexes.shape=[1026]
            past_key_values=past_key_values,     
            key_values_lens=key_values_lens,    # key_values_lens=tensor([9])
            packed_key_value_indexes=packed_key_value_indexes,   # packed_key_value_indexes.shape=[9]
            update_past_key_values=False,
            is_causal=False,
            **extra_inputs,
        )
        v_t = self.llm2vae(output.packed_query_sequence)
        v_t = v_t[packed_vae_token_indexes]
        if cfg_text_scale > 1.0:
            cfg_text_output = forward_model.forward(
                packed_query_sequence=packed_sequence,
                query_lens=packed_seqlens,
                packed_query_position_ids=cfg_text_packed_position_ids,   # cfg_text_packed_position_ids.shape=[1026]
                packed_query_indexes=cfg_text_packed_query_indexes,
                past_key_values=cfg_text_past_key_values,
                key_values_lens=cfg_text_key_values_lens,
                packed_key_value_indexes=cfg_text_packed_key_value_indexes,
                update_past_key_values=False,
                is_causal=False,
                **extra_inputs,
            )
            cfg_text_v_t = self.llm2vae(cfg_text_output.packed_query_sequence)
            cfg_text_v_t = cfg_text_v_t[packed_vae_token_indexes]
        if cfg_img_scale > 1.0:
            cfg_img_output = forward_model.forward(
                packed_query_sequence=packed_sequence,
                query_lens=packed_seqlens,
                packed_query_position_ids=cfg_img_packed_position_ids,
                packed_query_indexes=cfg_img_packed_query_indexes,
                past_key_values=cfg_img_past_key_values,
                key_values_lens=cfg_img_key_values_lens,
                packed_key_value_indexes=cfg_img_packed_key_value_indexes,
                update_past_key_values=False,
                is_causal=False,
                **extra_inputs,
            )
            cfg_img_v_t = self.llm2vae(cfg_img_output.packed_query_sequence)
            cfg_img_v_t = cfg_img_v_t[packed_vae_token_indexes]

        if cfg_text_scale > 1.0:
            if cfg_renorm_type == "text_channel":
                v_t_text_ = cfg_text_v_t + cfg_text_scale * (v_t - cfg_text_v_t)
                norm_v_t = torch.norm(v_t, dim=-1, keepdim=True)
                norm_v_t_text_ = torch.norm(v_t_text_, dim=-1, keepdim=True)
                scale = (norm_v_t / (norm_v_t_text_ + 1e-8)).clamp(min=cfg_renorm_min, max=1.0)
                v_t_text = v_t_text_ * scale
                if cfg_img_scale > 1.0:
                    v_t = cfg_img_v_t + cfg_img_scale * (v_t_text - cfg_img_v_t)
                else:
                    v_t = v_t_text
            else:
                v_t_text_ = cfg_text_v_t + cfg_text_scale * (v_t - cfg_text_v_t)
                
                if cfg_img_scale > 1.0:
                    v_t_ = cfg_img_v_t + cfg_img_scale * (v_t_text_ - cfg_img_v_t)
                else:
                    v_t_ = v_t_text_

                # NOTE norm is computed over all dimensions, thus currently only supports batch_size = 1 with navit
                if cfg_renorm_type == "global":
                    norm_v_t = torch.norm(v_t)
                    norm_v_t_ = torch.norm(v_t_)
                elif cfg_renorm_type == "channel":
                    norm_v_t = torch.norm(v_t, dim=-1, keepdim=True)
                    norm_v_t_ = torch.norm(v_t_, dim=-1, keepdim=True)
                else:
                    raise NotImplementedError(f"{cfg_renorm_type} is not suppoprted")
                scale = (norm_v_t / (norm_v_t_ + 1e-8)).clamp(min=cfg_renorm_min, max=1.0)
                v_t = v_t_ * scale
        else:
            # No CFG
            pass

        return v_t

    def prepare_start_tokens(self, curr_kvlens, curr_rope, new_token_ids):
        packed_start_tokens, packed_key_value_indexes = list(), list()
        packed_query_position_ids = list()

        curr = 0
        for curr_kvlen, curr_position_id in zip(curr_kvlens, curr_rope):
            packed_key_value_indexes.extend(range(curr, curr + curr_kvlen))
            packed_start_tokens.append(new_token_ids['bos_token_id'])
            packed_query_position_ids.append(curr_position_id)
            curr += curr_kvlen

        generation_input = {
            "packed_start_tokens": torch.tensor(packed_start_tokens, dtype=torch.long),
            "packed_query_position_ids": torch.tensor(packed_query_position_ids, dtype=torch.long),
            "key_values_lens": torch.tensor(curr_kvlens, dtype=torch.int),
            "packed_key_value_indexes": torch.tensor(packed_key_value_indexes, dtype=torch.long),
        }

        return generation_input

    @torch.no_grad
    def generate_text(
        self,
        past_key_values: NaiveCache,
        packed_key_value_indexes: torch.LongTensor,
        key_values_lens: torch.IntTensor,
        packed_start_tokens: torch.LongTensor,
        packed_query_position_ids: torch.LongTensor,
        max_length: int,
        do_sample: bool = False,
        temperature: float = 1.0,
        end_token_id: int = None,
    ):
        step = 0
        generated_sequence = []
        curr_tokens = packed_start_tokens
        while step < max_length:
            generated_sequence.append(curr_tokens)
            packed_text_embedding = self.language_model.model.embed_tokens(curr_tokens)
            query_lens = torch.ones_like(curr_tokens)
            packed_query_indexes = torch.cumsum(key_values_lens, dim=0) + torch.arange(
                0, len(key_values_lens), 
                device=key_values_lens.device, 
                dtype=key_values_lens.dtype
            )

            uppacked = list(packed_key_value_indexes.split(key_values_lens.tolist(), dim=0))
            for i in range(len(uppacked)):
                uppacked[i] += i
            packed_key_value_indexes = torch.cat(uppacked, dim=0)

            extra_inputs = {}
            if self.use_moe:
                extra_inputs = {"mode": "und"}

            output = self.language_model.forward(
                packed_query_sequence=packed_text_embedding,
                query_lens=query_lens,
                packed_query_position_ids=packed_query_position_ids,
                packed_query_indexes=packed_query_indexes,
                past_key_values=past_key_values,
                key_values_lens=key_values_lens,
                packed_key_value_indexes=packed_key_value_indexes,
                update_past_key_values=True,
                is_causal=True,
                **extra_inputs,
            )
            past_key_values = output.past_key_values
            packed_query_sequence = output.packed_query_sequence
            pred_logits = self.language_model.lm_head(packed_query_sequence)

            if do_sample:
                probs = nn.functional.softmax(pred_logits / temperature, dim=-1)
                curr_tokens = torch.multinomial(probs, num_samples=1).squeeze(1)
            else:
                curr_tokens = torch.argmax(pred_logits, dim=-1)

            uppacked = list(packed_key_value_indexes.split(key_values_lens.tolist(), dim=0))
            for i in range(len(uppacked)):
                uppacked[i] = torch.cat(
                    [uppacked[i], torch.tensor([uppacked[i][-1] + 1], device=uppacked[i].device)], dim=0
                )
            packed_key_value_indexes = torch.cat(uppacked, dim=0)
            key_values_lens = key_values_lens + 1
            packed_query_position_ids = packed_query_position_ids + 1
            step += 1

            if end_token_id is not None and curr_tokens[0] == end_token_id: # only support batch=1
                break

        output_device = generated_sequence[0].device
        return torch.stack([i.to(output_device) for i in generated_sequence], dim=0)

    # for evaluation
    @torch.no_grad()
    def chat(
        self,
        tokenizer,
        new_token_ids,
        image_transform,
        images,
        prompt,
        max_length: int,
        do_sample: bool = False,
        temperature: float = 1.0,
    ):
        device = next(self.parameters()).device

        if isinstance(new_token_ids, dict):
            for k, v in new_token_ids.items():
                if torch.is_tensor(v):
                    new_token_ids[k] = v.to(device)
        elif torch.is_tensor(new_token_ids):
            new_token_ids = new_token_ids.to(device)

        # prefill
        past_key_values = NaiveCache(self.config.llm_config.num_hidden_layers)
        newlens = [0]
        new_rope = [0]

        # add images
        for image in images:
            generation_input, newlens, new_rope = self.prepare_vit_images(
                curr_kvlens=newlens,
                curr_rope=new_rope, 
                images=[image], 
                transforms=image_transform,
                new_token_ids=new_token_ids,
            )
            for k, v in generation_input.items():
                if torch.is_tensor(v):
                    generation_input[k] = v.to(device)
            with torch.amp.autocast("cuda", enabled=True, dtype=torch.bfloat16):
                past_key_values = self.forward_cache_update_vit(past_key_values, **generation_input)

        # add text
        generation_input, newlens, new_rope = self.prepare_prompts(
            curr_kvlens=newlens,
            curr_rope=new_rope, 
            prompts=[prompt],
            tokenizer=tokenizer, 
            new_token_ids=new_token_ids,
        )
        for k, v in generation_input.items():
            if torch.is_tensor(v):
                generation_input[k] = v.to(device)
        with torch.amp.autocast("cuda", enabled=True, dtype=torch.bfloat16):
            past_key_values = self.forward_cache_update_text(past_key_values, **generation_input)

        # decode
        generation_input = self.prepare_start_tokens(newlens, new_rope, new_token_ids)
        for k, v in generation_input.items():
            if torch.is_tensor(v):
                generation_input[k] = v.to(device)
        with torch.amp.autocast("cuda", enabled=True, dtype=torch.bfloat16):
            unpacked_latent = self.generate_text(
                past_key_values=past_key_values,
                max_length=max_length,
                do_sample=do_sample,
                temperature=temperature,
                end_token_id=new_token_ids['eos_token_id'],
                **generation_input,
            )
        output = tokenizer.decode(unpacked_latent[:,0])
        output = output.split('<|im_end|>')[0].split('<|im_start|>')[1]

        return output