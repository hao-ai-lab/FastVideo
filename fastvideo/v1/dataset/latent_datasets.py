# SPDX-License-Identifier: Apache-2.0
import json
import os
import random

import torch
from torch.utils.data import Dataset


class LatentDataset(Dataset):

    def __init__(
        self,
        json_path,
        num_latent_t,
        cfg_rate,
    ) -> None:
        # data_merge_path: video_dir, latent_dir, prompt_embed_dir, json_path
        self.json_path = json_path
        self.cfg_rate = cfg_rate
        self.datase_dir_path = os.path.dirname(json_path)
        self.video_dir = os.path.join(self.datase_dir_path, "video")
        self.latent_dir = os.path.join(self.datase_dir_path, "latent")
        self.prompt_embed_dir = os.path.join(self.datase_dir_path,
                                             "prompt_embed")
        self.prompt_attention_mask_dir = os.path.join(self.datase_dir_path,
                                                      "prompt_attention_mask")
        with open(self.json_path) as f:
            self.data_anno = json.load(f)
        # json.load(f) already keeps the order
        # self.data_anno = sorted(self.data_anno, key=lambda x: x['latent_path'])
        self.num_latent_t = num_latent_t

        self.uncond_prompt_embed = torch.zeros(256, 4096).to(torch.float32)

        self.uncond_prompt_mask = torch.zeros(256).bool()
        self.lengths = [
            data_item.get("length", 1) for data_item in self.data_anno
        ]

    def __getitem__(self, idx):
        latent_file = self.data_anno[idx]["latent_path"]
        prompt_embed_file = self.data_anno[idx]["prompt_embed_path"]
        prompt_attention_mask_file = self.data_anno[idx][
            "prompt_attention_mask"]
        # load
        latent = torch.load(
            os.path.join(self.latent_dir, latent_file),
            map_location="cpu",
            weights_only=True,
        )
        latent = latent.squeeze(0)[:, -self.num_latent_t:]
        if random.random() < self.cfg_rate:
            prompt_embed = self.uncond_prompt_embed
            prompt_attention_mask = self.uncond_prompt_mask
        else:
            prompt_embed = torch.load(
                os.path.join(self.prompt_embed_dir, prompt_embed_file),
                map_location="cpu",
                weights_only=True,
            )
            prompt_attention_mask = torch.load(
                os.path.join(self.prompt_attention_mask_dir,
                             prompt_attention_mask_file),
                map_location="cpu",
                weights_only=True,
            )
        return latent, prompt_embed, prompt_attention_mask

    def __len__(self):
        return len(self.data_anno)


def latent_collate_function(batch):
    # return latent, prompt, latent_attn_mask, text_attn_mask
    # latent_attn_mask: # b t h w
    # text_attn_mask: b 1 l
    # needs to check if the latent/prompt' size and apply padding & attn mask
    latents, prompt_embeds, prompt_attention_masks = zip(*batch)
    # calculate max shape
    max_t = max([latent.shape[1] for latent in latents])
    max_h = max([latent.shape[2] for latent in latents])
    max_w = max([latent.shape[3] for latent in latents])

    # padding
    latent_list: list[torch.Tensor] = [
        torch.nn.functional.pad(
            latent,
            (
                0,
                max_t - latent.shape[1],
                0,
                max_h - latent.shape[2],
                0,
                max_w - latent.shape[3],
            ),
        ) for latent in latents
    ]
    # attn mask
    latent_attn_mask = torch.ones(len(latent_list), max_t, max_h, max_w)
    # set to 0 if padding
    for i, latent in enumerate(latent_list):
        latent_attn_mask[i, latent.shape[1]:, :, :] = 0
        latent_attn_mask[i, :, latent.shape[2]:, :] = 0
        latent_attn_mask[i, :, :, latent.shape[3]:] = 0

    prompt_embeds = torch.stack(prompt_embeds, dim=0)
    prompt_attention_masks = torch.stack(prompt_attention_masks, dim=0)
    latents = torch.stack(latent_list, dim=0)
    return latents, prompt_embeds, latent_attn_mask, prompt_attention_masks
