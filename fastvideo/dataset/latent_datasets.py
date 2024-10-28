import torch
from torch.utils.data import Dataset
import json
import os

class LatentDataset(Dataset):
    def __init__(self, data_merge_path):
        # data_merge_path: video_dir, latent_dir, prompt_embed_dir, json_path
        self.data_merge_path = data_merge_path
        video_dir, latent_dir, prompt_embed_dir, json_path = data_merge_path.split(",")
        self.video_dir = video_dir
        self.latent_dir = latent_dir
        self.prompt_embed_dir = prompt_embed_dir
        self.json_path = json_path
        with open(self.json_path, 'r') as f:
            self.data = json.load(f)
    def __getitem__(self, idx):
        latent_file = self.data[idx]["latent_path"]
        prompt_embed_file = self.data[idx]["prompt_embed_path"]
        # load 
        latent = torch.load(os.path.join(self.latent_dir, latent_file))
        prompt_embed = torch.load(os.path.join(self.prompt_embed_dir, prompt_embed_file))
        return latent, prompt_embed
    
    
def collate_function(batch):
    # return latent, prompt, attn_mask, text_attn_mask
    latents, prompt_embeds = zip(*batch)
    
    latent = torch.stack(latent, dim=0)
    prompt_embed = torch.stack(prompt_embed, dim=0)
    return latent, prompt_embed