import torch
from torch.utils.data import Dataset
import json
import os

class LatentDataset(Dataset):
    def __init__(self, data_merge_path, num_latent_t):
        # data_merge_path: video_dir, latent_dir, prompt_embed_dir, json_path
        self.data_merge_path = data_merge_path
        # read txt
        with open(data_merge_path, 'r') as f:
            merge_data = f.readlines()
        video_dir, latent_dir, prompt_embed_dir, prompt_attention_mask_dir, json_path = merge_data[0].split(",")
        self.video_dir = video_dir
        self.latent_dir = latent_dir
        self.prompt_embed_dir = prompt_embed_dir
        self.prompt_attention_mask_dir = prompt_attention_mask_dir
        self.json_path = json_path
        with open(self.json_path, 'r') as f:
            self.data_anno = json.load(f)
            
        self.num_latent_t = num_latent_t
    def __getitem__(self, idx):
        latent_file = self.data_anno[idx]["latent_path"]
        prompt_embed_file = self.data_anno[idx]["prompt_embed_path"]
        prompt_attention_mask_file = self.data_anno[idx]["prompt_attention_mask"]
        # load 
        latent = torch.load(os.path.join(self.latent_dir, latent_file), map_location="cpu", weights_only=True)[:, -self.num_latent_t:]
        prompt_embed = torch.load(os.path.join(self.prompt_embed_dir, prompt_embed_file), map_location="cpu", weights_only=True)
        prompt_attention_mask = torch.load(os.path.join(self.prompt_attention_mask_dir, prompt_attention_mask_file), map_location="cpu", weights_only=True)
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
    latents = [torch.nn.functional.pad(latent, (0, max_t - latent.shape[1], 0, max_h - latent.shape[2], 0, max_w - latent.shape[3])) for latent in latents]
    # attn mask
    latent_attn_mask = torch.ones(len(latents), max_t, max_h, max_w)
    # set to 0 if padding
    for i, latent in enumerate(latents):
        latent_attn_mask[i, latent.shape[1]:, :, :] = 0
        latent_attn_mask[i, :, latent.shape[2]:, :] = 0
        latent_attn_mask[i, :, :, latent.shape[3]:] = 0
    
    prompt_embeds = torch.stack(prompt_embeds, dim=0)
    prompt_attention_masks = torch.stack(prompt_attention_masks, dim=0)
    latents = torch.stack(latents, dim=0)
    return latents, prompt_embeds, latent_attn_mask, prompt_attention_masks

if __name__ == "__main__":
    dataset = LatentDataset("data/Mochi-Synthetic-Data/merge.txt", num_latent_t=28)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=2, shuffle=False, collate_fn=latent_collate_function)
    for latent, prompt_embed, latent_attn_mask, prompt_attention_mask in dataloader:
        print(latent.shape, prompt_embed.shape, latent_attn_mask.shape, prompt_attention_mask.shape)
        import pdb; pdb.set_trace()