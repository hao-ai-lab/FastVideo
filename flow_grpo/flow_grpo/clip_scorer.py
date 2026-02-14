# Based on https://github.com/RE-N-Y/imscore/blob/main/src/imscore/preference/model.py

from importlib import resources
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as T
from transformers import AutoImageProcessor,CLIPProcessor, CLIPModel
import numpy as np
from PIL import Image

def get_size(size):
    if isinstance(size, int):
        return (size, size)
    elif "height" in size and "width" in size:
        return (size["height"], size["width"])
    elif "shortest_edge" in size:
        return size["shortest_edge"]
    else:
        raise ValueError(f"Invalid size: {size}")
    
def get_image_transform(processor:AutoImageProcessor):
    config = processor.to_dict()
    resize = T.Resize(get_size(config.get("size"))) if config.get("do_resize") else nn.Identity()
    crop = T.CenterCrop(get_size(config.get("crop_size"))) if config.get("do_center_crop") else nn.Identity()
    normalise = T.Normalize(mean=processor.image_mean, std=processor.image_std) if config.get("do_normalize") else nn.Identity()

    return T.Compose([resize, crop, normalise])

class ClipScorer(torch.nn.Module):
    def __init__(self, device):
        super().__init__()
        self.device=device
        self.model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14").to(device)
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
        self.tform = get_image_transform(self.processor.image_processor)
        self.eval()
    
    def _process(self, pixels):
        dtype = pixels.dtype
        pixels = self.tform(pixels)
        pixels = pixels.to(dtype=dtype)

        return pixels

    @torch.no_grad()
    def __call__(self, pixels, prompts, return_img_embedding=False):
        texts = self.processor(text=prompts, padding='max_length', truncation=True, return_tensors="pt").to(self.device)
        pixels = self._process(pixels).to(self.device)
        outputs = self.model(pixel_values=pixels, **texts)
        if return_img_embedding:
            return outputs.logits_per_image.diagonal()/30, outputs.image_embeds
        return outputs.logits_per_image.diagonal()/30

    @torch.no_grad()
    def image_similarity(self, pixels, ref_pixels):
        pixels = self._process(pixels).to(self.device)
        ref_pixels = self._process(ref_pixels).to(self.device)

        pixel_embeds = self.model.get_image_features(pixel_values=pixels)
        ref_embeds = self.model.get_image_features(pixel_values=ref_pixels)

        pixel_embeds = pixel_embeds / pixel_embeds.norm(p=2, dim=-1, keepdim=True)
        ref_embeds = ref_embeds / ref_embeds.norm(p=2, dim=-1, keepdim=True)

        sim = pixel_embeds @ ref_embeds.T
        sim = torch.diagonal(sim, 0)
        return sim


def main():
    scorer = ClipScorer(
        device='cuda'
    )

    images=[
    "assets/test.jpg",
    "assets/test.jpg"
    ]
    pil_images = [Image.open(img) for img in images]
    prompts=[
        'an image of cat',
        'not an image of cat'
    ]
    images = [np.array(img) for img in pil_images]
    images = np.array(images)
    images = images.transpose(0, 3, 1, 2)  # NHWC -> NCHW
    images = torch.tensor(images, dtype=torch.uint8)/255.0
    print(scorer(images, prompts))

if __name__ == "__main__":
    main()