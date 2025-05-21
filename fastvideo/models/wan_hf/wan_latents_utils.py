import torch

wan_latents_mean = torch.tensor([
    -0.7571,
    -0.7089,
    -0.9113,
    0.1075,
    -0.1745,
    0.9653,
    -0.1517,
    1.5508,
    0.4134,
    -0.0715,
    0.5517,
    -0.3632,
    -0.1922,
    -0.9497,
    0.2503,
    -0.2921,
]).view(1, 16, 1, 1, 1)
wan_latents_std = torch.tensor([
    2.8184,
    1.4541,
    2.3275,
    2.6558,
    1.2196,
    1.7708,
    2.6052,
    2.0743,
    3.2687,
    2.1526,
    2.8652,
    1.5579,
    1.6382,
    1.1253,
    2.8251,
    1.916,
]).view(1, 16, 1, 1, 1)



def normalize_dit_input(model_type, latents):
    latents_mean = wan_latents_mean.to(latents.device, latents.dtype)
    latents_std = wan_latents_std.to(latents.device, latents.dtype)
    latents = (latents - latents_mean) / latents_std
    return latents

