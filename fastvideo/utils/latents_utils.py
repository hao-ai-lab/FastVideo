import torch

mochi_latents_mean = torch.tensor([
    -0.06730895953510081,
    -0.038011381506090416,
    -0.07477820912866141,
    -0.05565264470995561,
    0.012767231469026969,
    -0.04703542746246419,
    0.043896967884726704,
    -0.09346305707025976,
    -0.09918314763016893,
    -0.008729793427399178,
    -0.011931556316503654,
    -0.0321993391887285,
]).view(1, 12, 1, 1, 1)
mochi_latents_std = torch.tensor([
    0.9263795028493863,
    0.9248894543193766,
    0.9393059390890617,
    0.959253732819592,
    0.8244560132752793,
    0.917259975397747,
    0.9294154431013696,
    1.3720942357788521,
    0.881393668867029,
    0.9168315692124348,
    0.9185249279345552,
    0.9274757570805041,
]).view(1, 12, 1, 1, 1)
mochi_scaling_factor = 1.0


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
    if model_type == "mochi":
        latents_mean = mochi_latents_mean.to(latents.device, latents.dtype)
        latents_std = mochi_latents_std.to(latents.device, latents.dtype)
        latents = (latents - latents_mean) / latents_std
        return latents
    elif model_type == "hunyuan_hf":
        return latents * 0.476986
    elif model_type == "hunyuan":
        return latents * 0.476986
    elif model_type == "wan":
        latents_mean = wan_latents_mean.to(latents.device, latents.dtype)
        latents_std = wan_latents_std.to(latents.device, latents.dtype)
        latents = (latents - latents_mean) / latents_std
        return latents
    else:
        raise NotImplementedError(f"model_type {model_type} not supported")
