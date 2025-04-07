import os
import torch
import numpy as np
import random
from diffusers import WanPipeline, WanTransformer3DModel, AutoencoderKLWan
from transformers import AutoConfig, UMT5EncoderModel
from diffusers.utils import export_to_video
from fastvideo.v1.distributed import init_distributed_environment, initialize_model_parallel

if __name__ == "__main__":
    init_distributed_environment(world_size=1, rank=0, local_rank=0)
    initialize_model_parallel(tensor_model_parallel_size=1,
                              sequence_model_parallel_size=1)
    model_id = "/workspace/data/Wan2.1-T2V-1.3B-Diffusers"
    # torch.manual_seed(1024)
    # np.random.seed(1024)
    # random.seed(1024)
    generator = torch.Generator("cpu").manual_seed(1024)
    # from diffusers.utils.torch_utils import randn_tensor
    # latents = randn_tensor((1, 16, 21, 60, 104),
    #                     generator=generator,
    #                     device=torch.device("cuda:0"),
    #                     dtype=torch.float32)
    # print(latents[0, 0, 0, 0, :10])

    text_encoder = UMT5EncoderModel.from_pretrained(os.path.join(
        model_id, "text_encoder"),
                                                    torch_dtype=torch.bfloat16)
    transformer = WanTransformer3DModel.from_pretrained(
        os.path.join(model_id, "transformer"), torch_dtype=torch.bfloat16)
    vae = AutoencoderKLWan.from_pretrained(os.path.join(model_id, "vae"),
                                           torch_dtype=torch.float32)

    pipe = WanPipeline.from_pretrained(model_id,
                                       text_encoder=text_encoder,
                                       transformer=transformer,
                                       vae=vae,
                                       torch_dtype=torch.bfloat16)
    pipe.enable_model_cpu_offload()
    pipe.scheduler.set_shift(7.0)
    # dtypes = set(param.dtype for param in pipe.text_encoder.parameters())
    # print(dtypes)

    prompt = "Man walking his dog in the woods on a hot sunny day"
    negative_prompt = "Bright tones, overexposed, static, blurred details, subtitles, style, works, paintings, images, static, overall gray, worst quality, low quality, JPEG compression residue, ugly, incomplete, extra fingers, poorly drawn hands, poorly drawn faces, deformed, disfigured, misshapen limbs, fused fingers, still picture, messy background, three legs, many people in the background, walking backwards"
    num_frames = 81

    frames = pipe(prompt=prompt,
                  negative_prompt=negative_prompt,
                  num_frames=num_frames,
                  guidance_scale=3,
                  generator=generator).frames[0]
    export_to_video(frames, "new_shift.mp4", fps=16)
    raise Exception
    # torch.save(latents, "latents_diffuser.pt")
    # latents = torch.load("/workspace/FastVideo/latents_diffuser.pt")

    v1_latents = torch.load("/workspace/FastVideo/latents.pt")

    assert latents.dtype == v1_latents.dtype
    assert latents.shape == v1_latents.shape, print(
        f"{latents.shape}, {v1_latents.shape}")
    max_diff = torch.max(torch.abs(latents - v1_latents)).item()
    mean_diff = torch.mean(torch.abs(latents - v1_latents)).item()
    print("Max Diff: ", max_diff)
    print("Mean Diff: ", mean_diff)
