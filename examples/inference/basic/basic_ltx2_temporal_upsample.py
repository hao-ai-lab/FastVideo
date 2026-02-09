# SPDX-License-Identifier: Apache-2.0
from pathlib import Path

import torch

from fastvideo.configs.pipelines import PipelineConfig
from fastvideo.entrypoints.upsample import (_prepare_video, _read_video,
                                            _write_video)
from fastvideo.fastvideo_args import FastVideoArgs
from fastvideo.logger import init_logger
from fastvideo.models.loader.component_loader import UpsamplerLoader, VAELoader
from fastvideo.models.upsamplers import upsample_video
from fastvideo.utils import maybe_download_model

logger = init_logger(__name__)

# Input/output
INPUT_VIDEO = "outputs_video/ltx2_basic/output_ltx2_distilled_t2v.mp4"
OUTPUT_VIDEO = "outputs_video/ltx2_upscale/ltx2_temporal_upscale_x2.mp4"

# Diffusers-style LTX-2 repo with upsamplers included
MODEL_ID = "FastVideo/LTX2-Diffusers"

# Controls
DOUBLE_FPS = True


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    precision_str = "bf16" if torch.cuda.is_available() else "fp32"
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

    video, fps = _read_video(INPUT_VIDEO)
    video = _prepare_video(
        video,
        trim_frames=False,
        pad_frames=True,
        crop_multiple=32,
    )

    model_root = maybe_download_model(MODEL_ID)
    vae_path = str(Path(model_root) / "vae")
    temporal_upsampler_path = str(Path(model_root) / "temporal_upsampler")

    args = FastVideoArgs(
        model_path=vae_path,
        pipeline_config=PipelineConfig(vae_precision=precision_str),
        vae_cpu_offload=False,
    )
    vae = VAELoader().load(vae_path, args).to(device=device, dtype=dtype)
    temporal_upsampler = UpsamplerLoader().load(
        temporal_upsampler_path, args).to(device=device, dtype=dtype)

    if hasattr(vae.decoder, "decode_noise_scale"):
        vae.decoder.decode_noise_scale = 0.0

    # [F, C, H, W] -> [B, C, F, H, W]
    video = video.unsqueeze(0).permute(0, 2, 1, 3, 4).to(
        device=device, dtype=dtype)

    with torch.no_grad():
        latents = vae.encoder(video)
        up_latents = upsample_video(latents, vae.encoder,
                                    getattr(temporal_upsampler, "model",
                                            temporal_upsampler))

        timestep_value = getattr(vae.decoder, "decode_timestep", 0.05)
        timestep = torch.full((video.shape[0], ),
                              float(timestep_value),
                              device=device,
                              dtype=dtype)
        decoded = vae.decoder(up_latents, timestep=timestep)

    # [B, C, F, H, W] -> [F, C, H, W]
    decoded = decoded[0].permute(1, 0, 2, 3).detach().cpu()

    output_fps = fps * 2 if DOUBLE_FPS else fps
    Path(OUTPUT_VIDEO).parent.mkdir(parents=True, exist_ok=True)
    _write_video(decoded, OUTPUT_VIDEO, output_fps)
    logger.info("Temporal upsampled video saved to %s", OUTPUT_VIDEO)


if __name__ == "__main__":
    main()
