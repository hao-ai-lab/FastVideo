from unittest.mock import MagicMock

import numpy as np
import PIL.Image
import pytest
import torch

from fastvideo.fastvideo_args import ExecutionMode, FastVideoArgs
from fastvideo.pipelines.pipeline_batch_info import ForwardBatch
from fastvideo.pipelines.stages.image_encoding import SVIImageVAEEncodingStage


def _diffsynth_preprocess(img: PIL.Image.Image) -> torch.Tensor:
    """Reference (diffsynth) preprocess: uint8 * (2/255) - 1, permute to CHW."""
    arr = np.array(img, dtype=np.float32) * (2 / 255) - 1
    return torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)


def _reference_pre_vae(
    first_frames: list[PIL.Image.Image],
    random_ref_frame: PIL.Image.Image,
    num_frames: int,
    height: int,
    width: int,
    ref_pad_num: int,
    ref_pad_cfg: bool,
    vae_scale: int,
    temporal_compression: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Mirror of encode_images_adaptive mask + video_condition construction."""
    num_condition_frames = len(first_frames)
    remaining_frames = num_frames - num_condition_frames
    random_ref = _diffsynth_preprocess(random_ref_frame)

    msk = torch.ones(1, num_frames, height // vae_scale, width // vae_scale, dtype=torch.float32)
    if ref_pad_cfg:
        msk[:, num_condition_frames:] = 0
    else:
        msk[:, 1:] = 0
    msk = torch.concat(
        [torch.repeat_interleave(msk[:, 0:1], repeats=temporal_compression, dim=1), msk[:, 1:]],
        dim=1,
    )
    msk = msk.view(1, msk.shape[1] // temporal_compression, temporal_compression, height // vae_scale,
                   width // vae_scale)
    msk = msk.transpose(1, 2)[0]

    if len(first_frames) > 1:
        frame_tensors = [_diffsynth_preprocess(f) for f in first_frames]
        vae_input_condition = torch.cat(frame_tensors, dim=0).permute(1, 0, 2, 3)
    else:
        vae_input_condition = _diffsynth_preprocess(first_frames[0]).transpose(0, 1)

    if ref_pad_num == 0:
        vae_input_pad = torch.zeros(3, remaining_frames, height, width, dtype=torch.float32)
    elif ref_pad_num > 0:
        pad_imgs = [random_ref.transpose(0, 1)] * ref_pad_num
        if remaining_frames > ref_pad_num:
            pad_imgs += [torch.zeros(3, 1, height, width, dtype=torch.float32)] * (remaining_frames - ref_pad_num)
        vae_input_pad = torch.cat(pad_imgs, dim=1)
    elif ref_pad_num == -1:
        vae_input_pad = random_ref.transpose(0, 1).repeat(1, remaining_frames, 1, 1)
    else:
        raise ValueError(ref_pad_num)

    return msk, torch.concat([vae_input_condition, vae_input_pad], dim=1)


class _CaptureVAE:
    """Mock VAE that records its encode() input and returns a zero latent."""

    spatial_compression_ratio = 8
    temporal_compression_ratio = 4
    scaling_factor = 1.0
    shift_factor = None

    def __init__(self):
        self.captured_input: torch.Tensor | None = None

    def to(self, *_a, **_kw):
        return self

    def enable_tiling(self):
        pass

    def encode(self, x: torch.Tensor):
        self.captured_input = x.detach().clone()
        t_lat = (x.shape[2] - 1) // self.temporal_compression_ratio + 1
        h_lat = x.shape[3] // self.spatial_compression_ratio
        w_lat = x.shape[4] // self.spatial_compression_ratio
        latent = torch.zeros(x.shape[0], 16, t_lat, h_lat, w_lat, device=x.device, dtype=torch.float32)
        out = MagicMock()
        out.sample = lambda _g: latent
        out.mode = lambda: latent
        return out


def _make_pil(h: int, w: int, seed: int) -> PIL.Image.Image:
    arr = np.random.default_rng(seed).integers(0, 256, size=(h, w, 3), dtype=np.uint8)
    return PIL.Image.fromarray(arr)


def _make_fastvideo_args() -> FastVideoArgs:
    args = MagicMock(spec=FastVideoArgs)
    args.mode = ExecutionMode.INFERENCE
    args.disable_autocast = True
    args.pipeline_config = MagicMock()
    args.pipeline_config.vae_precision = "fp32"
    args.pipeline_config.vae_tiling = False
    return args


@pytest.mark.parametrize(
    "num_motion, ref_pad_num, ref_pad_cfg",
    [
        (1, -1, False),
        (5, 0, False),
        (1, 3, False),
        (5, -1, True),
    ],
)
def test_pre_vae_parity(num_motion: int, ref_pad_num: int, ref_pad_cfg: bool):
    height, width, num_frames = 48, 96, 17

    first_frames = [_make_pil(height, width, seed=i) for i in range(num_motion)]
    random_ref = _make_pil(height, width, seed=999)

    expected_msk, expected_video = _reference_pre_vae(
        first_frames=first_frames,
        random_ref_frame=random_ref,
        num_frames=num_frames,
        height=height,
        width=width,
        ref_pad_num=ref_pad_num,
        ref_pad_cfg=ref_pad_cfg,
        vae_scale=8,
        temporal_compression=4,
    )

    vae = _CaptureVAE()
    stage = SVIImageVAEEncodingStage(vae=vae)  # type: ignore[arg-type]

    batch = ForwardBatch(
        data_type="video",
        generator=torch.Generator().manual_seed(0),
        height=height,
        width=width,
        num_frames=num_frames,
        pil_image=first_frames[0],
        svi_first_frames=first_frames,
        svi_random_ref_frame=random_ref,
        svi_ref_pad_num=ref_pad_num,
        svi_ref_pad_cfg=ref_pad_cfg,
    )
    stage.forward(batch, _make_fastvideo_args())

    assert vae.captured_input is not None
    # Stage feeds (1, 3, T, H, W); reference produces (3, T, H, W).
    # .cpu() keeps the comparison device-agnostic when get_local_torch_device() picks CUDA.
    actual_video = vae.captured_input.squeeze(0).cpu()
    assert actual_video.shape == expected_video.shape
    # FastVideo preprocess is (uint8/255) -> 2x-1; diffsynth is uint8*(2/255) - 1. Same math, ~1 ULP apart.
    torch.testing.assert_close(actual_video, expected_video, atol=2e-7, rtol=1e-5)

    assert batch.image_latent is not None
    actual_msk = batch.image_latent[0, :4].cpu()
    assert actual_msk.shape == expected_msk.shape
    torch.testing.assert_close(actual_msk, expected_msk, atol=0.0, rtol=0.0)
