import numpy as np
import PIL.Image
import pytest
import torch

from fastvideo.pipelines.stages.image_encoding import ImageVAEEncodingStage


def make_stage() -> ImageVAEEncodingStage:
    # Bypass __init__: preprocess() does not use self.vae.
    return ImageVAEEncodingStage.__new__(ImageVAEEncodingStage)


def test_preprocess_pil_image():
    stage = make_stage()
    arr = np.array(
        [[[0, 0, 0], [128, 128, 128], [255, 255, 255]]],
        dtype=np.uint8,
    )
    image = PIL.Image.fromarray(arr, mode="RGB")

    out = stage.preprocess(image, vae_scale_factor=1, height=1, width=3)

    assert out.dtype == torch.float32
    assert out.shape == (1, 3, 1, 3)
    torch.testing.assert_close(
        out[0, 0, 0],
        torch.tensor([-1.0, 128.0 / 255.0 * 2 - 1, 1.0]),
        atol=1e-6,
        rtol=0,
    )


def test_preprocess_uint8_tensor():
    stage = make_stage()
    image = torch.tensor(
        [[[[0, 128, 255]], [[0, 128, 255]], [[0, 128, 255]]]],
        dtype=torch.uint8,
    )

    out = stage.preprocess(image, vae_scale_factor=1, height=1, width=3)

    assert out.dtype == torch.float32
    expected = torch.tensor([-1.0, 128.0 / 255.0 * 2 - 1, 1.0])
    torch.testing.assert_close(out[0, 0, 0], expected, atol=1e-6, rtol=0)
    assert out.max().item() <= 1.0
    assert out.min().item() >= -1.0


def test_preprocess_float01_tensor_matches_uint8_path():
    stage = make_stage()
    uint8_image = torch.tensor(
        [[[[0, 128, 255]], [[0, 128, 255]], [[0, 128, 255]]]],
        dtype=torch.uint8,
    )
    float_image = uint8_image.float() / 255.0

    out_uint8 = stage.preprocess(uint8_image, vae_scale_factor=1, height=1, width=3)
    out_float = stage.preprocess(float_image, vae_scale_factor=1, height=1, width=3)

    torch.testing.assert_close(out_uint8, out_float, atol=1e-6, rtol=0)


def test_preprocess_already_normalized_passthrough():
    stage = make_stage()
    # Already in [-1, 1]; do_normalize branch must be skipped.
    image = torch.tensor(
        [[[[-1.0, 0.0, 1.0]], [[-1.0, 0.0, 1.0]], [[-1.0, 0.0, 1.0]]]],
        dtype=torch.float32,
    )

    out = stage.preprocess(image, vae_scale_factor=1, height=1, width=3)

    torch.testing.assert_close(out, image, atol=0, rtol=0)


@pytest.mark.parametrize(
    "bad_input, expected_exc",
    [
        # Float tensor outside [-1, 1] / [0, 1].
        (torch.tensor([[[[0.0, 1.5]]]], dtype=torch.float32), ValueError),
        # Non-floating, non-uint8 tensor.
        (torch.tensor([[[[0, 1]]]], dtype=torch.int32), ValueError),
        # Wrong outer type.
        (np.zeros((1, 3, 1, 3), dtype=np.float32), TypeError),
    ],
)
def test_preprocess_rejects_invalid_inputs(bad_input, expected_exc):
    stage = make_stage()
    with pytest.raises(expected_exc):
        stage.preprocess(bad_input, vae_scale_factor=1, height=1, width=2)
