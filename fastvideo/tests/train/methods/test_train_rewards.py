import pytest
import torch

from fastvideo.train.methods.rl.rewards import (
    MultiRewardScorer,
    build_multi_reward_scorer,
    media_to_uint8_array,
    normalize_reward_weights,
    select_first_frame,
)


def test_select_first_frame_for_video_tensor():
    video = torch.arange(2 * 3 * 4 * 5 * 6).reshape(2, 3, 4, 5, 6)

    frame = select_first_frame(video)

    assert frame.shape == (2, 3, 5, 6)
    torch.testing.assert_close(frame, video[:, :, 0])


def test_select_first_frame_keeps_frame_tensor():
    frame = torch.randn(2, 3, 5, 6)

    selected = select_first_frame(frame)

    assert selected is frame


def test_multi_reward_weighted_sum_with_injected_scorers():
    def pickscore(media, prompts):
        assert media.shape == (2, 3, 4, 5, 6)
        assert prompts == ["a", "b"]
        return torch.tensor([1.0, 2.0])

    def clipscore(media, prompts):
        assert media.shape == (2, 3, 4, 5, 6)
        assert prompts == ["a", "b"]
        return torch.tensor([0.5, 1.5])

    scorer = MultiRewardScorer(
        {"pickscore": 2.0, "clipscore": 3.0},
        scorers={
            "pickscore": pickscore,
            "clipscore": clipscore,
        },
    )

    scores = scorer(torch.zeros(2, 3, 4, 5, 6), ["a", "b"])

    torch.testing.assert_close(scores["pickscore"], torch.tensor([1.0, 2.0]))
    torch.testing.assert_close(scores["clipscore"], torch.tensor([0.5, 1.5]))
    torch.testing.assert_close(scores["avg"], torch.tensor([3.5, 8.5]))


def test_build_multi_reward_accepts_nested_diffusion_nft_config():
    scorer = build_multi_reward_scorer(
        {"rewards": {
            "pickscore": 2.0,
        }},
        device="cpu",
        scorers={"pickscore": lambda media, prompts: torch.tensor([1.5, 2.5])},
    )

    scores = scorer(torch.zeros(2, 3, 4, 5, 6), ["a", "b"])

    torch.testing.assert_close(scores["avg"], torch.tensor([3.0, 5.0]))


def test_normalize_reward_weights_reports_nested_backend():
    weights, backend = normalize_reward_weights({
        "backend": "genrl",
        "rewards": {
            "videoalign_vq": 0.75,
            "hpsv3_general": 0.25,
        },
    })

    assert backend == "genrl"
    assert weights == {
        "videoalign_vq": 0.75,
        "hpsv3_general": 0.25,
    }


def test_media_to_uint8_array_converts_video_tensor_to_nfhwc():
    media = torch.zeros(2, 3, 4, 5, 6)
    media[:, 0] = 1.0

    array = media_to_uint8_array(media)

    assert array.shape == (2, 4, 5, 6, 3)
    assert array.dtype.name == "uint8"
    assert array[..., 0].max() == 255


@pytest.mark.parametrize("num_frames", [1, 3])
def test_media_to_uint8_array_prefers_canonical_channel_axis(num_frames):
    media = torch.zeros(2, 3, num_frames, 5, 6)
    media[:, 0] = 1.0

    array = media_to_uint8_array(media)

    assert array.shape == (2, num_frames, 5, 6, 3)
    assert array[..., 0].max() == 255


def test_build_multi_reward_instantiates_debug_reward_without_device_arg():
    scorer = build_multi_reward_scorer({"mean_luminance": 1.0}, device="cpu")

    scores = scorer(torch.ones(2, 3, 4, 5, 6), ["a", "b"])

    torch.testing.assert_close(scores["avg"], torch.ones(2))


def test_multi_reward_validates_score_shape():
    scorer = MultiRewardScorer(
        {"pickscore": 1.0},
        scorers={"pickscore": lambda media, prompts: torch.tensor([[1.0], [2.0]])},
    )

    with pytest.raises(ValueError, match="must return shape"):
        scorer(torch.zeros(2, 3, 4, 5, 6), ["a", "b"])


@pytest.mark.parametrize(
    "reward_model_path",
    [
        "HPSv3/hpsv3/model/reward_model.py",
        "VideoAlign/reward_model.py",
    ],
)
def test_qwen_reward_models_delegate_multimodal_forward(reward_model_path):
    from importlib import util
    from pathlib import Path
    from types import SimpleNamespace

    path = Path(__file__).resolve().parents[4] / "fastvideo/third_party/rl_rewards" / reward_model_path
    spec = util.spec_from_file_location(f"test_{path.parent.name}_reward_model", path)
    assert spec is not None and spec.loader is not None
    module = util.module_from_spec(spec)
    spec.loader.exec_module(module)

    class Backbone:

        def __call__(self, **kwargs):
            self.kwargs = kwargs
            return (torch.arange(6, dtype=torch.float32).reshape(2, 3, 1), )

    backbone = Backbone()
    reward_head = torch.nn.Linear(1, 1, bias=False)
    with torch.no_grad():
        reward_head.weight.fill_(1.0)
    reward_model = SimpleNamespace(
        config=SimpleNamespace(
            output_attentions=False,
            output_hidden_states=False,
            use_return_dict=True,
            pad_token_id=0,
        ),
        model=backbone,
        rm_head=reward_head,
        reward_token="last",
        special_token_ids=None,
    )
    input_ids = torch.tensor([[5, 6, 0], [7, 8, 9]])
    pixel_values = torch.ones(1)
    mm_token_type_ids = torch.ones_like(input_ids)

    result = module.Qwen2VLRewardModelBT.forward(
        reward_model,
        input_ids=input_ids,
        pixel_values=pixel_values,
        mm_token_type_ids=mm_token_type_ids,
    )

    assert backbone.kwargs["input_ids"] is input_ids
    assert backbone.kwargs["pixel_values"] is pixel_values
    assert backbone.kwargs["mm_token_type_ids"] is mm_token_type_ids
    assert backbone.kwargs["return_dict"] is True
    torch.testing.assert_close(result["logits"], torch.tensor([[1.0], [5.0]]))
