# SPDX-License-Identifier: Apache-2.0
"""CPU checks for the Open-H3-Regenerate-2K two-Spark opt-ins.

The peer Sol-Attn exchange itself needs two CUDA ranks and is covered by the two-Spark run recorded in the PR; these
tests pin the pieces that are exact by construction: fast-decoder token chunking, the native-block slice decode and
its assembly, the ``to_out`` M pad, and the LoRA adapter release gate.
"""

from types import SimpleNamespace

import pytest
import torch

import fastvideo.envs as envs
from fastvideo.models.dits.minimax_h3 import MiniMaxH3Attention
from fastvideo.models.vaes.minimax_h3_fast_vae import BLOCK, FRAMES_TO_TRIM, PATCH_T, FastH3VaeDecoder
from fastvideo.models.vaes.minimax_h3_fast_vae_decoder import NonCausalViTDecoder
from fastvideo.pipelines.basic.open_h3_regenerate_2k.stages import latent_preparation
from fastvideo.pipelines.lora_pipeline import LoRAPipeline


def _tiny_fast_decoder(token_chunk: int = 0) -> FastH3VaeDecoder:
    torch.manual_seed(0)
    model = NonCausalViTDecoder([0.0] * 24, [1.0] * 24, list(range(2)), time_norm=5, heads=2, dim_head=8)
    model.token_chunk = token_chunk
    decoder = FastH3VaeDecoder.__new__(FastH3VaeDecoder)
    torch.nn.Module.__init__(decoder)
    decoder.model = model.eval()
    decoder._forward = decoder.model
    decoder.latent_channels = 24
    decoder.spatial_compression_ratio = 16
    decoder.register_buffer("pixel_mean", torch.zeros(1, 3, 1, 1, 1), persistent=False)
    decoder.register_buffer("pixel_std", torch.ones(1, 3, 1, 1, 1), persistent=False)
    return decoder


@pytest.mark.parametrize("latent_frames", [5, 7, 12])
def test_fast_vae_token_chunk_matches_the_whole_layer(latent_frames):
    z = torch.randn(1, latent_frames, 24, 2, 3)
    whole = _tiny_fast_decoder()
    # 7 tokens is smaller than one latent frame's H*W*block, so both the FFN and the output projection slice.
    chunked = _tiny_fast_decoder(token_chunk=7)
    chunked.model.load_state_dict(whole.model.state_dict())
    with torch.no_grad():
        expected = whole.model(z)
        got = chunked.model(z)
    assert got.shape == expected.shape == whole.decoded_pixel_shape(z.shape)
    torch.testing.assert_close(got, expected, rtol=1e-5, atol=1e-6)


class _FakeSPGroup:
    """Single-process stand-in: ``all_gather`` returns the segments every rank contributed, in rank order."""

    def __init__(self, world_size: int, rank: int, segments: dict[int, torch.Tensor]):
        self.world_size = world_size
        self.rank_in_group = rank
        self.segments = segments

    def all_gather(self, tensor: torch.Tensor, dim: int) -> torch.Tensor:
        self.segments[self.rank_in_group] = tensor.clone()
        if len(self.segments) < self.world_size:
            return tensor
        return torch.cat([self.segments[r] for r in range(self.world_size)], dim=dim)


def _frame_coded_decoder(t_latent: int) -> FastH3VaeDecoder:
    """A decoder whose pixels encode the global output frame index, so misplaced frames are visible."""
    decoder = _tiny_fast_decoder()
    block_frames = BLOCK * PATCH_T - FRAMES_TO_TRIM

    def decode_range(z, start, stop):
        assert start % BLOCK == 0
        length = stop - start
        frames = length * PATCH_T - FRAMES_TO_TRIM * ((length + BLOCK - 1) // BLOCK)
        index = torch.arange(frames, dtype=torch.float32) + (start // BLOCK) * block_frames
        return (index / 1000.0).view(1, 1, -1, 1, 1).expand(z.shape[0], 3, -1, 2, 2).clone()

    decoder.decode_range = decode_range
    return decoder


@pytest.mark.parametrize(("t_latent", "world"), [(10, 2), (15, 2), (12, 2), (15, 3), (5, 2)])
def test_fast_vae_parallel_decode_places_every_rank_segment(t_latent, world):
    decoder = _frame_coded_decoder(t_latent)
    z = torch.zeros(1, t_latent, 24, 2, 2)
    num_blocks = (t_latent + BLOCK - 1) // BLOCK
    total_frames = t_latent * PATCH_T - FRAMES_TO_TRIM * num_blocks
    segments: dict[int, torch.Tensor] = {}
    with envs.FASTVIDEO_H3_VAE_UINT8_PIXELS.override(False):
        # The non-assembling ranks post first so the fake collective has every segment when rank 0 gathers.
        for rank in range(1, world):
            assert decoder.decode_to_pixels_parallel(z, None, _FakeSPGroup(world, rank, segments)) is None
        output = torch.full((1, 3, total_frames, 2, 2), -1.0)
        decoder.decode_to_pixels_parallel(z, output, _FakeSPGroup(world, 0, segments))
    expected = (torch.arange(total_frames, dtype=torch.float32) / 1000.0).view(1, 1, -1, 1, 1).expand_as(output)
    torch.testing.assert_close(output, expected)


def test_fast_vae_slice_decode_with_halo_matches_the_full_decode():
    decoder = _tiny_fast_decoder()
    # Keep the CPU check in fp32: the decode's CUDA autocast would otherwise pick bf16 for CPU linears.
    decoder.decode_autocast_dtype = torch.float32
    z = torch.randn(1, 15, 24, 2, 3)
    with torch.no_grad():
        full = decoder.decode_range(z, 0, 15)
    segments: dict[int, torch.Tensor] = {}
    output = torch.empty_like(full)
    with torch.no_grad(), envs.FASTVIDEO_H3_VAE_UINT8_PIXELS.override(False):
        decoder.decode_to_pixels_parallel(z, None, _FakeSPGroup(2, 1, segments))
        decoder.decode_to_pixels_parallel(z, output, _FakeSPGroup(2, 0, segments))
    torch.testing.assert_close(output, full, rtol=1e-4, atol=1e-5)


class _RecordingLinear:

    def __init__(self, weight: torch.Tensor):
        self.weight = weight
        self.rows: list[int] = []

    def __call__(self, hidden_states: torch.Tensor):
        self.rows.append(hidden_states.shape[1])
        return hidden_states @ self.weight, None


@pytest.mark.parametrize(("pad_m", "expected_rows"), [(0, 5), (16, 16), (4, 5)])
def test_sol_out_projection_pads_rows_and_drops_them(pad_m, expected_rows):
    torch.manual_seed(0)
    attn = SimpleNamespace(to_out=_RecordingLinear(torch.randn(8, 6)))
    hidden = torch.randn(1, 5, 8)
    with envs.FASTVIDEO_H3_SOLSP_OUT_PAD_M.override(pad_m):
        out = MiniMaxH3Attention._sol_out_projection(attn, hidden)
    assert attn.to_out.rows == [expected_rows]
    torch.testing.assert_close(out, hidden @ attn.to_out.weight)


class _FakeLoRALayer:

    def __init__(self, side_branch: bool = False):
        self.side_branch = side_branch
        self.lora_A = torch.ones(2, 4)
        self.lora_B = torch.ones(4, 2)


def _fake_lora_pipeline(*, constructor: bool, runtime_adapter=None, training: bool = False):
    merged, side = _FakeLoRALayer(), _FakeLoRALayer(side_branch=True)
    layers = SimpleNamespace(lora_layers_by_block=lambda: [(None, {"merged": merged, "side": side})])
    pipeline = SimpleNamespace(
        training_mode=training,
        _setting_constructor_adapter=constructor,
        _runtime_adapter=runtime_adapter,
        lora_layers={"transformer": layers},
        lora_adapters={"base": {"w": torch.ones(1)}},
        lora_adapter_paths={"base": "/weights/adapter"},
        cur_adapter_name="base",
        cur_adapter_path="/weights/adapter",
    )
    return pipeline, merged, side


def test_release_lora_adapter_is_off_by_default():
    pipeline, merged, _ = _fake_lora_pipeline(constructor=True)
    LoRAPipeline._maybe_release_adapter_factors(pipeline, "base")
    assert merged.lora_A is not None and "base" in pipeline.lora_adapters


def test_release_lora_adapter_drops_merged_constructor_factors():
    pipeline, merged, side = _fake_lora_pipeline(constructor=True)
    with envs.FASTVIDEO_H3_RELEASE_LORA_ADAPTER.override(True):
        LoRAPipeline._maybe_release_adapter_factors(pipeline, "base")
    assert merged.lora_A is None and merged.lora_B is None
    assert merged.lora_factors_released
    # Side-branch layers apply their factors on every forward, so they keep them.
    assert side.lora_A is not None
    # The path cache is cleared so a later rematerialize reloads the adapter instead of running unadapted.
    assert pipeline.lora_adapters == {} and pipeline.lora_adapter_paths == {}
    assert pipeline.cur_adapter_name == "" and pipeline.cur_adapter_path == ""


@pytest.mark.parametrize(("constructor", "runtime_adapter", "training"), [
    (False, None, False),
    (True, ("style", 1.0), False),
    (True, None, True),
])
def test_release_lora_adapter_keeps_runtime_and_training_factors(constructor, runtime_adapter, training):
    pipeline, merged, _ = _fake_lora_pipeline(constructor=constructor, runtime_adapter=runtime_adapter,
                                              training=training)
    with envs.FASTVIDEO_H3_RELEASE_LORA_ADAPTER.override(True):
        LoRAPipeline._maybe_release_adapter_factors(pipeline, "base")
    assert merged.lora_A is not None and "base" in pipeline.lora_adapters


class _FakeDraftCache:
    enabled = True

    def __init__(self, hit: torch.Tensor | None):
        self.hit = hit
        self.stored: list[torch.Tensor] = []

    def get(self, key):
        return self.hit

    def put(self, key, value):
        self.stored.append(value)


@pytest.mark.parametrize(("peer_hit", "expect_encode"), [(False, True), (True, False)])
def test_draft_cache_hit_is_used_only_when_every_parallel_rank_hit(monkeypatch, tmp_path, peer_hit, expect_encode):
    draft = tmp_path / "draft.mp4"
    draft.write_bytes(b"draft")
    cached = torch.ones(2, 3)
    encoded = torch.zeros(2, 3)
    calls: list[str] = []
    stage = SimpleNamespace(
        _draft_cache=_FakeDraftCache(cached),
        draft_size=(64, 64),
        _encode_draft=lambda path, device, args: calls.append(path) or encoded,
    )
    monkeypatch.setattr(latent_preparation, "_parallel_encode_group", lambda args: SimpleNamespace(device_group=None))

    def all_reduce(tensor, op=None, group=None):
        # This rank hit; the peer's flag decides the group minimum.
        tensor.fill_(min(int(tensor.item()), int(peer_hit)))

    monkeypatch.setattr(torch.distributed, "all_reduce", all_reduce)
    args = SimpleNamespace(model_path="/models/2k")
    got = latent_preparation.Regenerate2KLatentPreparationStage._encode_draft_cached(
        stage, str(draft), torch.device("cpu"), args)
    assert bool(calls) == expect_encode
    torch.testing.assert_close(got, encoded if expect_encode else cached)
