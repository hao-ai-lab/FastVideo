"""Phase 1a: per-model sampling defaults on the card + the kwargs>SamplingParam>card>generic precedence."""
from types import SimpleNamespace

from v2.card import SamplingDefaults
from v2.recipes.ltx2 import build_ltx2_3_card, build_ltx2_base_card, build_ltx2_card
from v2.recipes.wan21 import (
    build_wan21_card,
    build_wan22_a14b_card,
    build_wan22_ti2v_card,
    build_wan_t2v_14b_card,
)
from v2.recipes.wan_causal import build_wan_causal_card
from v2.registry import resolve
from v2.video_generator import _resolve_default


def test_wan_t2v_14b_defaults_and_registry():
    # Bucket-B port: reuses the Wan recipe/adapter, only defaults differ (720p, flow_shift 5.0).
    sd = build_wan_t2v_14b_card().sampling_defaults
    assert (sd.num_steps, sd.guidance_scale, sd.height, sd.width, sd.fps) == (50, 5.0, 720, 1280, 16)
    # the HF id resolves to the 14B card via the shared registry (not the 1.3B fallback defaults)
    build_card, _ = resolve("Wan-AI/Wan2.1-T2V-14B-Diffusers")
    assert build_card().sampling_defaults.height == 720


def test_wan21_defaults():
    sd = build_wan21_card().sampling_defaults
    assert (sd.num_steps, sd.guidance_scale, sd.height, sd.width, sd.num_frames, sd.fps) == (50, 3.0, 480, 832, 81, 16)
    assert "overexposed" in sd.negative_prompt


def test_wan22_ti2v_defaults():
    sd = build_wan22_ti2v_card().sampling_defaults
    assert (sd.num_steps, sd.guidance_scale, sd.height, sd.width, sd.num_frames, sd.fps) == (50, 5.0, 704, 1280, 121, 24)


def test_wan22_a14b_defaults():
    sd = build_wan22_a14b_card().sampling_defaults
    assert (sd.num_steps, sd.guidance_scale, sd.fps) == (40, 4.0, 16)


def test_wan_causal_defaults():
    sd = build_wan_causal_card().sampling_defaults
    assert (sd.num_steps, sd.guidance_scale, sd.height, sd.width, sd.num_frames, sd.fps) == (4, 1.0, 480, 832, 81, 16)


def test_ltx2_two_stage_defaults():
    sd = build_ltx2_card().sampling_defaults
    assert (sd.num_steps, sd.guidance_scale, sd.height, sd.width, sd.num_frames, sd.fps) == (8, 1.0, 1024, 1536, 121, 24)
    assert sd.negative_prompt == ""        # distilled: empty negative prompt


def test_ltx2_base_defaults():
    sd = build_ltx2_base_card().sampling_defaults
    assert (sd.num_steps, sd.guidance_scale, sd.height, sd.width) == (40, 3.0, 512, 768)
    assert "blurry" in sd.negative_prompt


def test_ltx2_3_defaults():
    sd = build_ltx2_3_card().sampling_defaults
    assert (sd.num_steps, sd.guidance_scale, sd.height, sd.width) == (30, 3.0, 512, 768)
    assert sd.guidance_per_modality == {"video": 3.0, "audio": 7.0}


def test_resolve_precedence():
    sd = SamplingDefaults(num_steps=8, guidance_scale=1.0, height=1024)
    # card default used when neither kwargs nor SamplingParam set it (note the num_steps alias)
    assert _resolve_default("num_inference_steps", 30, {}, None, sd, "num_steps") == 8
    assert _resolve_default("height", 480, {}, None, sd) == 1024
    # kwargs override card
    assert _resolve_default("num_inference_steps", 30, {"num_inference_steps": 4}, None, sd, "num_steps") == 4
    # SamplingParam override (no kwargs) beats card
    assert _resolve_default("height", 480, {}, SimpleNamespace(height=720), sd) == 720
    # generic fallback when card has no value and nothing else set
    assert _resolve_default("width", 832, {}, None, sd) == 832
    # empty-string negative prompt is a real value, not "unset"
    assert _resolve_default("negative_prompt", None, {}, None, SamplingDefaults(negative_prompt="")) == ""
