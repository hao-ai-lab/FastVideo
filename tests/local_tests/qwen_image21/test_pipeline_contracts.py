# SPDX-License-Identifier: Apache-2.0
"""CPU tests of the actual Qwen stages with small deterministic components.

Imports are isolated from FastVideo's optional CUDA runtime. These test request
flow and image/layout math; they do not stand in for released-weight parity.
"""

import ast
import importlib.util
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest
import torch
from PIL import Image
from torch import nn

ROOT = Path(__file__).resolve().parents[3]


@pytest.fixture
def inputs(monkeypatch):
    name = "qwen21_inputs_test"
    path = ROOT / "fastvideo/pipelines/basic/qwen_image21/inputs.py"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, name, module)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def stages(monkeypatch, inputs):
    class StripRuntimeImports(ast.NodeTransformer):

        def visit_ImportFrom(self, node):
            if node.level or (node.module and node.module.startswith("fastvideo")) or node.module == "torch.distributed.tensor":
                return ast.copy_location(ast.Pass(), node)
            return node

    class Stage:
        pass

    class Cache:
        instances = []

        def __init__(self, layers, storage_device):
            self.cleared = False
            self.instances.append(self)

        def clear(self):
            self.cleared = True

    class AllocationError(RuntimeError):
        pass

    class Validation:

        def add_check(self, *_args):
            return self

    module = ModuleType("qwen21_stages_test")
    module.__dict__.update({name: getattr(inputs, name) for name in dir(inputs) if not name.startswith("__")})
    module.__dict__.update(
        PipelineStage=Stage,
        ForwardBatch=SimpleNamespace,
        FastVideoArgs=SimpleNamespace,
        VerificationResult=Validation,
        V=SimpleNamespace(),
        init_logger=lambda _name: SimpleNamespace(warning=lambda *_args: None),
        get_local_torch_device=lambda: torch.device("cpu"),
        pinned_offload=SimpleNamespace(load=lambda *_args, **_kwargs: None, unload=lambda *_args: None),
        ModuleHookManager=SimpleNamespace(get_from=lambda _block: None),
        QwenImage21KVCache=Cache,
        QwenImage21KVCacheAllocationError=AllocationError,
        DTensor=type("DTensorSentinel", (), {}),
    )
    path = ROOT / "fastvideo/pipelines/basic/qwen_image21/stages.py"
    tree = StripRuntimeImports().visit(ast.parse(path.read_text()))
    exec(compile(ast.fix_missing_locations(tree), str(path), "exec"), module.__dict__)
    return module


def request(**updates):
    values = dict(prompt="draw", negative_prompt=None, references=None, image_path=None, pil_image=None,
                  height=64, width=64, reference_resolution=32, num_frames=1, num_videos_per_prompt=1,
                  num_inference_steps=2, guidance_scale=1, true_cfg_scale=1, video_path=None,
                  conditioning_mask=None, prompt_embeds=[], negative_prompt_embeds=None,
                  extra={}, latents=None, image_latent=None, generator=None, seed=42,
                  use_kv_cache=True, sigmas=None)
    values.update(updates)
    return SimpleNamespace(**values)


def arguments():
    return SimpleNamespace(tp_size=1, sp_size=1, num_gpus=1, text_encoder_cpu_offload=False,
                           vae_cpu_offload=False, dit_cpu_offload=False, dit_layerwise_offload=False,
                           use_fsdp_inference=False, pin_cpu_memory=False, output_type="pil",
                           pipeline_config=SimpleNamespace(dit_config=SimpleNamespace(arch_config=SimpleNamespace(in_channels=2)),
                                                           vae_tiling=True, kv_cache_device="cpu"))


class Processor:
    tokenizer = SimpleNamespace(encode=lambda _text: [9])

    def apply_chat_template(self, *_args, **_kwargs):
        return [[1, 2]]

    def __call__(self, text, images=None, **kwargs):
        assert kwargs["padding_side"] == "left"
        if images:
            assert all(image.mode == "RGB" for image in images)
        last_id = 5 if "bad" in text[0] else 4
        ids = torch.tensor([[1, 2, 3] + [9] * len(images or []) + [last_id]])
        return dict(input_ids=ids, attention_mask=torch.ones_like(ids))


class Encoder(nn.Module):

    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(1))

    def forward(self, input_ids, **_kwargs):
        return input_ids.float().unsqueeze(-1).repeat(1, 1, 2)


class VAE(nn.Module):

    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(1))
        self.config = SimpleNamespace(latents_mean=(0., 0.), latents_std=(1., 1.))

    def enable_tiling(self):
        pass

    def disable_tiling(self):
        pass

    def encode(self, pixels):
        latent = torch.nn.functional.avg_pool2d(pixels[:, :2, 0], 16).unsqueeze(2)
        return SimpleNamespace(latent_dist=SimpleNamespace(mode=lambda: latent))

    def decode(self, latents, return_dict=False):
        image = latents[:, :1].repeat_interleave(16, 3).repeat_interleave(16, 4)
        return (torch.cat((image.repeat(1, 3, 1, 1, 1), torch.full_like(image, -0.5)), dim=1),)


class Scheduler:
    config = dict(base_image_seq_len=256, max_image_seq_len=8192, base_shift=.5, max_shift=.9)

    def set_timesteps(self, sigmas, device, mu):
        self.timesteps = torch.tensor(sigmas, device=device) * 1000
        self.mu = mu

    def set_begin_index(self, index):
        assert index == 0

    def step(self, noise, timestep, latents, return_dict=False):
        return (latents - noise * .1,)


class Transformer:
    config = SimpleNamespace(causal_condition=True)
    transformer_blocks = [SimpleNamespace()]

    def __init__(self):
        self.calls = []

    def __call__(self, hidden_states, encoder_hidden_states, kv_cache_mode, **kwargs):
        self.calls.append((kv_cache_mode, kwargs["kv_cache"]))
        noise = encoder_hidden_states[:, -1, :].unsqueeze(1).expand_as(hidden_states)
        return (noise,)


def run_stages(stages, batch, transformer=None):
    args, vae, scheduler = arguments(), VAE(), Scheduler()
    transformer = transformer or Transformer()
    pipeline = [stages.QwenImage21InputStage(), stages.QwenImage21TextEncodingStage(Encoder(), Processor()),
                stages.QwenImage21LatentStage(vae), stages.QwenImage21ScheduleStage(scheduler),
                stages.QwenImage21DenoisingStage(transformer, scheduler), stages.QwenImage21DecodingStage(vae)]
    for stage in pipeline:
        batch = stage.forward(batch, args)
    return batch, transformer, scheduler


def test_t2i_rgba_and_request_cache_cleanup(stages):
    first, transformer, _ = run_stages(stages, request())
    second, _, _ = run_stages(stages, request())
    torch.testing.assert_close(first.output, second.output)
    assert first.output.shape == (1, 4, 1, 64, 64)
    torch.testing.assert_close(first.output[:, 3], torch.full((1, 1, 64, 64), .25))
    assert [mode for mode, _ in transformer.calls] == ["extract", "cached"]
    assert all(cache.cleared for cache in stages.QwenImage21KVCache.instances)


@pytest.mark.parametrize("count", [1, 2, 10])
def test_ordered_references_share_the_edit_pipeline(stages, inputs, count):
    refs = [Image.new("RGBA", (32, 32), (index * 20, 0, 0, 128)) for index in range(count)]
    batch, _, scheduler = run_stages(stages, request(references=refs))
    assert batch.image_latent.shape == (1, count * 4, 2)
    reds = batch.image_latent[0, ::4, 0]
    assert torch.all(reds[1:] > reds[:-1])
    assert batch.extra[inputs.STATE_KEY].img_shapes == [[(1, 2, 2)] * count + [(1, 4, 4)]]
    assert scheduler.mu == inputs.schedule_shift(16, scheduler.config)


def test_true_cfg_requires_explicit_negative_and_separate_caches(stages):
    initial = torch.zeros(1, 16, 2)
    guided, transformer, _ = run_stages(stages, request(latents=initial.clone(), true_cfg_scale=2, negative_prompt="bad"))
    torch.testing.assert_close(guided.latents, torch.full_like(initial, -.6))
    assert transformer.calls[0][1] is not transformer.calls[1][1]
    unguided, transformer, _ = run_stages(stages, request(latents=initial.clone(), true_cfg_scale=2))
    assert not unguided.do_classifier_free_guidance
    assert len(transformer.calls) == 2
    torch.testing.assert_close(unguided.latents, torch.full_like(initial, -.8))


def test_host_cache_allocation_failure_retries_uncached(stages):
    class FailingTransformer(Transformer):

        def __call__(self, **kwargs):
            if kwargs["kv_cache_mode"] == "extract":
                raise stages.QwenImage21KVCacheAllocationError("simulated host allocation failure")
            return super().__call__(**kwargs)

    batch, transformer, _ = run_stages(stages, request(), FailingTransformer())
    assert [mode for mode, _ in transformer.calls] == [None, None]
    assert all(cache.cleared for cache in stages.QwenImage21KVCache.instances)
    expected, _, _ = run_stages(stages, request(use_kv_cache=False))
    torch.testing.assert_close(batch.output, expected.output)


def test_unrelated_model_failure_propagates_and_clears_cache(stages):
    class BrokenTransformer(Transformer):

        def __call__(self, **kwargs):
            raise RuntimeError("model failure")

    with pytest.raises(RuntimeError, match="model failure"):
        run_stages(stages, request(), BrokenTransformer())
    assert all(cache.cleared for cache in stages.QwenImage21KVCache.instances)


@pytest.mark.parametrize("updates,match", [({"num_frames": 2}, "num_frames"), ({"guidance_scale": 3}, "true_cfg_scale"),
                                         ({"height": 63}, "multiple of 32"), ({"true_cfg_scale": float("nan")}, "finite"),
                                         ({"num_videos_per_prompt": 2}, "num_videos_per_prompt"),
                                         ({"conditioning_mask": torch.ones(1)}, "mask images")])
def test_unsupported_requests_fail_before_loading(stages, updates, match):
    with pytest.raises(ValueError, match=match):
        stages.QwenImage21InputStage().forward(request(**updates), arguments())


def test_rgba_png_roundtrip_preserves_alpha(inputs, tmp_path):
    rgba = Image.fromarray(np.array([[[11, 22, 33, 0], [44, 55, 66, 128]]], dtype=np.uint8), "RGBA")
    path = tmp_path / "reference.png"
    rgba.save(path)
    loaded = inputs.collect_references([path], None, None)[0]
    assert loaded.mode == "RGBA"
    np.testing.assert_array_equal(np.array(loaded), np.array(rgba))
    assert np.array(inputs.vision_reference(loaded))[0, 0].tolist() == [255, 255, 255]
    torch.testing.assert_close(inputs.reference_pixels(loaded)[0, 3, 0, 0], torch.tensor([-1., 128 / 127.5 - 1]))


def test_reference_limit_and_ambiguous_inputs(inputs):
    image = Image.new("RGB", (32, 32))
    with pytest.raises(ValueError, match="10"):
        inputs.collect_references([image] * 11, None, None)
    with pytest.raises(ValueError, match="single source"):
        inputs.collect_references([image], None, image)
    with pytest.raises(TypeError, match="ordered list"):
        inputs.collect_references("image.png", None, None)


def test_latent_layout_statistics_and_prompt_mask(inputs):
    latent = torch.arange(2 * 4 * 6).float().reshape(1, 2, 1, 4, 6)
    torch.testing.assert_close(inputs.unpack_latents(inputs.pack_latents(latent), 64, 96), latent)
    normalized = inputs.normalize_latents(latent, [1., 2.], [2., 3.])
    torch.testing.assert_close(inputs.normalize_latents(normalized, [1., 2.], [2., 3.], inverse=True), latent)
    ids = torch.tensor([[0, 1, 2, 9, 3], [0, 0, 1, 2, 3]])
    attention = ids.ne(0)
    hidden = ids.float().unsqueeze(-1)
    embeds, mask, images = inputs.extract_prompt(hidden, ids, attention, 2, 9)
    assert embeds[:, :, 0].tolist() == [[9., 3.], [3., 0.]]
    assert mask.tolist() == [[True, True], [True, False]]
    assert images.tolist() == [[True, False], [False, False]]
    template = inputs.prompt_template("edit", 10)
    assert template.index("<image1>") < template.index("<image2>") < template.index("<image10>")
