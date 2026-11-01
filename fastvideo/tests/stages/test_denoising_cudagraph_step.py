# SPDX-License-Identifier: Apache-2.0
"""DenoisingStage under torch.compile(mode="reduce-overhead"); no weights required.

The CPU tests check that each denoising step opens a new cudagraph-trees step
only when the DiT compile records CUDA graphs, and that such a compile warns
under layerwise offload. The CUDA test compiles each block of a tiny DiT the
way ``_compile_conditions`` compiles Wan blocks and runs the shared CFG loop.
"""

import pytest
import torch

from fastvideo.tests.stages._denoising_fixtures import (
    NullProgressBar, TinyDenoiser, TinyScheduler, _patch_denoising_module, _tiny_args, _tiny_batch,
)


def _compile_args(args, compile_kwargs, **overrides):
    args.enable_torch_compile = compile_kwargs is not None
    args.torch_compile_kwargs = compile_kwargs or {}
    args.training_mode = False
    args.inference_torch_compile = False
    args.use_fsdp_inference = False
    for key, value in overrides.items():
        setattr(args, key, value)
    return args


@pytest.mark.parametrize("compile_kwargs, overrides, expected_marks", [
    ({"mode": "reduce-overhead"}, {}, [0, 2, 4, 6]),
    ({"mode": "max-autotune"}, {}, [0, 2, 4, 6]),
    ({"options": {"triton.cudagraphs": True}}, {}, [0, 2, 4, 6]),
    ({"mode": "default"}, {}, []),
    ({"mode": "max-autotune-no-cudagraphs"}, {}, []),
    (None, {}, []),
    ({"mode": "reduce-overhead", "disable": True}, {}, []),
    # These configurations skip the pipeline-level DiT compile.
    ({"mode": "reduce-overhead"}, {"use_fsdp_inference": True}, []),
    ({"mode": "reduce-overhead"}, {"inference_torch_compile": True}, []),
    ({"mode": "reduce-overhead"}, {"training_mode": True}, []),
])
def test_denoising_marks_one_cudagraph_step_per_timestep(monkeypatch, env_overrides, compile_kwargs, overrides,
                                                         expected_marks):
    denoising, _ = _patch_denoising_module(monkeypatch, env_overrides, None)
    marks = []
    model = TinyDenoiser()
    monkeypatch.setattr(torch.compiler, "cudagraph_mark_step_begin", lambda: marks.append(len(model.calls)))
    stage = denoising.DenoisingStage(model, TinyScheduler())
    stage.progress_bar = lambda iterable=None, total=None: NullProgressBar()

    stage.forward(_tiny_batch(), _compile_args(_tiny_args(), compile_kwargs, **overrides))

    # With CUDA graphs: one mark before each step's cond+uncond pair, never
    # between them. Without: no mark, so other cudagraph users keep
    # automatic step boundaries.
    assert model.calls == ["cond", "uncond"] * 4
    assert marks == expected_marks


class _Block(torch.nn.Module):

    def __init__(self, dim):
        super().__init__()
        self.proj = torch.nn.Linear(dim, dim)

    def forward(self, hidden_states, context, temb):
        return hidden_states + torch.tanh(self.proj(hidden_states) + context.mean(dim=1, keepdim=True) + temb)


class _TinyBlockDiT(torch.nn.Module):
    """Eager embed/head around per-block compiled blocks, like Wan."""
    hidden_size = 1
    num_attention_heads = 1

    def __init__(self, dim=32, num_blocks=3):
        super().__init__()
        self.config = type("Cfg", (), {"use_meanflow": False})()
        self.embed = torch.nn.Linear(2, dim)
        self.time = torch.nn.Linear(1, dim)
        self.blocks = torch.nn.ModuleList(_Block(dim) for _ in range(num_blocks))
        self.head = torch.nn.Linear(dim, 2)

    def forward(self, latent_model_input, prompt_embeds, timestep, guidance=None):
        b, c, f, h, w = latent_model_input.shape
        x = self.embed(latent_model_input.flatten(2).transpose(1, 2).float())
        temb = self.time(timestep.float().reshape(b, 1, 1) / 1000)
        context = prompt_embeds[0].float()
        for block in self.blocks:
            x = block(x, context, temb)
        return self.head(x).transpose(1, 2).reshape(b, c, f, h, w)


def _run_compiled_stage(monkeypatch, env_overrides, mode):
    from fastvideo.tests.stages._denoising_fixtures import _args, _batch

    denoising, _ = _patch_denoising_module(monkeypatch, env_overrides, None)
    device = torch.device("cuda")
    monkeypatch.setattr(denoising, "get_local_torch_device", lambda: device)
    torch.manual_seed(0)
    model = _TinyBlockDiT().to(device)
    for block in model.blocks:
        block.forward = torch.compile(block.forward, mode=mode, dynamic=False)
    stage = denoising.DenoisingStage(model, TinyScheduler())
    stage.progress_bar = lambda iterable=None, total=None: NullProgressBar()
    outputs = []
    for _ in range(3):  # cudagraph trees warm up, record, then replay
        batch = _batch(steps=6, cfg=True)
        batch.latents = batch.latents.to(device)
        batch.prompt_embeds = [torch.full((1, 4, 32), 0.5, device=device)]
        batch.negative_prompt_embeds = [torch.full((1, 4, 32), -0.5, device=device)]
        with torch.inference_mode():
            outputs.append(stage.forward(batch, _compile_args(_args(), {"mode": mode})).latents.clone())
    return outputs


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA graphs need a CUDA device")
def test_reduce_overhead_cfg_loop_matches_default_compile(monkeypatch, env_overrides):
    torch._dynamo.reset()
    try:
        reference = _run_compiled_stage(monkeypatch, env_overrides, "default")
        torch._dynamo.reset()
        graphed = _run_compiled_stage(monkeypatch, env_overrides, "reduce-overhead")
    finally:
        torch._dynamo.reset()

    for ref, out in zip(reference, graphed, strict=True):
        assert torch.equal(ref, out)


@pytest.mark.parametrize("mode, offload, warned", [
    ("reduce-overhead", True, True),
    ("max-autotune", True, True),
    ("reduce-overhead", False, False),
    ("default", True, False),
])
def test_cudagraph_compile_warns_when_layerwise_offload_moves_weights(monkeypatch, mode, offload, warned):
    from types import SimpleNamespace

    import fastvideo.pipelines.composed_pipeline_base as base
    from fastvideo.tests.stages._denoising_fixtures import RecordingLogger

    class _Pipeline(base.ComposedPipelineBase):

        def __init__(self, fastvideo_args):  # deliberately does not call super()
            self.fastvideo_args = fastvideo_args
            self.modules = {"transformer": torch.nn.Linear(2, 2)}

        def create_pipeline_stages(self, fastvideo_args):
            raise NotImplementedError

    logger = RecordingLogger()
    monkeypatch.setattr(base, "logger", logger)
    monkeypatch.setattr(torch, "compile", lambda target, **kwargs: target)
    args = SimpleNamespace(enable_torch_compile=True, enable_torch_compile_text_encoder=False,
                           enable_torch_compile_vae=False, enable_torch_compile_audio_vae=False, training_mode=False,
                           inference_torch_compile=False, torch_compile_kwargs={}, torch_compile_kwargs_dit={"mode": mode},
                           torch_compile_kwargs_text_encoder={}, torch_compile_kwargs_vae={},
                           torch_compile_kwargs_audio_vae={}, dit_layerwise_offload=offload)

    pipeline = _Pipeline(args)
    pipeline._apply_inference_compile(("vae", ))  # no DiT selected: never warns
    assert logger.warnings == []
    # MiniMax-H3 sequential loading calls this again for late modules; warn once.
    pipeline._apply_inference_compile()
    pipeline._apply_inference_compile(("transformer", ))

    assert len([msg for msg in logger.warnings if "dit_layerwise_offload" in msg]) == (1 if warned else 0)
