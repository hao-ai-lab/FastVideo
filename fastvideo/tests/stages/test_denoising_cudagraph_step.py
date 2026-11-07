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


class _FakeFSDPModule:
    """Stands in for torch.distributed.fsdp.FSDPModule, which needs no GPU here."""


def _compiled_dit(monkeypatch, compile_kwargs, *, fsdp_wrapped=False):
    """A DiT put through the real pipeline compile path, so it carries its stamp.

    Going through ``_compile_pipeline_module_instance`` rather than setting
    flags is the point of these cases: the mark has to follow the compile that
    was actually attached, and that function decides from an ``isinstance``
    check, not from ``use_fsdp_inference``.
    """
    from fastvideo.pipelines.composed_pipeline_base import ComposedPipelineBase

    monkeypatch.setattr(torch, "compile", lambda target, **kwargs: target)
    model = TinyDenoiser()
    if fsdp_wrapped:
        model.__class__ = type("FSDPTinyDenoiser", (TinyDenoiser, _FakeFSDPModule), {})
    return ComposedPipelineBase._compile_pipeline_module_instance("transformer", model, _FakeFSDPModule,
                                                                  compile_kwargs or {})


@pytest.mark.parametrize("compile_kwargs, fsdp_wrapped, expected_marks", [
    ({"mode": "reduce-overhead"}, False, [0, 2, 4, 6]),
    ({"mode": "max-autotune"}, False, [0, 2, 4, 6]),
    ({"options": {"triton.cudagraphs": True}}, False, [0, 2, 4, 6]),
    ({"mode": "default"}, False, []),
    ({"mode": "max-autotune-no-cudagraphs"}, False, []),
    ({"mode": "reduce-overhead", "disable": True}, False, []),
    # Really FSDP-wrapped: the pipeline skips the compile, so nothing to mark.
    ({"mode": "reduce-overhead"}, True, []),
])
def test_denoising_marks_one_cudagraph_step_per_timestep(monkeypatch, env_overrides, compile_kwargs, fsdp_wrapped,
                                                         expected_marks):
    denoising, _ = _patch_denoising_module(monkeypatch, env_overrides, None)
    marks = []
    model = _compiled_dit(monkeypatch, compile_kwargs, fsdp_wrapped=fsdp_wrapped)
    monkeypatch.setattr(torch.compiler, "cudagraph_mark_step_begin", lambda: marks.append(len(model.calls)))
    stage = denoising.DenoisingStage(model, TinyScheduler())
    stage.progress_bar = lambda iterable=None, total=None: NullProgressBar()

    stage.forward(_tiny_batch(), _compile_args(_tiny_args(), compile_kwargs))

    # With CUDA graphs: one mark before each step's cond+uncond pair, never
    # between them. Without: no mark, so other cudagraph users keep
    # automatic step boundaries.
    assert model.calls == ["cond", "uncond"] * 4
    assert marks == expected_marks


def test_marks_follow_the_attached_compile_not_the_fsdp_flag(monkeypatch, env_overrides):
    """Regression: use_fsdp_inference=True on a DiT the pipeline still compiled.

    ``from_pretrained(loaded_modules=...)`` hands the pipeline a plain module,
    and a model that declares no shard conditions is never wrapped either. In
    both cases the FSDP instance check does not fire, the DiT is compiled with
    CUDA graphs, and the flag says otherwise. Marking has to happen anyway, or
    the crash this guards against comes back.
    """
    denoising, _ = _patch_denoising_module(monkeypatch, env_overrides, None)
    marks = []
    model = _compiled_dit(monkeypatch, {"mode": "reduce-overhead"}, fsdp_wrapped=False)
    monkeypatch.setattr(torch.compiler, "cudagraph_mark_step_begin", lambda: marks.append(len(model.calls)))
    stage = denoising.DenoisingStage(model, TinyScheduler())
    stage.progress_bar = lambda iterable=None, total=None: NullProgressBar()

    args = _compile_args(_tiny_args(), {"mode": "reduce-overhead"})
    args.use_fsdp_inference = True  # the flag says no compile; the module says otherwise

    stage.forward(_tiny_batch(), args)

    assert marks == [0, 2, 4, 6]


class _Block(torch.nn.Module):

    def __init__(self, dim):
        super().__init__()
        self.proj = torch.nn.Linear(dim, dim)

    def forward(self, hidden_states, context, temb):
        return hidden_states + torch.tanh(self.proj(hidden_states) + context.mean(dim=1, keepdim=True) + temb)


def _is_tiny_block(name: str, submodule: torch.nn.Module) -> bool:
    return isinstance(submodule, _Block)


class _TinyBlockDiT(torch.nn.Module):
    """Eager embed/head around per-block compiled blocks, like Wan."""
    hidden_size = 1
    num_attention_heads = 1
    # Declared so _compile_with_conditions compiles the blocks rather than the
    # whole module, which is the shape this test is about.
    _compile_conditions = [_is_tiny_block]

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
    # Compile through the pipeline's own entry point rather than calling
    # torch.compile on the blocks here. That is what production does, and it is
    # also what stamps the DiT with what the compile records; hand-rolling the
    # per-block compile skips the stamp, so the loop would not mark and this
    # test would reproduce the very crash it guards against.
    from fastvideo.pipelines.composed_pipeline_base import ComposedPipelineBase

    model = ComposedPipelineBase._compile_pipeline_module_instance(
        "transformer", model, None, {"mode": mode, "dynamic": False}
    )
    stage = denoising.DenoisingStage(model, TinyScheduler())
    stage.progress_bar = lambda iterable=None, total=None: NullProgressBar()
    outputs = []
    for _ in range(3):
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

        def __init__(self, fastvideo_args):
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
    pipeline._apply_inference_compile(("vae", ))
    assert logger.warnings == []
    # MiniMax-H3 sequential loading calls this again for late modules; warn once.
    pipeline._apply_inference_compile()
    pipeline._apply_inference_compile(("transformer", ))

    assert len([msg for msg in logger.warnings if "dit_layerwise_offload" in msg]) == (1 if warned else 0)


@pytest.mark.parametrize("kwargs, records", [
    ({"options": {"triton.cudagraphs": True}}, True),
    ({"fullgraph": True}, False),
])
def test_regional_compile_stamps_what_it_recorded(monkeypatch, kwargs, records):
    """The regional path reaches cudagraphs through options, so it must stamp too.

    _compile_model_regions rejects `mode` but merges the caller's `options`,
    which is how {"triton.cudagraphs": True} gets to inductor on the
    inference_torch_compile path. Before the stamp, the helper read False there
    and the loop never marked.
    """
    from fastvideo.fastvideo_args import DIT_COMPILE_CUDAGRAPHS_ATTR, dit_compile_uses_cudagraphs
    from fastvideo.models.loader.fsdp_load import _compile_model_regions

    monkeypatch.setattr(torch, "compile", lambda target, **kw: target)
    model = _TinyBlockDiT()

    compiled = _compile_model_regions(model, dict(kwargs))

    assert compiled == 3, "the tiny DiT's blocks should match _compile_conditions"
    assert getattr(model, DIT_COMPILE_CUDAGRAPHS_ATTR) is records
    assert dit_compile_uses_cudagraphs(model) is records
