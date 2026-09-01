# SPDX-License-Identifier: Apache-2.0
"""Deferred pipeline-module loading and release.

CPU only. Exercises the proxy contract and the release schedule the pipeline
derives from what its stages hold; no model weights are touched.
"""

import dataclasses

import torch
from types import SimpleNamespace

import pytest

from fastvideo.pipelines.composed_pipeline_base import ComposedPipelineBase
from fastvideo.pipelines.lazy_module import LazyModule, is_lazy_module
from fastvideo.pipelines.stages.base import PipelineStage


class _Component:

    def __init__(self, tag: str) -> None:
        self.tag = tag

    def __call__(self, value: int) -> int:
        return value * 2


def _counting_loader(tag: str = "c"):
    calls = []

    def loader():
        calls.append(tag)
        return _Component(tag)

    return loader, calls


def test_deferred_until_first_use():
    loader, calls = _counting_loader()
    module = LazyModule("transformer", loader)

    assert calls == []
    assert not module.is_materialized
    assert "deferred" in repr(module)

    assert module.tag == "c"
    assert calls == ["c"]
    assert module.is_materialized


def test_repr_does_not_materialize():
    loader, calls = _counting_loader()
    module = LazyModule("transformer", loader)

    repr(module)
    f"{module!r}"

    assert calls == []


def test_loads_exactly_once_across_many_accesses():
    loader, calls = _counting_loader()
    module = LazyModule("vae", loader)

    module.tag
    module.tag
    module(3)

    assert calls == ["c"]


def test_call_forwards_to_component():
    loader, _ = _counting_loader()
    module = LazyModule("vae", loader)

    assert module(21) == 42


def test_setattr_and_delattr_forward_to_component():
    loader, _ = _counting_loader()
    module = LazyModule("vae", loader)

    module.tag = "changed"
    assert module.materialize().tag == "changed"

    del module.tag
    assert not hasattr(module.materialize(), "tag")


def test_self_returning_methods_hand_back_the_proxy():
    # Stages write `self.vae = self.vae.to(device)`. Returning the component
    # would swap the proxy out and leave nothing releasable, with no error.
    import torch

    module = LazyModule("vae", lambda: torch.nn.Linear(2, 2))

    assert module.to("cpu") is module
    assert module.eval() is module
    assert module.float() is module
    assert module.requires_grad_(False) is module


def test_non_self_returning_methods_pass_their_result_through():
    import torch

    module = LazyModule("vae", lambda: torch.nn.Linear(2, 2))

    assert isinstance(module.state_dict(), dict)
    assert module.extra_repr() == "in_features=2, out_features=2, bias=True"


def test_callable_submodule_attribute_is_not_wrapped():
    # Only bound methods get the identity wrapper. A callable submodule must
    # come back as itself so attribute chains and further calls keep working.
    import torch

    inner = torch.nn.Linear(2, 2)
    module = LazyModule("vae", lambda: torch.nn.Sequential(inner))

    assert module.__getattr__("0") is inner


def test_isinstance_reports_the_real_class():
    # Callers branch on isinstance (FSDPModule, nn.Module). A proxy that
    # answered False would take the wrong branch silently.
    loader, _ = _counting_loader()
    module = LazyModule("transformer", loader)

    assert isinstance(module, _Component)
    assert is_lazy_module(module)


def test_is_lazy_module_does_not_materialize():
    loader, calls = _counting_loader()
    module = LazyModule("transformer", loader)

    assert is_lazy_module(module)
    assert calls == []
    assert not is_lazy_module(_Component("plain"))


def test_release_then_reload_is_correct_not_broken():
    loader, calls = _counting_loader()
    module = LazyModule("text_encoder", loader)

    first = module.materialize()
    assert module.release() is True
    assert not module.is_materialized

    second = module.materialize()
    assert calls == ["c", "c"]
    assert second is not first
    assert second.tag == "c"


def test_materialize_transform_applies_to_every_loaded_instance_without_loading_eagerly():
    loader, calls = _counting_loader()
    transformed = []
    module = LazyModule("vae", loader)

    def transform(component):
        transformed.append(component)
        component.tag = f"compiled-{component.tag}"
        return component

    module.set_materialize_transform(transform)
    assert calls == []

    first = module.materialize()
    assert first.tag == "compiled-c"
    assert module.release() is True

    second = module.materialize()
    assert second.tag == "compiled-c"
    assert second is not first
    assert calls == ["c", "c"]
    assert transformed == [first, second]


def test_materialize_transforms_compose_in_registration_order():
    loader, calls = _counting_loader()
    module = LazyModule("vae", loader)
    tags = []

    def inner(component):
        tags.append("inner")
        component.tag = f"inner-{component.tag}"
        return component

    def outer(component):
        tags.append("outer")
        component.tag = f"outer-{component.tag}"
        return component

    module.set_materialize_transform(inner)
    module.set_materialize_transform(outer)
    first = module.materialize()
    assert first.tag == "outer-inner-c"
    assert tags == ["inner", "outer"]
    assert calls == ["c"]


def test_release_without_materializing_is_a_noop():
    loader, calls = _counting_loader()
    module = LazyModule("text_encoder", loader)

    assert module.release() is False
    assert module.release() is False
    assert calls == []


def test_loader_returning_none_raises_instead_of_proxying_none():
    module = LazyModule("transformer", lambda: None)

    with pytest.raises(ValueError, match="returned None"):
        module.materialize()


# ----------------------------------------------------------------------
# Release schedule
# ----------------------------------------------------------------------


class _EchoStage(PipelineStage):

    def __init__(self, **held):
        for name, value in held.items():
            setattr(self, name, value)

    def forward(self, batch, fastvideo_args):
        return batch


class _FakePipeline(ComposedPipelineBase):
    """Just enough pipeline to exercise the schedule; no weights, no loading."""

    def __init__(self, modules, stages):  # deliberately does not call super()
        self.modules = modules
        self._stages = stages
        self._lazy_module_names = tuple(name for name, module in modules.items() if is_lazy_module(module))

    def create_pipeline_stages(self, fastvideo_args):
        raise NotImplementedError


def _schedule(modules, stages):
    return _FakePipeline(modules, stages)._build_lazy_release_schedule()


def _lazy(name):
    return LazyModule(name, lambda: _Component(name))


def test_pipeline_compile_is_reapplied_after_lazy_release(monkeypatch):
    loads = []
    compile_calls = []

    class _CompileAwareComponent(torch.nn.Module):
        _compile_conditions = (lambda name, module: name == "block", )

        def __init__(self):
            super().__init__()
            self.block = torch.nn.Linear(2, 2)
            self.prepare_calls = 0

        def prepare_for_compile(self):
            self.prepare_calls += 1

    def load_component():
        component = _CompileAwareComponent()
        loads.append(component)
        return component

    def fake_compile(target, **kwargs):
        compile_calls.append((target, kwargs))
        return target

    monkeypatch.setattr(torch, "compile", fake_compile)
    lazy = LazyModule("vae", load_component)
    pipeline = _FakePipeline({"vae": lazy}, [])

    pipeline._maybe_compile_pipeline_module("vae", None, {"mode": "reduce-overhead"})
    assert loads == []

    first = lazy.materialize()
    assert first.prepare_calls == 1
    assert lazy.release() is True

    second = lazy.materialize()
    assert second is not first
    assert second.prepare_calls == 1
    assert loads == [first, second]
    assert len(compile_calls) == 2
    assert [kwargs for _, kwargs in compile_calls] == [
        {"mode": "reduce-overhead"},
        {"mode": "reduce-overhead"},
    ]


def test_whole_module_compile_keeps_lazy_proxy_and_recompiles_after_release(monkeypatch):
    loads = []
    compile_calls = []

    class _WholeComponent(torch.nn.Module):

        def forward(self, value):
            return value

    class _Compiled:

        def __init__(self, original):
            self.original = original

    def load_component():
        component = _WholeComponent()
        loads.append(component)
        return component

    def fake_compile(target, **kwargs):
        compile_calls.append((target, kwargs))
        return _Compiled(target)

    monkeypatch.setattr(torch, "compile", fake_compile)
    lazy = LazyModule("text_encoder", load_component)
    pipeline = _FakePipeline({"text_encoder": lazy}, [])

    pipeline._maybe_compile_pipeline_module("text_encoder", None, {"dynamic": True})
    assert pipeline.modules["text_encoder"] is lazy
    assert loads == []

    first = lazy.materialize()
    assert first.original is loads[0]
    assert lazy.release() is True

    second = lazy.materialize()
    assert second.original is loads[1]
    assert second is not first
    assert len(compile_calls) == 2


def test_schedule_releases_after_the_last_stage_that_holds_a_module():
    text_encoder = _lazy("text_encoder")
    transformer = _lazy("transformer")
    vae = _lazy("vae")
    modules = {"text_encoder": text_encoder, "transformer": transformer, "vae": vae, "scheduler": object()}

    stages = [
        _EchoStage(vae=vae),  # 0 input prep
        _EchoStage(conditioner=text_encoder),  # 1 conditioning
        _EchoStage(transformer=transformer),  # 2 denoising
        _EchoStage(vae=vae, transformer=transformer),  # 3 decoding
    ]

    assert _schedule(modules, stages) == {1: ["text_encoder"], 3: ["transformer", "vae"]}


def test_building_the_schedule_does_not_materialize_anything():
    # isinstance() on a proxy forwards __class__, so a careless scan of stage
    # attributes would load every deferred module before the run starts and
    # silently undo the whole point of deferring.
    loaded = []

    def tracked(name):
        return LazyModule(name, lambda: loaded.append(name) or _Component(name))

    text_encoder, transformer = tracked("text_encoder"), tracked("transformer")
    stages = [
        _EchoStage(conditioner=text_encoder, flags=[1, 2], opts={"a": 1}, ref2va=False),
        _EchoStage(transformer=transformer),
    ]

    _schedule({"text_encoder": text_encoder, "transformer": transformer}, stages)

    assert loaded == []


def test_schedule_ignores_eager_modules():
    transformer = _lazy("transformer")
    scheduler = object()
    modules = {"transformer": transformer, "scheduler": scheduler}
    stages = [_EchoStage(transformer=transformer, scheduler=scheduler)]

    assert _schedule(modules, stages) == {0: ["transformer"]}


def test_schedule_finds_modules_held_inside_containers():
    text_encoder = _lazy("text_encoder")
    vae = _lazy("vae")
    modules = {"text_encoder": text_encoder, "vae": vae}
    stages = [
        _EchoStage(text_encoders=[text_encoder]),
        _EchoStage(by_name={"vae": vae}),
    ]

    assert _schedule(modules, stages) == {0: ["text_encoder"], 1: ["vae"]}


def test_unreferenced_module_is_never_released():
    # Safe direction: a module no stage holds stays loaded rather than
    # disappearing under a caller the schedule cannot see.
    orphan = _lazy("image_encoder")

    assert _schedule({"image_encoder": orphan}, [_EchoStage(other=1)]) == {}


def test_schedule_is_empty_without_lazy_modules():
    assert _schedule({"vae": object()}, [_EchoStage(vae=object())]) == {}


# ----------------------------------------------------------------------
# Enablement
# ----------------------------------------------------------------------


@pytest.mark.parametrize(("lazy", "training", "expected"), [
    (False, False, False),
    (True, False, True),
    (True, True, False),
    (False, True, False),
    (None, False, False),
])
def test_training_mode_never_defers(lazy, training, expected):
    args = SimpleNamespace(lazy_module_load=lazy, training_mode=training)

    assert ComposedPipelineBase._lazy_module_load_enabled(args) is expected


def test_flag_defaults_to_auto():
    from fastvideo.fastvideo_args import FastVideoArgs

    fields = {f.name: f for f in dataclasses.fields(FastVideoArgs)}
    assert fields["lazy_module_load"].default is None


# ----------------------------------------------------------------------
# Release hooks on the stages
# ----------------------------------------------------------------------


def test_hooks_land_on_the_last_stage_that_holds_each_module():
    text_encoder, transformer = _lazy("text_encoder"), _lazy("transformer")
    stages = [_EchoStage(conditioner=text_encoder), _EchoStage(transformer=transformer, extra=text_encoder)]
    pipeline = _FakePipeline({"text_encoder": text_encoder, "transformer": transformer}, stages)

    pipeline._install_lazy_release_hooks()

    assert stages[0]._lazy_modules_to_release == ()
    assert set(stages[1]._lazy_modules_to_release) == {text_encoder, transformer}


def test_installing_hooks_twice_does_not_shift_the_schedule():
    # The installed tuple is itself a container of proxies; a rebuild that
    # counted it as a use would keep pushing every release to the last stage.
    text_encoder = _lazy("text_encoder")
    stages = [_EchoStage(conditioner=text_encoder), _EchoStage(other=1)]
    pipeline = _FakePipeline({"text_encoder": text_encoder}, stages)

    pipeline._install_lazy_release_hooks()
    pipeline._install_lazy_release_hooks()

    assert stages[0]._lazy_modules_to_release == (text_encoder, )
    assert stages[1]._lazy_modules_to_release == ()


def test_stage_call_releases_its_modules():
    loader, calls = _counting_loader()
    text_encoder = LazyModule("text_encoder", loader)
    stages = [_EchoStage(conditioner=text_encoder), _EchoStage(other=1)]
    pipeline = _FakePipeline({"text_encoder": text_encoder}, stages)
    pipeline._install_lazy_release_hooks()

    text_encoder.tag  # the stage would use it
    assert text_encoder.is_materialized

    batch = object()
    args = SimpleNamespace(enable_stage_verification=False)
    assert stages[0](batch, args) is batch

    assert not text_encoder.is_materialized
    assert calls == ["c"]


def test_stage_without_hooks_releases_nothing():
    loader, _ = _counting_loader()
    module = LazyModule("vae", loader)
    stage = _EchoStage(vae=module)
    module.tag

    stage(object(), SimpleNamespace(enable_stage_verification=False))

    assert module.is_materialized


def test_pipeline_warns_when_no_stage_holds_a_deferred_module(caplog):
    # A silent no-op here would look exactly like a working run, so the flag
    # has to say when it cannot do anything.
    orphan = _lazy("image_encoder")
    pipeline = _FakePipeline({"image_encoder": orphan}, [_EchoStage(other=1)])

    with caplog.at_level("WARNING"):
        pipeline._install_lazy_release_hooks()

    assert "nothing will be freed" in caplog.text


def test_empty_opt_in_list_is_silent(caplog):
    pipeline = _FakePipeline({}, [_EchoStage(other=1)])
    with caplog.at_level("WARNING"):
        pipeline._install_lazy_release_hooks()
    assert caplog.text == ""


def test_a_stage_that_rebinds_through_to_can_still_be_released():
    # The end-to-end shape of the identity rule: a stage does the
    # `self.vae = self.vae.to(device)` dance, the pipeline still releases.
    import torch

    vae = LazyModule("vae", lambda: torch.nn.Linear(2, 2))
    stage = _EchoStage(vae=vae)
    pipeline = _FakePipeline({"vae": vae}, [stage])
    pipeline._install_lazy_release_hooks()

    stage.vae = stage.vae.to("cpu")
    assert stage.vae is vae
    assert vae.is_materialized

    stage(object(), SimpleNamespace(enable_stage_verification=False))

    assert not vae.is_materialized


def test_a_stage_added_after_the_schedule_rebuilds_it(caplog):
    # The schedule is derived from the stage list. A stage appended afterwards
    # could hold a module an earlier stage was already told to free, which
    # would hand it a released component mid-run.
    vae = _lazy("vae")
    first = _EchoStage(vae=vae)
    pipeline = _FakePipeline({"vae": vae}, [])
    pipeline._stage_name_mapping = {}
    pipeline.add_stage("first", first)
    pipeline._install_lazy_release_hooks()

    assert first._lazy_modules_to_release == (vae, )

    later = _EchoStage(vae=vae)
    with caplog.at_level("DEBUG"):
        pipeline.add_stage("later", later)

    assert "rebuilding the schedule" in caplog.text
    assert first._lazy_modules_to_release == ()
    assert later._lazy_modules_to_release == (vae, )


class _CompositeStage(PipelineStage):
    """Mirrors Cosmos25AutoDenoisingStage: the component lives in a child."""

    def __init__(self, **held):
        self._child = _EchoStage(**held)

    def forward(self, batch, fastvideo_args):
        return self._child.forward(batch, fastvideo_args)


def test_schedule_walks_into_nested_stages():
    # A stage can compose others rather than hold the component itself. Left
    # unwalked, the component reads as unreferenced and is never freed.
    transformer = _lazy("transformer")
    stages = [_EchoStage(other=1), _CompositeStage(transformer=transformer)]

    assert _schedule({"transformer": transformer}, stages) == {1: ["transformer"]}


def test_nested_walk_survives_a_cycle():
    vae = _lazy("vae")
    outer = _EchoStage(vae=vae)
    inner = _EchoStage(back=outer)
    outer.inner = inner

    assert _schedule({"vae": vae}, [outer]) == {0: ["vae"]}


def test_a_raising_stage_still_releases_its_modules():
    # The retry a memory-constrained caller attempts must not start from a
    # worse position than the request that just failed.
    class _Boom(_EchoStage):

        def forward(self, batch, fastvideo_args):
            raise RuntimeError("out of activation memory")

    vae = LazyModule("vae", lambda: torch.nn.Linear(2, 2))
    stage = _Boom(vae=vae)
    pipeline = _FakePipeline({"vae": vae}, [stage])
    pipeline._install_lazy_release_hooks()
    vae.materialize()

    with pytest.raises(RuntimeError, match="out of activation memory"):
        stage(object(), SimpleNamespace(enable_stage_verification=False))

    assert not vae.is_materialized


def test_a_failing_release_does_not_mask_the_original_error():
    class _Boom(_EchoStage):

        def forward(self, batch, fastvideo_args):
            raise RuntimeError("original")

    class _BadRelease(LazyModule):

        def release(self):
            raise ValueError("cleanup blew up")

    stage = _Boom(vae=None)
    stage._lazy_modules_to_release = (_BadRelease("vae", lambda: object()), )

    with pytest.raises(RuntimeError, match="original"):
        stage(object(), SimpleNamespace(enable_stage_verification=False))


def test_deferral_is_opt_in_per_pipeline():
    # A pipeline that has not been checked must get no deferral at all,
    # because releasing and reloading is only safe when nothing outside the
    # loader mutates the component or reads it while stages are built.
    from fastvideo.pipelines.basic.minimax_h3.minimax_h3_pipeline import MiniMaxH3BasePipeline

    assert ComposedPipelineBase._lazy_module_names == ()
    assert set(MiniMaxH3BasePipeline._lazy_module_names) == {"text_encoder", "transformer", "vae", "audio_vae"}


def test_an_aborted_run_releases_everything_already_materialized():
    # A stage frees only what it is the last user of. When the run aborts
    # earlier, the rest would stay for the life of the generator.
    vae = LazyModule("vae", lambda: torch.nn.Linear(2, 2))
    transformer = LazyModule("transformer", lambda: torch.nn.Linear(2, 2))

    class _Boom(_EchoStage):

        def forward(self, batch, fastvideo_args):
            raise RuntimeError("out of activation memory")

    early = _EchoStage(vae=vae)
    boom = _Boom(transformer=transformer)
    late = _EchoStage(vae=vae)
    pipeline = _FakePipeline({"vae": vae, "transformer": transformer}, [early, boom, late])
    pipeline._install_lazy_release_hooks()
    vae.materialize()
    transformer.materialize()

    assert vae.is_materialized and transformer.is_materialized

    pipeline._release_all_lazy_modules()

    assert not vae.is_materialized
    assert not transformer.is_materialized


# ----------------------------------------------------------------------
# Production wiring
#
# The tests above build stages and pipelines by hand. These two run the real
# code paths where the two defects review found would live: a `load_modules`
# that never reaches the deferral, and a stage constructor that reads a
# component's attributes and materializes it before the first request.
# ----------------------------------------------------------------------


class _StubLoader:
    """Stands in for PipelineComponentLoader and counts what it is asked for."""

    def __init__(self):
        self.loaded: list[str] = []

    def load_module(self, *, module_name, component_model_path, transformers_or_diffusers, fastvideo_args):
        self.loaded.append(module_name)
        return _Component(module_name)


def _run_real_load_modules(monkeypatch, lazy_names, manifest_modules):
    from fastvideo.pipelines import composed_pipeline_base as cpb

    stub = _StubLoader()
    monkeypatch.setattr(cpb.PipelineComponentLoader, "load_module", stub.load_module)

    class _Pipeline(ComposedPipelineBase):
        _required_config_modules = list(manifest_modules)
        _lazy_module_names = lazy_names

        def __init__(self):  # deliberately does not call super()
            self.model_path = "/nowhere"
            self.fastvideo_args = None

        def _load_config(self, model_path):
            index = {"_class_name": "X", "_diffusers_version": "0"}
            index.update({name: ["diffusers", "Cls", {}] for name in manifest_modules})
            return index

        def create_pipeline_stages(self, fastvideo_args):
            raise NotImplementedError

    args = SimpleNamespace(lazy_module_load=True, training_mode=False, revision=None)
    modules = _Pipeline().load_modules(args)
    return modules, stub.loaded


def test_real_load_modules_defers_only_the_opted_in_components(monkeypatch):
    modules, loaded = _run_real_load_modules(monkeypatch, ("transformer", "vae"), ["transformer", "vae", "scheduler"])

    assert is_lazy_module(modules["transformer"])
    assert is_lazy_module(modules["vae"])
    assert not is_lazy_module(modules["scheduler"])
    # The loader is asked only for what stays eager.
    assert loaded == ["scheduler"]


def test_real_load_modules_defers_nothing_when_the_pipeline_opts_out(monkeypatch):
    # The base class ships an empty list, so an unchecked pipeline must load
    # everything eagerly even with the flag on.
    modules, loaded = _run_real_load_modules(monkeypatch, (), ["transformer", "vae", "scheduler"])

    assert not any(is_lazy_module(m) for m in modules.values())
    assert sorted(loaded) == ["scheduler", "transformer", "vae"]


def test_building_the_real_h3_stages_materializes_nothing():
    # `DenoisingStage.__init__` in the shared stage set reads
    # `transformer.hidden_size` to pick an attention backend, which would pull
    # the DiT in during post_init. H3's stages must not acquire that habit.
    from fastvideo.configs.pipelines.minimax_h3 import MiniMaxH3PipelineConfig
    from fastvideo.pipelines.basic.minimax_h3.minimax_h3_pipeline import MiniMaxH3Pipeline
    from fastvideo.pipelines.composed_pipeline_base import _iter_held_objects

    loaded: list[str] = []

    def tracked(name):
        return LazyModule(name, lambda: loaded.append(name) or _Component(name))

    pipeline = MiniMaxH3Pipeline.__new__(MiniMaxH3Pipeline)
    pipeline._stages = []
    pipeline._stage_name_mapping = {}
    pipeline.modules = {
        "text_encoder": tracked("text_encoder"),
        "transformer": tracked("transformer"),
        "vae": tracked("vae"),
        "audio_vae": tracked("audio_vae"),
        "tokenizer": object(),
        "processor": object(),
        "scheduler": object(),
        "audio_scheduler": object(),
    }
    args = SimpleNamespace(pipeline_config=MiniMaxH3PipelineConfig())

    pipeline._add_stages(args, ref2va=False)

    assert loaded == [], f"building stages materialized {loaded}"
    assert len(pipeline._stages) == 6
    input_held = {id(obj) for obj in _iter_held_objects(pipeline._stage_name_mapping["input_preparation_stage"])}
    latent_held = {id(obj) for obj in _iter_held_objects(pipeline._stage_name_mapping["latent_preparation_stage"])}
    decode_held = {id(obj) for obj in _iter_held_objects(pipeline._stage_name_mapping["video_decoding_stage"])}
    denoise_held = {id(obj) for obj in _iter_held_objects(pipeline._stage_name_mapping["denoising_stage"])}
    assert id(pipeline.modules["vae"]) not in input_held
    assert id(pipeline.modules["transformer"]) not in input_held
    assert id(pipeline.modules["transformer"]) not in latent_held
    assert id(pipeline.modules["transformer"]) not in decode_held
    assert id(pipeline.modules["transformer"]) in denoise_held
    assert id(pipeline.modules["vae"]) in decode_held


def test_h3_lazy_release_drops_dit_before_vae_decode():
    from fastvideo.configs.pipelines.minimax_h3 import MiniMaxH3PipelineConfig
    from fastvideo.pipelines.basic.minimax_h3.minimax_h3_pipeline import MiniMaxH3Pipeline

    pipeline = MiniMaxH3Pipeline.__new__(MiniMaxH3Pipeline)
    pipeline._stages = []
    pipeline._stage_name_mapping = {}
    pipeline.modules = {
        "text_encoder": LazyModule("text_encoder", lambda: _Component("text_encoder")),
        "transformer": LazyModule("transformer", lambda: _Component("transformer")),
        "vae": LazyModule("vae", lambda: _Component("vae")),
        "audio_vae": LazyModule("audio_vae", lambda: _Component("audio_vae")),
        "tokenizer": object(),
        "processor": object(),
        "scheduler": object(),
        "audio_scheduler": object(),
    }
    args = SimpleNamespace(pipeline_config=MiniMaxH3PipelineConfig())
    pipeline._add_stages(args, ref2va=False)
    schedule = pipeline._build_lazy_release_schedule()
    names = {pipeline._stages[index]._pipeline_stage_name: modules for index, modules in schedule.items()}
    assert names["conditioning_stage"] == ["text_encoder"]
    assert names["denoising_stage"] == ["transformer"]
    assert "transformer" not in names.get("video_decoding_stage", [])
    assert "vae" in names["video_decoding_stage"]


def test_h3_checkpoint_json_updates_dit_patch_size_without_weights(tmp_path):
    from fastvideo.configs.pipelines.minimax_h3 import MiniMaxH3PipelineConfig
    from fastvideo.pipelines.basic.minimax_h3.minimax_h3_pipeline import _apply_h3_checkpoint_arch_configs

    transformer_dir = tmp_path / "transformer"
    transformer_dir.mkdir()
    (transformer_dir / "config.json").write_text('{"patch_size": [1, 1, 1]}')
    args = SimpleNamespace(pipeline_config=MiniMaxH3PipelineConfig())
    assert tuple(args.pipeline_config.dit_config.patch_size) == (1, 2, 2)
    _apply_h3_checkpoint_arch_configs(str(tmp_path), args, {})
    assert tuple(args.pipeline_config.dit_config.patch_size) == (1, 1, 1)


def test_h3_checkpoint_json_updates_audio_sampling_rate_without_weights(tmp_path):
    from fastvideo.configs.pipelines.minimax_h3 import MiniMaxH3PipelineConfig
    from fastvideo.pipelines.basic.minimax_h3.minimax_h3_pipeline import _apply_h3_checkpoint_arch_configs

    audio_dir = tmp_path / "audio_vae"
    audio_dir.mkdir()
    (audio_dir / "config.json").write_text('{"sampling_rate": 16000, "latent_channels": 16}')
    args = SimpleNamespace(pipeline_config=MiniMaxH3PipelineConfig())
    assert int(args.pipeline_config.audio_vae_config.arch_config.sampling_rate) == 32000
    _apply_h3_checkpoint_arch_configs(str(tmp_path), args, {})
    assert int(args.pipeline_config.audio_vae_config.arch_config.sampling_rate) == 16000
    assert int(args.pipeline_config.audio_vae_config.arch_config.latent_channels) == 16


class _LoRAConfigComponent(torch.nn.Module):

    def __init__(self, excluded_layers):
        super().__init__()
        self.config = SimpleNamespace(
            arch_config=SimpleNamespace(exclude_lora_layers=excluded_layers),
        )
        self.blocks = torch.nn.ModuleList([torch.nn.Linear(2, 2)])


def _build_stub_lora_pipeline(monkeypatch, transformer, excluded_layers=None):
    from fastvideo.pipelines import lora_pipeline as lora_module

    args = SimpleNamespace(
        lora_target_modules=None,
        lora_path=None,
        lora_nickname="default",
        lora_strength=1.0,
        training_mode=False,
        lora_training=False,
        dit_layerwise_offload=False,
        pipeline_config=SimpleNamespace(
            dit_config=SimpleNamespace(
                arch_config=SimpleNamespace(exclude_lora_layers=list(excluded_layers or [])),
            ),
        ),
    )

    def initialize_base(pipeline, *unused_args, **unused_kwargs):
        pipeline.fastvideo_args = args
        pipeline.modules = {"transformer": transformer}

    monkeypatch.setattr(ComposedPipelineBase, "__init__", initialize_base)
    monkeypatch.setattr(lora_module, "get_local_torch_device", lambda: torch.device("cpu"))

    class _Pipeline(lora_module.LoRAPipeline):

        def create_pipeline_stages(self, fastvideo_args):
            raise NotImplementedError

    return _Pipeline("unused", args)


def test_no_lora_setup_keeps_the_transformer_deferred(monkeypatch):
    loaded = []
    transformer = LazyModule(
        "transformer",
        lambda: loaded.append("transformer") or _LoRAConfigComponent(["proj_out"]),
    )

    pipeline = _build_stub_lora_pipeline(monkeypatch, transformer)

    assert loaded == []
    assert not transformer.is_materialized
    assert pipeline.exclude_lora_layers == {}
    assert pipeline.trainable_transformer_modules == {"transformer": transformer}


def test_lora_conversion_does_not_materialize_a_deferred_dit(monkeypatch):
    loaded = []
    transformer = LazyModule(
        "transformer",
        lambda: loaded.append("transformer") or _LoRAConfigComponent(["proj_out"]),
    )
    pipeline = _build_stub_lora_pipeline(monkeypatch, transformer, excluded_layers=["proj_out"])

    pipeline.convert_to_lora_layers()

    assert loaded == []
    assert not transformer.is_materialized
    assert pipeline.exclude_lora_layers == {}

    transformer.materialize()
    assert loaded == ["transformer"]
    assert transformer.is_materialized
    assert pipeline.exclude_lora_layers == {"transformer": ["proj_out"]}


def test_lora_release_drops_block_mapping_so_the_dit_can_free(monkeypatch):
    import gc
    import weakref

    holder = {}

    def load():
        component = _LoRAConfigComponent([])
        holder["component"] = component
        return component

    transformer = LazyModule("transformer", load)
    pipeline = _build_stub_lora_pipeline(monkeypatch, transformer)
    pipeline.convert_to_lora_layers()
    transformer.materialize()
    ref = weakref.ref(holder["component"])
    del holder["component"]

    assert transformer.release() is True
    gc.collect()
    assert ref() is None
    assert pipeline.lora_layers == {}


def test_lora_transformer_bookkeeping_is_per_pipeline(monkeypatch):
    first_transformer = LazyModule("transformer", lambda: _LoRAConfigComponent([]))
    first = _build_stub_lora_pipeline(monkeypatch, first_transformer)
    second_transformer = LazyModule("transformer", lambda: _LoRAConfigComponent([]))
    second = _build_stub_lora_pipeline(monkeypatch, second_transformer)

    assert first.trainable_transformer_modules == {"transformer": first_transformer}
    assert second.trainable_transformer_modules == {"transformer": second_transformer}
    assert first.trainable_transformer_modules is not second.trainable_transformer_modules
