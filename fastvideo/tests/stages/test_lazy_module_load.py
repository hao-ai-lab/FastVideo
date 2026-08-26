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

    def create_pipeline_stages(self, fastvideo_args):
        raise NotImplementedError


def _schedule(modules, stages):
    return _FakePipeline(modules, stages)._build_lazy_release_schedule()


def _lazy(name):
    return LazyModule(name, lambda: _Component(name))


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
])
def test_training_mode_never_defers(lazy, training, expected):
    args = SimpleNamespace(lazy_module_load=lazy, training_mode=training)

    assert ComposedPipelineBase._lazy_module_load_enabled(args) is expected


def test_flag_defaults_to_off():
    from fastvideo.fastvideo_args import FastVideoArgs

    fields = {f.name: f for f in dataclasses.fields(FastVideoArgs)}
    assert fields["lazy_module_load"].default is False


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
    with caplog.at_level("WARNING"):
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
