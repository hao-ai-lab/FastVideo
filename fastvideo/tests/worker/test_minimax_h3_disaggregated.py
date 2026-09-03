# SPDX-License-Identifier: Apache-2.0
"""CPU contracts for the two-node MiniMax-H3 component pipeline."""

from __future__ import annotations

import asyncio
from dataclasses import replace
import pickle
from types import SimpleNamespace
from typing import Any

import pytest
import torch

from fastvideo.configs.pipelines.minimax_h3 import MiniMaxH3PipelineConfig
from fastvideo.fastvideo_args import FastVideoArgs
from fastvideo.pipelines.basic.minimax_h3.disaggregated import (
    MINIMAX_H3_WIRE_SCHEMA_VERSION,
    MiniMaxH3DenoisedState,
    MiniMaxH3DiTPipeline,
    MiniMaxH3EncodedState,
    MiniMaxH3EncoderDecoderPipeline,
    MiniMaxH3RefDiTPipeline,
    MiniMaxH3RefEncoderDecoderPipeline,
)
from fastvideo.pipelines.basic.minimax_h3.packing import MiniMaxH3PackedLayout
from fastvideo.pipelines.basic.minimax_h3.stages.minimax_h3_latent_preparation import MINIMAX_H3_LAYOUT_KEY
from fastvideo.pipelines.pipeline_batch_info import ForwardBatch
import fastvideo.worker.minimax_h3_disaggregated as disaggregated_runtime
from fastvideo.worker.minimax_h3_disaggregated import (
    MiniMaxH3DisaggregatedExecutor,
    RayMiniMaxH3DisaggregatedRuntime,
    _node_resource,
    _resident_role_args,
    _validate_topology,
)
from fastvideo.worker.executor import Executor


def _layout(*, noncontiguous: bool = False) -> MiniMaxH3PackedLayout:
    position_ids = torch.arange(27, dtype=torch.float64).reshape(3, 9).transpose(0, 1)
    if not noncontiguous:
        position_ids = position_ids.contiguous()
    return MiniMaxH3PackedLayout(
        sequence_length=9,
        position_ids=position_ids,
        token_tags=torch.tensor([1, 1, 1, 2, 2, 0, 0, 0, 0]),
        video_indices=torch.tensor([5, 6, 7, 8]),
        audio_indices=torch.tensor([3, 4]),
        text_indices=torch.tensor([0, 1, 2]),
        num_condition_video_rows=0,
        num_condition_audio_rows=0,
        num_video_latent_frames=2,
        latent_height=2,
        latent_width=2,
        num_audio_latents=1,
    )


def _encoded_batch(*, request_id: str | None = None) -> ForwardBatch:
    prompt_embeds = torch.arange(12, dtype=torch.float32).reshape(1, 4, 3).transpose(1, 2)
    video_latents = torch.arange(20, dtype=torch.float32).reshape(5, 4).transpose(0, 1)
    audio_latents = torch.arange(12, dtype=torch.float32).reshape(6, 2).transpose(0, 1)
    extra: dict[str, Any] = {
        MINIMAX_H3_LAYOUT_KEY: _layout(noncontiguous=True),
        "vsa_mode": "compete",
        "vsa_dense_first_n_steps": 2,
        "vsa_dense_layers": [1, 7],
    }
    if request_id is not None:
        extra["request_id"] = request_id
    return ForwardBatch(
        data_type="video",
        prompt="must not cross the wire",
        prompt_embeds=[prompt_embeds],
        latents=video_latents,
        audio_latents=audio_latents,
        raw_latent_shape=(1, 16, 2, 2, 2),
        num_inference_steps=8,
        VSA_sparsity=0.75,
        extra=extra,
    )


def test_encoded_wire_state_is_minimal_cpu_contiguous_and_pickleable() -> None:
    batch = _encoded_batch()
    assert not batch.prompt_embeds[0].is_contiguous()
    assert not batch.latents.is_contiguous()
    assert not batch.audio_latents.is_contiguous()
    assert not batch.extra[MINIMAX_H3_LAYOUT_KEY].position_ids.is_contiguous()

    state = MiniMaxH3EncodedState.from_batch(batch, request_id="request-7")
    restored = pickle.loads(pickle.dumps(state))

    assert restored.request_id == "request-7"
    assert restored.schema_version == MINIMAX_H3_WIRE_SCHEMA_VERSION
    assert set(vars(restored)) == {
        "request_id",
        "prompt_embeds",
        "video_latents",
        "audio_latents",
        "layout",
        "raw_latent_shape",
        "num_inference_steps",
        "vsa_sparsity",
        "vsa_mode",
        "vsa_dense_first_n_steps",
        "vsa_dense_layers",
        "logging_info",
        "schema_version",
    }
    for tensor in (
        restored.prompt_embeds,
        restored.video_latents,
        restored.audio_latents,
        restored.layout.position_ids,
        restored.layout.token_tags,
        restored.layout.video_indices,
        restored.layout.audio_indices,
        restored.layout.text_indices,
    ):
        assert tensor.device.type == "cpu"
        assert tensor.is_contiguous()

    roundtrip = restored.to_batch(device="cpu")
    assert torch.equal(roundtrip.prompt_embeds[0], batch.prompt_embeds[0])
    assert torch.equal(roundtrip.latents, batch.latents)
    assert torch.equal(roundtrip.audio_latents, batch.audio_latents)
    assert roundtrip.raw_latent_shape == batch.raw_latent_shape
    assert roundtrip.num_inference_steps == 8
    assert roundtrip.VSA_sparsity == 0.75
    assert roundtrip.extra["vsa_mode"] == "compete"
    assert roundtrip.extra["vsa_dense_first_n_steps"] == 2
    assert roundtrip.extra["vsa_dense_layers"] == (1, 7)


def test_denoised_wire_state_roundtrips_only_decode_inputs() -> None:
    encoded = MiniMaxH3EncodedState.from_batch(_encoded_batch(), request_id="request-8")
    denoised_batch = encoded.to_batch(device="cpu")
    denoised_batch.latents = denoised_batch.latents + 1
    denoised_batch.audio_latents = denoised_batch.audio_latents - 1

    state = MiniMaxH3DenoisedState.from_batch(denoised_batch, request_id=encoded.request_id)
    restored = pickle.loads(pickle.dumps(state))
    roundtrip = restored.to_batch()

    assert restored.request_id == "request-8"
    assert set(vars(restored)) == {
        "request_id",
        "video_latents",
        "audio_latents",
        "layout",
        "raw_latent_shape",
        "logging_info",
        "schema_version",
    }
    assert torch.equal(roundtrip.latents, denoised_batch.latents)
    assert torch.equal(roundtrip.audio_latents, denoised_batch.audio_latents)
    assert roundtrip.raw_latent_shape == (1, 16, 2, 2, 2)
    assert roundtrip.prompt_embeds == []


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ({"schema_version": 999}, "schema version"),
        ({"request_id": ""}, "non-empty request_id"),
        ({"raw_latent_shape": (1, 16, 0, 2, 2)}, "raw latent shape"),
        ({"num_inference_steps": 0}, "at least one denoising step"),
        ({"vsa_mode": "unknown"}, "vsa_mode"),
        ({"video_latents": torch.zeros(3, 5)}, "video latent rows"),
        ({"audio_latents": torch.zeros(1, 6)}, "audio latent rows"),
        ({"prompt_embeds": torch.zeros(1, 2, 4)}, "prompt embedding rows"),
    ],
)
def test_encoded_wire_state_rejects_incompatible_payloads(mutation: dict[str, Any], message: str) -> None:
    state = MiniMaxH3EncodedState.from_batch(_encoded_batch(), request_id="valid")
    with pytest.raises(ValueError, match=message):
        replace(state, **mutation)


def test_wire_state_rejects_noncontiguous_or_malformed_layout() -> None:
    state = MiniMaxH3EncodedState.from_batch(_encoded_batch(), request_id="valid")
    with pytest.raises(ValueError, match="layout.position_ids must be a contiguous CPU tensor"):
        replace(state, layout=_layout(noncontiguous=True))

    malformed = replace(_layout(), token_tags=torch.zeros(8, dtype=torch.long))
    with pytest.raises(ValueError, match=r"layout.token_tags.*expected \(9,\)"):
        replace(state, layout=malformed)


def test_from_batch_requires_all_stage_boundary_fields() -> None:
    with pytest.raises(ValueError, match="one prompt embedding and both latent streams"):
        MiniMaxH3EncodedState.from_batch(ForwardBatch(data_type="video"))

    batch = _encoded_batch()
    batch.raw_latent_shape = None
    with pytest.raises(ValueError, match="five-dimensional raw latent shape"):
        MiniMaxH3EncodedState.from_batch(batch)

    batch = _encoded_batch()
    batch.extra.pop(MINIMAX_H3_LAYOUT_KEY)
    with pytest.raises(TypeError, match="missing its packed layout"):
        MiniMaxH3EncodedState.from_batch(batch)


def test_resident_roles_have_isolated_component_sets_and_no_lazy_modules() -> None:
    assert set(MiniMaxH3EncoderDecoderPipeline._required_config_modules) == {
        "text_encoder",
        "tokenizer",
        "processor",
        "vae",
        "audio_vae",
        "scheduler",
    }
    assert set(MiniMaxH3DiTPipeline._required_config_modules) == {
        "transformer",
        "scheduler",
        "audio_scheduler",
    }
    assert MiniMaxH3EncoderDecoderPipeline._lazy_module_names == ()
    assert MiniMaxH3DiTPipeline._lazy_module_names == ()


def test_role_downloads_include_only_owned_weights_plus_required_dit_metadata() -> None:
    encoder_dirs = set(MiniMaxH3EncoderDecoderPipeline.get_hf_download_component_dirs())
    assert encoder_dirs == {
        "text_encoder",
        "tokenizer",
        "processor",
        "vae",
        "audio_vae",
        "scheduler",
    }
    encoder_patterns = MiniMaxH3EncoderDecoderPipeline.get_hf_download_allow_patterns()
    assert "transformer/config.json" in encoder_patterns
    assert "transformer/**" not in encoder_patterns

    ref_encoder_patterns = MiniMaxH3RefEncoderDecoderPipeline.get_hf_download_allow_patterns()
    assert "transformer/config.json" in ref_encoder_patterns
    assert "transformer_ref/config.json" in ref_encoder_patterns
    assert "transformer/**" not in ref_encoder_patterns
    assert "transformer_ref/**" not in ref_encoder_patterns

    assert set(MiniMaxH3DiTPipeline.get_hf_download_component_dirs()) == {
        "transformer",
        "scheduler",
        "audio_scheduler",
    }
    assert set(MiniMaxH3RefDiTPipeline.get_hf_download_component_dirs()) == {
        "transformer_ref",
        "scheduler",
        "audio_scheduler",
    }


def _role_source_args() -> SimpleNamespace:
    return SimpleNamespace(
        num_gpus=8,
        tp_size=2,
        sp_size=4,
        hsdp_replicate_dim=2,
        hsdp_shard_dim=4,
        ray_placement_group=object(),
        ray_runtime_env={"env_vars": {"ORIGINAL": "1"}},
        distributed_executor_backend="ray",
        use_fsdp_inference=True,
        dit_cpu_offload=True,
        dit_layerwise_offload=True,
        text_encoder_cpu_offload=True,
        image_encoder_cpu_offload=True,
        vae_cpu_offload=True,
        lazy_module_load=True,
        h3_sequential_load=True,
        vae_parallel_encode=True,
        vae_parallel_decode=True,
        lora_path="adapter.safetensors",
        enable_torch_compile=True,
        enable_torch_compile_text_encoder=True,
        enable_torch_compile_vae=True,
        enable_torch_compile_audio_vae=True,
    )


@pytest.mark.parametrize("role", ["encoder_decoder", "dit"])
def test_resident_role_args_force_single_gpu_persistent_execution(role: str) -> None:
    source = _role_source_args()
    args = _resident_role_args(source, role=role)

    assert (args.num_gpus, args.tp_size, args.sp_size) == (1, 1, 1)
    assert (args.hsdp_replicate_dim, args.hsdp_shard_dim) == (1, 1)
    assert args.distributed_executor_backend == "mp"
    assert args.ray_placement_group is None
    assert args.ray_runtime_env is None
    assert args.use_fsdp_inference is False
    assert args.lazy_module_load is False
    assert args.h3_sequential_load is False
    assert args.vae_parallel_encode is False
    assert args.vae_parallel_decode is False
    assert all(
        getattr(args, name) is False for name in (
            "dit_cpu_offload",
            "dit_layerwise_offload",
            "text_encoder_cpu_offload",
            "image_encoder_cpu_offload",
            "vae_cpu_offload",
        )
    )
    assert source.num_gpus == 8
    assert source.lazy_module_load is True
    assert source.ray_runtime_env == {"env_vars": {"ORIGINAL": "1"}}

    if role == "encoder_decoder":
        assert args.lora_path is None
        assert args.enable_torch_compile is False
        assert args.enable_torch_compile_text_encoder is True
    else:
        assert args.lora_path == "adapter.safetensors"
        assert args.enable_torch_compile is True
        assert args.enable_torch_compile_text_encoder is False
        assert args.enable_torch_compile_vae is False
        assert args.enable_torch_compile_audio_vae is False


def test_resident_role_args_reject_unknown_role() -> None:
    with pytest.raises(ValueError, match="Unknown MiniMax-H3 worker role"):
        _resident_role_args(_role_source_args(), role="both")


def test_topology_requires_two_distinct_live_ray_nodes() -> None:
    resources = {
        _node_resource("10.0.0.1"): 1.0,
        _node_resource("10.0.0.2"): 1.0,
        "CPU": 64.0,
    }
    _validate_topology("10.0.0.1", "10.0.0.2", resources)

    with pytest.raises(ValueError, match="both encoder and DiT node IPs"):
        _validate_topology("", "10.0.0.2", resources)
    with pytest.raises(ValueError, match="must use different Ray nodes"):
        _validate_topology("10.0.0.1", "10.0.0.1", resources)
    with pytest.raises(RuntimeError, match=r"10\.0\.0\.3.*available node IPs"):
        _validate_topology("10.0.0.1", "10.0.0.3", resources)


def test_executor_selector_routes_disaggregated_h3_away_from_full_pipeline_ray_workers() -> None:
    args = SimpleNamespace(h3_disaggregated=True, distributed_executor_backend="ray")
    assert Executor.get_class(args) is MiniMaxH3DisaggregatedExecutor


def _valid_disaggregated_args(**overrides: Any) -> FastVideoArgs:
    values: dict[str, Any] = {
        "model_path": "unused/for-this-test",
        "pipeline_config": MiniMaxH3PipelineConfig(),
        "distributed_executor_backend": "ray",
        "h3_disaggregated": True,
        "h3_encoder_node_ip": "10.0.0.1",
        "h3_dit_node_ip": "10.0.0.2",
    }
    values.update(overrides)
    return FastVideoArgs(**values)


def test_disaggregated_args_are_opt_in_and_trim_node_addresses() -> None:
    args = _valid_disaggregated_args(
        h3_encoder_node_ip=" 10.0.0.1 ",
        h3_dit_node_ip=" 10.0.0.2 ",
        h3_ray_address=" 10.0.0.1:6379 ",
    )

    assert args.h3_encoder_node_ip == "10.0.0.1"
    assert args.h3_dit_node_ip == "10.0.0.2"
    assert args.h3_ray_address == "10.0.0.1:6379"
    assert Executor.get_class(args) is MiniMaxH3DisaggregatedExecutor


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"distributed_executor_backend": "mp"}, "requires distributed_executor_backend='ray'"),
        ({"h3_dit_node_ip": "10.0.0.1"}, "must use different Ray node"),
        ({"h3_encoder_node_ip": None}, "requires h3_encoder_node_ip"),
        ({"num_gpus": 2}, "requires num_gpus=1"),
        ({"tp_size": 2, "num_gpus": 2}, "requires num_gpus=1"),
        ({"sp_size": 2, "num_gpus": 2}, "requires num_gpus=1"),
        ({"use_fsdp_inference": True}, "does not support FSDP"),
        ({"lazy_module_load": True}, "keeps role components resident"),
        ({"h3_sequential_load": True}, "keeps role components resident"),
        ({"vae_parallel_decode": True}, "does not support sequence-parallel VAE"),
        ({"video_decode_backend": "taeh3"}, "requires video_decode_backend='h3-vae'"),
    ],
)
def test_disaggregated_args_reject_incompatible_execution(overrides: dict[str, Any], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        _valid_disaggregated_args(**overrides)


class _FakeRef:

    def __init__(self, kind: str, ordinal: int, value: Any = None) -> None:
        self.kind = kind
        self.ordinal = ordinal
        self.value = value


class _RemoteMethod:

    def __init__(self, name: str, events: list[tuple[str, tuple[Any, ...]]], result_factory=None) -> None:
        self.name = name
        self.events = events
        self.calls: list[tuple[Any, ...]] = []
        self.refs: list[_FakeRef] = []
        self.result_factory = result_factory

    def remote(self, *args: Any) -> _FakeRef:
        self.calls.append(args)
        self.events.append((self.name, args))
        value = None if self.result_factory is None else self.result_factory(*args)
        ref = _FakeRef(self.name, len(self.calls), value)
        self.refs.append(ref)
        return ref


class _FakeEncoderActor:

    def __init__(self, events: list[tuple[str, tuple[Any, ...]]]) -> None:
        self.encode = _RemoteMethod("encode", events)
        self.decode = _RemoteMethod(
            "decode",
            events,
            lambda denoised_ref: ForwardBatch(data_type="video", extra={"decoded_from": denoised_ref}),
        )


class _FakeDiTActor:

    def __init__(self, events: list[tuple[str, tuple[Any, ...]]]) -> None:
        self.denoise = _RemoteMethod("denoise", events)


class _FakeRay:

    def __init__(self, events: list[tuple[str, tuple[Any, ...]]]) -> None:
        self.events = events
        self.get_calls: list[_FakeRef] = []
        self.wait_calls: list[tuple[list[_FakeRef], int, bool]] = []

    def get(self, ref: _FakeRef):
        self.get_calls.append(ref)
        self.events.append(("get", (ref, )))
        return ref.value

    def wait(self, refs: list[_FakeRef], *, num_returns: int, fetch_local: bool):
        self.wait_calls.append((refs, num_returns, fetch_local))
        self.events.append(("wait", (refs[0], )))
        return refs[:num_returns], refs[num_returns:]


def _fake_runtime(monkeypatch):
    events: list[tuple[str, tuple[Any, ...]]] = []
    fake_ray = _FakeRay(events)
    monkeypatch.setattr(disaggregated_runtime, "ray", fake_ray)
    runtime = RayMiniMaxH3DisaggregatedRuntime.__new__(RayMiniMaxH3DisaggregatedRuntime)
    runtime._closed = False
    runtime.encoder_decoder = _FakeEncoderActor(events)
    runtime.dit = _FakeDiTActor(events)
    return runtime, fake_ray, events


def test_submit_builds_actor_dag_without_fetching_intermediate_payloads(monkeypatch) -> None:
    runtime, fake_ray, _ = _fake_runtime(monkeypatch)
    batch = _encoded_batch(request_id="batch-id")

    decoded_ref = runtime.submit(batch)

    encode_args = runtime.encoder_decoder.encode.calls[0]
    assert encode_args == (batch, "batch-id")
    assert runtime.dit.denoise.calls[0] == (runtime.encoder_decoder.encode.refs[0], )
    assert runtime.encoder_decoder.decode.calls[0] == (runtime.dit.denoise.refs[0], )
    assert decoded_ref.kind == "decode"
    assert fake_ray.get_calls == []


def test_execute_forward_fetches_only_the_final_decode(monkeypatch) -> None:
    runtime, fake_ray, _ = _fake_runtime(monkeypatch)

    output = runtime.execute_forward(_encoded_batch(), request_id="explicit-id")

    assert isinstance(output, ForwardBatch)
    assert runtime.encoder_decoder.encode.calls[0][1] == "explicit-id"
    assert [ref.kind for ref in fake_ray.get_calls] == ["decode"]


def test_iter_forward_overlaps_encode_denoise_decode_with_bounded_lookahead(monkeypatch) -> None:
    runtime, fake_ray, events = _fake_runtime(monkeypatch)
    batches = [_encoded_batch(request_id=f"request-{index}") for index in range(3)]

    outputs = list(runtime.iter_forward(batches))

    assert len(outputs) == 3
    assert [call[1] for call in runtime.encoder_decoder.encode.calls] == [
        "request-0",
        "request-1",
        "request-2",
    ]
    assert [ref.kind for ref in fake_ray.get_calls] == ["decode", "decode", "decode"]
    assert len(fake_ray.wait_calls) == 2
    assert all(refs[0].kind == "denoise" and num_returns == 1 and fetch_local is False
               for refs, num_returns, fetch_local in fake_ray.wait_calls)

    event_names = [name for name, _ in events]
    assert event_names == [
        "encode",
        "denoise",
        "encode",
        "wait",
        "decode",
        "denoise",
        "get",
        "encode",
        "wait",
        "decode",
        "denoise",
        "get",
        "decode",
        "get",
    ]


def test_iter_forward_async_exposes_the_same_bounded_pipeline(monkeypatch) -> None:
    runtime, fake_ray, _ = _fake_runtime(monkeypatch)
    batches = [_encoded_batch(request_id=f"async-{index}") for index in range(2)]

    async def collect() -> list[ForwardBatch]:
        return [batch async for batch in runtime.iter_forward_async(batches)]

    outputs = asyncio.run(collect())

    assert len(outputs) == 2
    assert [call[1] for call in runtime.encoder_decoder.encode.calls] == ["async-0", "async-1"]
    assert [ref.kind for ref in fake_ray.get_calls] == ["decode", "decode"]
