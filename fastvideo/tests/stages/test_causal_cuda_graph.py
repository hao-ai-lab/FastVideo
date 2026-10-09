# SPDX-License-Identifier: Apache-2.0
"""Graph staging contracts; gpu-marked cases require one CUDA GPU."""

from types import SimpleNamespace

import pytest
import torch

from fastvideo.pipelines.basic.wan.stages import causal_cuda_graph as graphs


def _inputs(dtype=torch.float32):
    return (torch.ones(1, 2, dtype=dtype), [torch.ones(1, 2)], torch.ones(1, dtype=dtype))


def test_staging_reuses_storage_and_detects_layout_and_dtype_changes():
    wrapper = graphs.CausalCudaGraphWrapper(lambda *args: args[0])
    args = _inputs()
    wrapper._stage(args, {})
    pointer = wrapper._static_args[0].data_ptr()
    wrapper._stage((args[0] * 3, args[1], args[2] * 2), {})
    assert wrapper._static_args[0].data_ptr() == pointer
    assert torch.equal(wrapper._static_args[0], args[0] * 3)
    assert wrapper.reset_count == 0
    wrapper._stage((args[0].double(), args[1], args[2]), {})
    assert wrapper.reset_count == 1
    wrapper._stage((torch.ones(2, 1, dtype=torch.float64).t(), args[1], args[2]), {})
    assert wrapper.reset_count == 2


def test_cache_positions_do_not_reset_but_replaced_storage_does():
    wrapper = graphs.CausalCudaGraphWrapper(lambda *args: args[0])
    args = _inputs()
    cache = [{"k": torch.zeros(1, 4), "v": torch.zeros(1, 4), "global_end_index": 4, "local_end_index": 4}]
    wrapper._stage(args, {"kv_cache": cache, "current_start": 4})
    cache[0]["global_end_index"] = 8
    wrapper._stage(args, {"kv_cache": cache, "current_start": 8})
    assert wrapper.reset_count == 0
    cache[0]["k"] = torch.zeros(1, 4)
    wrapper._stage(args, {"kv_cache": cache, "current_start": 8})
    assert wrapper.reset_count == 1


def test_changed_conditioning_resets_recording():
    wrapper = graphs.CausalCudaGraphWrapper(lambda *args: args[0])
    args = _inputs()
    wrapper._stage(args, {})
    wrapper._stage((args[0], [torch.zeros_like(args[1][0])], args[2]), {})
    assert wrapper.reset_count == 1


def test_context_dtype_does_not_reset_denoising_wrapper(monkeypatch):
    class RecordingWrapper:
        def __init__(self, fn, **kwargs):
            self.dtypes = []

        def __call__(self, *args, **kwargs):
            self.dtypes.append(args[2].dtype)

    monkeypatch.setattr(graphs, "CausalCudaGraphWrapper", RecordingWrapper)
    model = lambda *args, **kwargs: None
    dispatch = graphs.CausalCudaGraphDispatch(model, enabled=True)
    for _ in range(30):
        for first in (True, False, False, False):
            dispatch.call(model, *_inputs(), is_chunk_start=first, is_steady_state=True)
        dispatch.call(model, *_inputs(torch.int64), is_chunk_start=False, is_context=True, is_steady_state=True)
    assert set(dispatch.wrappers["continuation"].dtypes) == {torch.float32}
    assert set(dispatch.wrappers["context"].dtypes) == {torch.int64}


def test_dispatch_rejects_wrong_transformer():
    model = lambda *args, **kwargs: None
    dispatch = graphs.CausalCudaGraphDispatch(model, enabled=True)
    with pytest.raises(ValueError, match="different transformer"):
        dispatch.call(lambda *args: None, *_inputs(), is_chunk_start=True, is_steady_state=True)


def test_dispatch_requires_explicit_chunk_start_for_graph_execution():
    model = lambda *args, **kwargs: None
    dispatch = graphs.CausalCudaGraphDispatch(model, enabled=True)
    with pytest.raises(ValueError, match="explicit chunk-start"):
        dispatch.call(model, *_inputs(), is_chunk_start=None, is_steady_state=True)


@pytest.mark.parametrize("current_start,local_ends,enabled,expected_steady", [
    (0, [8, 8], True, False),
    (8, [4, 8], True, False),
    (8, [8, 4], True, False),
    (8, [8, 8], True, True),
    (12, [8, 8], True, True),
    (8, [8, 8], False, False),
])
def test_graphs_wait_for_every_cache_layer_and_finalize_replay_counters(
    current_start, local_ends, enabled, expected_steady,
):
    from fastvideo.pipelines.basic.wan.stages.causal_denoising import WanCausalDenoisingBase

    stage = object.__new__(WanCausalDenoisingBase)
    stage.frame_seq_length = 2
    cache = [{"k": torch.zeros(1, 8), "global_end_index": current_start, "local_end_index": end}
             for end in local_ends]
    calls = []
    output = object()

    def record_call(*args, **kwargs):
        calls.append(kwargs)
        return output

    dispatch = SimpleNamespace(enabled=enabled, call=record_call)
    latents = torch.ones(1, 1, 2, 1, 1)
    result = stage._call_transformer(
        dispatch, object(), latents, [], torch.ones(1), kv_cache=cache,
        current_start=current_start, is_chunk_start=True,
    )
    assert result is output
    assert calls[0]["is_steady_state"] is expected_steady
    assert calls[0]["is_chunk_start"] is (True if enabled else None)
    assert [entry["global_end_index"] for entry in cache] == [
        current_start + 4 if expected_steady else current_start
    ] * 2
    assert [entry["local_end_index"] for entry in cache] == local_ends


def test_wrapper_rejects_cpu_inputs():
    with pytest.raises(ValueError, match="CUDA latent"):
        graphs.CausalCudaGraphWrapper(lambda x: x)(torch.ones(1))


@pytest.mark.parametrize("setting,value", [
    ("num_gpus", 2), ("sp_size", 2), ("tp_size", 2), ("use_fsdp_inference", True),
    ("dit_cpu_offload", True), ("dit_layerwise_offload", True), ("enable_torch_compile", True),
    ("inference_torch_compile", True),
])
def test_unsupported_engine_settings_fall_back(monkeypatch, setting, value):
    from fastvideo.pipelines.basic.wan.stages.causal_denoising import WanCausalDenoisingBase
    stage = object.__new__(WanCausalDenoisingBase)
    prepared = []
    stage.transformer = SimpleNamespace(rope_cache_policy="relativistic", prepare_causal_cuda_graph=prepared.append)
    stage.transformer_2 = None
    stage.local_attn_size, stage.sink_size, stage.num_frames_per_block = 9, 0, 3
    args = SimpleNamespace(enable_causal_cuda_graph=True, **{setting: value})
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    dispatch = stage._make_graph_dispatch(args, SimpleNamespace(device=torch.device("cuda")))
    assert dispatch.enabled is False
    assert prepared == [False]


@pytest.mark.parametrize("unsupported", ["cpu", "no_cuda", "global_window", "small_window", "absolute", "dual", "boundary", "model"])
def test_unsupported_model_or_request_falls_back(monkeypatch, unsupported):
    from fastvideo.pipelines.basic.wan.stages.causal_denoising import WanCausalDenoisingBase
    stage = object.__new__(WanCausalDenoisingBase)
    prepared = []
    stage.transformer = SimpleNamespace(rope_cache_policy="relativistic", prepare_causal_cuda_graph=prepared.append)
    stage.transformer_2 = None
    stage.local_attn_size, stage.sink_size, stage.num_frames_per_block = 9, 1, 3
    device = "cpu" if unsupported == "cpu" else "cuda"
    monkeypatch.setattr(torch.cuda, "is_available", lambda: unsupported != "no_cuda")
    if unsupported == "global_window":
        stage.local_attn_size = -1
    elif unsupported == "small_window":
        stage.local_attn_size = 3
    elif unsupported == "absolute":
        stage.transformer.rope_cache_policy = "absolute"
    elif unsupported == "dual":
        stage.transformer_2 = object()
    elif unsupported == "model":
        del stage.transformer.prepare_causal_cuda_graph
    dispatch = stage._make_graph_dispatch(
        SimpleNamespace(enable_causal_cuda_graph=True), SimpleNamespace(device=torch.device(device)),
        boundary_timestep=500 if unsupported == "boundary" else None,
    )
    assert dispatch.enabled is False
    assert prepared == ([] if unsupported == "model" else [False])


def test_dispatch_disabled_preserves_legacy_transformer_signature():
    # Non-Wan callers need not accept the added graph-only argument.
    model = lambda x: x + 1
    dispatch = graphs.CausalCudaGraphDispatch(model, enabled=False)
    assert dispatch.call(model, 3, is_chunk_start=None, is_steady_state=False) == 4


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires one CUDA GPU")
def test_capture_executes_once_and_returns_owned_outputs():
    cache = {"count": torch.zeros(1, device="cuda")}

    def forward(x, prompts, t, *, kv_cache):
        kv_cache["count"].add_(x.sum() + t.sum())
        return kv_cache["count"].clone()

    wrapper = graphs.CausalCudaGraphWrapper(forward, warmup_iters=0)
    x = torch.ones(1, device="cuda")
    t = torch.zeros(1, device="cuda")
    first = wrapper(x, [], t, kv_cache=cache)
    second = wrapper(x * 2, [], t, kv_cache=cache)
    assert first.item() == 1
    assert second.item() == 3
    assert cache["count"].item() == 3
    assert wrapper.capture_count == 1 and wrapper.replay_count == 2


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires one CUDA GPU")
@pytest.mark.parametrize("change", ["shape", "dtype", "stride", "conditioning", "cache", "scalar"])
def test_graph_rebuilds_safely_when_inputs_or_storage_change(change):
    def forward(x, prompts, t, *, kv_cache, extra):
        kv_cache["count"].add_(x.sum())
        return {"prediction": [x * extra["scale"] + t + prompts[0].sum() + extra["offsets"][0]],
                "state": kv_cache["count"].clone()}

    wrapper = graphs.CausalCudaGraphWrapper(forward, warmup_iters=0)
    x = torch.arange(4., device="cuda").reshape(2, 2)
    prompts = [torch.zeros(2, 2, device="cuda")]
    timestep = torch.ones(1, device="cuda")
    cache = {"count": torch.zeros(1, device="cuda")}
    extra = {"scale": 2., "offsets": (torch.ones(1, device="cuda"),)}
    first = wrapper(x, prompts, timestep, kv_cache=cache, extra=extra)
    saved_prediction, saved_state = first["prediction"][0].clone(), first["state"].clone()
    if change == "shape":
        x = torch.arange(6., device="cuda").reshape(3, 2)
    elif change == "dtype":
        x = x.double()
    elif change == "stride":
        x = x.t()
        assert not x.is_contiguous()
    elif change == "conditioning":
        prompts = [torch.ones_like(prompts[0])]
    elif change == "cache":
        cache = {"count": torch.zeros_like(cache["count"])}
    else:
        extra["scale"] = 3.

    expected_count = cache["count"].clone()
    for step in (2., 3.):
        fresh_timestep = timestep * step
        fresh_extra = {"scale": extra["scale"], "offsets": (extra["offsets"][0] * step,)}
        expected_prediction = x * fresh_extra["scale"] + fresh_timestep + prompts[0].sum() + fresh_extra["offsets"][0]
        expected_count.add_(x.sum())
        result = wrapper(x, prompts, fresh_timestep, kv_cache=cache, extra=fresh_extra)
        torch.testing.assert_close(result["prediction"][0], expected_prediction, atol=0, rtol=0)
        torch.testing.assert_close(result["state"], expected_count, atol=0, rtol=0)
    torch.testing.assert_close(first["prediction"][0], saved_prediction, atol=0, rtol=0)
    torch.testing.assert_close(first["state"], saved_state, atol=0, rtol=0)
    assert wrapper.capture_count == 2
    assert wrapper.replay_count == 3
    assert wrapper.reset_count == 1


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires one CUDA GPU")
@pytest.mark.parametrize("failure", ["capture", "first_replay"])
def test_recording_failure_leaves_no_invalid_graph_and_can_retry(monkeypatch, failure):
    cache = {"count": torch.zeros(1, device="cuda")}

    def forward(x, prompts, t, *, kv_cache):
        kv_cache["count"].add_(x.sum())
        return kv_cache["count"].clone()

    if failure == "capture":
        original = torch.cuda.graph
        attribute = "graph"
        owner = torch.cuda
    else:
        original = torch.cuda.CUDAGraph.replay
        attribute = "replay"
        owner = torch.cuda.CUDAGraph
    failed = False

    def fail_once(*args, **kwargs):
        nonlocal failed
        if not failed:
            failed = True
            raise RuntimeError("injected recording failure")
        return original(*args, **kwargs)

    monkeypatch.setattr(owner, attribute, fail_once)
    wrapper = graphs.CausalCudaGraphWrapper(forward, warmup_iters=0)
    x, timestep = torch.ones(1, device="cuda"), torch.zeros(1, device="cuda")
    with pytest.raises(RuntimeError, match="injected recording failure"):
        wrapper(x, [], timestep, kv_cache=cache)
    assert wrapper._graph is None
    assert wrapper.capture_count == wrapper.replay_count == 0
    assert cache["count"].item() == 0
    first = wrapper(x, [], timestep, kv_cache=cache)
    second = wrapper(x * 2, [], timestep, kv_cache=cache)
    assert first.item() == 1
    assert second.item() == 3
    assert cache["count"].item() == 3
    assert wrapper.capture_count == 1 and wrapper.replay_count == 2


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires one CUDA GPU")
def test_stateful_rollout_records_three_graphs_without_dtype_resets():
    cache = {"sum": torch.zeros(1, device="cuda")}

    def forward(x, prompts, t, *, kv_cache, is_chunk_start, current_start):
        if is_chunk_start:
            kv_cache["sum"].mul_(0.5)
        kv_cache["sum"].add_(x.sum() + t.float().sum())
        return kv_cache["sum"].clone()

    dispatch = graphs.CausalCudaGraphDispatch(forward, enabled=True)
    expected = torch.zeros(1, device="cuda")
    for chunk in range(32):
        for step in range(4):
            x = torch.full((1,), chunk + step, device="cuda", dtype=torch.float32)
            t = torch.tensor([0.25 * step], device="cuda")
            if step == 0:
                expected.mul_(0.5)
            expected.add_(x.sum() + t.sum())
            output = dispatch.call(forward, x, [], t, kv_cache=cache, current_start=chunk,
                                   is_chunk_start=step == 0, is_steady_state=True)
            torch.testing.assert_close(output, expected, atol=0, rtol=0)
        x = torch.full((1,), chunk, device="cuda", dtype=torch.float32)
        t = torch.zeros(1, device="cuda", dtype=torch.int64)
        expected.add_(x.sum())
        dispatch.call(forward, x, [], t, kv_cache=cache, current_start=chunk, is_chunk_start=False,
                      is_context=True, is_steady_state=True)
        torch.testing.assert_close(cache["sum"], expected, atol=0, rtol=0)
    assert dispatch.statistics() == {"capture_count": 3, "replay_count": 154, "reset_count": 0}


def _real_transformer_pair(window, sink):
    from fastvideo.models.loader.utils import set_default_torch_dtype
    from fastvideo.models.wan.causal_transformer import CausalWanTransformer3DModel
    from fastvideo.models.wan.config import WanVideoArchConfig, WanVideoConfig
    arch = WanVideoArchConfig(num_attention_heads=2, attention_head_dim=64, in_channels=2, out_channels=2,
                              text_dim=16, freq_dim=16, ffn_dim=256, num_layers=1, num_frames_per_block=1,
                              local_attn_size=window, sink_size=sink, rope_cache_policy="relativistic")
    arch.text_len = 8
    with set_default_torch_dtype(torch.bfloat16):
        eager = CausalWanTransformer3DModel(WanVideoConfig(arch_config=arch), {}).cuda().eval()
        captured = CausalWanTransformer3DModel(WanVideoConfig(arch_config=arch), {}).cuda().eval()
    # FastVideo linears allocate empty weights for the checkpoint loader.
    # Initialize every parameter explicitly in this weight-free fixture.
    with torch.no_grad():
        for parameter in eager.parameters():
            parameter.normal_(mean=0, std=0.02)
    captured.load_state_dict(eager.state_dict())
    return eager, captured


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires one CUDA GPU")
def test_disabling_graphs_clears_owned_device_tables_and_preserves_output(distributed_setup, env_overrides):
    from fastvideo import envs
    from fastvideo.forward_context import set_forward_context
    from fastvideo.pipelines.basic.wan.stages.causal_denoising import WanCausalDenoisingBase

    env_overrides.enter_context(envs.FASTVIDEO_ATTENTION_BACKEND.override("TORCH_SDPA"))
    model, _ = _real_transformer_pair(window=3, sink=1)
    stage = WanCausalDenoisingBase(model, SimpleNamespace())
    stage.frame_seq_length = 4
    keys = set(model.state_dict())
    latents = torch.randn(1, 2, 1, 4, 4, device="cuda", dtype=torch.bfloat16)
    prompts = [torch.randn(1, 8, 16, device="cuda", dtype=torch.bfloat16)]
    timestep = torch.ones(1, 1, device="cuda")
    results = []
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16), \
            set_forward_context(current_timestep=0, attn_metadata=None):
        for enabled in (True, False):
            dispatch = stage._make_graph_dispatch(SimpleNamespace(enable_causal_cuda_graph=enabled), latents)
            assert dispatch.enabled is enabled
            if not enabled:
                assert model._causal_rope_cache is None
                assert model.condition_embedder.time_embedder._frequency_cache is None
                assert model.condition_embedder.time_embedder.cache_frequencies is False
            cache = stage._initialize_kv_cache(1, torch.bfloat16, torch.device("cuda"))
            cross = stage._initialize_crossattn_cache(1, 8, torch.bfloat16, torch.device("cuda"))
            results.append(stage._call_transformer(
                dispatch, model, latents, prompts, timestep, kv_cache=cache, crossattn_cache=cross,
                current_start=0, start_frame=0, is_chunk_start=True,
            ))
            if enabled:
                assert model._causal_rope_cache is not None
                assert model.condition_embedder.time_embedder._frequency_cache is not None
    torch.testing.assert_close(results[0], results[1], atol=0, rtol=0)
    assert set(model.state_dict()) == keys


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires one CUDA GPU")
@pytest.mark.parametrize("backend", ["TORCH_SDPA", "FLASH_ATTN"])
@pytest.mark.parametrize("window,sink", [(3, 1), (6, 0)])
def test_real_transformer_long_rollout_matches_eager(distributed_setup, env_overrides, backend, window, sink):
    from fastvideo import envs
    from fastvideo.forward_context import set_forward_context
    from fastvideo.pipelines.basic.wan.stages.causal_denoising import WanCausalDenoisingBase

    env_overrides.enter_context(envs.FASTVIDEO_ATTENTION_BACKEND.override(backend))
    eager, captured = _real_transformer_pair(window, sink)
    assert eager.blocks[0].attn1.attn.backend.name == backend
    assert captured.blocks[0].attn1.attn.backend.name == backend
    captured.prepare_causal_cuda_graph(True)
    stage = WanCausalDenoisingBase(captured, SimpleNamespace())
    stage.frame_seq_length = 4
    generator = torch.Generator(device="cuda").manual_seed(11)
    prompts = [torch.randn(1, 8, 16, device="cuda", dtype=torch.bfloat16, generator=generator)]
    caches = [stage._initialize_kv_cache(1, torch.bfloat16, torch.device("cuda")) for _ in range(2)]
    crosses = [stage._initialize_crossattn_cache(1, 8, torch.bfloat16, torch.device("cuda")) for _ in range(2)]
    dispatch = graphs.CausalCudaGraphDispatch(captured, enabled=True)
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16), \
            set_forward_context(current_timestep=0, attn_metadata=None):
        for chunk in range(35):
            for step in range(4):
                context = step == 3
                latents = torch.randn(1, 2, 1, 4, 4, device="cuda", dtype=torch.bfloat16, generator=generator)
                timestep = torch.zeros(1, 1, device="cuda", dtype=torch.int64) if context else torch.full(
                    (1, 1), 749.25 - 250 * step, device="cuda", dtype=torch.float32)
                expected = eager(latents, prompts, timestep, kv_cache=caches[0], crossattn_cache=crosses[0],
                                 current_start=chunk * 4, start_frame=chunk)
                assert torch.isfinite(expected).all()
                actual = stage._call_transformer(
                    dispatch, captured, latents, prompts, timestep, kv_cache=caches[1], crossattn_cache=crosses[1],
                    current_start=chunk * 4, start_frame=chunk, is_chunk_start=step == 0, is_context=context)
                if not context or actual is not None:
                    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
                for first, second in zip(caches[0], caches[1], strict=True):
                    assert first["global_end_index"] == second["global_end_index"] == (chunk + 1) * 4
                    assert first["local_end_index"] == second["local_end_index"]
                    torch.testing.assert_close(first["k"], second["k"], atol=0, rtol=0)
                    torch.testing.assert_close(first["v"], second["v"], atol=0, rtol=0)
    assert dispatch.statistics()["capture_count"] == 3
    assert dispatch.statistics()["replay_count"] > 0
    assert dispatch.statistics()["reset_count"] == 0


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires one CUDA GPU")
@pytest.mark.parametrize("sampler", ["standard_50", "dmd", "dmd_warped"])
def test_real_sampler_preserves_latents_and_rng(distributed_setup, env_overrides, sampler):
    from fastvideo import envs
    from fastvideo.models.schedulers.scheduling_flow_match_euler_discrete import FlowMatchEulerDiscreteScheduler
    from fastvideo.models.schedulers.scheduling_flow_unipc_multistep import FlowUniPCMultistepScheduler
    from fastvideo.pipelines.basic.wan.stages.causal_denoising import CausalDMDDenosingStage, CausalDenoisingStage
    from fastvideo.tests.stages._denoising_fixtures import NullProgressBar, _args, _batch

    env_overrides.enter_context(envs.FASTVIDEO_ATTENTION_BACKEND.override("FLASH_ATTN"))
    models = _real_transformer_pair(window=3, sink=1)
    inputs = torch.randn(1, 2, 12, 4, 4, device="cuda")
    prompts = [torch.randn(1, 8, 16, device="cuda", dtype=torch.bfloat16)]
    results, random_states = [], []
    for enabled, model in zip((False, True), models, strict=True):
        assert model.blocks[0].attn1.attn.backend.name == "FLASH_ATTN"
        args, batch = _args(), _batch(steps=50 if sampler == "standard_50" else 4, cfg=False)
        args.disable_autocast = False
        args.enable_causal_cuda_graph = enabled
        args.pipeline_config.text_encoder_configs = [SimpleNamespace(arch_config=SimpleNamespace(text_len=8))]
        args.pipeline_config.context_noise = 0
        args.pipeline_config.dmd_denoising_steps = [1000, 750, 500, 250]
        args.pipeline_config.warp_denoising_step = sampler == "dmd_warped"
        batch.latents, batch.prompt_embeds = inputs.clone(), prompts
        batch.generator = [torch.Generator().manual_seed(123)]
        stage = (CausalDenoisingStage(model, FlowUniPCMultistepScheduler(shift=3.0)) if sampler == "standard_50"
                 else CausalDMDDenosingStage(model, FlowMatchEulerDiscreteScheduler(shift=8.0)))
        stage.progress_bar = lambda **kwargs: NullProgressBar()
        # Match the generator's no-grad execution: the DMD scheduler replaces
        # its device tables and relies on normal tensor version counters.
        with torch.no_grad():
            result = stage.forward(batch, args)
        assert torch.isfinite(result.latents).all()
        results.append(result.latents)
        random_states.append(batch.generator[0].get_state())
        if enabled:
            assert result.extra["causal_cuda_graph"]["capture_count"] == 3
            assert result.extra["causal_cuda_graph"]["reset_count"] == 0
            assert result.extra["causal_cuda_graph"]["replay_count"] > 0
    torch.testing.assert_close(results[0], results[1], atol=0, rtol=0)
    assert torch.equal(random_states[0], random_states[1])
