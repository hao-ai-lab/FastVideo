# SPDX-License-Identifier: Apache-2.0
import contextlib
from types import SimpleNamespace

import torch
import pytest

from fastvideo.fastvideo_args import FastVideoArgs
from fastvideo.pipelines.pipeline_batch_info import ForwardBatch
from fastvideo.pipelines.stages.denoising_engine import DenoisingEngine
from fastvideo.pipelines.stages.denoising_strategies import (
    BlockDenoisingStrategy)
from fastvideo.pipelines.stages.denoising_standard_strategy import (
    StandardStrategy)
from fastvideo.pipelines.stages.denoising_cosmos_strategy import (
    CosmosStrategy)
from fastvideo.pipelines.stages.denoising_dmd_strategy import DmdStrategy
from fastvideo.pipelines.stages.denoising_longcat_strategy import (
    LongCatStrategy,
    LongCatI2VStrategy,
    LongCatVCStrategy,
)
from fastvideo.pipelines.stages.denoising_causal_strategy import (
    CausalBlockStrategy)
from fastvideo.pipelines.stages.denoising_matrixgame_strategy import (
    MatrixGameBlockStrategy)


class DummyProgressBar:
    def __init__(self):
        self.updates = 0

    def update(self):
        self.updates += 1

    def close(self):
        return None


class DummyScheduler:
    def __init__(self, timesteps: torch.Tensor):
        self.order = 1
        self.timesteps = timesteps
        self.sigmas = self._make_sigmas(timesteps)
        self.num_train_timesteps = int(timesteps.max().item()
                                       ) if timesteps.numel() else 1
        self.init_noise_sigma = 1.0
        self.config = SimpleNamespace(final_sigmas_type=None)

    def _make_sigmas(self, timesteps: torch.Tensor) -> torch.Tensor:
        if timesteps.numel() == 0:
            return timesteps
        max_val = float(timesteps.max().item()) or 1.0
        return timesteps.float() / max_val

    def register_to_config(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self.config, key, value)

    def set_timesteps(self, num_steps: int, device=None):
        self.timesteps = torch.linspace(1.0, 0.0, num_steps, device=device)
        self.sigmas = self._make_sigmas(self.timesteps)
        return self.timesteps

    def scale_model_input(self, latents: torch.Tensor,
                          t: torch.Tensor) -> torch.Tensor:
        return latents

    def step(self, noise_pred: torch.Tensor, t: torch.Tensor,
             latents: torch.Tensor, **kwargs):
        return (latents - noise_pred * 0.1, )

    def add_noise(self, latents: torch.Tensor, noise: torch.Tensor,
                  t: torch.Tensor) -> torch.Tensor:
        return latents + noise * 0.01

    def add_noise_high(self, latents: torch.Tensor, noise: torch.Tensor,
                       t: torch.Tensor, boundary_timestep: torch.Tensor
                       ) -> torch.Tensor:
        return latents + noise * 0.01


class DummyTransformer(torch.nn.Module):
    def __init__(self,
                 hidden_size: int = 8,
                 num_attention_heads: int = 2,
                 patch_size: tuple[int, int, int] = (1, 1, 1),
                 num_frames_per_block: int = 1,
                 sliding_window_num_frames: int = 1,
                 local_attn_size: int = 4) -> None:
        super().__init__()
        self.dummy = torch.nn.Parameter(torch.zeros(1))
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.attention_head_dim = hidden_size // num_attention_heads
        self.blocks = [object(), object()]
        self.config = SimpleNamespace(
            arch_config=SimpleNamespace(
                patch_size=patch_size,
                num_frames_per_block=num_frames_per_block,
                sliding_window_num_frames=sliding_window_num_frames,
            ))
        self.patch_size = patch_size
        self.local_attn_size = local_attn_size
        self.model = SimpleNamespace(local_attn_size=local_attn_size)
        self.independent_first_frame = False

    def forward(self, *args, **kwargs):
        latent = kwargs.get("hidden_states")
        if latent is None and args:
            latent = args[0]
        if latent is None:
            raise ValueError("Missing latent input")
        timestep = kwargs.get("timestep")
        if timestep is None and len(args) > 2:
            timestep = args[2]
        scale = float(torch.as_tensor(timestep).float().mean().item()
                      ) if timestep is not None else 0.0
        out = latent * 0.01 + scale
        if kwargs.get("return_dict") is False:
            return (out, )
        return out


class DummyStandardStage:
    def __init__(self, transformer, scheduler):
        self.transformer = transformer
        self.transformer_2 = None
        self.scheduler = scheduler
        self.vae = None
        self.pipeline = None
        self.attn_backend = object()

    def prepare_extra_func_kwargs(self, func, kwargs):
        extra_step_kwargs = {}
        for key, value in kwargs.items():
            if key in set(func.__code__.co_varnames):
                extra_step_kwargs[key] = value
        return extra_step_kwargs

    def progress_bar(self, iterable=None, total=None):
        return DummyProgressBar()

    def rescale_noise_cfg(self, noise_cfg, noise_pred_text,
                          guidance_rescale=0.0) -> torch.Tensor:
        std_text = noise_pred_text.std(dim=list(range(1, noise_pred_text.ndim)),
                                       keepdim=True)
        std_cfg = noise_cfg.std(dim=list(range(1, noise_cfg.ndim)),
                                keepdim=True)
        noise_pred_rescaled = noise_cfg * (std_text / std_cfg)
        return (guidance_rescale * noise_pred_rescaled +
                (1 - guidance_rescale) * noise_cfg)

    def prepare_sta_param(self, batch, fastvideo_args):
        return None

    def save_sta_search_results(self, batch):
        return None


class DummyLongCatStage:
    def __init__(self, transformer, scheduler):
        self.transformer = transformer
        self.scheduler = scheduler

    def optimized_scale(self, positive_flat, negative_flat) -> torch.Tensor:
        dot_product = torch.sum(positive_flat * negative_flat,
                                dim=1,
                                keepdim=True)
        squared_norm = torch.sum(negative_flat**2, dim=1, keepdim=True) + 1e-8
        return dot_product / squared_norm


class DummyDmdStage(DummyStandardStage):
    @property
    def device(self) -> torch.device:
        return torch.device("cpu")


class DummyCausalStage:
    def __init__(self, transformer, scheduler):
        self.transformer = transformer
        self.transformer_2 = None
        self.scheduler = scheduler
        self.vae = None
        self.num_transformer_blocks = len(transformer.blocks)
        self.num_frames_per_block = transformer.config.arch_config.num_frames_per_block
        self.sliding_window_num_frames = (
            transformer.config.arch_config.sliding_window_num_frames)
        self.local_attn_size = transformer.model.local_attn_size
        self.attn_backend = object()
        self.frame_seq_length = 1

    @property
    def device(self) -> torch.device:
        return next(self.transformer.parameters()).device

    def prepare_extra_func_kwargs(self, func, kwargs):
        extra_step_kwargs = {}
        for key, value in kwargs.items():
            if key in set(func.__code__.co_varnames):
                extra_step_kwargs[key] = value
        return extra_step_kwargs

    def prepare_sta_param(self, batch, fastvideo_args):
        return None

    def progress_bar(self, iterable=None, total=None):
        return DummyProgressBar()

    def _initialize_kv_cache(self, batch_size, dtype, device):
        return [{} for _ in range(self.num_transformer_blocks)]

    def _initialize_crossattn_cache(self, batch_size, max_text_len, dtype,
                                    device):
        return [{} for _ in range(self.num_transformer_blocks)]


class DummyMatrixGameStage:
    def __init__(self, transformer, scheduler):
        self.transformer = transformer
        self.transformer_2 = None
        self.scheduler = scheduler
        self.vae = None
        self.num_transformer_blocks = len(transformer.blocks)
        self.num_frame_per_block = transformer.config.arch_config.num_frames_per_block
        self.sliding_window_num_frames = (
            transformer.config.arch_config.sliding_window_num_frames)
        self.local_attn_size = transformer.local_attn_size
        self.use_action_module = False
        self.attn_backend = object()
        self.frame_seq_length = 1

    @property
    def device(self) -> torch.device:
        return torch.device("cpu")

    def progress_bar(self, iterable=None, total=None):
        return DummyProgressBar()

    def prepare_sta_param(self, batch, fastvideo_args):
        return None

    def _initialize_kv_cache(self, batch_size, dtype, device):
        return [{} for _ in range(self.num_transformer_blocks)]

    def _initialize_action_kv_cache(self, batch_size, dtype, device):
        return None, None

    def _initialize_crossattn_cache(self, batch_size, max_text_len, dtype,
                                    device):
        return [{} for _ in range(self.num_transformer_blocks)]

    def _prepare_action_kwargs(self, batch: ForwardBatch, start_index: int,
                               num_frames: int) -> dict:
        return {}

    def _process_single_block(self, current_latents: torch.Tensor,
                              batch: ForwardBatch, start_index: int,
                              current_num_frames: int, timesteps: torch.Tensor,
                              ctx, action_kwargs, noise_generator=None,
                              progress_bar=None):
        latents = current_latents
        for t_cur in timesteps:
            noise_pred = self.transformer(latents, batch.prompt_embeds, t_cur)
            latents = self.scheduler.step(noise_pred,
                                          t_cur,
                                          latents,
                                          return_dict=False)[0]
            if progress_bar is not None:
                progress_bar.update()
        return latents

    def _update_context_cache(self, current_latents: torch.Tensor,
                              batch: ForwardBatch, start_index: int,
                              current_num_frames: int, ctx, action_kwargs,
                              context_noise: float):
        return None


@pytest.fixture(autouse=True)
def _patch_autocast(monkeypatch, request):
    if request.node.get_closest_marker("gpu"):
        return

    @contextlib.contextmanager
    def _autocast(*args, **kwargs):
        yield

    monkeypatch.setattr(torch, "autocast", _autocast)


@pytest.fixture(autouse=True)
def _patch_local_device(monkeypatch, request):
    if request.node.get_closest_marker("gpu"):
        return

    def _cpu_device():
        return torch.device("cpu")

    import fastvideo.distributed as dist
    import fastvideo.pipelines.stages.denoising_standard_strategy as standard
    import fastvideo.pipelines.stages.denoising_dmd_strategy as dmd
    import fastvideo.pipelines.stages.denoising_causal_strategy as causal
    import fastvideo.pipelines.stages.denoising_matrixgame_strategy as matrixgame

    monkeypatch.setattr(dist, "get_local_torch_device", _cpu_device)
    monkeypatch.setattr(standard, "get_local_torch_device", _cpu_device)
    monkeypatch.setattr(dmd, "get_local_torch_device", _cpu_device)
    monkeypatch.setattr(causal, "get_local_torch_device", _cpu_device)
    monkeypatch.setattr(matrixgame, "get_local_torch_device", _cpu_device)


def _make_args() -> FastVideoArgs:
    args = FastVideoArgs(model_path="dummy")
    args.pipeline_config.dit_config.arch_config.patch_size = (1, 1, 1)
    args.pipeline_config.dit_config.boundary_ratio = None
    args.pipeline_config.ti2v_task = False
    args.pipeline_config.embedded_cfg_scale = None
    return args


def _make_batch(latents: torch.Tensor,
                timesteps: torch.Tensor,
                guidance_scale: float,
                seed: int,
                device: torch.device | str | None = None,
                generator_device: torch.device | str = "cpu") -> ForwardBatch:
    if device is None:
        device = latents.device
    embed_generator = torch.Generator(device=device).manual_seed(seed)
    batch_generator = torch.Generator(device=generator_device).manual_seed(seed)
    batch = ForwardBatch(data_type="test")
    batch.latents = latents.clone()
    batch.timesteps = timesteps.clone()
    batch.num_inference_steps = len(timesteps)
    batch.num_frames = latents.shape[2]
    batch.height = latents.shape[-2]
    batch.width = latents.shape[-1]
    batch.raw_latent_shape = latents.shape
    batch.prompt_embeds = [
        torch.randn(1, 4, 8, generator=embed_generator, device=device)
    ]
    batch.negative_prompt_embeds = [
        torch.randn(1, 4, 8, generator=embed_generator, device=device)
    ]
    batch.prompt_attention_mask = [
        torch.ones(1, 4, dtype=torch.long, device=device)
    ]
    batch.negative_attention_mask = [
        torch.ones(1, 4, dtype=torch.long, device=device)
    ]
    batch.guidance_scale = guidance_scale
    batch.guidance_scale_2 = guidance_scale
    batch.do_classifier_free_guidance = guidance_scale > 1.0
    batch.guidance_rescale = 0.0
    batch.generator = batch_generator
    batch.image_embeds = []
    batch.clip_embedding_pos = [
        torch.randn(1, 4, 8, generator=embed_generator, device=device)
    ]
    batch.clip_embedding_neg = [
        torch.randn(1, 4, 8, generator=embed_generator, device=device)
    ]
    return batch


def _run_manual(strategy, batch, args):
    state = strategy.prepare(batch, args)
    if isinstance(strategy, BlockDenoisingStrategy):
        block_plan = strategy.block_plan(state)
        for block_idx, block_item in enumerate(block_plan.items):
            block_ctx = strategy.init_block_context(state, block_item, block_idx)
            strategy.process_block(state, block_ctx, block_item)
            strategy.update_context(state, block_ctx, block_item)
    else:
        for i, t in enumerate(state.timesteps):
            model_inputs = strategy.make_model_inputs(state, t, i)
            noise_pred = strategy.forward(state, model_inputs)
            noise_pred = strategy.cfg_combine(state, noise_pred)
            state.latents = strategy.scheduler_step(state, noise_pred, t)
    return strategy.postprocess(state)


def _cuda_bf16_available() -> bool:
    if not torch.cuda.is_available():
        return False
    is_bf16_supported = getattr(torch.cuda, "is_bf16_supported", None)
    if is_bf16_supported is None:
        return False
    return torch.cuda.is_bf16_supported()


def test_standard_strategy_parity_cpu():
    timesteps = torch.tensor([2, 1], dtype=torch.long)
    latents = torch.randn(1, 2, 2, 2, 2)
    args = _make_args()
    scheduler = DummyScheduler(timesteps)
    stage_a = DummyStandardStage(DummyTransformer(), scheduler)
    stage_b = DummyStandardStage(DummyTransformer(), DummyScheduler(timesteps))

    batch_a = _make_batch(latents, timesteps, guidance_scale=2.0, seed=0)
    batch_b = _make_batch(latents, timesteps, guidance_scale=2.0, seed=0)

    engine_out = DenoisingEngine(StandardStrategy(stage_a)).run(batch_a, args)
    manual_out = _run_manual(StandardStrategy(stage_b), batch_b, args)

    assert torch.allclose(engine_out.latents, manual_out.latents)


def test_cosmos_strategy_parity_cpu():
    timesteps = torch.tensor([2, 1], dtype=torch.long)
    latents = torch.randn(1, 2, 2, 2, 2)
    args = _make_args()
    scheduler = DummyScheduler(timesteps)
    stage_a = DummyStandardStage(DummyTransformer(), scheduler)
    stage_b = DummyStandardStage(DummyTransformer(), DummyScheduler(timesteps))

    batch_a = _make_batch(latents, timesteps, guidance_scale=2.0, seed=1)
    batch_b = _make_batch(latents, timesteps, guidance_scale=2.0, seed=1)

    engine_out = DenoisingEngine(CosmosStrategy(stage_a)).run(batch_a, args)
    manual_out = _run_manual(CosmosStrategy(stage_b), batch_b, args)

    assert torch.allclose(engine_out.latents, manual_out.latents)


def test_dmd_strategy_parity_cpu():
    timesteps = torch.tensor([4, 2], dtype=torch.long)
    latents = torch.randn(1, 2, 2, 2, 2)
    args = _make_args()
    args.pipeline_config.dmd_denoising_steps = [4, 2]
    scheduler = DummyScheduler(timesteps)
    scheduler.timesteps = timesteps
    scheduler.sigmas = scheduler._make_sigmas(timesteps)
    stage_a = DummyDmdStage(DummyTransformer(), scheduler)
    stage_b = DummyDmdStage(DummyTransformer(),
                            DummyScheduler(timesteps))

    batch_a = _make_batch(latents, timesteps, guidance_scale=1.0, seed=2)
    batch_b = _make_batch(latents, timesteps, guidance_scale=1.0, seed=2)

    engine_out = DenoisingEngine(DmdStrategy(stage_a)).run(batch_a, args)
    manual_out = _run_manual(DmdStrategy(stage_b), batch_b, args)

    assert torch.allclose(engine_out.latents, manual_out.latents)


def test_longcat_strategy_parity_cpu():
    timesteps = torch.tensor([2, 1], dtype=torch.long)
    latents = torch.randn(1, 2, 2, 2, 2)
    args = _make_args()
    scheduler = DummyScheduler(timesteps)
    stage_a = DummyLongCatStage(DummyTransformer(), scheduler)
    stage_b = DummyLongCatStage(DummyTransformer(), DummyScheduler(timesteps))

    batch_a = _make_batch(latents, timesteps, guidance_scale=2.0, seed=3)
    batch_b = _make_batch(latents, timesteps, guidance_scale=2.0, seed=3)

    engine_out = DenoisingEngine(LongCatStrategy(stage_a)).run(batch_a, args)
    manual_out = _run_manual(LongCatStrategy(stage_b), batch_b, args)

    assert torch.allclose(engine_out.latents, manual_out.latents)


def test_longcat_i2v_strategy_parity_cpu():
    timesteps = torch.tensor([2, 1], dtype=torch.long)
    latents = torch.randn(1, 2, 3, 2, 2)
    args = _make_args()
    scheduler = DummyScheduler(timesteps)
    stage_a = DummyLongCatStage(DummyTransformer(), scheduler)
    stage_b = DummyLongCatStage(DummyTransformer(), DummyScheduler(timesteps))

    batch_a = _make_batch(latents, timesteps, guidance_scale=2.0, seed=4)
    batch_b = _make_batch(latents, timesteps, guidance_scale=2.0, seed=4)
    batch_a.num_cond_latents = 1
    batch_b.num_cond_latents = 1

    engine_out = DenoisingEngine(LongCatI2VStrategy(stage_a)).run(batch_a,
                                                                  args)
    manual_out = _run_manual(LongCatI2VStrategy(stage_b), batch_b, args)

    assert torch.allclose(engine_out.latents, manual_out.latents)


def test_longcat_vc_strategy_parity_cpu():
    timesteps = torch.tensor([2, 1], dtype=torch.long)
    latents = torch.randn(1, 2, 3, 2, 2)
    args = _make_args()
    scheduler = DummyScheduler(timesteps)
    stage_a = DummyLongCatStage(DummyTransformer(), scheduler)
    stage_b = DummyLongCatStage(DummyTransformer(), DummyScheduler(timesteps))

    batch_a = _make_batch(latents, timesteps, guidance_scale=2.0, seed=5)
    batch_b = _make_batch(latents, timesteps, guidance_scale=2.0, seed=5)
    batch_a.num_cond_latents = 1
    batch_b.num_cond_latents = 1
    batch_a.use_kv_cache = True
    batch_b.use_kv_cache = True
    batch_a.kv_cache_dict = {}
    batch_b.kv_cache_dict = {}
    batch_a.cond_latents = torch.randn(1, 2, 1, 2, 2)
    batch_b.cond_latents = batch_a.cond_latents.clone()

    engine_out = DenoisingEngine(LongCatVCStrategy(stage_a)).run(batch_a,
                                                                 args)
    manual_out = _run_manual(LongCatVCStrategy(stage_b), batch_b, args)

    assert torch.allclose(engine_out.latents, manual_out.latents)


def test_causal_block_strategy_parity_cpu():
    timesteps = torch.tensor([4, 2], dtype=torch.long)
    latents = torch.randn(1, 2, 2, 2, 2)
    args = _make_args()
    args.pipeline_config.dmd_denoising_steps = [4, 2]
    scheduler = DummyScheduler(timesteps)
    scheduler.timesteps = timesteps
    scheduler.sigmas = scheduler._make_sigmas(timesteps)
    stage_a = DummyCausalStage(DummyTransformer(num_frames_per_block=1),
                               scheduler)
    stage_b = DummyCausalStage(DummyTransformer(num_frames_per_block=1),
                               DummyScheduler(timesteps))

    batch_a = _make_batch(latents, timesteps, guidance_scale=1.0, seed=6)
    batch_b = _make_batch(latents, timesteps, guidance_scale=1.0, seed=6)

    engine_out = DenoisingEngine(CausalBlockStrategy(stage_a)).run(batch_a,
                                                                   args)
    manual_out = _run_manual(CausalBlockStrategy(stage_b), batch_b, args)

    assert torch.allclose(engine_out.latents, manual_out.latents)


def test_matrixgame_block_strategy_parity_cpu():
    timesteps = torch.tensor([3, 1], dtype=torch.long)
    latents = torch.randn(1, 2, 2, 2, 2)
    args = _make_args()
    args.pipeline_config.dmd_denoising_steps = [3, 1]
    scheduler = DummyScheduler(timesteps)
    scheduler.timesteps = timesteps
    scheduler.sigmas = scheduler._make_sigmas(timesteps)
    stage_a = DummyMatrixGameStage(DummyTransformer(num_frames_per_block=1),
                                   scheduler)
    stage_b = DummyMatrixGameStage(DummyTransformer(num_frames_per_block=1),
                                   DummyScheduler(timesteps))

    batch_a = _make_batch(latents, timesteps, guidance_scale=1.0, seed=7)
    batch_b = _make_batch(latents, timesteps, guidance_scale=1.0, seed=7)

    engine_out = DenoisingEngine(MatrixGameBlockStrategy(stage_a)).run(
        batch_a, args)
    manual_out = _run_manual(MatrixGameBlockStrategy(stage_b), batch_b, args)

    assert torch.allclose(engine_out.latents, manual_out.latents)


def test_matrixgame_block_strategy_streaming_parity_cpu():
    timesteps = torch.tensor([3, 1], dtype=torch.long)
    latents = torch.randn(1, 2, 2, 2, 2)
    args = _make_args()
    args.pipeline_config.dmd_denoising_steps = [3, 1]
    scheduler = DummyScheduler(timesteps)
    scheduler.timesteps = timesteps
    scheduler.sigmas = scheduler._make_sigmas(timesteps)

    stage_full = DummyMatrixGameStage(DummyTransformer(num_frames_per_block=1),
                                      scheduler)
    stage_stream = DummyMatrixGameStage(
        DummyTransformer(num_frames_per_block=1), DummyScheduler(timesteps))

    batch_full = _make_batch(latents, timesteps, guidance_scale=1.0, seed=10)
    batch_stream = _make_batch(latents, timesteps, guidance_scale=1.0, seed=10)
    batch_full.generator = None
    batch_stream.generator = None

    torch.manual_seed(1234)
    noise_shape = (latents.shape[0], stage_full.num_frame_per_block,
                   latents.shape[1], latents.shape[3], latents.shape[4])
    noise_pool = [
        torch.randn(noise_shape, dtype=latents.dtype)
        for _ in range(max(len(timesteps) - 1, 0))
    ]

    strategy_full = MatrixGameBlockStrategy(stage_full)
    state_full = strategy_full.prepare(batch_full, args)
    state_full.extra["ctx"].noise_pool = [t.clone() for t in noise_pool]
    block_plan_full = strategy_full.block_plan(state_full)

    strategy_stream = MatrixGameBlockStrategy(stage_stream)
    state_stream = strategy_stream.prepare(batch_stream, args)
    state_stream.extra["ctx"].noise_pool = [t.clone() for t in noise_pool]
    block_plan_stream = strategy_stream.block_plan(state_stream)

    engine_full = DenoisingEngine(strategy_full)
    engine_full.run_blocks(state_full, block_plan=block_plan_full)

    engine_stream = DenoisingEngine(strategy_stream)
    for block_idx in range(len(block_plan_stream.items)):
        engine_stream.run_blocks(
            state_stream,
            block_plan=block_plan_stream,
            start_block=block_idx,
            num_blocks=1,
        )

    batch_full = strategy_full.postprocess(state_full)
    batch_stream = strategy_stream.postprocess(state_stream)

    assert torch.allclose(batch_full.latents, batch_stream.latents)


@pytest.mark.gpu
@pytest.mark.skipif(not _cuda_bf16_available(),
                    reason="CUDA BF16 not available")
def test_standard_strategy_parity_cuda():
    device = torch.device("cuda")
    timesteps = torch.tensor([2, 1], dtype=torch.long, device=device)
    latents = torch.randn(1, 2, 2, 2, 2, device=device)
    args = _make_args()
    scheduler = DummyScheduler(timesteps)
    stage_a = DummyStandardStage(DummyTransformer().to(device), scheduler)
    stage_b = DummyStandardStage(DummyTransformer().to(device),
                                 DummyScheduler(timesteps))

    batch_a = _make_batch(latents, timesteps, guidance_scale=2.0, seed=8)
    batch_b = _make_batch(latents, timesteps, guidance_scale=2.0, seed=8)

    engine_out = DenoisingEngine(StandardStrategy(stage_a)).run(batch_a, args)
    manual_out = _run_manual(StandardStrategy(stage_b), batch_b, args)

    assert torch.allclose(engine_out.latents, manual_out.latents)


@pytest.mark.gpu
@pytest.mark.skipif(not _cuda_bf16_available(),
                    reason="CUDA BF16 not available")
def test_causal_block_strategy_parity_cuda():
    device = torch.device("cuda")
    timesteps = torch.tensor([4, 2], dtype=torch.long, device=device)
    latents = torch.randn(1, 2, 2, 2, 2, device=device)
    args = _make_args()
    args.pipeline_config.dmd_denoising_steps = [4, 2]
    scheduler = DummyScheduler(timesteps)
    scheduler.timesteps = timesteps
    scheduler.sigmas = scheduler._make_sigmas(timesteps)
    stage_a = DummyCausalStage(
        DummyTransformer(num_frames_per_block=1).to(device), scheduler)
    stage_b = DummyCausalStage(
        DummyTransformer(num_frames_per_block=1).to(device),
        DummyScheduler(timesteps))

    batch_a = _make_batch(latents, timesteps, guidance_scale=1.0, seed=9)
    batch_b = _make_batch(latents, timesteps, guidance_scale=1.0, seed=9)

    engine_out = DenoisingEngine(CausalBlockStrategy(stage_a)).run(batch_a,
                                                                   args)
    manual_out = _run_manual(CausalBlockStrategy(stage_b), batch_b, args)

    assert torch.allclose(engine_out.latents, manual_out.latents)
