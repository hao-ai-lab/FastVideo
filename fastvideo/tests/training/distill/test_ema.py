import types

import pytest  # type: ignore
import torch  # type: ignore
import torch.nn as nn  # type: ignore

from fastvideo.training.training_utils import EMA_FSDP


class TinyNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.lin = nn.Linear(4, 3, bias=True)
        # Non-trainable param should be ignored by EMA
        self.register_parameter("frozen", nn.Parameter(torch.randn(2), requires_grad=False))
        # Buffer should be ignored by EMA
        self.register_buffer("buf", torch.ones(1))


def named_params_to_cpu_float_dict(module: nn.Module):
    return {n: p.detach().clone().float().cpu() for n, p in module.named_parameters() if p.requires_grad}


def test_ema_init_local_shard_copies_params_cpu_float():
    torch.manual_seed(0)
    model = TinyNet()
    ema = EMA_FSDP(model, decay=0.9, mode="local_shard")

    expected = named_params_to_cpu_float_dict(model)
    assert set(ema.shadow.keys()) == set(expected.keys())
    for n, v in expected.items():
        assert torch.equal(ema.shadow[n], v)


def test_ema_update_local_shard_matches_formula():
    torch.manual_seed(0)
    model = TinyNet()
    decay = 0.9
    ema = EMA_FSDP(model, decay=decay, mode="local_shard")

    # Save initial snapshot
    start = {n: p.detach().clone().float().cpu() for n, p in model.named_parameters() if p.requires_grad}

    # Mutate model parameters to new values
    with torch.no_grad():
        for _, p in model.named_parameters():
            if p.requires_grad:
                p.add_(torch.randn_like(p))

    after = {n: p.detach().clone().float().cpu() for n, p in model.named_parameters() if p.requires_grad}

    ema.update(model)

    for n in start.keys():
        expected = start[n] * decay + after[n] * (1.0 - decay)
        assert torch.allclose(ema.shadow[n], expected, rtol=1e-6, atol=1e-8)


def test_apply_to_model_swaps_and_restores():
    torch.manual_seed(0)
    model = TinyNet()
    ema = EMA_FSDP(model, decay=0.0, mode="local_shard")  # decay 0 -> shadow becomes last param on update

    # Change model and update EMA so shadow != current
    with torch.no_grad():
        for _, p in model.named_parameters():
            if p.requires_grad:
                p.add_(1.2345)
    ema.update(model)

    # Change model again so current != shadow
    with torch.no_grad():
        for _, p in model.named_parameters():
            if p.requires_grad:
                p.add_(2.0)

    # Snapshot of values at context entry
    entry_values = {n: p.detach().clone() for n, p in model.named_parameters() if p.requires_grad}

    # Inside context, params should equal EMA shadow; outside, restored
    with ema.apply_to_model(model):
        for n, p in model.named_parameters():
            if p.requires_grad:
                assert torch.allclose(p.detach().cpu().float(), ema.shadow[n])

    for n, p in model.named_parameters():
        if p.requires_grad:
            assert torch.equal(p.detach(), entry_values[n])


def test_state_dict_roundtrip():
    torch.manual_seed(0)
    model = TinyNet()
    ema = EMA_FSDP(model, decay=0.8, mode="local_shard")

    with torch.no_grad():
        for _, p in model.named_parameters():
            if p.requires_grad:
                p.add_(torch.randn_like(p))
    ema.update(model)

    sd = ema.state_dict()
    ema2 = EMA_FSDP(model, decay=0.8, mode="local_shard")
    ema2.load_state_dict(sd)

    for k in sd.keys():
        assert torch.equal(ema2.shadow[k], sd[k])


def test_copy_to_unwrapped_and_rank0_full_guard():
    torch.manual_seed(0)
    src = TinyNet()
    ema = EMA_FSDP(src, decay=0.7, mode="local_shard")

    # Make EMA shadow distinct from a fresh target
    with torch.no_grad():
        for _, p in src.named_parameters():
            if p.requires_grad:
                p.mul_(3.14)
    ema.update(src)

    tgt = TinyNet()
    # copy in local_shard mode always applies
    ema.copy_to_unwrapped(tgt)
    for n, p in tgt.named_parameters():
        if p.requires_grad:
            assert torch.allclose(p.detach().cpu().float(), ema.shadow[n])

    # If mode is rank0_full but rank != 0, copy is a no-op
    ema.mode = "rank0_full"
    ema.rank = 1
    before = {n: p.detach().clone() for n, p in tgt.named_parameters()}
    ema.copy_to_unwrapped(tgt)
    for n, p in tgt.named_parameters():
        assert torch.equal(p.detach(), before[n])


def test_rank0_full_init_and_update_with_stubbed_gather(monkeypatch):
    torch.manual_seed(0)
    model = TinyNet()

    # Stub gather_state_dict_on_cpu_rank0 to avoid requiring initialized dist
    def fake_gather(mod, device=None):
        return {n: p.detach().clone() for n, p in mod.named_parameters() if p.requires_grad}

    # Patch in the module where EMA_FSDP is defined
    import fastvideo.training.training_utils as tu
    monkeypatch.setattr(tu, "gather_state_dict_on_cpu_rank0", fake_gather, raising=True)

    ema = EMA_FSDP(model, decay=0.5, mode="rank0_full")
    assert len(ema.shadow) > 0

    start = {n: t.clone().float().cpu() for n, t in ema.shadow.items()}

    with torch.no_grad():
        for _, p in model.named_parameters():
            if p.requires_grad:
                p.add_(1.0)

    ema.update(model)

    cpu_state = {n: p.detach().clone().float().cpu() for n, p in model.named_parameters() if p.requires_grad}
    for n in start.keys():
        expected = start[n] * 0.5 + cpu_state[n] * 0.5
        assert torch.allclose(ema.shadow[n], expected)

    # state_dict should be empty if rank != 0 in rank0_full mode
    ema.rank = 1
    assert ema.state_dict() == {}


def test_apply_to_model_raises_when_rank0_full():
    model = TinyNet()
    ema = EMA_FSDP(model, decay=0.9, mode="rank0_full")
    with pytest.raises(RuntimeError):
        with ema.apply_to_model(model):
            pass


def test_copy_to_unwrapped_rank0_full_rank0(monkeypatch):
    torch.manual_seed(0)
    model = TinyNet()

    # Stub gather to produce keys matching named_parameters
    def fake_gather(mod, device=None):
        return {n: p.detach().clone() for n, p in mod.named_parameters() if p.requires_grad}

    import fastvideo.training.training_utils as tu
    monkeypatch.setattr(tu, "gather_state_dict_on_cpu_rank0", fake_gather, raising=True)

    ema = EMA_FSDP(model, decay=0.0, mode="rank0_full")
    ema.rank = 0

    # Modify model and update so EMA has distinct weights
    with torch.no_grad():
        for _, p in model.named_parameters():
            if p.requires_grad:
                p.add_(1.0)
    ema.update(model)

    tgt = TinyNet()
    ema.copy_to_unwrapped(tgt)
    for n, p in tgt.named_parameters():
        if p.requires_grad:
            assert torch.allclose(p.detach().cpu().float(), ema.shadow[n])


