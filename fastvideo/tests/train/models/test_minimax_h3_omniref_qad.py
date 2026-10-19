# SPDX-License-Identifier: Apache-2.0
"""OmniRef PDD QAD: student initialization, deployed NVFP4 numerics, rows and metrics.

CPU tests: a disabled QAD student is bit-identical to the untouched model,
static scales come from the calibration table, the row plan is the latent
generator's plan, and the eval statistics are consistent. GPU tests (GB200 or
any FlashInfer FP4 GPU): the QAD linear forward is bit-identical to the
deployed ``NVFP4QuantizeMethod`` on a converter-packed weight, and the STE
backward is the dense gradient.
"""
from __future__ import annotations

import copy
import math

import pytest
import torch

from fastvideo.configs.models.dits.minimax_h3 import MiniMaxH3ArchConfig, MiniMaxH3Config
from fastvideo.forward_context import set_forward_context
from fastvideo.layers.quantization.nvfp4_qad import (NVFP4QADLinearMethod, NVFP4QADPlan, install_nvfp4_qad,
                                                     nvfp4_qad_export_scales, set_nvfp4_qad_enabled)
from fastvideo.models.dits import minimax_h3
from fastvideo.pipelines.basic.minimax_h3.packing import MINIMAX_H3_TEXT_TAG, build_packed_sequence, build_row_timesteps
from fastvideo.pipelines.pipeline_batch_info import ForwardBatch
from fastvideo.tests.loader.test_minimax_h3_pdd_loading import TINY_ARCH, VIDEO_PATCH_DIM, cpu_loader  # noqa: F401
from fastvideo.tests.loader.test_minimax_h3_pdd_loading import single_process_group  # noqa: F401
from fastvideo.train.methods.knowledge_distillation.pdd_qad_metrics import RungStats, summarize
from fastvideo.train.models.minimax_h3.omniref_data import RowSpec, RowStream, select_rows

needs_fp4 = pytest.mark.skipif(not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] < 10,
                               reason="NVFP4 GEMMs need a Blackwell GPU with FlashInfer")


def _tiny_model() -> torch.nn.Module:
    arch = MiniMaxH3ArchConfig(**{**TINY_ARCH, "patch_size": tuple(TINY_ARCH["patch_size"])})
    model = minimax_h3.MiniMaxH3Transformer3DModel(MiniMaxH3Config(arch_config=arch), hf_config={})
    generator = torch.Generator().manual_seed(0)
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.copy_(torch.randn(parameter.shape, generator=generator) * 0.1)
    return model.eval()


def _amax_table(num_blocks: int = 2) -> dict[str, float]:
    return {f"b{b}.ff.{sub}": 3.0 + b for b in range(num_blocks) for sub in ("fc_in", "fc_out")}


def _forward(model: torch.nn.Module) -> tuple[torch.Tensor, torch.Tensor]:
    layout = build_packed_sequence(torch.full((3, ), MINIMAX_H3_TEXT_TAG, dtype=torch.long), 2, 4, 4, 2, (1, 2, 2))
    unique, inverse = build_row_timesteps(layout, 0.3, 0.4, 0.999, 1.0)
    generator = torch.Generator().manual_seed(1)
    inputs = dict(hidden_states=torch.randn(1, int(layout.video_indices.numel()), VIDEO_PATCH_DIM, generator=generator),
                  audio_hidden_states=torch.randn(1, int(layout.audio_indices.numel()), TINY_ARCH["audio_in_channels"],
                                                  generator=generator),
                  encoder_hidden_states=torch.randn(1, 3, TINY_ARCH["text_dim"], generator=generator),
                  timestep=unique, timestep_indices=inverse, token_tags=layout.token_tags,
                  position_ids=layout.position_ids, video_indices=layout.video_indices,
                  audio_indices=layout.audio_indices, text_indices=layout.text_indices)
    with torch.no_grad(), set_forward_context(current_timestep=0, attn_metadata=None,
                                              forward_batch=ForwardBatch(data_type="dummy")):
        return model(**inputs)


def test_disabled_qad_student_is_bit_identical_to_teacher(cpu_loader, single_process_group):  # noqa: F811
    teacher = _tiny_model()
    student = copy.deepcopy(teacher)
    methods = install_nvfp4_qad(student, _amax_table(), NVFP4QADPlan())
    # Two blocks x (q, k, v, out, fc_in, fc_out); SDPA builds no VSA gates.
    assert len(methods) == 12
    assert all(isinstance(m, NVFP4QADLinearMethod) for m in methods.values())
    set_nvfp4_qad_enabled(methods, False)
    for expected, got in zip(_forward(teacher), _forward(student), strict=True):
        assert torch.equal(expected, got)
    assert all(torch.equal(a, b) for a, b in zip(teacher.state_dict().values(), student.state_dict().values()))


def test_static_scales_require_calibration_and_round_trip(cpu_loader, single_process_group):  # noqa: F811
    model = _tiny_model()
    table = _amax_table()
    with pytest.raises(KeyError, match="calibrated input amax"):
        install_nvfp4_qad(copy.deepcopy(model), {k: v for k, v in table.items() if k != "b1.ff.fc_out"},
                          NVFP4QADPlan())
    methods = install_nvfp4_qad(model, table, NVFP4QADPlan())
    assert nvfp4_qad_export_scales(methods) == table
    unit = [name for name, m in methods.items() if m.act_scale == "unit"]
    assert len(unit) == 8 and all(".attn." in name for name in unit)
    assert float(methods["transformer_blocks.1.ff.fc_out"].input_global_sf(torch.device("cpu"))) == pytest.approx(
        2688.0 / 4.0)
    # Re-installing wraps the original method, never a QAD method.
    again = install_nvfp4_qad(model, table, NVFP4QADPlan(quantize_attention=False))
    assert len(again) == 4 and not isinstance(again["transformer_blocks.0.ff.fc_in"].original, NVFP4QADLinearMethod)


def _reference_plan(groups: dict[tuple[str, str], list[dict]], seed: int, per_group: int) -> list[dict]:
    """The latent generator's selection: per-group seeded shuffle, then interleave."""
    import random
    picked = []
    for key in sorted(groups):
        entries = sorted(groups[key], key=lambda entry: entry["id"])
        random.Random(f"{seed}:{key[0]}:{key[1]}").shuffle(entries)
        picked.append(entries[:per_group])
    plan = []
    for index in range(max(len(entries) for entries in picked)):
        for entries in picked:
            if index < len(entries):
                plan.append(entries[index])
    return plan


def test_qad_student_can_be_deep_copied_and_pickled(cpu_loader, single_process_group):  # noqa: F811
    """EMA copies and teacher snapshots deep-copy the student after install_nvfp4_qad."""
    import pickle
    student = _tiny_model()
    install_nvfp4_qad(student, _amax_table(), NVFP4QADPlan())
    clone = copy.deepcopy(student)
    method = clone.transformer_blocks[0].ff.fc_in.quant_method
    assert isinstance(method, NVFP4QADLinearMethod) and method.input_amax == 3.0
    assert method.original is not student.transformer_blocks[0].ff.fc_in.quant_method.original
    restored = pickle.loads(pickle.dumps(student.transformer_blocks[0].ff.fc_out.quant_method))
    assert restored.prefix.endswith("transformer_blocks.0.ff.fc_out") and restored.input_amax == 3.0
    with pytest.raises(AttributeError):
        object.__new__(NVFP4QADLinearMethod).anything


def test_row_plan_is_the_latent_generators_plan():
    groups = {(case, res): [{"id": f"{case}:src{i}", "case": case, "resolution": res, "parquet": f"/p/{case}{i}"}
                            for i in range(12)] for case in ("first_frame", "storyboard") for res in ("480p", "768p")}
    ours = select_rows(groups, 20261007, {"480p": 5, "768p": 5})
    theirs = _reference_plan(groups, 20261007, 5)
    assert [(r.id, r.parquet) for r in ours] == [(c["id"], c["parquet"]) for c in theirs]
    excluded = select_rows(groups, 20261007, {"480p": 12}, frozenset({"src0", "src1"}))
    assert not any(r.source in {"src0", "src1"} for r in excluded) and all(r.resolution == "480p" for r in excluded)


def test_row_stream_feeds_each_data_parallel_group_distinct_rows():
    plan = [RowSpec(f"r{i}", f"s{i}", "first_frame", "480p", f"/p{i}", i) for i in range(10)]
    streams = [iter(RowStream(plan, dp_rank=r, dp_size=2, seed=3)) for r in range(2)]
    firsts = [[next(s)["row"].id for _ in range(5)] for s in streams]
    assert set(firsts[0]).isdisjoint(firsts[1]) and len(set(firsts[0] + firsts[1])) == 10
    replay = iter(RowStream(plan, dp_rank=0, dp_size=2, seed=3))
    assert [next(replay)["row"].id for _ in range(5)] == firsts[0]


def test_rung_stats_summaries():
    stats = RungStats(2)
    teacher = torch.ones(4, 3)
    for rung in range(2):
        pair = {m: {"student": teacher * (1.1 if m == "video" else 1.0), "teacher": teacher,
                    "student_v": teacher, "teacher_v": teacher} for m in ("video", "audio")}
        stats.add_rung(rung, pair)
    summary = summarize(stats.reduce(contribute=True), 2, {"video": 1.0, "audio": 1.0})
    assert summary["x0_rel_l2/video/rung1"] == pytest.approx(0.1)
    assert summary["x0_rel_l2/audio/rung0"] == 0.0
    assert summary["score"] == pytest.approx(0.05)


# ----------------------------------------------------------------------------- GPU: deployed numerics
def _runtime_layer(weight: torch.Tensor, input_amax: float | None):
    """A ``ReplicatedLinear`` packed the way the H3 NVFP4 converter packs a dense weight."""
    from fastvideo.layers.linear import ReplicatedLinear
    from fastvideo.layers.quantization.nvfp4_config import (H3_NVFP4_DIT_INPUT_SF_NAME, NVFP4QuantizeMethod,
                                                            _nvfp4_quantize, _require_flashinfer)
    sf_layout, _, _ = _require_flashinfer()
    layer = ReplicatedLinear(weight.shape[1], weight.shape[0], bias=False).cuda()
    global_sf = (448 * 6) / weight.float().abs().nan_to_num().max()
    packed, scale = _nvfp4_quantize(weight, global_sf, sfLayout=sf_layout.layout_128x4, do_shuffle=False)
    layer._nvfp4_weight, layer._nvfp4_weight_scale = packed, scale
    layer._nvfp4_alpha = (1.0 / torch.as_tensor(global_sf, device="cuda", dtype=torch.float32)).float()
    layer._weight_global_sf = torch.as_tensor(global_sf, device="cuda").to(torch.bfloat16)
    if input_amax is not None:
        setattr(layer, H3_NVFP4_DIT_INPUT_SF_NAME, torch.tensor(2688.0 / input_amax, device="cuda"))
    return layer, NVFP4QuantizeMethod("transformer_blocks.0.ff.fc_out")


@needs_fp4
@pytest.mark.parametrize("input_amax", [None, 40.0])
def test_qad_linear_forward_is_the_deployed_nvfp4_linear(input_amax):
    from fastvideo.layers.quantization.nvfp4_qad import _NVFP4STE, nvfp4_gemm
    torch.manual_seed(0)
    weight = (torch.randn(512, 1024, device="cuda") * 0.05).to(torch.bfloat16)
    x = (torch.randn(300, 1024, device="cuda") * 8).to(torch.bfloat16)
    layer, runtime = _runtime_layer(weight, input_amax)
    expected = runtime.apply(layer, x)
    sf = torch.tensor(1.0 if input_amax is None else 2688.0 / input_amax, device="cuda")
    assert torch.equal(nvfp4_gemm(x, weight, None, sf), expected)
    weight_master = weight.float().requires_grad_(True)
    x_grad = x.clone().requires_grad_(True)
    out = _NVFP4STE.apply(x_grad, weight_master, None, sf)
    assert torch.equal(out, expected)
    grad = torch.randn_like(out)
    out.backward(grad)
    torch.testing.assert_close(weight_master.grad, (grad.float().t() @ x.float()), rtol=2e-2, atol=1e-2)
    torch.testing.assert_close(x_grad.grad.float(), grad.float() @ weight.float(), rtol=2e-2, atol=1e-2)


def test_block_reconstruction_is_local_and_zero_at_identity(cpu_loader, single_process_group):  # noqa: F811
    from fastvideo.train.methods.knowledge_distillation.pdd_qad_recon import BlockReconstruction
    teacher = _tiny_model()
    student = copy.deepcopy(teacher).train().requires_grad_(True)
    teacher.requires_grad_(False)
    with BlockReconstruction(teacher, student, scale=1.0) as recon:
        _forward(teacher)
    assert len(recon.attn_rel) == 2 and float(recon.loss()) == 0.0
    assert all(p.grad is None or float(p.grad.abs().max()) == 0.0 for p in student.parameters())
    # Perturb block 1's FFN only: its FFN unit errs; block 0 and every attention unit stay exact.
    with torch.no_grad():
        student.transformer_blocks[1].ff.fc_out.weight.mul_(1.1)
    student.zero_grad(set_to_none=True)
    with BlockReconstruction(teacher, student, scale=1.0) as recon:
        _forward(teacher)
    assert float(recon.ff_rel[1]) > 0 and float(recon.ff_rel[0]) == 0.0
    assert all(float(a) == 0.0 for a in recon.attn_rel)
    grads = {n: p.grad for n, p in student.named_parameters() if p.grad is not None and p.grad.abs().max() > 0}
    assert grads and all(n.startswith("transformer_blocks.1.ff.") for n in grads), sorted(grads)


# ----------------------------------------------------------------------------- E0: recipe-review fixes
def test_x0_matches_scheduler_sign():
    """x0 = x_t + (1 - t) v, exactly what MiniMaxH3Scheduler.step denoises to (a final step lands on it)."""
    from fastvideo.models.schedulers.scheduling_minimax_h3 import MiniMaxH3Scheduler
    from fastvideo.train.methods.knowledge_distillation.pdd_qad_metrics import denoised
    scheduler = MiniMaxH3Scheduler(shift=12.0)
    scheduler.set_timesteps(sigmas=torch.tensor([0.7, 0.0]))
    x_t, v = torch.randn(5, 3), torch.randn(5, 3)
    final = scheduler.step(v, scheduler.timesteps[0], x_t, return_dict=False)[0]
    torch.testing.assert_close(denoised(x_t, v, scheduler.timesteps[0]), final)
    assert not torch.allclose(final, x_t - 0.7 * v)


def test_grouped_rel_error_weights_groups_and_drops_absent():
    from fastvideo.train.methods.knowledge_distillation.pdd_qad_recon import GroupedRelError
    groups = torch.tensor([0, 0, 1, 2, 2, -1])  # video, video, audio, cond, cond, padding
    teacher = torch.tensor([[1.0, 0], [1, 0], [0, 2], [0, 100], [100, 0], [7, 7]])[None]
    student = teacher.clone()
    student[0, 2] += torch.tensor([0.0, 1.0])  # audio rel = 1/4
    student[0, 5] += 9.0  # padding row never counts
    loss, per_group = GroupedRelError(groups, {"video": 1.0, "audio": 1.0, "condition": 0.25})(student, teacher)
    assert per_group.tolist() == [0.0, 0.25, 0.0]
    assert float(loss) == pytest.approx(0.25 / 2.25)  # the huge conditioning rows do not dilute audio
    no_audio = torch.tensor([0, 0, 2, 2, 2, -1])
    _, per_group = GroupedRelError(no_audio, {"video": 1.0, "audio": 1.0})(student, teacher)
    assert math.isnan(float(per_group[1]))


def test_grouped_rel_error_sp_gradient_matches_unsharded(monkeypatch):
    """FSDP averages gradients over SP ranks; the mean of the shards' gradients must be the unsharded gradient.

    The x0 output loss already has this scale (its all-gather backward sums over SP ranks), so reconstruction
    must too, or ``recon_lambda`` is silently divided by ``sp_size``.
    """
    import fastvideo.distributed as distributed
    from fastvideo.train.methods.knowledge_distillation.pdd_qad_recon import GroupedRelError
    torch.manual_seed(0)
    weights = {"video": 1.0, "audio": 1.0, "condition": 0.25}
    groups = torch.tensor([0, 0, 0, 1, 2, 2, 0, 1, 2, 0, 0, -1])  # 2 shards of 6 rows, last row is SP padding
    x, teacher = torch.randn(12, 5), torch.randn(12, 3)
    weight = torch.randn(3, 5, requires_grad=True)

    def grad(loss: torch.Tensor) -> torch.Tensor:
        weight.grad = None
        loss.backward()
        return weight.grad.clone()

    monkeypatch.setattr(distributed, "model_parallel_is_initialized", lambda: False)
    full_loss, full_rel = GroupedRelError(groups, weights)(x @ weight.T, teacher)
    full_grad = grad(full_loss)

    sp_size, shard = 2, 6
    local_totals: list[torch.Tensor] = []

    class _Group:  # all_reduce = the sum of every shard's totals, recorded in a first pass

        def all_reduce(self, totals: torch.Tensor) -> torch.Tensor:
            local_totals.append(totals)
            return sum(local_totals[:sp_size]) if len(local_totals) > sp_size else totals

    monkeypatch.setattr(distributed, "model_parallel_is_initialized", lambda: True)
    monkeypatch.setattr(distributed, "get_sp_world_size", lambda: sp_size)
    monkeypatch.setattr(distributed, "get_sp_group", lambda: _Group())
    rows = [slice(rank * shard, (rank + 1) * shard) for rank in range(sp_size)]
    for part in rows:  # first pass: record each shard's local totals
        GroupedRelError(groups[part], weights)(x[part] @ weight.T, teacher[part])
    shard_grads = []
    for part in rows:
        loss, rel = GroupedRelError(groups[part], weights)(x[part] @ weight.T, teacher[part])
        torch.testing.assert_close(rel, full_rel)
        shard_grads.append(grad(loss))
    torch.testing.assert_close(sum(shard_grads) / sp_size, full_grad)  # FSDP's mean over the SP ranks


def test_packed_row_groups_from_layout():
    from types import SimpleNamespace
    from fastvideo.train.methods.knowledge_distillation.pdd_qad_recon import local_row_groups, packed_row_groups
    layout = SimpleNamespace(video_indices=torch.tensor([3, 4, 5, 6]), num_condition_video_rows=1,
                             audio_indices=torch.tensor([7, 8]), num_condition_audio_rows=1)
    groups = packed_row_groups(layout, 9)
    assert groups.tolist() == [2, 2, 2, 2, 0, 0, 0, 2, 1]
    assert torch.equal(local_row_groups(groups, torch.device("cpu")), groups)


def test_grouped_block_reconstruction_measure_only_has_no_grads(cpu_loader, single_process_group):  # noqa: F811
    from fastvideo.train.methods.knowledge_distillation.pdd_qad_recon import BlockReconstruction, GroupedRelError
    teacher = _tiny_model()
    student = copy.deepcopy(teacher).train().requires_grad_(True)
    teacher.requires_grad_(False)
    with torch.no_grad():
        student.transformer_blocks[0].attn.to_v.weight.mul_(1.2)
    layout = build_packed_sequence(torch.full((3, ), MINIMAX_H3_TEXT_TAG, dtype=torch.long), 2, 4, 4, 2, (1, 2, 2))
    groups = torch.full((int(layout.position_ids.shape[0]), ), 2, dtype=torch.long)
    groups[layout.video_indices] = 0
    groups[layout.audio_indices] = 1
    criterion = GroupedRelError(groups, {"video": 1.0, "audio": 1.0, "condition": 0.25})
    with BlockReconstruction(teacher, student, 1.0, criterion, measure_only=True) as recon:
        _forward(teacher)
    assert float(recon.loss()) > 0 and all(p.grad is None for p in student.parameters())
    with BlockReconstruction(teacher, student, 1.0, criterion) as recon:
        _forward(teacher)
    grads = {n for n, p in student.named_parameters() if p.grad is not None and p.grad.abs().max() > 0}
    assert grads and all(n.startswith("transformer_blocks.0.attn.") for n in grads), sorted(grads)
    assert set(recon.metrics("m")) >= {"m/video_rel", "m/audio_rel", "m/condition_rel"}


def _bare_method(**attrs):
    from fastvideo.train.methods.knowledge_distillation.pdd_qad import OmniRefPDDQADMethod
    method = object.__new__(OmniRefPDDQADMethod)
    defaults = dict(eligibility_keys=("endpoint/audio", "keyframe/vs_teacher/first"), tolerances={},
                    fidelity_tolerance=0.05, min_eligibility_tolerance=0.02, select_metric="score",
                    best_score=1.0, save_best=True, patience=3, bad_evals=0, bad_score_evals=0,
                    bad_alignment_evals=0, stop_requested=False, _checkpoint_request=None)
    method.__dict__.update({**defaults, **attrs})
    return method


def test_qad_data_parallel_groups_follow_the_same_collective_schedule():
    """Different rows must not select different student forwards or reconstruction backwards under FSDP."""
    rows = [f"omniref-case{group}-asset{group}-480p" for group in range(4)]
    methods = [_bare_method(dp_size=4, dp_rank=group, sampling_seed=1234, rungs_per_row=4,
                            rollout_start_step=20, rollout_fraction=0.25) for group in range(4)]
    schedules = []
    for iteration in range(1, 201):
        choices = [(method._trained_rungs(row, iteration, 8), method._use_rollout(row, iteration))
                   for method, row in zip(methods, rows, strict=True)]
        assert all(choice == choices[0] for choice in choices)
        assert len(choices[0][0]) == 4
        if iteration < 20:
            assert not choices[0][1]
        schedules.append((tuple(sorted(choices[0][0])), choices[0][1]))
    assert len(set(schedules)) > 8
    assert {rollout for _, rollout in schedules[19:]} == {False, True}


def test_qad_single_data_parallel_group_retains_row_sampling():
    from fastvideo.train.methods.knowledge_distillation.pdd_qad import _unit_hash
    method = _bare_method(dp_size=1, sampling_seed=1234, rungs_per_row=4,
                          rollout_start_step=20, rollout_fraction=0.25)
    for row in ("storyboard-a", "first_frame-b"):
        for iteration in (1, 20, 31):
            order = sorted(range(8), key=lambda rung: _unit_hash(row, iteration, "rung", rung))
            assert method._trained_rungs(row, iteration, 8) == set(order[:4])
            assert method._use_rollout(row, iteration) == (iteration >= 20 and _unit_hash(row, iteration) < 0.25)


def test_noise_floor_sets_two_sigma_tolerances():
    method = _bare_method()
    step0 = {"endpoint/audio": 0.40, "keyframe/vs_teacher/first": 0.30, "score": 0.30, "endpoint/video": 0.42}
    method._evaluate = lambda rows, rollout, save_tag="": {"endpoint/audio": 0.44, "keyframe/vs_teacher/first": 0.30,
                                                          "score": 0.30, "endpoint/video": 0.42}
    out = method._noise_floor([], step0)
    sigma = 0.04 / (math.sqrt(2) * 0.42)
    assert out["init/noise_sigma/endpoint/audio"] == pytest.approx(sigma)
    assert method.tolerances["endpoint/audio"] == pytest.approx(2 * sigma)
    assert method.tolerances["keyframe/vs_teacher/first"] == 0.02  # floor when two seeds agree


def test_best_requires_eligibility_on_full_eval():
    method = _bare_method(step0={"score": 0.30, "endpoint/audio": 0.40, "keyframe/vs_teacher/first": 0.30},
                          best_score=0.30, tolerances={"endpoint/audio": 0.10, "keyframe/vs_teacher/first": 0.10})
    teacher_forced_only = {"score": 0.20}
    method._guardrails(25, teacher_forced_only)
    assert method._checkpoint_request is None  # better score, but alignment unknown
    audio_regressed = {"score": 0.20, "endpoint/video": 0.4, "endpoint/audio": 0.45, "keyframe/vs_teacher/first": 0.3}
    out = method._guardrails(50, audio_regressed)
    assert method._checkpoint_request is None and out["eval/eligible"] == 0.0 and method.bad_evals == 1
    fine = {"score": 0.21, "endpoint/video": 0.4, "endpoint/audio": 0.43, "keyframe/vs_teacher/first": 0.31}
    out = method._guardrails(75, fine)
    assert method._checkpoint_request == "best" and method.best_score == 0.21 and method.bad_evals == 0


def test_teacher_forced_evals_do_not_reset_alignment_failures():
    """eval_every 15 / full_eval_every 30 / patience 2: persistent alignment regressions must still stop the run."""
    method = _bare_method(step0={"score": 0.30, "endpoint/video": 0.40, "endpoint/audio": 0.40},
                          best_score=0.30, patience=2)
    full_regressed = {"score": 0.25, "endpoint/video": 0.40, "endpoint/audio": 0.60}
    method._guardrails(30, full_regressed)
    assert (method.bad_alignment_evals, method.bad_evals, method.stop_requested) == (1, 1, False)
    method._guardrails(45, {"score": 0.24})  # teacher-forced only: better score, alignment unmeasured
    assert (method.bad_score_evals, method.bad_alignment_evals, method.stop_requested) == (0, 1, False)
    out = method._guardrails(60, full_regressed)
    assert method.stop_requested and out["eval/bad_alignment_evals"] == 2.0
    method = _bare_method(step0={"score": 0.30, "endpoint/audio": 0.40}, patience=2)
    method._guardrails(30, {"score": 0.25, "endpoint/video": 0.4, "endpoint/audio": 0.60})
    method._guardrails(60, {"score": 0.25, "endpoint/video": 0.4, "endpoint/audio": 0.40})  # recovered
    assert method.bad_alignment_evals == 0 and not method.stop_requested
    method._guardrails(75, {"score": 0.31})
    method._guardrails(90, {"score": 0.32, "endpoint/video": 0.4, "endpoint/audio": 0.40})
    assert method.bad_score_evals == 2 and method.stop_requested  # the score count still runs on every eval


def test_baseline_state_round_trips_through_dcp(tmp_path, single_process_group):  # noqa: F811
    """Resume restores the step-0 baseline, guard counters and provisional scales instead of re-measuring."""
    import torch.distributed.checkpoint as dcp
    from types import SimpleNamespace
    from fastvideo.train.methods.knowledge_distillation.pdd_qad import _BaselineState

    def student(amax):
        linears = {"transformer_blocks.0.ff.fc_in": NVFP4QADLinearMethod(object(), "fc_in", "static", amax,
                                                                         allow_uncalibrated=True)}
        return SimpleNamespace(provisional_rows=2, qad_linears=linears)

    saved = _bare_method(normalizers={"video": [0.2, 0.1], "audio": [0.4, 0.3]}, step0={"score": 0.3},
                         recon_norm=[0.05, 0.04], tolerances={"endpoint/audio": 0.1}, best_score=0.27,
                         bad_score_evals=1, bad_alignment_evals=2, student=student(7.5))
    dcp.save({"method.qad_baseline": _BaselineState(saved)}, checkpoint_id=str(tmp_path))
    resumed = _bare_method(normalizers=None, step0={}, recon_norm=None, best_score=math.inf, student=student(None))
    dcp.load({"method.qad_baseline": _BaselineState(resumed)}, checkpoint_id=str(tmp_path))
    for key in ("normalizers", "step0", "recon_norm", "tolerances", "best_score", "bad_score_evals",
                "bad_alignment_evals"):
        assert getattr(resumed, key) == getattr(saved, key), key
    assert resumed.bad_evals == 2
    assert resumed.student.qad_linears["transformer_blocks.0.ff.fc_in"].input_amax == 7.5
    with pytest.raises(ValueError, match="no step-0 baseline"):
        _bare_method(student=student(None))._load_baseline_state({**saved._baseline_state(), "normalizers": None})


def test_provisional_calibration_runs_on_a_step0_cache_hit(single_process_group):  # noqa: F811
    from types import SimpleNamespace
    calls = []
    cached = {"eval/score": 0.3, "eval/x0_mse/video/rung0": 0.2, "eval/x0_mse/audio/rung0": 0.4}
    method = _bare_method(normalizers=None, step0={}, eval_every=15, student=SimpleNamespace(num_rungs=1),
                          _rank_rows=lambda: [], _load_step0=lambda: cached,
                          _restore_step0=lambda c: {k[len("eval/"):]: v for k, v in c.items()},
                          _provisional_calibration=lambda: calls.append(1) or {"init/provisional_calibration_rows": 2})
    metrics = method.on_validation_begin(0)
    assert calls == [1] and metrics["init/provisional_calibration_rows"] == 2
    assert method.normalizers == {"video": [0.2], "audio": [0.4]}
    method.on_validation_begin(7)  # not an eval step (e.g. a resume): calibration still gets its chance
    assert calls == [1, 1]


def test_manifest_groups_keep_each_resolution_of_a_shared_id(tmp_path):
    import json
    from fastvideo.train.models.minimax_h3.omniref_data import load_manifest_groups

    def entry(res, frames, case="first_frame", source="a"):
        return {"id": f"{case}:{source}", "case": case, "resolution": res,
                "parquet": f"/d/bucket=832x480-{frames}f/{source}-{res}.parquet"}

    first, second = tmp_path / "v4.jsonl", tmp_path / "v5.jsonl"
    first.write_text("\n".join(json.dumps(e) for e in (entry("480p", 121), entry("768p", 121))) + "\n")
    second.write_text("\n".join(json.dumps(e) for e in (entry("480p", 121), entry("480p", 241, source="b"))) + "\n")
    groups = load_manifest_groups([str(first), str(second)], ("first_frame", ), 243)
    assert {key: [e["id"] for e in entries] for key, entries in groups.items()} == {
        ("first_frame", "480p"): ["first_frame:a", "first_frame:b"],  # the v5 repeat of (a, 480p) is dropped
        ("first_frame", "768p"): ["first_frame:a"],
    }


def test_eval_top_up_respects_the_eval_frame_cap():
    from fastvideo.train.models.minimax_h3.omniref_data import limit_frames, top_up_eval_rows
    groups = {("storyboard", "480p"): [{"id": f"storyboard:s{i}", "case": "storyboard", "resolution": "480p",
                                        "parquet": f"/d/bucket=832x480-{121 if i % 2 else 241}f/s{i}.parquet"}
                                       for i in range(8)]}
    assert limit_frames(groups, 0) == groups
    capped = limit_frames(groups, 121)
    assert len(capped[("storyboard", "480p")]) == 4 and len(groups[("storyboard", "480p")]) == 8
    rows = top_up_eval_rows([], capped, 7, {"480p": 3})
    assert len(rows) == 3 and all(int(row.source[1:]) % 2 for row in rows)


def test_eval_top_up_is_held_out_and_interleaved():
    from fastvideo.train.models.minimax_h3.omniref_data import top_up_eval_rows
    groups = {(case, res): [{"id": f"{case}:{res}{i}", "case": case, "resolution": res, "parquet": f"/p{i}"}
                            for i in range(10)] for case in ("first_frame", "storyboard") for res in ("480p", "768p")}
    existing = [RowSpec("e1", "480p0", "first_frame", "480p", "/p", 1)]
    rows = top_up_eval_rows(existing, groups, 7, {"480p": 3, "768p": 1})
    count = {}
    for row in rows:
        count[(row.case, row.resolution)] = count.get((row.case, row.resolution), 0) + 1
    assert count == {("first_frame", "480p"): 3, ("first_frame", "768p"): 1, ("storyboard", "480p"): 3,
                     ("storyboard", "768p"): 1}
    assert len({row.source for row in rows if row.resolution == "480p" and row.case == "first_frame"}) == 3
    assert [(r.case, r.resolution) for r in rows[:4]] == sorted({(r.case, r.resolution) for r in rows})
    plan = select_rows(groups, 20261007, {"480p": 10}, frozenset(row.source for row in rows))
    assert not {r.source for r in plan} & {r.source for r in rows}


def test_pinned_amax_checks_sha_and_copies(tmp_path, single_process_group):  # noqa: F811
    import hashlib
    from fastvideo.train.models.minimax_h3.omniref_pdd import _pin_amax
    table = tmp_path / "amax.json"
    table.write_text('{"b0.ff.fc_in": 3.0}')
    digest = hashlib.sha256(table.read_bytes()).hexdigest()
    assert _pin_amax(str(table), digest, str(tmp_path / "run")) == digest
    assert (tmp_path / "run" / f"amax-{digest[:12]}.json").is_file()
    with pytest.raises(ValueError, match="pins"):
        _pin_amax(str(table), "0" * 64, "")


def test_skip_blocks_keep_whole_blocks_bf16(cpu_loader, single_process_group):  # noqa: F811
    methods = install_nvfp4_qad(_tiny_model(), _amax_table(), NVFP4QADPlan(skip_blocks=(1, )))
    assert methods and all(name.startswith("transformer_blocks.0.") for name in methods)


def test_audio_stats_and_seed_floor_gate():
    from fastvideo.train.methods.knowledge_distillation.pdd_qad_audio_stats import (audio_stats, log_mel_distance,
                                                                                    seed_floor_gate)
    rate = 32000
    t = torch.arange(rate) / rate
    low, high = torch.sin(2 * math.pi * 1000 * t), torch.sin(2 * math.pi * 10000 * t)
    stats_low, stats_high = audio_stats(low, rate), audio_stats(high[:, None].repeat(1, 2), rate)
    assert stats_low["spectral_centroid_hz"] == pytest.approx(1000, rel=0.05)
    assert stats_low["share_above_8khz"] < 0.01 and stats_high["share_above_8khz"] > 0.99
    assert log_mel_distance(low, low, rate) == 0.0 and log_mel_distance(low, high, rate) > 1.0
    teacher = [{"spectral_centroid_hz": 950.0, "share_above_8khz": 0.1, "dynamic_range_db": 30.0,
                "loudness_iqr_db": 6.0}] * 3
    other = [{**row, "spectral_centroid_hz": 900.0, "dynamic_range_db": 28.0, "loudness_iqr_db": 5.0,
              "share_above_8khz": 0.09} for row in teacher]
    good = [{**row, "spectral_centroid_hz": 930.0, "share_above_8khz": 0.095} for row in teacher]
    dull = [{**row, "spectral_centroid_hz": 530.0} for row in teacher]
    assert seed_floor_gate(good, teacher, other)["pass"]
    result = seed_floor_gate(dull, teacher, other)
    assert not result["pass"] and not result["spectral_centroid_hz/pass"]


def test_fp32_boundary_modules_match_inference_and_stay_frozen(cpu_loader, single_process_group):  # noqa: F811
    """The QAD loader keeps H3's FP32 boundary modules FP32 (as inference) and refuses to train them."""
    from types import SimpleNamespace
    from fastvideo.train.models.minimax_h3.omniref_pdd import MiniMaxH3OmniRefPDDModel
    model = _tiny_model().to(torch.float32)
    assert not model.config.uniform_parameter_dtype
    fp32 = {n for n, _ in model.named_parameters() if model._get_parameter_dtype(n, torch.bfloat16) == torch.float32}
    assert fp32 and {n.split(".", 1)[0] for n in fp32} <= set(model._keep_in_fp32_modules)
    model.requires_grad_(False)
    fake = SimpleNamespace(transformer=model)
    MiniMaxH3OmniRefPDDModel._check_fp32_modules_frozen(fake)
    model.proj_in.weight.requires_grad_(True)
    with pytest.raises(ValueError, match="must stay frozen"):
        MiniMaxH3OmniRefPDDModel._check_fp32_modules_frozen(fake)


def test_block_first_call_keeps_the_fsdp_root(single_process_group):  # noqa: F811
    """Stage 2 from a checkpoint calibrates reconstruction (student blocks called directly) before any full student
    forward; without pinning the root first, FSDP2 makes block 0 the root and the next full forward fails."""
    from torch.distributed.device_mesh import init_device_mesh
    from torch.distributed.fsdp import fully_shard

    from fastvideo.train.methods.knowledge_distillation.pdd_qad_recon import ensure_fsdp_root_initialized
    device = "cuda" if torch.cuda.is_available() else "cpu"

    class Toy(torch.nn.Module):

        def __init__(self) -> None:
            super().__init__()
            self.transformer_blocks = torch.nn.ModuleList(torch.nn.Linear(4, 4) for _ in range(2))

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            for block in self.transformer_blocks:
                x = block(x)
            return x

    def sharded() -> Toy:
        mesh = init_device_mesh(device, (1, ))
        model = Toy().to(device)
        for block in model.transformer_blocks:
            fully_shard(block, mesh=mesh)
        return fully_shard(model, mesh=mesh)

    x = torch.randn(2, 4, device=device)
    broken = sharded()
    broken.transformer_blocks[0](x)
    with pytest.raises(RuntimeError, match="already been lazily initialized"):
        broken(x)
    fixed = sharded()
    ensure_fsdp_root_initialized(fixed)
    ensure_fsdp_root_initialized(fixed)  # idempotent
    fixed.transformer_blocks[0](x)
    assert fixed(x).shape == (2, 4)
    ensure_fsdp_root_initialized(torch.nn.Linear(2, 2))  # no FSDP: no-op


def test_step0_cache_round_trip_and_fingerprint(tmp_path):
    from fastvideo.train.methods.knowledge_distillation.pdd_qad_step0 import (load_step0, restore_step0, save_step0,
                                                                              step0_fingerprint_inputs)
    method = {"init_student_dcp": "/runs/e3/best", "noise_floor_eval": True, "eligibility_keys": ["endpoint/audio"],
              "recon_group_weights": {"video": 1.0, "audio": 1.0}, "eval_every": 20}
    data = {"eval_manifest": "/m.json", "eval_per_group": {"480p": 4, "768p": 2}, "cases": ["first_frame"]}
    qad = {"linears": {"skip_blocks": [42, 43]}, "fp4_attention": {"two_level_p": True}}
    models = dict(student_init_from="/ckpt/student", teacher_init_from="/ckpt/teacher")
    inputs = step0_fingerprint_inputs(method, data, qad, "ab" * 32, 30, **models)
    metrics = {"eval/score": 0.31, "eval/x0_mse/video/rung0": 0.2, "eval/seconds": 6000.0,
               "init/recon_step0/rung0": 0.04, "init/recon_step0/rung1": 0.03,
               "init/noise_sigma/endpoint/audio": 0.05, "init/noise_sigma/score": 0.001}
    path = tmp_path / "step0.json"
    assert load_step0(path, inputs) is None
    save_step0(path, inputs, metrics)
    # Settings that cannot change step 0 (eval cadence) do not invalidate the cache.
    same = step0_fingerprint_inputs({**method, "eval_every": 40}, data, qad, "ab" * 32, 30, **models)
    cached = load_step0(path, same)
    assert cached is not None and "eval/seconds" not in cached and cached["eval/score"] == 0.31
    summary, recon_norm, tolerances = restore_step0(cached, 2, recon=True, noise_floor=True, min_tolerance=0.02)
    assert summary == {"score": 0.31, "x0_mse/video/rung0": 0.2}
    assert recon_norm == [0.04, 0.03]
    assert tolerances == {"endpoint/audio": pytest.approx(0.1), "score": 0.02}
    # Training-plan keys only matter with provisional calibration (no amax_json).
    assert load_step0(path, step0_fingerprint_inputs(method, {**data, "seed": 5}, qad, "ab" * 32, 30, **models))
    provisional = {**qad, "linears": {"skip_blocks": [42, 43], "provisional_calibration_rows": 2}}
    for changed in (step0_fingerprint_inputs({**method, "init_student_dcp": "/runs/e3b/best"}, data, qad, "ab" * 32, 30,
                                             **models),
                    step0_fingerprint_inputs(method, {**data, "eval_per_group": 1}, qad, "ab" * 32, 30, **models),
                    step0_fingerprint_inputs(method, data, {**qad, "linears": {"skip_blocks": []}}, "ab" * 32, 30,
                                             **models),
                    step0_fingerprint_inputs(method, data, qad, "cd" * 32, 30, **models),
                    step0_fingerprint_inputs(method, data, qad, "ab" * 32, 30, **{**models, "student_init_from": "/x"}),
                    step0_fingerprint_inputs(method, data, qad, "ab" * 32, 30, **{**models, "teacher_init_from": "/x"}),
                    step0_fingerprint_inputs({**method, "audio_weight": 2.0}, data, qad, "ab" * 32, 30, **models),
                    step0_fingerprint_inputs(method, data, provisional, "ab" * 32, 30, **models)):
        with pytest.raises(ValueError, match="other inputs"):
            load_step0(path, changed)
