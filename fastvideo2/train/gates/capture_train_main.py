"""Capture per-step finetune goldens from fastvideo-main's LEGACY
wan_training_pipeline (the authority for shipped models).

Run on the cluster venv (fastvideo main installed) under torchrun 1 GPU:

    torchrun --nproc_per_node=1 fastvideo2/train/gates/capture_train_main.py

Gate config (pinned here, recorded in the manifest): crush-smol processed
parquet, 5 steps, batch 1, grad_accum 1, num_latent_t 8 (shorter than the
recipe's 20 to keep step time small — the math is length-independent),
lr 5e-5, wd 1e-4, betas (0.9, 0.999), grad clip 1.0, uniform timestep
sampling, target = noise - latents, dit fp32 + bf16 batch,
training_cfg_rate 0 (the CFG coin-flip would add an RNG stream the gate
doesn't need), seed 42, no validation, wandb offline.

Goldens (evidence/goldens/train-finetune-main/):
    step{i}: hashes of (latents, embeds, noise) + timesteps/sigmas values +
             loss + grad_norm + the batch caption (for row identification)
    step0.npz: full tensors (latents, embeds, mask, noise) for triage
    params.npz: proj_out.weight[:8,:8] before training and after 5 steps
                (optimizer-chain parity)
    manifest.json

Our trainer replays: same rows (by caption), same CUDA/CPU generator seeds
(same GPU model => same streams), same math => per-step hashes must match.
"""
from __future__ import annotations

import hashlib
import json
import os
import sys

MODE = sys.argv[1] if len(sys.argv) > 1 and not sys.argv[1].startswith("-") else "finetune"
VSA = MODE == "vsa"
DMD2 = MODE == "dmd2"
os.environ["FASTVIDEO_ATTENTION_BACKEND"] = "VIDEO_SPARSE_ATTN" if VSA else "FLASH_ATTN"
os.environ.setdefault("WANDB_MODE", "offline")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

SEED = 42
STEPS = 5
NUM_LATENT_T = 8  # dmd2 uses 4 (three models on one GPU; math is length-free)
DATASET = "wlsaidhi/crush-smol_processed_t2v"
# vsa gate trains the FastWan checkpoint (has to_gate_compress weights) so
# all parameters are deterministic; base-checkpoint VSA finetune random-
# initializes the gate projections — an init-RNG parity gate for later
MODEL = ("FastVideo/FastWan2.1-T2V-1.3B-Diffusers" if VSA
         else "Wan-AI/Wan2.1-T2V-1.3B-Diffusers")


def _hash(t) -> str:
    import torch
    return hashlib.sha256(t.detach().to(torch.float32).cpu().numpy().tobytes()
                          ).hexdigest()[:16]


def main() -> None:
    import numpy as np
    import torch
    from huggingface_hub import snapshot_download

    here = os.path.dirname(os.path.abspath(__file__))
    out = os.path.normpath(os.path.join(here, "..", "..", "evidence", "goldens",
                                        f"train-{MODE}-main"))
    os.makedirs(out, exist_ok=True)

    data_root = snapshot_download(DATASET, repo_type="dataset", token=False)
    # the processed dataset carries the parquet dir at its root or one level in
    data_path = data_root
    for cand in ("combined_parquet_dataset",):
        p = os.path.join(data_root, cand)
        if os.path.isdir(p):
            data_path = p

    from fastvideo.fastvideo_args import FastVideoArgs, TrainingArgs
    from fastvideo.utils import FlexibleArgumentParser

    argv = [
        "capture",
        "--model_path", MODEL,
        "--pretrained_model_name_or_path", MODEL,
        "--data_path", data_path,
        "--dataloader_num_workers", "1",
        "--num_gpus", "1", "--sp_size", "1", "--tp_size", "1",
        "--hsdp_replicate_dim", "1", "--hsdp_shard_dim", "1",
        "--train_batch_size", "1", "--train_sp_batch_size", "1",
        "--gradient_accumulation_steps", "1",
        "--max_train_steps", str(STEPS),
        "--num_latent_t", str(NUM_LATENT_T),
        "--num_height", "480", "--num_width", "832", "--num_frames", "77",
        "--learning_rate", "5e-5", "--weight_decay", "1e-4",
        "--max_grad_norm", "1.0",
        "--mixed_precision", "bf16", "--dit_precision", "fp32",
        "--training_cfg_rate", "0.0",
        "--seed", str(SEED),
        "--output_dir", "/tmp/fv2_train_capture",
        "--tracker_project_name", "fv2_capture",
        "--inference_mode", "False",
        "--weight_only_checkpointing_steps", "100000",
        "--training_state_checkpointing_steps", "100000",
    ]
    if VSA:
        argv += ["--VSA_sparsity", "0.8", "--VSA_decay_rate", "0.2",
                 "--VSA_decay_interval_steps", "1"]
    if DMD2:
        base = "Wan-AI/Wan2.1-T2V-1.3B-Diffusers"
        argv[argv.index(str(NUM_LATENT_T))] = "4"
        argv += ["--real_score_model_path", base, "--fake_score_model_path", base,
                 "--dmd_denoising_steps", "1000,757,522",
                 "--simulate_generator_forward",
                 "--generator_update_interval", "2",
                 "--real_score_guidance_scale", "3.5",
                 "--min_timestep_ratio", "0.02", "--max_timestep_ratio", "0.98",
                 "--flow_shift", "8.0"]
        argv[argv.index("5e-5")] = "2e-6"  # learning_rate per the shipped recipe
    sys.argv = argv
    parser = FlexibleArgumentParser()
    parser = TrainingArgs.add_cli_args(parser)
    parser = FastVideoArgs.add_cli_args(parser)
    args = parser.parse_args()
    args.dit_cpu_offload = False

    import fastvideo.training.training_pipeline as TP
    records: list[dict] = []
    step0: dict = {}

    orig_batch = TP.TrainingPipeline._get_next_batch
    orig_prepare = TP.TrainingPipeline._prepare_dit_inputs
    orig_step = TP.TrainingPipeline.train_one_step

    batches: list[dict] = []

    def get_batch(self, tb):
        tb = orig_batch(self, tb)
        rec = {"caption": str(tb.infos[0].get("caption", tb.infos[0]) if
                              isinstance(tb.infos[0], dict) else tb.infos[0]),
               "latents_hash": _hash(tb.latents),
               "embeds_hash": _hash(tb.encoder_hidden_states)}
        records.append(rec)
        batches.append({})
        return tb

    def prepare(self, tb):
        tb = orig_prepare(self, tb)
        rec = records[-1]
        rec.update(noise_hash=_hash(tb.noise),
                   noisy_hash=_hash(tb.noisy_model_input),
                   timesteps=[float(v) for v in tb.timesteps.flatten().tolist()],
                   sigmas=[float(v) for v in tb.sigmas.flatten().tolist()])
        # record here, NOT in _get_next_batch: train_one_step runs
        # _normalize_dit_input (VAE scaling) between fetch and prepare, and
        # the loss consumes the NORMALIZED latents
        batches[-1].update(
            latents=tb.latents.detach().to(torch.float32).cpu().numpy(),
            embeds=tb.encoder_hidden_states.detach().to(torch.float32).cpu().numpy(),
            mask=tb.encoder_attention_mask.detach().to(torch.float32).cpu().numpy(),
            noise=tb.noise.detach().to(torch.float32).cpu().numpy())
        return tb

    def one_step(self, tb):
        tb = orig_step(self, tb)
        records[-1].update(loss=float(tb.total_loss.item() if torch.is_tensor(tb.total_loss)
                                      else tb.total_loss),
                           grad_norm=float(tb.grad_norm),
                           vsa_sparsity=float(getattr(tb, "current_vsa_sparsity", 0.0) or 0.0))
        return tb

    TP.TrainingPipeline._get_next_batch = get_batch
    TP.TrainingPipeline._prepare_dit_inputs = prepare
    TP.TrainingPipeline.train_one_step = one_step

    def _slice(w):
        w = w.full_tensor() if hasattr(w, "full_tensor") else w  # FSDP2 DTensor
        return w.detach()[:8, :8].to(torch.float32).cpu().numpy().copy()

    def _np(t):
        return t.detach().to(torch.float32).cpu().numpy()

    if DMD2:
        import fastvideo.training.distillation_pipeline as DP
        from fastvideo.training.wan_distillation_pipeline import WanDistillationPipeline

        dmd_steps: list[dict] = []

        def _cur() -> dict:
            if not dmd_steps or dmd_steps[-1].get("closed"):
                dmd_steps.append({"rollouts": []})
            return dmd_steps[-1]

        orig_roll = DP.DistillationPipeline._generator_multi_step_simulation_forward
        orig_dmd = DP.DistillationPipeline._dmd_forward
        orig_fake = DP.DistillationPipeline.faker_score_forward
        orig_dstep = DP.DistillationPipeline.train_one_step

        def rec_roll(self, tb):
            draws = {"step_noises": []}
            orig_randint, orig_randn = torch.randint, torch.randn

            def randint(*a, **k):
                out = orig_randint(*a, **k)
                draws.setdefault("target_idx", int(out.item()))
                return out

            def randn(*a, **k):
                out = orig_randn(*a, **k)
                if "init_noise" not in draws:
                    draws["init_noise"] = _np(out)
                else:
                    draws["step_noises"].append(_np(out))
                return out

            torch.randint, torch.randn = randint, randn
            try:
                res = orig_roll(self, tb)
            finally:
                torch.randint, torch.randn = orig_randint, orig_randn
            _cur()["rollouts"].append(draws)
            _cur().setdefault("x0_student_hash", _hash(res))
            return res

        def rec_dmd(self, gpv, tb):
            rec = _cur()
            orig_randint, orig_randn = torch.randint, torch.randn
            raw = {}

            def randint(*a, **k):
                out = orig_randint(*a, **k)
                raw["t_raw"] = int(out.item())
                return out

            def randn(*a, **k):
                out = orig_randn(*a, **k)
                raw["noise"] = _np(out)
                return out

            torch.randint, torch.randn = randint, randn
            try:
                loss = orig_dmd(self, gpv, tb)
            finally:
                torch.randint, torch.randn = orig_randint, orig_randn
            rec["dmd"] = {"t_final": float(tb.dmd_latent_vis_dict["dmd_timestep"].item()),
                          "noise": raw["noise"]}
            rec["x0_student"] = _np(gpv)
            rec["uncond_embeds"] = _np(tb.unconditional_dict["encoder_hidden_states"])
            return loss

        def rec_fake(self, tb):
            rec = _cur()
            orig_randint, orig_randn = torch.randint, torch.randn
            raw = {"noises": []}

            def randint(*a, **k):
                out = orig_randint(*a, **k)
                raw.setdefault("ints", []).append(int(out.item()))
                return out

            def randn(*a, **k):
                out = orig_randn(*a, **k)
                raw["noises"].append(_np(out))
                return out

            torch.randint, torch.randn = randint, randn
            try:
                tb, loss = orig_fake(self, tb)
            finally:
                torch.randint, torch.randn = orig_randint, orig_randn
            # last randn is the critic noise; earlier ones belong to the
            # rollout wrapper (already recorded); timestep from vis dict
            rec["critic"] = {"t_final": float(tb.fake_score_latent_vis_dict[
                                "fake_score_timestep"].item()),
                             "noise": raw["noises"][-1]}
            rec["fake_loss"] = float(loss.detach().item())
            rec["critic_x0_hash"] = _hash(tb.fake_score_latent_vis_dict[
                "generator_pred_video"])
            return tb, loss

        def rec_dstep(self, tb):
            tb = orig_dstep(self, tb)
            rec = _cur()
            rec["gen_loss"] = float(tb.generator_loss)
            rec["embeds"] = _np(tb.encoder_hidden_states) if tb.encoder_hidden_states is not None else None
            rec["closed"] = True
            return tb

        DP.DistillationPipeline._generator_multi_step_simulation_forward = rec_roll
        DP.DistillationPipeline._dmd_forward = rec_dmd
        DP.DistillationPipeline.faker_score_forward = rec_fake
        DP.DistillationPipeline.train_one_step = rec_dstep

        pipeline = WanDistillationPipeline.from_pretrained(
            args.pretrained_model_name_or_path, args=args)
        student = pipeline.get_module("transformer")
        critic = pipeline.fake_score_transformer
        w0 = {"student": _slice(student.proj_out.weight),
              "critic": _slice(critic.proj_out.weight)}
        pipeline.train()
        w5 = {"student": _slice(student.proj_out.weight),
              "critic": _slice(critic.proj_out.weight)}

        uncond = next((r["uncond_embeds"] for r in dmd_steps
                       if "uncond_embeds" in r), None)
        np.savez(os.path.join(out, "params.npz"),
                 w0_student=w0["student"], w5_student=w5["student"],
                 w0_critic=w0["critic"], w5_critic=w5["critic"],
                 **({"neg_embeds": uncond} if uncond is not None else {}))
        meta_steps = []
        for i, rec in enumerate(dmd_steps):
            arrs = {"embeds": rec["embeds"]}
            if "x0_student" in rec:
                arrs["x0_student"] = rec["x0_student"]
            for j, ro in enumerate(rec["rollouts"]):
                arrs[f"ro{j}_init"] = ro["init_noise"]
                for k, sn in enumerate(ro["step_noises"]):
                    arrs[f"ro{j}_n{k}"] = sn
            if "dmd" in rec:
                arrs["dmd_noise"] = rec["dmd"]["noise"]
            arrs["critic_noise"] = rec["critic"]["noise"]
            np.savez(os.path.join(out, f"step{i}.npz"), **arrs)
            meta_steps.append({
                "targets": [ro.get("target_idx") for ro in rec["rollouts"]],
                "dmd_t": rec.get("dmd", {}).get("t_final"),
                "critic_t": rec["critic"]["t_final"],
                "gen_loss": rec["gen_loss"], "fake_loss": rec["fake_loss"],
                "x0_student_hash": rec.get("x0_student_hash"),
            })
        with open(os.path.join(out, "steps.json"), "w") as f:
            json.dump(meta_steps, f, indent=1)
        import subprocess
        import fastvideo
        src = os.path.dirname(os.path.dirname(os.path.abspath(fastvideo.__file__)))
        commit = subprocess.run(["git", "-C", src, "rev-parse", "HEAD"],
                                capture_output=True, text=True).stdout.strip()
        with open(os.path.join(out, "manifest.json"), "w") as f:
            json.dump({"fastvideo_commit": commit, "mode": MODE, "seed": SEED,
                       "gen_losses": [r["gen_loss"] for r in dmd_steps],
                       "fake_losses": [r["fake_loss"] for r in dmd_steps],
                       "torch": torch.__version__,
                       "gpu": torch.cuda.get_device_name(0),
                       "config": "dmd2 legacy: interval2 gw3.5 lr2e-6 shift8 "
                                 "steps[1000,757,522] simulate nlt4 1gpu"}, f, indent=2)
        print("gen:", [round(r["gen_loss"], 6) for r in dmd_steps], flush=True)
        print("fake:", [round(r["fake_loss"], 6) for r in dmd_steps], flush=True)
        print("capture complete ->", out, flush=True)
        return

    from fastvideo.training.wan_training_pipeline import WanTrainingPipeline
    pipeline = WanTrainingPipeline.from_pretrained(args.pretrained_model_name_or_path,
                                                   args=args)
    tf = pipeline.get_module("transformer")

    def fwd_hook(module, args_in, kwargs_in, output):
        o = output[0] if isinstance(output, (tuple, list)) else output
        i = len([r for r in records if "pred_hash" in r])
        if i < len(records):
            records[i]["pred_hash"] = _hash(o)
            if i == 0:
                batches[0]["pred"] = o.detach().to(torch.float32).cpu().numpy()

    hh = tf.register_forward_hook(fwd_hook, with_kwargs=True)
    w0 = _slice(tf.proj_out.weight)
    pipeline.train()
    hh.remove()
    w5 = _slice(tf.proj_out.weight)

    for i, b in enumerate(batches):
        np.savez(os.path.join(out, f"step{i}.npz"), **b)
    np.savez(os.path.join(out, "params.npz"), w0=w0, w5=w5)
    with open(os.path.join(out, "steps.json"), "w") as f:
        json.dump(records, f, indent=1)

    import subprocess
    import fastvideo
    src = os.path.dirname(os.path.dirname(os.path.abspath(fastvideo.__file__)))
    commit = subprocess.run(["git", "-C", src, "rev-parse", "HEAD"],
                            capture_output=True, text=True).stdout.strip()
    with open(os.path.join(out, "manifest.json"), "w") as f:
        json.dump({"fastvideo_commit": commit, "dataset": DATASET, "mode": MODE,
                   "model": MODEL,
                   "seed": SEED, "steps": STEPS, "num_latent_t": NUM_LATENT_T,
                   "config": "1gpu bs1 accum1 lr5e-5 wd1e-4 betas(0.9,0.999) "
                             "clip1.0 uniform-t cfg_rate0 dit_fp32 mixed_bf16 "
                             "flow-match target=noise-latents flash_attn",
                   "torch": torch.__version__,
                   "gpu": torch.cuda.get_device_name(0),
                   "losses": [r.get("loss") for r in records]}, f, indent=2)
    print("losses:", [round(r.get("loss", -1), 6) for r in records], flush=True)
    print("capture complete ->", out, flush=True)


if __name__ == "__main__":
    main()
