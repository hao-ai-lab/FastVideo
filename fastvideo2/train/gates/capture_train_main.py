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

os.environ["FASTVIDEO_ATTENTION_BACKEND"] = "FLASH_ATTN"
os.environ.setdefault("WANDB_MODE", "offline")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

SEED = 42
STEPS = 5
NUM_LATENT_T = 8
DATASET = "wlsaidhi/crush-smol_processed_t2v"


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
                                        "train-finetune-main"))
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
        "--model_path", "Wan-AI/Wan2.1-T2V-1.3B-Diffusers",
        "--pretrained_model_name_or_path", "Wan-AI/Wan2.1-T2V-1.3B-Diffusers",
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

    def get_batch(self, tb):
        tb = orig_batch(self, tb)
        rec = {"caption": str(tb.infos[0].get("caption", tb.infos[0]) if
                              isinstance(tb.infos[0], dict) else tb.infos[0]),
               "latents_hash": _hash(tb.latents),
               "embeds_hash": _hash(tb.encoder_hidden_states)}
        records.append(rec)
        if len(records) == 1:
            step0.update(latents=tb.latents.detach().to(torch.float32).cpu().numpy(),
                         embeds=tb.encoder_hidden_states.detach().to(torch.float32).cpu().numpy(),
                         mask=tb.encoder_attention_mask.detach().to(torch.float32).cpu().numpy())
        return tb

    def prepare(self, tb):
        tb = orig_prepare(self, tb)
        rec = records[-1]
        rec.update(noise_hash=_hash(tb.noise),
                   noisy_hash=_hash(tb.noisy_model_input),
                   timesteps=[float(v) for v in tb.timesteps.flatten().tolist()],
                   sigmas=[float(v) for v in tb.sigmas.flatten().tolist()])
        if len(records) == 1:
            step0.update(noise=tb.noise.detach().to(torch.float32).cpu().numpy())
        return tb

    def one_step(self, tb):
        tb = orig_step(self, tb)
        records[-1].update(loss=float(tb.total_loss.item() if torch.is_tensor(tb.total_loss)
                                      else tb.total_loss),
                           grad_norm=float(tb.grad_norm))
        return tb

    TP.TrainingPipeline._get_next_batch = get_batch
    TP.TrainingPipeline._prepare_dit_inputs = prepare
    TP.TrainingPipeline.train_one_step = one_step

    from fastvideo.training.wan_training_pipeline import WanTrainingPipeline
    pipeline = WanTrainingPipeline.from_pretrained(args.pretrained_model_name_or_path,
                                                   args=args)
    tf = pipeline.get_module("transformer")
    w0 = tf.proj_out.weight.detach()[:8, :8].to(torch.float32).cpu().numpy().copy()
    pipeline.train()
    w5 = tf.proj_out.weight.detach()[:8, :8].to(torch.float32).cpu().numpy().copy()

    np.savez(os.path.join(out, "step0.npz"), **step0)
    np.savez(os.path.join(out, "params.npz"), w0=w0, w5=w5)
    with open(os.path.join(out, "steps.json"), "w") as f:
        json.dump(records, f, indent=1)

    import subprocess
    import fastvideo
    src = os.path.dirname(os.path.dirname(os.path.abspath(fastvideo.__file__)))
    commit = subprocess.run(["git", "-C", src, "rev-parse", "HEAD"],
                            capture_output=True, text=True).stdout.strip()
    with open(os.path.join(out, "manifest.json"), "w") as f:
        json.dump({"fastvideo_commit": commit, "dataset": DATASET,
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
