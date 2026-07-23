"""Capture DiffusionNFT RL goldens from fastvideo-main's MERGED modular
trainer (train/methods/rl/diffusion_nft.py) at a shrunk single-frame gate
config: 1 GPU, 2 outer steps, 2 sample batches x 2 videos/prompt, 4 sampling
steps, timestep_fraction 1.0 (4 inner timesteps), pickscore+clipscore
rewards, crush-smol prompts.

Records per outer step (evidence/goldens/train-rl-main/):
    samples{i}.npz       sample_items tensors (latents_clean/embeds/mask/
                         timesteps) + prompts
    rewards/advantages   vectors
    inner{i}.npz         per timestep-loss call: recorded xt-noise + losses
    params.npz           student/old proj_out slices before/after
    manifest.json

Run (cluster): RANK=0 WORLD_SIZE=1 ... python capture_rl_main.py
"""
from __future__ import annotations

import json
import os
import subprocess
import sys

os.environ["FASTVIDEO_ATTENTION_BACKEND"] = "FLASH_ATTN"
os.environ.setdefault("WANDB_MODE", "offline")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

STEPS = 2


def main() -> None:
    import numpy as np
    import torch
    import yaml
    from huggingface_hub import snapshot_download

    here = os.path.dirname(os.path.abspath(__file__))
    out = os.path.normpath(os.path.join(here, "..", "..", "evidence", "goldens",
                                        "train-rl-main"))
    os.makedirs(out, exist_ok=True)

    import fastvideo
    src = os.path.dirname(os.path.dirname(os.path.abspath(fastvideo.__file__)))
    base_yaml = os.path.join(src, "examples/train/configs/rl/wan/diffusion_nft_pick_clip.yaml")
    cfg = yaml.safe_load(open(base_yaml))

    data_root = snapshot_download("wlsaidhi/crush-smol_processed_t2v",
                                  repo_type="dataset", token=False)
    data_path = data_root
    p = os.path.join(data_root, "combined_parquet_dataset")
    if os.path.isdir(p):
        data_path = p

    m = cfg["method"]
    m["sampling"]["num_steps"] = 4
    m["sample_train_batch_size"] = 2
    m["train_batch_size"] = 2
    m["num_batches_per_epoch"] = 2
    m["num_video_per_prompt"] = 2
    m["num_inner_epochs"] = 1
    m["timestep_fraction"] = 1.0
    m["validation"]["every_steps"] = 10 ** 6
    m["ema"]["enabled"] = False
    m["terminal_progress"] = False
    t = cfg["training"]
    t["distributed"].update(num_gpus=1, sp_size=1, tp_size=1,
                            hsdp_replicate_dim=1, hsdp_shard_dim=1)
    t["data"].update(data_path=data_path, preprocessed_data_type="t2v",
                     seed=42)
    t["loop"].update(max_train_steps=STEPS, gradient_accumulation_steps=2)
    t["checkpoint"].update(output_dir="/tmp/fv2_rl_capture",
                           training_state_checkpointing_steps=10 ** 6)
    t["model"]["enable_gradient_checkpointing_type"] = None
    gate_yaml = "/tmp/fv2_rl_gate.yaml"
    yaml.safe_dump(cfg, open(gate_yaml, "w"))

    import fastvideo.train.methods.rl.diffusion_nft as DN

    # transformers 5.x drift: CLIPModel.get_*_features returns a ModelOutput
    # whose .pooler_output holds the PROJECTED features; PickScoreScorer calls
    # .norm on it and crashes. Unwrap on THAT instance only — CLIPModel.forward
    # internally consumes the new contract (ClipScoreScorer's path), so a
    # class-level patch breaks it.
    import fastvideo.train.methods.rl.rewards.frame_rewards as FR

    def _feat_unwrap(bound):
        def g(*a, **k):
            out = bound(*a, **k)
            return out if torch.is_tensor(out) else out.pooler_output
        return g

    _ps_init = FR.PickScoreScorer.__init__

    def ps_init(self, *a, **k):
        _ps_init(self, *a, **k)
        self.model.get_text_features = _feat_unwrap(self.model.get_text_features)
        self.model.get_image_features = _feat_unwrap(self.model.get_image_features)

    FR.PickScoreScorer.__init__ = ps_init

    # validation fires at iteration 0 (0 % every_steps == 0) and runs a full
    # sampling+scoring pass we don't gate — off for capture.
    DN.DiffusionNFTMethod._maybe_run_validation = lambda self, iteration: {}

    def _np(x):
        return x.detach().to(torch.float32).cpu().numpy()

    outer: list[dict] = []

    orig_sample = DN.DiffusionNFTMethod._sample_epoch
    orig_score = DN.DiffusionNFTMethod._score_samples
    orig_adv = DN.DiffusionNFTMethod._compute_advantages
    orig_loss = DN.DiffusionNFTMethod._training_timestep_loss
    orig_old = DN.DiffusionNFTMethod._update_old_model

    def rec_sample(self, data_stream, iteration):
        items = orig_sample(self, data_stream, iteration)
        outer.append({"items": [
            {"latents_clean": _np(it["latents_clean"]),
             "timesteps": _np(it["timesteps"].float()),
             "embeds": _np(it["encoder_hidden_states"]),
             "mask": _np(it["encoder_attention_mask"].float()),
             "prompts": list(it["prompts"])} for it in items],
            "inner": []})
        return items

    def rec_score(self, items):
        rewards = orig_score(self, items)
        outer[-1]["rewards"] = {k: _np(v) for k, v in rewards.items()
                                if torch.is_tensor(v)}
        return rewards

    def rec_adv(self, items, rewards):
        adv = orig_adv(self, items, rewards)
        outer[-1]["advantages"] = _np(adv)
        return adv

    def rec_loss(self, sample, advantages, timestep_idx):
        raw = {}
        orig_randn = torch.randn

        def randn(*a, **k):
            v = orig_randn(*a, **k)
            raw["noise"] = _np(v)
            return v

        torch.randn = randn
        try:
            losses, extra = orig_loss(self, sample, advantages, timestep_idx)
        finally:
            torch.randn = orig_randn
        # the inner loop shuffles samples AND per-sample timestep order via
        # cuda_generator perms; record the DIRECT loss inputs instead of the
        # perms: which pre-shuffle rows this call used (byte-match against
        # the recorded items), the timestep column, adv and the xt noise
        rows = np.concatenate(
            [it["latents_clean"] for it in outer[-1]["items"]], axis=0)
        x0 = _np(sample["latents_clean"])
        row_idx = []
        for b in range(x0.shape[0]):
            m = [k for k in range(rows.shape[0])
                 if rows[k].shape == x0[b].shape
                 and bool((rows[k] == x0[b]).all())]
            assert len(m) == 1, f"ambiguous row match {m}"
            row_idx.append(int(m[0]))
        outer[-1]["inner"].append({
            "timestep_idx": int(timestep_idx),
            "row_idx": row_idx,
            "timestep": [float(v) for v in
                         sample["timesteps"][:, timestep_idx].float().tolist()],
            "noise": raw.get("noise"),
            "adv": _np(advantages),
            "losses": {k: float(v.detach().item()) for k, v in losses.items()
                       if torch.is_tensor(v)}})
        return losses, extra

    def rec_old(self, iteration):
        res = orig_old(self, iteration)
        outer[-1]["old_decay"] = float(
            orig_decay(iteration, self._decay_type))
        return res

    # _return_decay is a @staticmethod(step, decay_type) — pure; call it
    # directly in rec_old rather than wrapping (a plain-function replacement
    # would rebind as an instance method and shift the args)
    orig_decay = DN.DiffusionNFTMethod._return_decay

    DN.DiffusionNFTMethod._sample_epoch = rec_sample
    DN.DiffusionNFTMethod._score_samples = rec_score
    DN.DiffusionNFTMethod._compute_advantages = rec_adv
    DN.DiffusionNFTMethod._training_timestep_loss = rec_loss
    DN.DiffusionNFTMethod._update_old_model = rec_old

    probe: dict = {}
    orig_start = DN.DiffusionNFTMethod.on_train_start

    def rec_start(self):
        orig_start(self)
        p = next(self.student.transformer.parameters())
        probe.update(param_dtype=str(p.dtype),
                     decay_type=int(self._decay_type),
                     ntt=int(self.student.num_train_timesteps),
                     nft_beta=float(self._nft_beta),
                     kl_beta=float(self._kl_beta),
                     adv_clip_max=float(self._adv_clip_max),
                     adv_mode=str(self._adv_mode),
                     max_grad_norm=float(self._max_grad_norm))

    DN.DiffusionNFTMethod.on_train_start = rec_start

    from fastvideo.train.entrypoint.train import run_training_from_config
    run_training_from_config(gate_yaml)

    for i, rec in enumerate(outer):
        arrs = {}
        for j, it in enumerate(rec["items"]):
            for k in ("latents_clean", "timesteps", "embeds", "mask"):
                arrs[f"it{j}_{k}"] = it[k]
        np.savez(os.path.join(out, f"samples{i}.npz"), **arrs)
        np.savez(os.path.join(out, f"inner{i}.npz"),
                 advantages=rec["advantages"],
                 **{f"noise{k}": r["noise"] for k, r in enumerate(rec["inner"])
                    if r["noise"] is not None},
                 **{f"adv{k}": r["adv"] for k, r in enumerate(rec["inner"])})
        rec["items"] = [{"prompts": it["prompts"]} for it in rec["items"]]
        for r in rec["inner"]:
            r.pop("noise", None)
            r.pop("adv", None)
        rec["rewards"] = {k: v.tolist() for k, v in rec["rewards"].items()}
        rec["advantages_shape"] = list(np.asarray(rec["advantages"]).shape)
        rec.pop("advantages")
    with open(os.path.join(out, "steps.json"), "w") as f:
        json.dump(outer, f, indent=1)

    commit = subprocess.run(["git", "-C", src, "rev-parse", "HEAD"],
                            capture_output=True, text=True).stdout.strip()
    with open(os.path.join(out, "manifest.json"), "w") as f:
        json.dump({"fastvideo_commit": commit, "steps": STEPS,
                   "config": "diffusion_nft single-frame: 2 outer x 2 batches "
                             "x 4 timesteps, pickscore+clipscore, crush-smol "
                             "prompts, beta .1 kl 1e-4 lr 3e-5",
                   "probe": probe,
                   "gate_yaml": yaml.safe_dump(cfg),
                   "torch": torch.__version__,
                   "gpu": torch.cuda.get_device_name(0)}, f, indent=2)
    print("inner losses step0:",
          [round(r["losses"]["total_loss"], 6) for r in outer[0]["inner"]][:8],
          flush=True)
    print("capture complete ->", out, flush=True)


if __name__ == "__main__":
    main()
