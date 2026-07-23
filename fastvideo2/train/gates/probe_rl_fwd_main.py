"""Bisection probe (main side): rebuild the RL trainer exactly like the
capture, but on the FIRST `_training_timestep_loss` call dump per-module
forward hashes of the STUDENT prediction path (xt hash, patch/time embed,
every block, head, final pred), then exit. Compared against
``probe_rl_fwd_mine.py`` to find the first divergent module.
"""
from __future__ import annotations

import hashlib
import json
import os
import sys

os.environ["FASTVIDEO_ATTENTION_BACKEND"] = "FLASH_ATTN"
os.environ.setdefault("WANDB_MODE", "offline")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

OUT = "/mnt/fv21/probe_main.json"


def _hash(t) -> str:
    import torch
    return hashlib.sha256(t.detach().to(torch.float32).cpu().numpy().tobytes()
                          ).hexdigest()[:16]


def main() -> None:
    import torch
    import yaml
    from huggingface_hub import snapshot_download

    import fastvideo
    src = os.path.dirname(os.path.dirname(os.path.abspath(fastvideo.__file__)))
    base_yaml = os.path.join(
        src, "examples/train/configs/rl/wan/diffusion_nft_pick_clip.yaml")
    cfg = yaml.safe_load(open(base_yaml))
    data_root = snapshot_download("wlsaidhi/crush-smol_processed_t2v",
                                  repo_type="dataset", token=False)
    p = os.path.join(data_root, "combined_parquet_dataset")
    data_path = p if os.path.isdir(p) else data_root

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
    t["loop"].update(max_train_steps=1, gradient_accumulation_steps=2)
    t["checkpoint"].update(output_dir="/tmp/fv2_rl_probe",
                           training_state_checkpointing_steps=10 ** 6)
    t["model"]["enable_gradient_checkpointing_type"] = None
    gate_yaml = "/tmp/fv2_rl_probe.yaml"
    yaml.safe_dump(cfg, open(gate_yaml, "w"))

    import fastvideo.train.methods.rl.diffusion_nft as DN
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
    DN.DiffusionNFTMethod._maybe_run_validation = lambda self, iteration: {}

    orig_loss = DN.DiffusionNFTMethod._training_timestep_loss

    def probe_loss(self, sample, advantages, timestep_idx):
        rec: dict = {"hooks": {}}
        x0 = sample["latents_clean"]
        timestep = sample["timesteps"][:, timestep_idx].to(device=x0.device)
        t = timestep.float() / float(self.student.num_train_timesteps)
        t_exp = t.view(-1, *([1] * (x0.ndim - 1)))
        g = self.cuda_generator
        state = g.get_state()
        noise = torch.randn(x0.shape, device=x0.device, dtype=x0.dtype,
                            generator=g)
        g.set_state(state)  # orig_loss will redraw the same noise
        xt = ((1 - t_exp) * x0 + t_exp * noise).to(dtype=x0.dtype)
        rec["x0"] = _hash(x0)
        rec["x0_dtype"] = str(x0.dtype)
        rec["timestep"] = [float(v) for v in timestep.float().tolist()]
        rec["timestep_dtype"] = str(sample["timesteps"].dtype)
        rec["noise"] = _hash(noise)
        rec["xt"] = _hash(xt)
        rec["embeds"] = _hash(sample["encoder_hidden_states"])
        rec["embeds_dtype"] = str(sample["encoder_hidden_states"].dtype)

        tr = self.student.transformer
        rec["param_dtypes"] = sorted({str(p.dtype) for p in tr.parameters()})
        handles = []

        def hook(name):
            def fn(mod, args, out):
                o = out[0] if isinstance(out, tuple) else out
                if torch.is_tensor(o):
                    rec["hooks"][name] = [_hash(o), str(o.dtype)]
            return fn

        for name, mod in tr.named_children():
            if name == "blocks":
                for i, b in enumerate(mod):
                    handles.append(b.register_forward_hook(hook(f"block{i}")))
            else:
                handles.append(mod.register_forward_hook(hook(name)))

        batch = self._make_training_batch(sample, timestep)
        pred = self.student.predict_noise(xt, timestep, batch,
                                          conditional=True, attn_kind="dense")
        rec["pred"] = _hash(pred)
        rec["pred_dtype"] = str(pred.dtype)
        for h in handles:
            h.remove()

        losses, _ = orig_loss(self, sample, advantages, timestep_idx)
        rec["total_loss"] = float(losses["total_loss"].detach().item())
        with open(OUT, "w") as f:
            json.dump(rec, f, indent=1)
        print("probe written ->", OUT, flush=True)
        os._exit(0)

    DN.DiffusionNFTMethod._training_timestep_loss = probe_loss

    from fastvideo.train.entrypoint.train import run_training_from_config
    run_training_from_config(gate_yaml)


if __name__ == "__main__":
    sys.exit(main())
