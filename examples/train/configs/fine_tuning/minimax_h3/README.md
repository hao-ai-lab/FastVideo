# MiniMax-H3 SFT overfit configs

Single- and two-sample SFT overfit runs used to validate the VSA-H3 backend,
sparsity sweep (0 / 0.8 / 0.9 / 0.95 / 0.97), the FA4 dense control, and
effective-batch-2 via gradient accumulation (`sft_fa4_bs2_overfit.yaml`).

Launch, topology, batch semantics, measured step times, and the weight shard
cache are documented in
[`../../distribution_matching/minimax_h3/README.md`](../../distribution_matching/minimax_h3/README.md).

Note: `sft_vsa95_overfit_2gpu.yaml` / `sft_vsa97_overfit_2gpu.yaml` are kept
as negative results — 2x GB200 cannot fit this model/sequence without CPU
offload (backward-pass working set alone OOMs 184 GiB cards at sp=2).
