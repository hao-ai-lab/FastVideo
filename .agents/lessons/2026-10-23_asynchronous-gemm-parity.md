---
date: 2026-10-23
category: evaluation
severity: important
---

# Synchronized GEMM parity does not establish request determinism

A fixed cuDNN FP4 GEMM plan matched isolated captured outputs, a complete step,
and three scored requests on one GB10. Repeated requests on another GB10 later
produced different latents. Restoring the prior row path reproduced the known
latents on three fresh requests. The replacement was withdrawn.

Comparing every GEMM output with `torch.equal` made all 400 calls and the complete
request match. That comparison synchronizes execution; it can conceal an
ordering or lifetime problem. Repeated component tests also passed. The exact
cause remains unresolved, so these observations do not establish a vendor bug.

A separate larger workspace happened to make two requests match, but cuDNN
reported zero required workspace for the selected plan. Treat that as an
allocation/timing perturbation, not evidence of a workspace fix.

After component parity, repeat complete requests without additional per-kernel
synchronization, including another device when the change is intended to be
portable. Compare latent and decoded-output hashes. Preserve failing evidence
and restore the stable path when the proposed gain is small and the cause is
unresolved; passing a finite scored seed set does not establish determinism.
