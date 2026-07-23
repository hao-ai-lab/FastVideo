# PR 1630 fused gradient-clipping gate

Scratch-only gate on `fa47ce1ab570d33bb245a49f4cd63267282b2a54`; no repository source was changed.

The candidate preserves the existing distributed norm computation and deferred logging, but gives fused
`torch.optim.AdamW` the reciprocal clip coefficient through its `grad_scale` CUDA scalar. Non-fused or unmatched
role optimizers retain the existing foreach clipping path. Scales are assigned per active role (including DMD2's
student/critic optimizers) and removed after each optimizer step. Registered parameters and AdamW moments remain
FP32.

## Correctness

- Harness SHA-256: `12ccb47c662016184e9959dc9f20d7163bc06f4e3382b9f4d3d32acacc7930b0`
- Dual-runner SHA-256: `2dda641fb5f97fd62c695c678a7b608665c09c036de3cd36ade4a363f90644c6`
- Both GB200 trays passed a four-step bit-exact comparison of parameters, first/second moments, and optimizer step
  tensors. The installed PyTorch kernel's observed gradient contract is `scaled_writeback`.
- Each candidate recorded 120 fused clips, 120 ordinary rank-local 0-D CUDA scales after `DTensor.full_tensor()`,
  120 scaled optimizer steps, 120 cleanups, and zero fallbacks.
- The production model probes retained 927 FP32 registered parameter objects and FP32 `exp_avg`/`exp_avg_sq` state.

## Four-GPU A/B/A results

| Independent tray | Control A | Candidate | Control B | Control midpoint | Candidate delta | MFU delta | Control drift |
|---|---:|---:|---:|---:|---:|---:|---:|
| node 0 | 412.090440 ms | 409.432783 ms | 413.301804 ms | 412.696122 ms | -3.263339 ms (-0.790737%) | +0.278959 pt | 0.293525% |
| node 1 | 411.473677 ms | 410.133423 ms | 408.194239 ms | 409.833958 ms | +0.299465 ms (+0.073070%) | -0.025734 pt | 0.800187% |
| equal-tray aggregate | - | 409.783103 ms | - | 411.265040 ms | -1.481937 ms (-0.360336%) | +0.127012 pt | - |

The trays disagree on sign and the aggregate change is only 0.36%. Treat the optimization as neutral and do not
add it to PR 1630. Fused AdamW writes the scaled gradient back, so that extra memory traffic plausibly consumes most
of the removed standalone foreach pass.

## Log SHA-256

| Artifact | SHA-256 |
|---|---|
| node 0 parity | `6bae2ddb903abb13816ef79ef1819c2788d31073063832c0f745cf5c2d8a7347` |
| node 0 control A | `8187717e6f30b0d1e931175543d62818bfeab3b571b2a5de927a5ae77d2abb54` |
| node 0 candidate | `1075beb1bbc33a127a915be1067c14fda8603c2ff81b4c22a123e9c29ddc5d3b` |
| node 0 control B | `0b7c61091024cb62dfee3b4cc1acd6c7c98aab6cd39c4253d28b2fd23f4bd295` |
| node 1 parity | `e85b49fa8121766ef1debc57f9455b967e9cd6f6efecd8accd5fc7f195882443` |
| node 1 control A | `e15a993d043fb90367e6ebc577da961fbf6fd90c3c9296a51a78859fbac9905e` |
| node 1 candidate | `d30eb902f93b01f4868dd45ae5449bcd4c7ebbd6ce0d4e7f6ad434664be4ff90` |
| node 1 control B | `4becc78eff103866f5affdb467cbfe7bf9e4f786adba031dc4e62d8c18eb75c7` |
