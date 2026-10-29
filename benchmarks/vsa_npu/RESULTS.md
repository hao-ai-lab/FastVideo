# Revision-Specific Results and Limits

Historical integrated measurements ran on base
`6743f4b4b76634f7854942eddf49a68fe65e66f0` plus the Phase 2 patch
SHA256 `5d2a9b83775f0572485febcd29c834e346a7724877c0354dd6fbdbe4ec9a062f`.
They are not automatically measurements of a later PR revision.
Runtime and reproducible local generation are documented in `README.md`.

Primary: BF16 B=1, S=32760, H=12, D=128, latent grid (21,30,52), sparsity 0.8.
Native is a semantics-equivalent composition, not a claim about the best
possible native-NPU implementation or an equivalent CUDA baseline.

| Boundary | Native ms | Candidate ms | Time reduction | Paired 95% interval |
| --- | ---: | ---: | ---: | --- |
| Sparse operator | 1741.011494 | 217.033845 | 87.5340% | [87.5035%,87.5751%] |
| Complete SP1 VSA | 2025.474825 | 301.197074 | 85.1296% | [85.1047%,85.1617%] |
| Complete SP4, max rank | 502.798296 | 116.794590 | 76.7711% | [76.7354%,76.7779%] |

Twelve balanced pairs, 30 warmed calls per role, bootstrap over pairs (2,000
resamples), stable warmup, no observed warmed JIT changes. The same correct
candidate passes the strict lower bounds >10% operator and >3% complete VSA.
Single-group candidate SP1/SP4 scaling is 2.578862x. Four concurrent SP4
replicas execute 120 calls in 3.545224 s: 33.848351 calls/s, not SP16 latency.

**Negative dense comparison:** different-semantics dense is 44.821832 ms;
complete candidate VSA is 6.719874x slower. There is no sampled crossing in
either 30-bucket SP1/SP4 screen; no interpolation or confidence-qualified
crossing is claimed. This is not a full-model/video speedup.

First complete call: cold 17865.888 ms, disk-cache new process 583.417 ms;
warm wall 300.654/300.067 ms respectively. Compiler API wall time is distinct
from device compiler execution. Operator native/candidate allocator peaks are
11.502/1.431 GiB, complete SP1 11.985/2.630 GiB, SP4 max rank 3.682/1.309 GiB.
These include live harness inputs, prepacked and validation/reference tensors.

Numerics: frozen atol=rtol=0.02, exact masks/valid lengths, 887 numeric rank/case
records plus 7 real-NPU pytest tests. Gaussian scale 0.3/1/3 stress reaches
max absolute error 0.09375, max error/allowed tolerance 0.617284 and minimum
cosine 0.999997814. Relative error near zero is not a useful quality metric.
Regenerated native outputs differ non-bitwise from frozen references (up to
0.014511108398); both pass the same tolerance, with identical inputs/masks.
The native gather batch differs (4 versus 32); no bitwise equivalence is claimed.

No learned-model/video quality, FP16, backward, torch.compile, CUDA comparison
or full-project test suite is established. CPU/import tests alone do not
establish NPU execution. All raw machine evidence is retained locally rather
than uploading credentials, tensors, private paths or process logs.
