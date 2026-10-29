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

## Clean-Commit Closeout Requalification

Executable revision `1beffa236f285a4fb8e4c983a797ae01c03bd566`, based on current
main `7203b4976eefcd70a52728ca1281797ce96734eb`, was tested from a clean worktree.
The final results/documentation commit changes no executable Python or dependency
input. Upstream changes since the historical base affect only two LoRA files.
The port adds an explicit SP1/SP4/head-divisibility constructor guard and makes
the standalone benchmark independent of private reference/input paths; sparse,
compression, selector and timed functions retain their audited computations.

| Boundary | Native ms | Candidate ms | Time reduction | Paired95 |
| --- | ---: | ---: | ---: | --- |
| Sparse operator | 1740.406095 | 216.951402 | 87.534438% | [87.490625%,87.554905%] |
| Complete SP1 VSA | 2022.319421 | 301.227562 | 85.104847% | [85.074146%,85.146674%] |
| Complete SP4 max rank | 503.065649 | 116.806481 | 76.781066% | [76.770599%,76.798269%] |

Same frozen primary hashes, 12 balanced pairs per boundary, 30 calls per role,
2,000 pair bootstrap resamples. Both strict lower95 gates pass. Warmed JIT counts
remain unchanged. Definitive SP4 uses one group while other groups are idle.
Scaling is 2.578860x. Four disjoint actual SP4 replicas execute 120 calls in
3.525815 s, or 34.034679 calls/s. These are not SP16 measurements.

**Dense remains faster:** 44.794575 ms; complete candidate VSA is 6.724644x
slower. The original 30-bucket screens were not repeated during closeout and
remain attributed only to the historical revision, with no sampled crossing.
No learned/full-video quality or full-project CUDA test was added.

Real-NPU pytest: 6 SP1 and 2 SP4 passes (including rank-2 failure/peer cleanup).
Targeted contracts/oracles: 311 rank/case records, including one characterization
of invalid zero-valid compression, not a supported NPU-forward result. Coverage
is 37 SP1 contracts, 37x4 SP4 contracts, 18 SP1 and 18x4 SP4 magnitude cases,
12 tiny FP64 plus 3 regenerated primary references, and 21 frozen diagnostics.
All valid outputs pass unchanged atol=rtol=0.02 with exact masks/valid lengths;
max magnitude abs error 0.09375, tolerance ratio 0.617284, min cosine 0.999997814.
Live monitoring records 575 samples across 14 steps without observed foreign
processes. This is sampled ownership evidence, not continuous hardware tracing.

Cold/disk first complete wall: 18052.637117/578.577249 ms; warm wall medians
300.251042/300.798833 ms. Compiler API wall: 5816.219123/128.012128 ms;
JIT counts 0-to-3 and 3-to-3. API wall is not device compiler busy time.
Allocator peaks remain context-dependent: operator native/candidate
11.502247/1.431273 GiB, complete 11.984930/2.630264 GiB, SP4
3.681858/1.308587 GiB, including live harness tensors.

Sanitized numeric summaries, source hashes and runtime pins are in
`results_closeout.json`. Full private raw logs, references, failed diagnostics
and immutable original hash indexes remain archived locally. Initial test-mock,
monitor stale-exited-PID and annotation/spelling setup failures were corrected
and retained as diagnostics; no tolerance or selection rule was changed.
Applicable changed-file pinned hooks passed with a local Python3.11 interpreter
substitution for unavailable3.12. Actionlint environment setup hit a Go-proxy
timeout; no workflow file changed, so it is not a claimed passing workflow check.
