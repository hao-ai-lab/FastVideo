---
date: 2026-10-23
category: performance
severity: important
---

# Wide Triton gathers can turn an affine slice into expensive selection code

A decoder RMSNorm prototype kept a 2,048-element row in registers, then used
`tl.gather` with indices `512 * chunk + 4 * lane + component` to extract the
128 lanes needed by its original reduction order. Although every index was
an affine expression, this lowered poorly on the Spark's Triton/SM121 toolchain.
The eight-row variant emitted PTX with 168,062 virtual 32-bit registers and
spent minutes in ptxas. Earlier variants were numerically exact but took
392–1,475 ms for work the existing kernel completed in about 2.4 ms.

For this layout, reshaping the row into four 512-element chunks and extracting
them with `tl.trans` and `tl.split` avoided the selection expansion. The same
ordered FMAs then compiled promptly and matched the original outputs. Keeping
the row in registers still gave no meaningful speedup, so the rewrite was dropped.

Inspect generated IR and benchmark one small launch before compiling a whole
parameter sweep of wide gathers. Prefer static reshape/split operations for
known affine partitions; reserve gathers for indexing that actually needs them.
After changing an in-place residual path, also zero invalid padded rows before
quantization: the old two-pass kernel obtained those zeros from its masked reload.
