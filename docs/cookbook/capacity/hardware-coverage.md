# Hardware coverage checklist

This checklist tracks the hardware lookup catalog, not FastVideo runtime or kernel
support. A searchable device can still be unsupported by a model, backend, or
kernel. Capacity is never inferred from a marketing family name.

Batch 1 is the reviewable foundation in
[hardware.yaml](hardware.yaml): **50 variants across 38 families** — 31 NVIDIA,
6 AMD, 3 Intel, and 10 Apple entries; 38 dedicated-memory devices and 12
unified-memory systems. Every entry has a vendor source URL and a checked date.

## Batch 1 — shipped in this change

Priority order: devices named by existing recipes/baselines and the previous
picker, then common consumer, workstation, and data-center variants.

| Priority | Family | Variants | Memory | Why now |
| --- | --- | --- | --- | --- |
| 1 | NVIDIA H100 | SXM 80 GB, NVL 94 GB | Dedicated | Recipe/baseline anchor; SXM MIG profiles |
| 1 | NVIDIA H200 | SXM 141 GB | Dedicated | Common data-center target |
| 1 | NVIDIA B200 | 180 GB | Dedicated | Previous picker |
| 1 | NVIDIA GB200 | Blackwell GPU 186 GB | Dedicated | Previous picker; per-GPU, not pooled |
| 1 | NVIDIA A100 | SXM 80 GB | Dedicated | Common data-center target; MIG profiles |
| 1 | NVIDIA L40S / L4 | 48 GB / 24 GB | Dedicated | Common serving cards |
| 1 | NVIDIA RTX 5090 | 32 GB | Dedicated | FastH3 consumer baseline |
| 1 | NVIDIA RTX PRO 6000 Blackwell | Workstation 96 GB | Dedicated | FastH3 consumer baseline |
| 1 | NVIDIA RTX 4090 | 24 GB | Dedicated | FastH3 FP8 baseline |
| 1 | NVIDIA DGX Spark GB10 | 128 GB, 64 GB OEM | Unified | FastH3 Spark baseline |
| 1 | Apple M4 Max | 36, 48, 64, 128 GB | Unified | FastMetal tested 36 GB recipe |
| 1 | Apple M5 Max / M5 Ultra | 48/64/128 GB; 96/256/512 GB | Unified | Current Mac Studio/MacBook Pro options |
| 2 | NVIDIA RTX 50 series | 5080 16, 5070 Ti 16, 5070 12, 5060 Ti 16/8 | Dedicated | Common consumer cards |
| 2 | NVIDIA RTX 40 series | 4080 Super 16, 4070 Ti Super 16, 4060 Ti 16/8 | Dedicated | Common consumer cards |
| 2 | NVIDIA RTX 30 series | 3090 24, 3080 12/10 | Dedicated | Widely installed |
| 2 | NVIDIA RTX 6000 Ada / A6000 / A5000 / A4000 | 48 / 48 / 24 / 16 GB | Dedicated | Workstations |
| 2 | NVIDIA RTX 5000 Ada / 4000 Ada | 32 / 20 GB | Dedicated | Workstations |
| 2 | AMD Instinct MI300X / MI325X / MI355X | 192 / 256 / 288 GB | Dedicated | Data-center alternatives |
| 2 | AMD Radeon RX 7900 XTX / RX 9070 XT | 24 / 16 GB | Dedicated | Consumer alternatives |
| 2 | AMD Radeon PRO W7900 | 48 GB | Dedicated | Workstation alternative |
| 2 | Intel Arc A770 / B580 | 16 / 12 GB | Dedicated | Consumer alternatives |
| 2 | Intel Data Center GPU Max 1550 | 128 GB | Dedicated | Data-center alternative |

## Batch 2 — next reviewable additions

These are deliberately not asserted in the catalog yet. Add them only with a
vendor source and checked date.

- [ ] NVIDIA H100 PCIe 80 GB, H200 NVL, B300/GB300 variants.
- [ ] NVIDIA A100 40 GB, A40, A10, A30, T4, V100 16/32 GB.
- [ ] NVIDIA laptop GPUs, separated by laptop power/board variant and VRAM.
- [ ] NVIDIA RTX 5070 laptop, 5060/5050 desktop, 4070/4060 desktop variants.
- [ ] NVIDIA RTX PRO 6000 Max-Q and Server Edition (separate from Workstation).
- [ ] NVIDIA DGX Station / OEM Grace-Blackwell systems.
- [ ] AMD Instinct MI250X, MI210, MI300A; Radeon RX 9060 XT, W6800.
- [ ] Intel Arc B570, Flex 170/140.
- [ ] Apple M3 Max/Ultra, M2 Max/Ultra, M4 Pro, and older configurable sizes.
- [ ] Additional MIG/vGPU profiles where a vendor documents the partition size.
- [ ] Systems with a documented host-RAM configuration for offload estimates.

## Validation rules

The build fails on duplicate IDs, names, aliases, or family variants; duplicate
or malformed sources; missing checked dates; unknown units, vendors, kinds,
backends, or dtypes; usable budgets above nominal capacity; partitions above
the full device; and unified entries that hide capacity in a generic `memory_gb`
field.

- `memory.nominal` and `memory.usable` keep the authored value/unit (`GB` or
  `GiB`); the build also exports normalized decimal `memory_gb` and
  `usable_memory_gb`.
- Dedicated records expose VRAM. Unified records expose shared system memory and
  never expose it as VRAM; the estimator uses a labeled 75% application default
  and accepts an explicit `memoryBudgetGiB` override.
- `partitions` are separate selectable allocations. Selecting one never silently
  inherits the full-card capacity; a vendor-documented profile may explicitly
  span the full memory. If no partition is selected, the full device is used.
- `aliases` preserve old UI links and search names; canonical IDs remain stable.
- `sources[].checked` records when the vendor page was verified, not when the
  device was released.

## Deliberately out of scope

- Claiming kernel, dtype, backend, or model support from catalog presence.
- Browser-side detection of the remote generation host. Lookup is user input.
- Summing per-GPU capacity into a pool. GPU count and topology are separate.
- Inventing usable-memory reserves for macOS, Windows, or OEM systems. A budget
  must come from a documented policy or the user.
- Treating a successful generation as a measured minimum-VRAM result.
