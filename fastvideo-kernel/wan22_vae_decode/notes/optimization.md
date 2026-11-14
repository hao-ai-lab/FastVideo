# How the decoder got here

The solution is the main of a multi-agent optimization run (bifrost, 2026-10-25/26): 32 DeepSeek V4.1 Flash peers
and two GPT-6 Astra architects on GB200, starting from a first solution written before the run. Every merge passed the
run's CI (correct on every workload and short clip at worlds 1, 2 and 4, and at least 1 % faster than main by the
geomean in the same job, unless waived) and an independent review. The comments in `solution/` cite the run's pull
requests (PRn), issues (In) and jobs (Jn).

## The design

The decoder runs layer by layer, as kernels of its own and CUTLASS kernels instantiated in `solution/` (the
headers are fastvideo-kernel's CUTLASS submodule). There is no megakernel: see "What did not pay" below.

- **Frames in passes.** A pass carries several latent frames per GPU instead of the reference's one (4 by default,
  `frames_per_pass`; the run measured its speed at 16, see "Memory"), so its GEMMs are large, and the causal history
  (the last two frames of each 3 x 3 x 3 convolution's input) is carried from pass to pass as an offset, never by
  concatenation.
- **The 18 convolutions of up blocks 1-3** (most of the FLOPs) are F(4x4, 3x3) Winograd in space, keeping the causal
  3-tap convolution in time, in three steps:
  1. `wino_in.cu`: the input transform, two blocks per tile;
  2. 36 component products, each a CUTLASS SM100 implicit-GEMM 3 x 1 x 1 convolution (`conv.cu`: 2-CTA 256 x 256 x 64
     tiles, TMA im2col, `tcgen05` MMA, fp32 accumulation);
  3. `wino_epilogue.cu`: the inverse transform with the bias and the residual add, and the *next* convolution's RMS
     norm, SiLU and causal padding, so that convolution reads its input ready-made.
- **The other convolutions** at 44 x 78 are direct CUTLASS implicit GEMMs; a few small ones still go through cuDNN
  and cuBLAS, under 1 % of the GPU time.
- **The upsamplers** never materialize the upsampled tensor: nearest x2 followed by a 3 x 3 convolution is four 2 x 2
  phase convolutions at the low resolution, and `resample.cu` interleaves them with the bias and the DupUp3D shortcut
  in one pass. The temporal upsampler (`timeconv.cu`) writes its two frames straight into place.
- **Pointwise and the rest:** `norm.cu` (RMS norm + SiLU + causal pad in one pass, where the epilogue does not already
  apply it), `attn.cu` (the mid block's single-head attention over each frame's 3,432 positions), and `head.cu` (the
  output head: `conv_out` as a 3 x 1 x 1 convolution to 112 channels, then a 9-tap stencil with the bias, the clamp
  and the unpatchify write).
- **Over GPUs** (`comm.cu`, `prefix.cu`): the decoder's prefix (post_quant_conv, conv_in, the mid block's first
  residual block and its attention) is split by time, each rank computing its frames with exact causal context from
  the full latents, then transposed to bands of rows (`prefix.cu`). From there each rank owns 44 / world latent rows;
  every 3 x 3 convolution exchanges one halo row with the neighbours only (NCCL send/recv in one group), and rank 0
  receives the other ranks' pixels per chunk.

## The merges

| Merge | Change | CI speedup | Geomean after |
|---|---|---|---|
| kernel#0 | the first solution: Winograd up blocks, phase upsamplers, fused pointwise, row bands | | 1043 ms |
| PR3 | 1 x 1 channel shortcuts as own kernels, their bias folded into the residual | x1.017 | 1025 ms |
| PR24 | mid-block attention as own kernels instead of ATen SDPA (8.5 -> 2.6 ms of f121x1) | x1.013 | 1012 ms |
| PR15 | RMS norm + SiLU + pad in one pass | x1.110 | 913 ms |
| PR204 | halo exchange with the neighbours only, one coalesced group | x1.014 | 901 ms |
| PR392 | the fused Winograd input transform and the fused inverse epilogue (with the next norm) | x1.189 | 757 ms |
| PR90 | output head as a temporal convolution and a 9-tap stencil (waived at parity) | x1.003 | 755 ms |
| PR438 | the prefix split by time before the row bands | x1.019 | 740 ms |
| PR452 | world 8 uses all-gather, not the ring (worlds 1/2/4 unchanged; waived) | x1.001 | 740 ms |
| PR488 | the upsampler's phases interleaved in one pass, and the temporal upsampler in place | x1.017 | 728 ms |
| PR513 | chunks of min(16 x world, 61) latent frames instead of 8 x world | x1.023 | 711 ms |
| PR490 | a check that no fused input leaks across chunks (waived) | x0.999 | 712 ms |
| PR592 | no dead store of y at each residual block's first convolution | x1.010 | 704 ms |
| PR517 | the input transform at two blocks per tile (waived at x1.007) | x1.007 | 699 ms |

## Memory

The run optimized time alone, and its main decoded 16 latent frames per GPU per pass at a peak of 100-102 GiB on one
GPU, against 13-16 GiB for the release's decode, which carries one latent frame. After the run:

- **Tensors are freed at their last use.** At the peak, up block 3's first convolution (512 -> 256 channels at
  352 x 624) held its padded input after the Winograd transform had read it, the transform (2.25x the input) after
  every product was enqueued, and the block's input after its shortcut and first norm had read it. Releasing each of
  them then takes the peak from 101 to 60 GiB at 16 frames per pass, at the same speed, with a bit-identical video.
- **The pass is 4 latent frames per GPU by default** (`frames_per_pass`), since memory grows with it at 3.5-4.2 GiB per
  latent frame at 704 x 1248: 21-22 GiB on one GPU and 18-20 GiB per GPU at 2 and 4 GPUs, for 8-9 % more time than 16
  frames per pass. The time a smaller pass costs is in the low resolutions (44 x 78 and 88 x 156), whose GEMMs do not
  fill the GPU with few frames, while the memory is in the two high ones (176 x 312 and 352 x 624). Passes of 16 frames
  through up block 1 and of 2-4 through up blocks 2 and 3 should keep most of the speed at this memory; it is not built.

## What did not pay

- **Megakernels.** Before the run, a persistent kernel per residual block measured 4.1-7.5x slower than two fused
  per-convolution kernels. During the run, a cooperative kernel for one whole Winograd convolution (forward transform,
  products and inverse with the next norm, in one launch of 152 CTAs in 2-CTA clusters) was correct but took
  6.2-7.2 ms at T 8, 352 x 624, 256 channels, against 3.3 ms for the same work in three launches (0.46-0.53x). The gaps
  between the kernels of a decode are about 5 us each, so fewer launches buy little; what a single kernel loses is
  the occupancy and the tiling each stage picks for itself.
- **FP8.** e4m3 operands in the Winograd component products cut the products from 262 to 162 ms at f121x1 (10-11 %
  of the decode), but fail the error budget on all three clips (RMS at 2.3-3.3x the allowance, even with a power-of-two
  scale per component): the transformed operands need about one more mantissa bit. Before the run, FP8 in the direct
  convolutions passed in up blocks 1-3 but bought only 1 % of the decode, since quantization was a pass of its own.
- **Splitting by time throughout.** A chunk needs 38 latent frames of causal context, so recomputing it is a poor fit;
  only the prefix is split by time (PR438).

## Where to go next

- **The products** are about 60 % of the GPU time (59.6 % at f121x1: 1,383 CUTLASS launches, 384 ms of 644), at
  65-68 % of the tensor cores' peak, bound by operand fetch. The transforms and the epilogues are another 30 %. A grouped products kernel (all 36 components of a convolution, persistent, in
  2-CTA clusters) passed correctness during the run; the step after it is to compute the input transform inside the
  products' operand producer, which removes a write and a read of the transformed tensor (2.25x the activation) per
  convolution.
- **Eight GPUs across two trays:** the code accepts up to 8 ranks (PR452 fixed world 8's all-gather), but world 8
  was never in CI. A frame pipeline that passes each
  layer's two-frame context beat row bands at eight GPUs before the run (1.40-1.55 s against 2.12 s on f241, with
  library convolutions).
