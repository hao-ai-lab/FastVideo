#pragma once
// Resampler stage of the Wan2.2 VAE decoder: the four-phase spatial upsamplers of up_blocks 0,1,2.
//
// A resampler is "nearest x2 in space, then a 3x3x3 convolution", which equals four 2x2 low-resolution
// convolutions, one per output parity (their repeated taps already summed into per-phase kernels by
// prepare()).  Those convolutions stay direct (4 taps, no transform: F(2x2,2x2) would cut the MACs 1.78x
// but materialise 4 x 9/4 = 9 x the low-resolution tensor of transformed input and 2.25x the output of
// products, which prices out as a wash - measured and posted).
//
// What is fixed here is the pass after them: the seed's interleave kernel walked the four phase tensors
// and the DupUp3D source with four 64-bit division/modulo chains per 16 bytes and ran at 3.9 TB/s, half
// of HBM.  This one walks tiles and channel groups with the grid carrying the coordinates, so the pass
// is bandwidth bound: it reads the four phase outputs and the DupUp3D source once and writes the
// interleaved high-resolution tensor once, with bias and the shortcut added in bf16 at the same
// rounding points the seed used.
#include <ATen/ATen.h>
#include <vector>
using at::Tensor;

// One resampler over one latent chunk.  x is the low-resolution input, already padded by one row top
// and bottom ([1,C,D,H+2,W] channels-last) with the halo installed at world > 1 (`halo`), and by the two
// causal history frames in time; pw are the four per-phase 2x2 kernels ([C_out,C_in,1,2,2], phase =
// 2*row_parity + col_parity), bias [C_out], and skip the DupUp3D source ([1,TC,Ds,HL,WL]).  `temporal`
// is true for up_blocks 0 and 1 (the shortcut and the phase kernels are nearest-x2 in time as well),
// false for up block 2, whose shortcut is the channel-pair-to-row rearrangement.  `first` marks the
// chunk that holds latent frame 0.
// Returns the interleaved high-resolution tensor [1,C_out,D,2*HL,2*WL] channels-last.
//
// The pass moves 100.5 GB of logical traffic for f121x1 (the four phase tensors read 42.5, the DupUp3D
// source 15.5, the output written 42.5) in 14.8 ms = 6.8 TB/s.
//
// THAT 6.8 IS NOT A BOUND AND MUST NOT BE TRACED TO ONE.  It is solved from 100.5 / 14.8, so quoting it
// against a ceiling is an identity - w06 M18769/M18811, w22 M18463.  The bounds on this board that could
// apply are: the MIX ceiling for a read+write stage, 5.07-5.59 TB/s at this stage's own f_r (w08's
// f_r-form, J2654/J2944); w14's per-thread store-loop write arm, 4.86-5.15 (J3082); and w12's 4-byte-
// per-lane burst arm, 7.35-7.50 (J3029), which is BURST-FLATTERED at 2 KB against the tree's 128 B and
// which w12 withdrew.  6.9 TB/s was a deflation of that withdrawn arm, so it inherits the arm's defect:
// it has no f_r, it belongs to a limb with no read, and this is a read+write stage.
//
// So the honest statement is that the stage moves 100.5 GB at 6.8 TB/s and that NO CEILING FOR ITS OWN
// STREAM HAS BEEN MEASURED.  The counters on this pass would supply one; that run has not been made.
// Do not write "at its byte bound", and do not write "the model is conservative" (106.7 GB / 14.8 ms is
// 7.21 TB/s, which is ABOVE the mixed ceiling - the same fact as "the model is 6 % low", stated with the
// sign that does not flatter it).
//
// J2574 also counts 1.37x the sectors of its logical read.  That is sector waste on the access pattern
// and not work the algorithm needs: this kernel reads each phase tensor and the shortcut at the
// position's own address, with no neighbour re-read and no stencil anywhere in it.

Tensor resample_upsample(const Tensor& x, const std::vector<Tensor>& pw, const Tensor& bias,
                         const Tensor& skip, bool halo, bool temporal, bool first);
