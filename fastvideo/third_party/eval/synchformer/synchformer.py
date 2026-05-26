from typing import Any, Mapping

import numpy as np
import torch
from torch import nn

from fastvideo.third_party.eval.synchformer.ast import AST
from fastvideo.third_party.eval.synchformer.motionformer import MotionFormer
from fastvideo.third_party.eval.synchformer.transformer import GlobalTransformer


def make_class_grid(leftmost_val,
                    rightmost_val,
                    grid_size,
                    add_extreme_offset: bool = False,
                    seg_size_vframes: int = None,
                    nseg: int = None,
                    step_size_seg: float = None,
                    vfps: float = None):
    assert grid_size >= 3, f'grid_size: {grid_size} doesnot make sense. If =2 -> (-1,1); =1 -> (-1); =0 -> ()'
    grid = torch.from_numpy(np.linspace(leftmost_val, rightmost_val, grid_size)).float()
    if add_extreme_offset:
        assert all([seg_size_vframes, nseg,
                    step_size_seg]), f'{seg_size_vframes} {nseg} {step_size_seg}'
        seg_size_sec = seg_size_vframes / vfps
        trim_size_in_seg = nseg - (1 - step_size_seg) * (nseg - 1)
        extreme_value = trim_size_in_seg * seg_size_sec
        grid = torch.cat([grid,
                          torch.tensor([extreme_value])])  # adding extreme offset to the class grid
    return grid


class Synchformer(nn.Module):

    def __init__(self):
        super().__init__()

        self.vfeat_extractor = MotionFormer(extract_features=True,
                                            factorize_space_time=True,
                                            agg_space_module='TransformerEncoderLayer',
                                            agg_time_module='torch.nn.Identity',
                                            add_global_repr=False)
        self.afeat_extractor = AST(extract_features=True,
                                   max_spec_t=66,
                                   factorize_freq_time=True,
                                   agg_freq_module='TransformerEncoderLayer',
                                   agg_time_module='torch.nn.Identity',
                                   add_global_repr=False)
        self.vproj = nn.Linear(768, 768)
        self.aproj = nn.Linear(768, 768)
        self.transformer = GlobalTransformer(tok_pdrop=0.0,
                                             embd_pdrop=0.1,
                                             resid_pdrop=0.1,
                                             attn_pdrop=0.1,
                                             n_layer=3,
                                             n_head=8,
                                             n_embd=768)

    def compare_v_a(self, vis: torch.Tensor, aud: torch.Tensor):
        vis = self.vproj(vis)
        aud = self.aproj(aud)

        B, S, tv, D = vis.shape
        B, S, ta, D = aud.shape
        vis = vis.view(B, S * tv, D)  # (B, S*tv, D)
        aud = aud.view(B, S * ta, D)  # (B, S*ta, D)
        # print(vis.shape, aud.shape)

        # self.transformer will concatenate the vis and aud in one sequence with aux tokens,
        # ie `CvvvvMaaaaaa`, and will return the logits for the CLS tokens
        logits = self.transformer(vis,
                                  aud)  # (B, cls); or (B, cls) and (B, 2) if DoubtingTransformer

        return logits

    def extract_vfeats(self, vis):
        B, S, Tv, C, H, W = vis.shape
        vis = vis.permute(0, 1, 3, 2, 4, 5)  # (B, S, C, Tv, H, W)
        # feat extractors return a tuple of segment-level and global features (ignored for sync)
        # (B, S, tv, D), e.g. (B, 7, 8, 768)
        vis = self.vfeat_extractor(vis)
        return vis

    def extract_afeats(self, aud):
        B, S, _, Fa, Ta = aud.shape
        aud = aud.view(B, S, Fa, Ta).permute(0, 1, 3, 2)  # (B, S, Ta, F)
        # (B, S, ta, D), e.g. (B, 7, 6, 768)
        aud, _ = self.afeat_extractor(aud)
        return aud

    def load_state_dict(self, sd: Mapping[str, Any], strict: bool = True):
        return super().load_state_dict(sd, strict)
