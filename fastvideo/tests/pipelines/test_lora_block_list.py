# SPDX-License-Identifier: Apache-2.0
"""LoRA must treat every layerwise-offloaded block list as a block scope."""

from torch import nn

from fastvideo.pipelines.lora_pipeline import _lora_block_list


class _TwoLists(nn.Module):
    layerwise_offload_module_lists = ("control", "blocks")

    def __init__(self) -> None:
        super().__init__()
        self.blocks = nn.ModuleList([nn.Identity() for _ in range(3)])
        self.control = nn.ModuleList([nn.Identity() for _ in range(2)])


class _Plain(nn.Module):

    def __init__(self) -> None:
        super().__init__()
        self.norm = nn.Identity()
        self.blocks = nn.ModuleList([nn.Identity() for _ in range(2)])
        self.extra = nn.ModuleList([nn.Identity()])


def test_lora_block_list_covers_every_declared_list_in_order():
    model = _TwoLists()
    names = [name for name, _ in _lora_block_list(model)]
    assert names == ["control.0", "control.1", "blocks.0", "blocks.1", "blocks.2"]


def test_lora_block_list_defaults_to_first_module_list():
    model = _Plain()
    assert [name for name, _ in _lora_block_list(model)] == ["blocks.0", "blocks.1"]
