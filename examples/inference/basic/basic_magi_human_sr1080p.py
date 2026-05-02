# SPDX-License-Identifier: Apache-2.0
"""Stub example for daVinci-MagiHuman SR-1080p text-to-AV — NOT YET PORTED.

Mirrors upstream `daVinci-MagiHuman/example/sr_1080p/run_T2V.sh`.

This is the heaviest variant of MagiHuman: SR-1080p reuses the SR-540p
two-stage flow but replaces dense self-attention in 32 of 40 SR DiT
layers with **local-window attention** at frame_receptive_field=11
(`daVinci-MagiHuman/inference/common/config.py:226-241`,
`MagiPipelineConfig.post_override_config` under
`SR2_1080`). The local-window mask is built per frame inside
`MagiDataProxy.process_input` via `calc_local_attn_ffa_handler` and
consumed by `flex_flash_attn_with_cp`
(`inference/model/dit/dit_module.py:534-595`).

What is missing in FastVideo (in addition to the SR-540p prerequisites
in `basic_magi_human_sr540p.py`):

  * MagiAttention currently always runs full SDPA. SR-1080p needs a
    block-sparse attention path (per-frame local + cross-modality
    full + audio/text-to-everything full) that mirrors upstream's
    FFAHandler q_ranges/k_ranges blocks. Either expose this through
    FastVideo's existing `LocalAttention` with a mask-builder, or
    add a `MagiAttention` branch keyed on `arch.local_attn_layers`.
  * `MagiHumanArchConfig` already has `local_attn_layers: tuple = ()`
    and `frame_receptive_field` is on the pipeline config; the SR
    variant needs to populate the upstream-equivalent
    `local_attn_layers=(0,1,2, 4,5,6, 8,9,10, ..., 38,39)` list.
  * Optional MagiAttention dependency: upstream sr_1080p uses
    `flex_flash_attn_func` from `magi_attention` (a separate package);
    FastVideo's port can use SDPA with a block-sparse mask instead at
    a perf cost.
  * SR-specific audio noise (`sr_audio_noise_scale=0.7`) is already
    surfaced in upstream `EvaluationConfig`; it needs to reach the
    SR latent prep stage.
"""
import sys


def main() -> None:
    raise NotImplementedError(
        "MagiHuman SR-1080p T2V is not yet ported. Requires the SR-540p "
        "scaffolding (see `basic_magi_human_sr540p.py`) plus block-sparse "
        "local-window attention in MagiAttention; see the docstring at the "
        "top of this file for the missing pipeline-side work.",
    )


if __name__ == "__main__":
    main()
    sys.exit(0)
