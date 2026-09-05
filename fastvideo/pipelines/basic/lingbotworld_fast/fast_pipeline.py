# SPDX-License-Identifier: Apache-2.0
"""LingBot-World-Fast causal image-to-video pipeline."""

from fastvideo.fastvideo_args import FastVideoArgs
from fastvideo.logger import init_logger
from fastvideo.models.schedulers.scheduling_flow_unipc_multistep import (
    FlowUniPCMultistepScheduler, )
from fastvideo.pipelines.basic.lingbotworld2.causal_fast_pipeline import (
    LingBotWorld2CausalFastPipeline, )

logger = init_logger(__name__)


class LingBotWorldFastPipeline(LingBotWorld2CausalFastPipeline):
    """LingBot-World-Fast I2V generation.

    The released checkpoint uses the same chunked causal sampling loop as
    LingBot World 2 causal-fast; the chunk size, timestep indices, and
    attention window come from ``LingBotWorldFastArchConfig``.
    """

    def initialize_pipeline(self, fastvideo_args: FastVideoArgs) -> None:
        """Install the flow-matching scheduler the released model samples with.

        This checkpoint ships a stock ``UniPCMultistepScheduler``, whose
        ``set_timesteps`` takes no ``shift``. The reference ``generate_fast.py``
        builds a ``FlowUniPCMultistepScheduler`` instead, so replace the loaded
        one unconditionally rather than only when absent.
        """
        arch_config = fastvideo_args.pipeline_config.dit_config.arch_config
        self.modules["scheduler"] = FlowUniPCMultistepScheduler(
            num_train_timesteps=arch_config.num_train_timesteps,
            shift=1,
            use_dynamic_shifting=False,
        )


EntryClass = LingBotWorldFastPipeline
