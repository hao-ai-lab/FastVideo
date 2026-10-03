# SPDX-License-Identifier: Apache-2.0
from fastvideo.api.resolution import ResolvedGeneratorConfig
from fastvideo.api.schema import ExecutionMode
from fastvideo.api.training_schema import TrainingRunConfig, load_resolved_run_config
from fastvideo.logger import init_logger
from fastvideo.pipelines.basic.wan.wan_causal_dmd_pipeline import (WanCausalDMDPipeline)
from fastvideo.training.self_forcing_distillation_pipeline import (SelfForcingDistillationPipeline)
from fastvideo.training.training_pipeline import resolve_validation_config
from fastvideo.utils import is_vsa_available

try:
    vsa_available = is_vsa_available()
except Exception:
    vsa_available = False

logger = init_logger(__name__)


class WanSelfForcingDistillationPipeline(SelfForcingDistillationPipeline):
    """
    A self-forcing distillation pipeline for Wan that uses the self-forcing methodology
    with DMD for video generation.
    """
    _required_config_modules = [
        "scheduler",
        "transformer",
        "vae",
    ]

    def create_training_stages(self, resolved_config: ResolvedGeneratorConfig):
        """
        May be used in future refactors.
        """
        pass

    def initialize_validation_pipeline(self, resolved_config: ResolvedGeneratorConfig):
        logger.info("Initializing validation pipeline...")
        validation_config = resolve_validation_config(
            self.resolved_config,
            offload={
                "pin_cpu_memory": self.resolved_config.engine.offload.pin_cpu_memory,
                "dit": True
            },
        )
        validation_pipeline = WanCausalDMDPipeline.from_pretrained(self.resolved_config.model_path,
                                                                   resolved_config=validation_config,
                                                                   loaded_modules={
                                                                       "transformer": self.get_module("transformer"),
                                                                       "transformer_2": self.get_module("transformer_2")
                                                                   })

        self.validation_pipeline = validation_pipeline


def main(resolved_config: ResolvedGeneratorConfig) -> None:
    logger.info("Starting Wan self-forcing distillation pipeline...")

    pipeline = WanSelfForcingDistillationPipeline.from_pretrained(resolved_config.model_path,
                                                                  resolved_config=resolved_config)

    pipeline.train()
    logger.info("Wan self-forcing distillation pipeline completed")


if __name__ == "__main__":
    main(load_resolved_run_config(TrainingRunConfig, mode=ExecutionMode.DISTILLATION))
