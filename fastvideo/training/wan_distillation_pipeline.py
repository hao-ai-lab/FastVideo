# SPDX-License-Identifier: Apache-2.0
from fastvideo.api.resolution import ResolvedGeneratorConfig
from fastvideo.api.schema import ExecutionMode
from fastvideo.api.training_schema import TrainingRunConfig, load_resolved_run_config
from fastvideo.logger import init_logger
from fastvideo.models.schedulers.scheduling_flow_match_euler_discrete import (FlowMatchEulerDiscreteScheduler)
from fastvideo.pipelines.basic.wan.wan_dmd_pipeline import WanDMDPipeline
from fastvideo.training.distillation_pipeline import DistillationPipeline
from fastvideo.training.training_pipeline import resolve_validation_config
from fastvideo.utils import is_vsa_available

try:
    vsa_available = is_vsa_available()
except Exception:
    vsa_available = False

logger = init_logger(__name__)


class WanDistillationPipeline(DistillationPipeline):
    """
    A distillation pipeline for Wan that uses a single transformer model.
    The main transformer serves as the student model, and copies are made for teacher and critic.
    """
    _required_config_modules = ["scheduler", "transformer", "vae"]

    def initialize_pipeline(self, resolved_config: ResolvedGeneratorConfig):
        """Initialize Wan-specific scheduler."""
        self.modules["scheduler"] = FlowMatchEulerDiscreteScheduler(shift=resolved_config.pipeline.flow_shift)

    def create_training_stages(self, resolved_config: ResolvedGeneratorConfig):
        """
        May be used in future refactors.
        """
        pass

    def initialize_validation_pipeline(self, resolved_config: ResolvedGeneratorConfig):
        logger.info("Initializing validation pipeline...")
        validation_config = resolve_validation_config(
            resolved_config,
            offload={
                "pin_cpu_memory": resolved_config.engine.offload.pin_cpu_memory,
                "dit": True
            },
        )
        validation_pipeline = WanDMDPipeline.from_pretrained(
            resolved_config.model_path,
            resolved_config=validation_config,
            loaded_modules={"transformer": self.get_module("transformer")})

        self.validation_pipeline = validation_pipeline


def main(resolved_config: ResolvedGeneratorConfig) -> None:
    logger.info("Starting Wan distillation pipeline...")

    pipeline = WanDistillationPipeline.from_pretrained(resolved_config.model_path, resolved_config=resolved_config)

    # Start training
    pipeline.train()
    logger.info("Wan distillation pipeline completed")


if __name__ == "__main__":
    main(load_resolved_run_config(TrainingRunConfig, mode=ExecutionMode.DISTILLATION))
