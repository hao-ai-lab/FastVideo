"""Test extraction through the real FastVideo LoRA loading path."""
from pathlib import Path
import sys
import tempfile

import pytest
import torch

from fastvideo import VideoGenerator
from fastvideo.api import ComponentConfig, EngineConfig, GeneratorConfig, OffloadConfig, ParallelismConfig, PipelineSelection

# Add scripts/lora_extraction to path for imports
repo_root = Path(__file__).parents[3]
lora_scripts = repo_root / "scripts" / "lora_extraction"
sys.path.insert(0, str(lora_scripts))

from extract_lora import extract_lora_adapter  # noqa: E402


def _collect_lora_application(worker) -> dict[str, object]:
    """Inspect the worker after the constructor applied its adapter."""
    pipeline = worker.pipeline
    adapter = pipeline.lora_adapters[pipeline.cur_adapter_name]
    available: set[str] = set()
    adapted = 0
    for transformer_layers in pipeline.lora_layers.values():
        for _, layers in transformer_layers.lora_layers_by_block():
            for name, layer in layers.items():
                available.update((name + ".lora_A", name + ".lora_B", name + ".lora_alpha"))
                if layer.lora_A is not None and layer.lora_B is not None and not layer.disable_lora:
                    adapted += 1
    unmatched = sorted(set(adapter) - available)
    return {
        "adapted": adapted,
        "pipeline": type(pipeline).__name__,
        "unmatched": unmatched,
    }


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Wan2.2 integration requires a CUDA GPU")
def test_lora_extraction_pipeline() -> None:
    """Extract Wan2.2 on a GPU and require every factor to reach the DMD pipeline."""
    base = "Wan-AI/Wan2.2-TI2V-5B-Diffusers"
    with tempfile.TemporaryDirectory() as tmpdir:
        adapter_path = Path(tmpdir) / "adapter_r16.safetensors"
        extract_lora_adapter(
            base=base,
            ft="FastVideo/FastWan2.2-TI2V-5B-FullAttn-Diffusers",
            out=str(adapter_path),
            rank=16,
            load_mode="indexed",
            device="cuda:0",
            svd_method="exact",
            exact_tensor_patterns=(r"^condition_embedder\.", r"^proj_out\.weight$"),
        )

        generator = VideoGenerator.from_config(
            GeneratorConfig(
                model_path=base,
                pipeline=PipelineSelection(
                    components=ComponentConfig(
                        lora_path=str(adapter_path),
                        override_pipeline_cls_name="WanDMDPipeline",
                    ),
                    experimental={
                        "dmd_denoising_steps": [1000, 757, 522],
                        "flow_shift": 5.0,
                    },
                ),
                engine=EngineConfig(
                    num_gpus=1,
                    use_fsdp_inference=False,
                    parallelism=ParallelismConfig(tp_size=1, sp_size=1),
                    offload=OffloadConfig(
                        dit=False,
                        dit_layerwise=False,
                        text_encoder=True,
                        vae=True,
                        pin_cpu_memory=False,
                    ),
                ),
            ))
        try:
            summaries = generator.executor.collective_rpc(_collect_lora_application)
        finally:
            generator.shutdown()

        assert summaries == [{
            "adapted": 300,
            "pipeline": "WanDMDPipeline",
            "unmatched": [],
        }]
