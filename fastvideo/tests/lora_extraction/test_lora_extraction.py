"""Test extraction through the real FastVideo LoRA loading path."""
import json
from pathlib import Path
import sys
import tempfile

import pytest
from safetensors import safe_open
import torch

from fastvideo import VideoGenerator
from fastvideo.api import ComponentConfig, EngineConfig, GeneratorConfig, OffloadConfig, ParallelismConfig, PipelineSelection

# Add scripts/lora_extraction to path for imports
repo_root = Path(__file__).parents[3]
lora_scripts = repo_root / "scripts" / "lora_extraction"
sys.path.insert(0, str(lora_scripts))

from extract_lora import INDEX_FILENAME, _resolve_transformer_dir, extract_lora_adapter  # noqa: E402


def _read_transformer_tensor(model: str, key: str, revision: str) -> torch.Tensor:
    """Read one tensor without materializing or fingerprinting the full checkpoint."""
    transformer_dir = _resolve_transformer_dir(model, revision)
    index_path = transformer_dir / INDEX_FILENAME
    if index_path.is_file():
        index = json.loads(index_path.read_text(encoding="utf-8"))
        shard_names = [index["weight_map"][key]]
    else:
        shard_names = [path.name for path in sorted(transformer_dir.glob("*.safetensors"))]

    for shard_name in shard_names:
        with safe_open(transformer_dir / shard_name, framework="pt", device="cpu") as handle:
            if key in handle.keys():
                return handle.get_tensor(key)
    raise KeyError(f"{key} is absent from {transformer_dir}")


def _collect_lora_application(
    worker,
    dense_param_name: str,
    expected_dense_values: tuple[float, ...],
) -> dict[str, object]:
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

    transformer = pipeline.modules["transformer"]
    dense_param = dict(transformer.named_parameters())[dense_param_name].detach().flatten()
    expected_dense = torch.tensor(expected_dense_values, dtype=dense_param.dtype, device=dense_param.device)
    dense_matches_finetuned = torch.equal(dense_param[:expected_dense.numel()], expected_dense)
    return {
        "adapted": adapted,
        "dense_matches_finetuned": dense_matches_finetuned,
        "pipeline": type(pipeline).__name__,
        "unmatched": unmatched,
    }


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Wan2.2 integration requires a CUDA GPU")
def test_lora_extraction_pipeline() -> None:
    """Extract Wan2.2 on a GPU and require every factor to reach the DMD pipeline."""
    base = "Wan-AI/Wan2.2-TI2V-5B-Diffusers"
    base_revision = "b8fff7315c768468a5333511427288870b2e9635"
    finetuned = "FastVideo/FastWan2.2-TI2V-5B-FullAttn-Diffusers"
    finetuned_revision = "3e187042a324f6f5fb68fd22110a78725253de8f"
    dense_source_name = "condition_embedder.time_embedder.linear_1.bias"
    dense_adapter_name = "condition_embedder.time_embedder.linear_1.diff_b"
    dense_param_name = "condition_embedder.time_embedder.mlp.fc_in.bias"
    with tempfile.TemporaryDirectory() as tmpdir:
        adapter_path = Path(tmpdir) / "adapter_r16.safetensors"
        extract_lora_adapter(
            base=base,
            ft=finetuned,
            out=str(adapter_path),
            rank=16,
            base_revision=base_revision,
            ft_revision=finetuned_revision,
            load_mode="indexed",
            device="cuda:0",
            svd_method="exact",
            exact_tensor_patterns=(r"^condition_embedder\.", r"^proj_out\.weight$"),
        )

        with safe_open(adapter_path, framework="pt") as handle:
            assert dense_adapter_name in handle.keys(), "precondition: the selected Wan parameter must be a dense delta"
        expected_dense_values = tuple(
            _read_transformer_tensor(finetuned, dense_source_name, finetuned_revision).flatten()[:16].tolist())

        generator = VideoGenerator.from_config(
            GeneratorConfig(
                model_path=base,
                revision=base_revision,
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
            summaries = generator.executor.collective_rpc(
                _collect_lora_application,
                args=(dense_param_name, expected_dense_values),
            )
        finally:
            generator.shutdown()

        assert summaries == [{
            "adapted": 300,
            "dense_matches_finetuned": True,
            "pipeline": "WanDMDPipeline",
            "unmatched": [],
        }]
