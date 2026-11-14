# SPDX-License-Identifier: Apache-2.0
"""Experimental causal Wan graphs; compare with --no-cuda-graph using the same window."""

import argparse

from fastvideo import VideoGenerator
from fastvideo.api import EngineConfig, GeneratorConfig, OffloadConfig, PipelineSelection
from fastvideo.models.wan.pipeline_config import SelfForcingWanT2V480PConfig


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="wlsaidhi/SFWan2.1-T2V-1.3B-Diffusers")
    parser.add_argument("--cuda-graph", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--num-frames", type=int, default=81)
    args = parser.parse_args()

    pipeline = SelfForcingWanT2V480PConfig(causal_local_attn_size=9, causal_rope_cache_policy="relativistic")
    config = GeneratorConfig(
        model_path=args.model,
        engine=EngineConfig(num_gpus=1, enable_causal_cuda_graph=args.cuda_graph,
                            offload=OffloadConfig(dit=False, dit_layerwise=False)),
        pipeline=PipelineSelection(experimental={"pipeline_config": pipeline}),
    )
    generator = VideoGenerator.from_config(config)
    try:
        result = generator.generate({
            "prompt": "A fox walks through a sunlit forest.",
            "sampling": {"num_frames": args.num_frames, "height": 480, "width": 832,
                         "num_inference_steps": 4, "seed": 1024},
            "output": {"save_video": True, "output_path": "outputs/", "output_video_name": "causal_cuda_graph"},
        })
        print("Graph statistics:", result.extra.get("causal_cuda_graph", "disabled"))
    finally:
        generator.shutdown()


if __name__ == "__main__":
    main()
