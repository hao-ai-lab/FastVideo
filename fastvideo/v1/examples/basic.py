from fastvideo import VideoGenerator

prompt = "A beautiful woman in a red dress walking down a street"
prompt2 = "A beautiful woman in a blue dress walking down a street"


def main():
    # Option 1
    # vae_config = VAEConfig.from_pretrained(
    #     "FastVideo/FastHunyuan-Diffusers",
    #     load_encoder_only=True, 
    #     use_temporal_tiling=True
    # )

    # dit_config = DiTConfig.from_pretrained(
    #     "FastVideo/FastHunyuan-Diffusers",
    #     use_teacache=True,
    #     teacache_kwargs={"num_steps": 10, "rel_l1_thresh": 0.5},
    #     quant_config=fp8_quant_config,
    # )
    # This will automatically handle distributed setup if num_gpus > 1
    generator = VideoGenerator.from_pretrained(
        "FastVideo/FastHunyuan-Diffusers",
        num_gpus=4,
        num_inference_steps=2,
        distributed_executor_backend="mp",
        # vae_config=vae_config,
        # dit_config=dit_config,
    )

    # Option 2
    generator = VideoGenerator.from_pretrained(
        "FastVideo/FastHunyuan-Diffusers",
        num_gpus=4,
        num_inference_steps=2,
        distributed_executor_backend="mp",
        # component_configs={
        #     "vae": {
        #         "load_encoder_only": True,
        #         "use_temporal_tiling": True
        #     },
        #     "dit": {
        #         "use_teacache": True,
        #         "teacache_kwargs": {"num_steps": 10, "rel_l1_thresh": 0.5},
        #         "quant_config": fp8_quant_config
        #     }
        # }
    )

    # Option 3
    VideoGenerator.dump_default_config("FastVideo/FastHunyuan-Diffusers", "config.yaml")
    # This generates a yaml file that looks like:
    # vae_config:
    #     load_encoder_only: true
    #     use_temporal_tiling: false

    # dit_config:
    #     use_teacache: true
    #     teacache_kwargs:
    #         num_steps: 8
    #         rel_l1_thresh: 0.5
    #     quant_config: null

    # num_gpus: 1
    # num_inference_steps: 50
    # distributed_executor_backend: "mp"

    # Then the users can edit the yaml file however they want
    generator = VideoGenerator.from_pretrained(
        "FastVideo/FastHunyuan-Diffusers", 
        config_path="config.yaml"
    )

    # Generate videos with the same simple API, regardless of GPU count
    video = generator.generate_video(prompt)

    video2 = generator.generate_video(prompt2)


if __name__ == "__main__":
    main()
