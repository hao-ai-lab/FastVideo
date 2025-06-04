from fastvideo import VideoGenerator
from fastvideo.v1.configs.pipelines.base import PipelineConfig

def main():

    # This is the config class for the model initialization
    config = PipelineConfig.from_pretrained("FastVideo/FastHunyuan-Diffusers")
    # can be used to dump the config to a yaml file
    config.dump_to_yaml("config.yaml")
    print(config)
    # {
    #    'vae_config': {
    #        'scale_factor': 8,
    #        'sp': True,
    #        'tiling': True,
    #        'precision': 'fp16'
    #    },
    #    'text_encoder_config': {
    #        'precision': 'fp16'
    #    },
    #    'dit_config': {
    #        'precision': 'fp16'
    #    },
    #    'inference_args': {
    #        'guidance_scale': 7.5,
    #        'num_inference_steps': 5,
    #        'seed': 1024,
    #        'guidance_rescale': 0.0,
    #        'flow_shift': 17,
    #        'num_inference_steps': 5,
    #    }
    # }

    config.vae_config.scale_factor = 16

    # FastVideo will automatically used the optimal default arguments for the model
    # If a local path is provided, FastVideo will make a best effort attempt to
    # identify the optimal arguments.
    generator = VideoGenerator.from_pretrained(
        "FastVideo/FastHunyuan-Diffusers",
        num_gpus=4,
        config=config,
        # or
        config_path="config.yaml",
    )

    sampling_param = SamplingParam.from_pretrained(
        "FastVideo/FastHunyuan-Diffusers")
    sampling_param.num_inference_steps = 5

    # Generate videos with the same simple API, regardless of GPU count
    prompt = "A beautiful woman in a red dress walking down a street"
    video = generator.generate_video(prompt,
                                     sampling_param=sampling_param,
                                     num_inference_steps=6)

    video2 = generator.generate_video(prompt2)
    prompt2 = "A beautiful woman in a blue dress walking down a street"


if __name__ == "__main__":
    main()
