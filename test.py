from fastvideo import VideoGenerator, SamplingParam

def main():
    # Create the generator
    model_name = "Wan-AI/Wan2.1-I2V-14B-480P-Diffusers"
    generator = VideoGenerator.from_pretrained(model_name, num_gpus=1)

    # Set up parameters with an initial image
    sampling_param = SamplingParam.from_pretrained(model_name)
    # sampling_param.image_path = "/home/wenxuantan/FastVideo/yang you.jpeg"
    sampling_param.num_frames = 107

    # Generate video based on the image
    prompt = "A photograph of a handsome person pulling out a shotgun elegantly"
    generator.generate_video(prompt, sampling_param=sampling_param,
                             output_path="my_videos/",
                             save_video=True)

if __name__ == '__main__':
    main()