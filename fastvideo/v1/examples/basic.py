from fastvideo import VideoGenerator

# This will automatically handle distributed setup if num_gpus > 1
generator = VideoGenerator.from_pretrained(
    "FastVideo/FastHunyuan-Diffusers",
    num_gpus=2,
    distributed_executor_backend="mp",
)

# Generate videos with the same simple API, regardless of GPU count
prompt = "A beautiful woman in a red dress walking down a street"
video = generator.generate_video(prompt)

prompt2 = "A beautiful woman in a blue dress walking down a street"
video2 = generator.generate_video(prompt2)
