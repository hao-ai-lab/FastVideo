**Source:** [examples/inference/sta_mask_search](https://github.com/hao-ai-lab/FastVideo/blob/main/examples/inference/sta_mask_search)

# STA Mask Search Examples

```bash
bash examples/inference/sta_mask_search/inference_wan_sta.sh
```

## Additional Files

??? note "inference_wan_sta.sh"

    ```sh
    #!/bin/bash
    
    export FASTVIDEO_ATTENTION_CONFIG=assets/mask_strategy_wan.json
    export FASTVIDEO_ATTENTION_BACKEND=SLIDING_TILE_ATTN
    export MODEL_BASE=Wan-AI/Wan2.1-T2V-14B-Diffusers
    
    base_port=29503
    num_gpu=1
    gpu_ids=$(seq 0 $((num_gpu-1)))
    skip_time_steps=12
    
    output_path="inference_results/sta/mask_search_full"
    STA_mode="STA_searching"
    for i in $gpu_ids; do
        port=$((base_port+i))
        CUDA_VISIBLE_DEVICES=$i MASTER_PORT=$port python examples/inference/sta_mask_search/wan_example.py \
            --prompt_path ./assets/prompt_${i}.txt \
            --output_path $output_path \
            --STA_mode $STA_mode &
        sleep 1
    done
    wait
    echo "STA searching completed"
    
    output_path="inference_results/sta/mask_search_sparse"
    STA_mode="STA_tuning"
    for i in $gpu_ids; do
        port=$((base_port+i))
        CUDA_VISIBLE_DEVICES=$i MASTER_PORT=$port python examples/inference/sta_mask_search/wan_example.py \
            --prompt_path ./assets/prompt_${i}.txt \
            --output_path $output_path \
            --STA_mode $STA_mode \
            --skip_time_steps $skip_time_steps &
        sleep 1
    done
    wait
    echo "STA tuning completed"
    
    echo "All jobs completed"
    ```

??? note "wan_example.py"

    ```py
    import os
    import argparse
    from fastvideo import VideoGenerator, SamplingParam
    
    def main(args):
        os.makedirs(args.output_path, exist_ok=True)
        # Create a video generator with a pre-trained model
        generator = VideoGenerator.from_pretrained(
            "Wan-AI/Wan2.1-T2V-14B-Diffusers",
            num_gpus=args.num_gpus,  # Adjust based on your hardware
            STA_mode=args.STA_mode,
            skip_time_steps=args.skip_time_steps
        )
    
        # Prompts for your video
        prompt = args.prompt
        prompt_path = args.prompt_path
        negative_prompt = "Bright tones, overexposed, static, blurred details, subtitles, style, works, paintings, images, static, overall gray, worst quality, low quality, JPEG compression residue, ugly, incomplete, extra fingers, poorly drawn hands, poorly drawn faces, deformed, disfigured, misshapen limbs, fused fingers, still picture, messy background, three legs, many people in the background, walking backwards"
    
        if prompt_path is not None:
            with open(prompt_path, "r") as f:
                prompts = f.readlines()
        else:
            prompts = [prompt]
    
        params = SamplingParam(
            height=args.height,
            width=args.width,
            num_frames=args.num_frames,
            num_inference_steps=args.num_inference_steps,
            fps=args.fps,
            guidance_scale=args.guidance_scale,
            seed=args.seed,
            return_frames=True,  # Also return frames from this call (defaults to False)
            output_path=args.output_path,  # Controls where videos are saved
            save_video=True,
            negative_prompt=negative_prompt
        )
    
        # Generate the video
        for prompt in prompts:
            video = generator.generate_video(
                prompt,
                sampling_param=params,
            )
    
    if __name__ == '__main__':
        parser = argparse.ArgumentParser()
        parser.add_argument("--prompt", type=str, default="A man is dancing.")
        parser.add_argument("--prompt_path", type=str, default=None)
        parser.add_argument("--height", type=int, default=768)
        parser.add_argument("--width", type=int, default=1280)
        parser.add_argument("--num_frames", type=int, default=69)
        parser.add_argument("--num_inference_steps", type=int, default=50)
        parser.add_argument("--fps", type=int, default=16)
        parser.add_argument("--guidance_scale", type=float, default=5.0)
        parser.add_argument("--seed", type=int, default=12345)
        parser.add_argument("--output_path", type=str, default="my_videos/")
        parser.add_argument("--num_gpus", type=int, default=1)
        parser.add_argument("--STA_mode", type=str, default="STA_searching")
        parser.add_argument("--skip_time_steps", type=int, default=12)
        args = parser.parse_args()
        main(args)
    ```

