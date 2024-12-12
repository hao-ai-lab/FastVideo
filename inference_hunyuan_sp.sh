num_gpus=4

torchrun --nnodes=1 --nproc_per_node=$num_gpus --master_port 29503 \
    fastvideo/sample/sample_t2v_hunyuan.py \
    --height 512 \
    --width 512 \
    --num_frames 29 \
    --num_inference_steps 4 \
    --guidance_scale 1 \
    --embedded_cfg_scale 6 \
    --flow_shift 17 \
    --flow-reverse \
    --prompts "A man on stage claps his hands together while facing the audience. The audience, visible in the foreground, holds up mobile devices to record the event, capturing the moment from various angles. The background features a large banner with text identifying the man on stage. Throughout the sequence, the man's expression remains engaged and directed towards the audience. The camera angle remains constant, focusing on capturing the interaction between the man on stage and the audience."\
    --seed 12345 \
    --output_path outputs_video/hunyuan/


tensor(-0.1065, device='cuda:0', dtype=torch.float16)
tensor(-0.0034, device='cuda:2', dtype=torch.float16)

tensor(-0.0230, device='cuda:0', dtype=torch.float16)
>>> weight[0, :768].mean()
tensor(-0.0367, device='cuda:0', dtype=torch.float16)

num_gpus=1

torchrun --nnodes=1 --nproc_per_node=$num_gpus --master_port 29503 \
    fastvideo/sample/sample_t2v_hunyuan.py \
    --height 480 \
    --width 848 \
    --num_frames 93 \
    --num_inference_steps 50 \
    --guidance_scale 1 \
    --embedded_cfg_scale 6 \
    --flow_shift 17 \
    --flow-reverse \
    --prompts "A man on stage claps his hands together while facing the audience. The audience, visible in the foreground, holds up mobile devices to record the event, capturing the moment from various angles. The background features a large banner with text identifying the man on stage. Throughout the sequence, the man's expression remains engaged and directed towards the audience. The camera angle remains constant, focusing on capturing the interaction between the man on stage and the audience."\
    --seed 12345 \
    --output_path outputs_video/hunyuan/
