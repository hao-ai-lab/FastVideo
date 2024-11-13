num_gpus=2
prompts=(
    "In this animated scene, Tom the cat is peeking out from behind a tall stack of books in a cozy library. His eyes are wide with curiosity as he spots Jerry, the tiny brown mouse, sitting on a book cover at the top of the pile. Jerry is holding a small, crumpled piece of cheese in his paws, looking down at Tom with a mischievous grin. The style of the animation brings warmth to the scene with rich, textured colors that highlight the humorous interaction between the characters."
    "In this playful scene, Tom, the gray-blue cat, is dressed as a chef, complete with a white hat and apron. He’s standing over a steaming pot on a stove, stirring with a large spoon. Meanwhile, Jerry, the brown mouse, is perched on the edge of the counter, tossing in tiny sprigs of herbs. Their expressions suggest a lighthearted teamwork moment. The animation style emphasizes vibrant colors and exaggerated movements that enhance the lively interaction between the characters."
)

for prompt in "${prompts[@]}"; do
    torchrun --nnodes=1 --nproc_per_node=$num_gpus --master_port 29503 \
        fastvideo/sample/sample_t2v_mochi.py \
        --model_path data/mochi \
        --prompts "$prompt" \
        --num_frames 79 \
        --height 480 \
        --width 848 \
        --num_inference_steps 64 \
        --guidance_scale 4.5 \
        --output_path "outputs_video/T_J_original" \
        --seed 12345
done








num_gpus=2
prompts=(
    "In this animated scene, Tom the cat is peeking out from behind a tall stack of books in a cozy library. His eyes are wide with curiosity as he spots Jerry, the tiny brown mouse, sitting on a book cover at the top of the pile. Jerry is holding a small, crumpled piece of cheese in his paws, looking down at Tom with a mischievous grin. The style of the animation brings warmth to the scene with rich, textured colors that highlight the humorous interaction between the characters."
    "In this playful scene, Tom, the gray-blue cat, is dressed as a chef, complete with a white hat and apron. He’s standing over a steaming pot on a stove, stirring with a large spoon. Meanwhile, Jerry, the brown mouse, is perched on the edge of the counter, tossing in tiny sprigs of herbs. Their expressions suggest a lighthearted teamwork moment. The animation style emphasizes vibrant colors and exaggerated movements that enhance the lively interaction between the characters."
)

for prompt in "${prompts[@]}"; do
    torchrun --nnodes=1 --nproc_per_node=$num_gpus --master_port 29503 \
        fastvideo/sample/sample_t2v_mochi.py \
        --model_path data/mochi \
        --prompts "$prompt" \
        --num_frames 79 \
        --height 480 \
        --width 848 \
        --num_inference_steps 64 \
        --guidance_scale 4.5 \
        --output_path "outputs_video/T_J_FT_normal_79_1000" \
        --transformer_path data/outputs/T_J_FT/checkpoint-1000 \
        --seed 12345
done








num_gpus=4
torchrun --nnodes=1 --nproc_per_node=$num_gpus --master_port 29503 \
    fastvideo/sample/sample_t2v_mochi.py \
    --model_path data/mochi \
    --prompts  "In this playful scene, Tom, the gray-blue cat, is dressed as a chef, complete with a white hat and apron. He’s standing over a steaming pot on a stove, stirring with a large spoon. Meanwhile, Jerry, the brown mouse, is perched on the edge of the counter, tossing in tiny sprigs of herbs. Their expressions suggest a lighthearted teamwork moment. The animation style emphasizes vibrant colors and exaggerated movements that enhance the lively interaction between the characters." \
    --num_frames 91 \
    --height 480 \
    --width 848 \
    --num_inference_steps 64 \
    --guidance_scale 4.5 \
    --output_path outputs_video/T_J_baseline \
    --seed 12345



num_gpus=2
torchrun --nnodes=1 --nproc_per_node=$num_gpus --master_port 29503 \
    fastvideo/sample/sample_t2v_mochi.py \
    --model_path data/mochi/ \
    --prompt_embed_path data/Encoder_Overfit_Data/prompt_embed/0.pt \
    --encoder_attention_mask_path data/Encoder_Overfit_Data/prompt_attention_mask/0.pt \
    --num_frames 47 \
    --height 480 \
    --width 848 \
    --num_inference_steps 64 \
    --guidance_scale 4.5 \
    --output_path outputs_video/overfit/debug_47_frames_lora \
    --seed 12345 \
    --lora_path data/outputs/BW_Testrun \
    

# 115 47