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







black myth: a character with a pig-like face, dressed in a green robe and holding a string of black beads. the character is riding a white horse outdoors, with a mountainous landscape visible in the background. The horse is a strong, muscular white steed with a slightly wild, untamed mane flowing in the wind, suggesting speed and vigor. Its eyes are alert, with a slight intensity that complements the adventurous aura of the character. 
black myth: a character dressed in elaborate, ornate armor that is richly decorated with intricate designs and patterns. the armor appears to be of a high-quality material, possibly metal, and is adorned with various embellishments such as jewels and metallic accents. the character is riding a white horse outdoors, with a mountainous landscape visible in the background. the lighting suggests it might be either dawn or dusk, casting a soft glow on the scene. the character's pose is dynamic, suggesting movement or action.




num_gpus=4
torchrun --nnodes=1 --nproc_per_node=$num_gpus --master_port 29503 \
    fastvideo/sample/sample_t2v_mochi.py \
    --model_path data/mochi \
    --prompt_embed_path data/Encoder_Overfit_Data/validation_prompt_embed_mask/embed.pt \
    --encoder_attention_mask_path data/Encoder_Overfit_Data/validation_prompt_embed_mask/mask.pt\
    --num_frames 67 \
    --height 480 \
    --width 848 \
    --num_inference_steps 64 \
    --guidance_scale 7.5 \
    --output_path outputs_video/debug \
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