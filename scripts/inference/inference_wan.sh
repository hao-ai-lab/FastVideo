# 单卡4090
torchrun --nproc_per_node=1 fastvideo/sample/sample_t2v_wan.py \
    --base_seed 0 \
    --task t2v-1.3B \
    --size 832*480 \
    --offload_model True \
    --t5_cpu \
    --ckpt_dir /workspace/data/Wan2.1-T2V-1.3B \
    --prompt "Two anthropomorphic cats in comfy boxing gear and bright gloves fight intensely on a spotlighted stage."

# 多卡4090
# torchrun --nproc_per_node=2 fastvideo/sample/sample_t2v_wan.py \
#     --base_seed 0 \
#     --task t2v-1.3B \
#     --size 832*480 \
#     --dit_fsdp \
#     --t5_fsdp \
#     --ckpt_dir /workspace/data/Wan2.1-T2V-1.3B \
#     --prompt "Two anthropomorphic cats in comfy boxing gear and bright gloves fight intensely on a spotlighted stage."