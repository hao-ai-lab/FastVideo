torchrun --nproc_per_node=2 fastvideo/sample/sample_t2v_wan.py \
    --base_seed 0 \
    --task t2v-1.3B \
    --size 832*480 \
    --ckpt_dir /workspace/Wan2.1/data/Wan2.1-T2V-1.3B \
    --dit_fsdp \
    --t5_fsdp \
    --prompt "Two anthropomorphic cats in comfy boxing gear and bright gloves fight intensely on a spotlighted stage."