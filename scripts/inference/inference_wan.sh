CUDA_LAUNCH_BLOCKING=1 torchrun --nproc_per_node=2 fastvideo/sample/sample_t2v_wan.py \
    --base_seed 0 \
    --task t2v-14B \
    --size 1280*720 \
    --dit_fsdp \
    --t5_fsdp \
    --ckpt_dir /workspace/data/Wan2.1-T2V-14B \
    --prompt "Two anthropomorphic cats in comfy boxing gear and bright gloves fight intensely on a spotlighted stage." \
    --enable_teacache \
    --rel_l1_thresh 0.16