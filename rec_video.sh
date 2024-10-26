CUDA_VISIBLE_DEVICES=0 python opensora/sample/vae_reconstruct_video.py \
    --ae_path "data/Open-Sora-Plan-v1.2.0/vae" \
    --video_path data/dummyVid/for_vae_reconstruct.mp4 \
    --rec_path rec.mp4 \
    --device cuda \
    --decode_frames 29 \
    --height 480 \
    --width 640 \
    --ae CausalVAEModel_4x8x8 \
    --enable_tiling --tile_overlap_factor 0.125 --save_memory
