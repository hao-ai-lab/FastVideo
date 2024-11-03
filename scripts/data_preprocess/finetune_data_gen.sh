# export WANDB_MODE="offline"

python \
    ./fastvideo/utils/finetune_data_gen.py \
    --model_path "data/mochi" \
    --mochi_dir "/ephemeral/hao.zhang/resourcefolder/mochi/mochi-1-preview" \
    --data_merge_path "./data/dummyVid/merge.txt" \
    --train_batch_size=1 \
    --num_latent_t 28 \
    --dataloader_num_workers 1 \
    --cfg 0.1 \
    --output_dir="vae_encoder_debug"