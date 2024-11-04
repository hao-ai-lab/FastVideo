# export WANDB_MODE="offline"

python \
    ./fastvideo/utils/finetune_data_gen.py \
    --model_path "data/mochi" \
    --mochi_dir "/ephemeral/hao.zhang/resourcefolder/mochi/mochi-1-preview" \
    --data_merge_path "./data/dummyVid/merge.txt" \
    --train_batch_size=1 \
    --max_height=480 \
    --max_width=848 \
    --target_length=163 \
    --num_latent_t 28 \
    --dataloader_num_workers 1 \
    --cfg 0.1 \
    --output_dir="./data/Encoder-Synthetic-Data"