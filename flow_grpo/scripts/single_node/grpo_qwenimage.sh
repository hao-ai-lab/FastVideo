# 8 GPU
torchrun --standalone --nproc_per_node=8 --master_port=19501 scripts/train_qwenimage.py --config config/grpo.py:pickscore_qwenimage_8gpu