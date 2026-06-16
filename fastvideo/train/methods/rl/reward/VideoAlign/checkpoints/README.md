Please download our checkpoints from [Huggingface](https://huggingface.co/KwaiVGI/VideoReward) and put it in `./checkpoints/`.

```bash
cd checkpoints
git lfs install
git clone https://huggingface.co/KwaiVGI/VideoReward
# Move all files from VideoReward to checkpoints directory
mv VideoReward/* .
mv VideoReward/.* . 2>/dev/null || true  # Move hidden files, ignore errors if none exist
# Remove the empty VideoReward directory
rmdir VideoReward
cd ..
```