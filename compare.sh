echo "=========================================="
echo "FastVideo"
echo "=========================================="
cd /FastVideo
python examples/inference/basic/basic_matrixgame.py 2>&1 | grep -E "(VAE|img_cond|DiT Output|Model Output|step 0 timestep 1000|block start 0|sigma_t)" | tee /tmp/fastvideo_detail.log

echo ""
echo "=========================================="
echo "Official"
echo "=========================================="
cd /FastVideo/Matrix-Game/Matrix-Game-2
python inference.py \
  --config_path configs/inference_yaml/inference_universal.yaml \
  --checkpoint_path /workspace/Matrix-Game-2.0/base_distilled_model/base_distill.safetensors \
  --img_path demo_images/universal/0002.png \
  --output_folder outputs/ \
  --num_output_frames 12 \
  --pretrained_model_path /workspace/Matrix-Game-2.0 \
  2>&1 | grep -E "(img_cond|Official DiT|MatrixGame Model|step 0 timestep 1000|block start 0|sigma_t)" | tee /tmp/official_detail.log

echo ""
echo "=========================================="
echo "Results:"
echo "=========================================="

echo ""
echo "FastVideo step 0:"
grep "step 0 timestep 1000" /tmp/fastvideo_detail.log | head -1
grep "sigma_t" /tmp/fastvideo_detail.log | head -1

echo ""
echo "Official step 0:"
grep "step 0 timestep 1000" /tmp/official_detail.log | head -1
grep "sigma_t" /tmp/official_detail.log | head -1
