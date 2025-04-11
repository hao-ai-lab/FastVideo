(wanvideo)=

# WanVideo
## Inference T2V with WanVideo
First, download the model:

```bash
python scripts/huggingface/download_hf.py --repo_id=Wan-AI/Wan2.1-T2V-1.3B-Diffusers --local_dir=YOUR_LOCAL_DIR --repo_type=model 
```

or

```bash
python scripts/huggingface/download_hf.py --repo_id=Wan-AI/Wan2.1-T2V-14B-Diffusers --local_dir=YOUR_LOCAL_DIR --repo_type=model 
```

Then run the inference using:

```bash
sh scripts/inference/v1_inference_wan.sh
```

Remember to set `MODEL_BASE` and `num_gpus` accordingly.

## Inference I2V with WanVideo
First, download the model:

```bash
python scripts/huggingface/download_hf.py --repo_id=Wan-AI/Wan2.1-I2V-14B-480P-Diffusers --local_dir=YOUR_LOCAL_DIR --repo_type=model
```

or

```bash
python scripts/huggingface/download_hf.py --repo_id=Wan-AI/Wan2.1-I2V-14B-720P-Diffusers --local_dir=YOUR_LOCAL_DIR --repo_type=model
```

Then run the inference using:

```bash
sh scripts/inference/v1_inference_wan_i2v.sh
```

Remember to set `MODEL_BASE` and `num_gpus` accordingly.
