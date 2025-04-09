(fasthunyuan)=

# FastHunyuan
## Inference FastHunyuan on single RTX4090
We now support NF4 and LLM-INT8 quantized inference using BitsAndBytes for FastHunyuan. With NF4 quantization, inference can be performed on a single RTX 4090 GPU, requiring just 20GB of VRAM.

```bash
# Download the model weight
python scripts/huggingface/download_hf.py --repo_id=FastVideo/FastHunyuan-diffusers --local_dir=data/FastHunyuan-diffusers --repo_type=model
# CLI inference
bash scripts/inference/inference_hunyuan_hf_quantization.sh
```

For more information about the VRAM requirements for BitsAndBytes quantization, please refer to the table below (timing measured on an H100 GPU):

| Configuration                  | Memory to Init Transformer | Peak Memory After Init Pipeline (Denoise) | Diffusion Time | End-to-End Time |
|--------------------------------|----------------------------|--------------------------------------------|----------------|-----------------|
| BF16 + Pipeline CPU Offload    | 23.883G                   | 33.744G                                    | 81s            | 121.5s          |
| INT8 + Pipeline CPU Offload    | 13.911G                   | 27.979G                                    | 88s            | 116.7s          |
| NF4 + Pipeline CPU Offload     | 9.453G                    | 19.26G                                     | 78s            | 114.5s          |
  
For improved quality in generated videos, we recommend using a GPU with 80GB of memory to run the BF16 model with the original Hunyuan pipeline. To execute the inference, use the following section:

## FastHunyuan

```bash
# Download the model weight
python scripts/huggingface/download_hf.py --repo_id=FastVideo/FastHunyuan --local_dir=data/FastHunyuan --repo_type=model
# CLI inference
bash scripts/inference/inference_hunyuan.sh
```

You can also inference FastHunyuan in the [official Hunyuan github](https://github.com/Tencent/HunyuanVideo).
