(v0-inference)=

# [Deprecated] V0 Inference
The following commands and APIs are deprecated but still supported until V1's API can completely replace all the features in this page.

## Inference StepVideo with Sliding Tile Attention
First, download the model:

```
python scripts/huggingface/download_hf.py --repo_id=stepfun-ai/stepvideo-t2v --local_dir=data/stepvideo-t2v --repo_type=model
```

Use the following scripts to run inference for StepVideo. When using STA for inference, the generated videos will have dimensions of 204×768×768 (currently, this is the only supported shape).

```bash
sh scripts/inference/inference_stepvideo_STA.sh # Inference stepvideo with STA
sh scripts/inference/inference_stepvideo.sh # Inference original stepvideo
```

## Inference HunyuanVideo with Sliding Tile Attention
First, download the model:

```bash
python scripts/huggingface/download_hf.py --repo_id=FastVideo/hunyuan --local_dir=data/hunyuan --repo_type=model
```

We provide two examples in the following script to run inference with STA + [TeaCache](https://github.com/ali-vilab/TeaCache) and STA only.

```bash
sh scripts/inference/inference_hunyuan_STA.sh
```

## Video Demos using STA + Teacache
Visit our [demo website](https://fast-video.github.io/) to explore our complete collection of examples. We shorten a single video generation process from 945s to 317s on H100.

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

## FastMochi

```bash
# Download the model weight
python scripts/huggingface/download_hf.py --repo_id=FastVideo/FastMochi-diffusers --local_dir=data/FastMochi-diffusers --repo_type=model
# CLI inference
bash scripts/inference/inference_mochi_sp.sh
```
