(hunyuanvideo)=

# HunyuanVideo
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
