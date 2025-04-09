(stepvideo)=

# StepVideo
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
