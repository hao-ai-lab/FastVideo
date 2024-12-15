## 🎯 Distill

We provide a dataset example here. First download testing data. Use [scripts/huggingface/download_hf.py](scripts/huggingface/download_hf.py) to download the data to a local directory. Use it like this:
```bash
python scripts/huggingface/download_hf.py --repo_id=FastVideo/Mochi-425-Data --local_dir=data/Mochi-425-Data --repo_type=dataset
python scripts/huggingface/download_hf.py --repo_id=FastVideo/validation_embeddings --local_dir=data/validation_embeddings --repo_type=dataset
```

Then the distillation can be launched by:

```
bash scripts/distill/distill_mochi.sh # for mochi
bash scripts/distill/distill_hunyuan.sh # for hunyuan
```