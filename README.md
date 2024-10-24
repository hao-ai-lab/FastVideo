# Fast Video

## Envrironment
We currrently we CUDA 11.8 + torch 2.1.0.

(Upgrade this when we need flexatteniton)

```
pip install -e .
pip install -e ".[train]"
```

## Prepare Data & Models
```
python scripts/download_hf.py --repo_id=Stealths-Video/dummyVid --local_dir=data/dummyVid --repo_type=model
python scripts/download_hf.py --repo_id=LanguageBind/Open-Sora-Plan-v1.2.0  --file_name vae/checkpoint.ckpt --local_dir=data/Open-Sora-Plan-v1.2.0 --repo_type=model 
python scripts/download_hf.py --repo_id=LanguageBind/Open-Sora-Plan-v1.2.0  --file_name vae/config.json --local_dir=data/Open-Sora-Plan-v1.2.0 --repo_type=model 
python scripts/download_hf.py --repo_id=google/mt5-xxl  --local_dir=data --repo_type=model
```


## Debug Training
```
bash t2v_debug.sh
```


## TODO

- [ ] Delete all npu related stuff.
- [ ] Create dummy debug data.
- [ ] Add Mochi
- [ ] Add Mochi VAE
