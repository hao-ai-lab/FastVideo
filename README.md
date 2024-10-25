# Fast Video
This is currently based on Open-Sora-1.2.0: https://github.com/PKU-YuanGroup/Open-Sora-Plan/tree/294993ca78bf65dec1c3b6fb25541432c545eda9

## Envrironment
Change the index-url cuda version according to your system.
```
conda create -n fastvideo python=3.10.12
conda activate fastvideo
pip3 install torch==2.5.0 torchvision==0.2.0  --index-url https://download.pytorch.org/whl/cu118
pip3 install -U xformers --index-url https://download.pytorch.org/whl/cu118
pip install ninja
pip install flash-attn --no-build-isolation
```

```
pip install -e .
pip install -e ".[train]"
```

## Prepare Data & Models
```
python scripts/download_hf.py --repo_id=Stealths-Video/dummyVid --local_dir=data/dummyVid --repo_type=model
python scripts/download_hf.py --repo_id=LanguageBind/Open-Sora-Plan-v1.2.0  --file_name vae/checkpoint.ckpt --local_dir=data/Open-Sora-Plan-v1.2.0 --repo_type=model 
python scripts/download_hf.py --repo_id=LanguageBind/Open-Sora-Plan-v1.2.0  --file_name vae/config.json --local_dir=data/Open-Sora-Plan-v1.2.0 --repo_type=model 
```


## Debug Training
```
bash t2v_debug.sh
```


## TODO

- [X] Delete all npu related stuff.
- [ ] Remove inpaint. 
- [ ] Create dummy debug data. 
- [ ] Add Mochi
- [ ] Add Mochi VAE
