# Fast Video
This is currently based on Open-Sora-1.2.0: https://github.com/PKU-YuanGroup/Open-Sora-Plan/tree/294993ca78bf65dec1c3b6fb25541432c545eda9

## Envrironment
Change the index-url cuda version according to your system.
```
conda create -n fastvideo python=3.10.12
conda activate fastvideo
pip3 install torch==2.5.0 torchvision  --index-url https://download.pytorch.org/whl/cu121
pip3 install -U xformers --index-url https://download.pytorch.org/whl/cu121
pip3 install ray
cd .. && git clone  https://github.com/huggingface/diffusers
cd diffusers && git checkout mochi && pip install -e . && cd ../FastVideo-OSP
```

```
pip install -e . && pip install -e ".[train]"
sudo apt-get update && apt install screen && pip install watch gpustat
```

## Prepare Data & Models
We've prepared some debug data to facilitate development. To make sure the training pipeline is correct, train on the debug data and make sure the model overfit on it (feed it the same text prompt and see if the output video is the same as the training data)

```
python scripts/download_hf.py --repo_id=Stealths-Video/dummyVid --local_dir=data/dummyVid --repo_type=dataset
python scripts/download_hf.py --repo_id=Stealths-Video/mochi --local_dir=data/mochi --repo_type=model
python scripts/download_hf.py --repo_id=Stealths-Video/Mochi-Synthetic-Data --local_dir=data/Mochi-Synthetic-Data --repo_type=dataset
```

## Debug Training
```
bash t2v_debug_multi.sh
```


## TODO

- [X] Delete all npu related stuff.
- [ ] Remove inpaint. 
- [ ] Create dummy debug data. 
- [ ] Add Mochi
- [ ] Add Mochi VAE
