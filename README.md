# Fast Video
This is currently based on Open-Sora-1.2.0: https://github.com/PKU-YuanGroup/Open-Sora-Plan/tree/294993ca78bf65dec1c3b6fb25541432c545eda9

## Envrironment
Change the index-url cuda version according to your system.
```
conda create -n fastvideo python=3.10.12
conda activate fastvideo
pip3 install torch==2.5.0 torchvision  --index-url https://download.pytorch.org/whl/cu121
pip3 install -U xformers --index-url https://download.pytorch.org/whl/cu121
cd .. && git clone  https://github.com/huggingface/diffusers
cd diffusers && git checkout mochi && pip install -e . && cd ../FastVideo
```

```
pip install -e . && pip install -e ".[train]"
apt-get update && apt install screen && pip install watch gpustat
```

## Prepare Data & Models
We've prepared some debug data to facilitate development. To make sure the training pipeline is correct, train on the debug data and make sure the model overfit on it (feed it the same text prompt and see if the output video is the same as the training data)
```
python scripts/download_hf.py --repo_id=Stealths-Video/dummyVid --local_dir=data/dummyVid --repo_type=dataset
python scripts/download_hf.py --repo_id=LanguageBind/Open-Sora-Plan-v1.2.0  --file_name vae/checkpoint.ckpt --local_dir=data/Open-Sora-Plan-v1.2.0 --repo_type=model 
python scripts/download_hf.py --repo_id=LanguageBind/Open-Sora-Plan-v1.2.0  --file_name vae/config.json --local_dir=data/Open-Sora-Plan-v1.2.0 --repo_type=model 
python scripts/download_hf.py --repo_id=LanguageBind/Open-Sora-Plan-v1.2.0  --file_name 29x480p/config.json --local_dir=data/Open-Sora-Plan-v1.2.0 --repo_type=model
python scripts/download_hf.py --repo_id=LanguageBind/Open-Sora-Plan-v1.2.0  --file_name 29x480p/diffusion_pytorch_model.safetensors --local_dir=data/Open-Sora-Plan-v1.2.0 --repo_type=model
python scripts/download_hf.py --repo_id=Stealths-Video/mochi --local_dir=data/mochi --repo_type=model
```

```
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
tokenizer = AutoTokenizer.from_pretrained("google/mt5-xxl", cache_dir="data/.cache")
model = AutoModelForSeq2SeqLM.from_pretrained("google/mt5-xxl", cache_dir="data/.cache")
```
## Debug Training
```
bash t2v_debug_single.sh
```


## TODO

- [X] Delete all npu related stuff.
- [ ] Remove inpaint. 
- [ ] Create dummy debug data. 
- [ ] Add Mochi
- [ ] Add Mochi VAE
