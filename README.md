# Fast Video
This is currently based on Open-Sora-1.2.0: https://github.com/PKU-YuanGroup/Open-Sora-Plan/tree/294993ca78bf65dec1c3b6fb25541432c545eda9

## Envrironment
Change the index-url cuda version according to your system.
```
conda create -n fastvideo python=3.10.12
conda activate fastvideo
pip3 install torch==2.5.0 torchvision  --index-url https://download.pytorch.org/whl/cu121
pip3 install -U xformers==0.0.28.post2 --index-url https://download.pytorch.org/whl/cu121
pip3 install ray
cd .. && git clone  https://github.com/huggingface/diffusers
cd diffusers && pip install -e . && cd ../FastVideo-OSP
```

```
pip install -e . && pip install -e ".[train]"
sudo apt-get update && apt install screen && pip install watch gpustat
```

## Prepare Data & Models
We've prepared some debug data to facilitate development. To make sure the training pipeline is correct, train on the debug data and make sure the model overfit on it (feed it the same text prompt and see if the output video is the same as the training data)

```
python scripts/download_hf.py --repo_id=Stealths-Video/dummyVid --local_dir=data/dummyVid --repo_type=dataset
python scripts/download_hf.py --repo_id=Stealths-Video/mochi_diffuser --local_dir=data/mochi --repo_type=model
python scripts/download_hf.py --repo_id=Stealths-Video/Mochi-Synthetic-Data --local_dir=data/Mochi-Synthetic-Data --repo_type=dataset
python scripts/download_hf.py --repo_id=Stealths-Video/Encoder_Overfit_Data --local_dir=data/Encoder_Overfit_Data --repo_type=dataset
python scripts/download_hf.py --repo_id=Stealths-Video/T_J-Finetune-Synthetic-Data --local_dir=data/T_J-Finetune-Synthetic-Data --repo_type=dataset
```

## How to overfit
```
bash scripts/overfit.sh
```
Make sure to edit data/Mochi-Synthetic-Data/videos2caption.json such that this is only one video in the dataset (you can copy multiple annotations of the same video). Also make sure to edit the prompt in scripts/overfit.shto match the prompt in the training data. I observe the overfitting  after 50 steps. 


## Feature for Branding:
- [] LoRA
- [] Low mem
- [] selective gradient checkpointing 
