# Fast Video
This is currently based on Open-Sora-1.2.0: https://github.com/PKU-YuanGroup/Open-Sora-Plan/tree/294993ca78bf65dec1c3b6fb25541432c545eda9

## Envrironment
Change the index-url cuda version according to your system.
```
conda create -n fastvideo python=3.10.12
conda activate fastvideo
pip3 install torch==2.5.0 torchvision  --index-url https://download.pytorch.org/whl/cu121
pip3 install -U xformers==0.0.28.post2 --index-url https://download.pytorch.org/whl/cu121
pip install git+https://github.com/huggingface/diffusers.git@76b7d86a9a5c0c2186efa09c4a67b5f5666ac9e3
```

```
pip install -e . && pip install -e ".[train]"
sudo apt-get update && apt install screen && pip install watch gpustat
```

## Prepare Data & Models
We've prepared some debug data to facilitate development. To make sure the training pipeline is correct, train on the debug data and make sure the model overfit on it (feed it the same text prompt and see if the output video is the same as the training data)

```
mkdir data && mkdir data/outputs/
python scripts/download_hf.py --repo_id=Stealths-Video/mochi_diffuser --local_dir=data/mochi --repo_type=model
python scripts/download_hf.py --repo_id=Stealths-Video/Merge-30k-Data --local_dir=data/Merge-30k-Data --repo_type=dataset
python scripts/download_hf.py --repo_id=Stealths-Video/validation_embeddings --local_dir=data/validation_embeddings --repo_type=dataset
cd data/Merge-30k-Data
cat Merged30K.tar.gz.part.* > Merged30K.tar.gz
rm Merged30K.tar.gz.part.*
tar --use-compress-program="pigz --processes 64" -xvf Merged30K.tar.gz
mv ephemeral/hao.zhang/codefolder/FastVideo-OSP/data/Merged-30K-Data/* .
rm -r ephemeral
rm Merged30K.tar.gz
cd ../..
```

## How to overfit
```
bash scripts/overfit.sh
```
Make sure to edit data/Mochi-Synthetic-Data/videos2caption.json such that this is only one video in the dataset (you can copy multiple annotations of the same video). Also make sure to edit the prompt in scripts/overfit.shto match the prompt in the training data. I observe the overfitting  after 50 steps. 

# Experiments
1. pcm_linear_quadratic， euler_steps 50, 0.025
2. pcm_linear_quadratic， euler_steps 50, 0.05
3. shift 8, euler_steps 100
4. shift 8, euler_steps 100, adv
5. pcm_linear_quadratic， euler_steps 50, 0.025, adv