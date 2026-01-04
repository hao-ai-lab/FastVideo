apt-get update && apt-get install -y git-lfs && git lfs install
export HF_HOME=/workspace/.cache/huggingface
git clone https://huggingface.co/datasets/H1yori233/footsies-dataset /workspace/data/footsies-dataset
cd /workspace/data/footsies-dataset
git lfs pull
cd data && for f in episodes_*.tar.gz; do tar -xzf "$f"; done

huggingface-cli download Skywork/SkyReels-V2-I2V-1.3B-540P-Diffusers --local-dir SkyReels-V2-I2V-1.3B-540P-Diffusers