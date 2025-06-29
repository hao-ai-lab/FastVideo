from huggingface_hub import HfApi

api = HfApi()

api.upload_folder(
    folder_path="/mnt/weka/home/hao.zhang/wei/FastVideo/data/Wan2.1-Fun-1.3B-InP-Diffusers",
    repo_id="weizhou03/Wan2.1-Fun-1.3B-InP-Diffusers",
    repo_type="model",
)
