from huggingface_hub import HfApi

api = HfApi()

api.upload_folder(
    folder_path="Wan2.2-TI2V-5B-Diffusers",
    repo_id="FastVideo/FastWan2.2-TI2V-5B-Diffusers",
    repo_type="model",
)
