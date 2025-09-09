from huggingface_hub import HfApi

api = HfApi()

api.upload_folder(
    folder_path="sfwan",
    repo_id="wlsaidhi/SFWan2.1-T2V-1.3B-Diffusers",
    repo_type="model",
)
