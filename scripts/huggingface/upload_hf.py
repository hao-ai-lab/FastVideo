from huggingface_hub import HfApi

api = HfApi()

repo_id = "FastVideo/LTX2-Diffusers"

# Create the repo if it doesn't exist
api.create_repo(repo_id=repo_id, repo_type="model", exist_ok=True)

api.upload_folder(
    folder_path="ltx_base_diffusers",
    repo_id=repo_id,
    repo_type="model",
)
