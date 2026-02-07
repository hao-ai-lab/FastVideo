from huggingface_hub import HfApi

api = HfApi()

repo_id = "FastVideo/LTX2-Distilled-LoRA"

# Create the repo if it doesn't exist
api.create_repo(repo_id=repo_id, repo_type="model", exist_ok=True)

api.upload_folder(
    folder_path="/mnt/user_storage/dev/FastVideo/ltx_distilled_lora",
    repo_id=repo_id,
    repo_type="model",
)
