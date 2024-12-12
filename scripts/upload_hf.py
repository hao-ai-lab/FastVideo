from huggingface_hub import HfApi

api = HfApi()

api.upload_folder(
    folder_path="data/Mochi-Synthetic-Data",
    repo_id="FastVideo/HY-Distill-Debug",
    repo_type="dataset",
    token=True
)
