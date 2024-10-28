from huggingface_hub import HfApi
api = HfApi()

api.upload_folder(
    folder_path="data/dummySynthetic",
    repo_id="Stealths-Video/Mochi-Synthetic-Data",
    repo_type="dataset",
)