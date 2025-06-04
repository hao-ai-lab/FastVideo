from huggingface_hub import HfApi

api = HfApi()

api.upload_folder(
    folder_path="mini_i2v_dataset",
    repo_id="FastVideo/mini-i2v-dataset",
    repo_type="dataset",
)
