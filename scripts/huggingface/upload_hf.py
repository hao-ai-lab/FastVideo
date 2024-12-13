from huggingface_hub import HfApi

api = HfApi()

api.upload_folder(
    folder_path="/ephemeral/hao.zhang/codefolder/FastVideo-OSP/data/Black-Myth-Taylor-Src",
    repo_id="FastVideo/Image-Vid-Finetune-Src",
    repo_type="dataset",
)
