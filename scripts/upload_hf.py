from huggingface_hub import HfApi

api = HfApi()

api.upload_folder(
    folder_path="/ephemeral/hao.zhang/codefolder/FastVideo-OSP/data/Hunyuan-Mixkit-Data",
    repo_id="FastVideo/Hunyuan-Distill-Data",
    repo_type="dataset",
)
