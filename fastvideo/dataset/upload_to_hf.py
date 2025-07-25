# SPDX-License-Identifier: Apache-2.0
from huggingface_hub import HfApi, upload_folder

api = HfApi()
repo_id = "weizhou03/HD-Mixkit-Finetune-Wan"  # customize this
api.create_repo(repo_id=repo_id, repo_type="dataset")

upload_folder(repo_id=repo_id,
              folder_path="/workspace/data/HD-Mixkit-Finetune-Wan",
              repo_type="dataset",
              path_in_repo="")
