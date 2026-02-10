import argparse

from huggingface_hub import hf_hub_download, snapshot_download

# set args for repo_id, local_dir, repo_type,

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download a dataset or model from the Hugging Face Hub")
    parser.add_argument("--repo_id", type=str, help="The ID of the repository to download")
    parser.add_argument(
        "--local_dir",
        type=str,
        help="The local directory to download the repository to",
    )
    parser.add_argument(
        "--repo_type",
        type=str,
        help="The type of repository to download (dataset or model)",
    )
    parser.add_argument("--file_name", type=str, help="The file name to download")
    args = parser.parse_args()
    if args.file_name:
        hf_hub_download(
            repo_id=args.repo_id,
            filename=args.file_name,
            repo_type=args.repo_type,
            local_dir=args.local_dir,
        )
    else:
        snapshot_download(
            repo_id=args.repo_id,
            local_dir=args.local_dir,
            repo_type=args.repo_type,
            local_dir_use_symlinks=False,
            resume_download=True,
        )
