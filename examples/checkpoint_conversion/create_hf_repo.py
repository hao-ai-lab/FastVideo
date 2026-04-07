import argparse
import os
import shutil
import glob
from huggingface_hub import snapshot_download, upload_folder, create_repo

def main():
    parser = argparse.ArgumentParser(description="Download a HF Diffusers repo and replace its transformer weights.")
    parser.add_argument(
        "--repo_id", 
        type=str, 
        required=True, 
        help="The Hugging Face repository ID to download (e.g., 'stabilityai/stable-diffusion-xl-base-1.0')."
    )
    parser.add_argument(
        "--local_dir", 
        type=str, 
        required=True, 
        help="The local directory where the repository will be downloaded and modified."
    )
    parser.add_argument(
        "--checkpoint_dir", 
        type=str, 
        required=True, 
        help="The directory containing the new transformer model weights (checkpoints) to inject."
    )
    parser.add_argument(
        "--component_name", 
        type=str, 
        default="transformer", 
        help="The name of the component subfolder in the diffusers repo to replace weights for (default: 'transformer')."
    )
    parser.add_argument(
        "--weight_file_name",
        type=str,
        default="diffusion_pytorch_model.safetensors",
        help="The target weight filename in the component directory (default: 'diffusion_pytorch_model.safetensors')."
    )
    parser.add_argument(
        "--ignore_patterns",
        nargs="+",
        default=[],
        help="Patterns to ignore when downloading the repo (passed to snapshot_download)."
    )
    parser.add_argument(
        "--push_to_hub",
        action="store_true",
        help="If set, upload the modified local directory back to the Hugging Face Hub (to --repo_id)."
    )
    parser.add_argument(
        "--hub_commit_message",
        type=str,
        default="Update transformer weights",
        help="Commit message for the Hub upload."
    )
    parser.add_argument(
        "--upload_repo_id",
        type=str,
        default=None,
        help="The Hugging Face repository ID to upload to. If not provided, defaults to --repo_id. Creates the repo if it doesn't exist."
    )

    args = parser.parse_args()

    print(f"Downloading repo '{args.repo_id}' to '{args.local_dir}'...")
    snapshot_download(
        repo_id=args.repo_id,
        local_dir=args.local_dir,
        ignore_patterns=args.ignore_patterns,
        local_dir_use_symlinks=False  # We want actual files to modify them
    )
    print("Download complete.")

    target_component_dir = os.path.join(args.local_dir, args.component_name)
    if not os.path.exists(target_component_dir):
        print(f"Warning: Target component directory '{target_component_dir}' does not exist in the downloaded repo.")
        # We might still want to create it if it's a new component, but usually it should exist for replacement.
        # Proceeding assuming the user knows what they are doing.
        os.makedirs(target_component_dir, exist_ok=True)

    # Find weight files in checkpoint_dir
    source_weights = glob.glob(os.path.join(args.checkpoint_dir, "*.safetensors")) + \
                     glob.glob(os.path.join(args.checkpoint_dir, "*.bin"))
    
    if not source_weights:
        print(f"Error: No .safetensors or .bin weights found in '{args.checkpoint_dir}'.")
        return

    print(f"Found weights in checkpoint dir: {[os.path.basename(x) for x in source_weights]}")

    # Remove existing weights in target component dir to avoid mixing shards/files
    existing_weights = glob.glob(os.path.join(target_component_dir, "*.safetensors")) + \
                       glob.glob(os.path.join(target_component_dir, "*.bin")) + \
                       glob.glob(os.path.join(target_component_dir, "*.index.json")) # Remove index if present
    
    for ew in existing_weights:
        print(f"Removing existing weight file: {ew}")
        os.remove(ew)

    # Copy new weights
    for src in source_weights:
        filename = os.path.basename(src)
        dst = os.path.join(target_component_dir, filename)
        
        # If there is only one file and it doesn't match the standard name, maybe we should rename it?
        if len(source_weights) == 1 and filename != args.weight_file_name:
             print(f"Renaming '{filename}' to '{args.weight_file_name}'...")
             dst = os.path.join(target_component_dir, args.weight_file_name)
        
        print(f"Copying '{src}' to '{dst}'...")
        shutil.copy2(src, dst)

    print("Replacement complete.")
    print(f"New repo ready at: {args.local_dir}")

    if args.push_to_hub:
        target_repo_id = args.upload_repo_id if args.upload_repo_id else args.repo_id
        print(f"Uploading to Hugging Face Hub: {target_repo_id}...")
        
        try:
            # Ensure the repo exists
            create_repo(target_repo_id, exist_ok=True)
            
            upload_folder(
                repo_id=target_repo_id,
                folder_path=args.local_dir,
                commit_message=args.hub_commit_message
            )
            print("Upload complete!")
        except Exception as e:
            print(f"Error uploading to Hub: {e}")

if __name__ == "__main__":
    main()
