from pathlib import Path
import argparse
from collections import defaultdict

def parse_video_name(filename: str) -> tuple:
    """Parse video filename of format A-B.mp4 into tuple(A, B)"""
    try:
        base = filename.rsplit('.', 1)[0]  # Remove extension
        a, b = base.split('-')
        return (a, int(b))
    except:
        return None

def find_duplicate_videos(folder_path: str, dry_run: bool = True):
    """
    Find and optionally delete videos where A has both B=1 and B=2,
    deleting the B=2 version.
    
    Args:
        folder_path: Path to folder containing videos
        dry_run: If True, only print actions without deleting
    """
    folder = Path(folder_path)
    if not folder.exists():
        print(f"Error: Folder {folder_path} does not exist")
        return
    
    # Group videos by A value
    videos_by_a = defaultdict(list)
    for video_path in folder.glob("*.mp4"):
        result = parse_video_name(video_path.name)
        if result:
            a, b = result
            videos_by_a[a].append((b, video_path))
    
    # Find cases where A has both B=1 and B=2
    to_delete = []
    for a, videos in videos_by_a.items():
        # Convert to dict for easy lookup
        b_values = {b: path for b, path in videos}
        
        # Check if both B=1 and B=2 exist
        if 1 in b_values and 2 in b_values:
            to_delete.append(b_values[2])  # Delete the B=2 version
    
    # Report findings
    print(f"\nFound {len(to_delete)} videos to delete:")
    for path in to_delete:
        print(f"- {path.name}")
    
    # Perform deletion if not dry run
    if not dry_run and to_delete:
        print("\nDeleting files...")
        for path in to_delete:
            try:
                path.unlink()
                print(f"Deleted: {path.name}")
            except Exception as e:
                print(f"Error deleting {path.name}: {str(e)}")
        print(f"\nSuccessfully deleted {len(to_delete)} files")
    elif dry_run and to_delete:
        print("\nDry run - no files were deleted")
        print("To delete files, run again without --dry-run")

def parse_args():
    parser = argparse.ArgumentParser(description='Delete duplicate videos where B=2 when both B=1 and B=2 exist for same A')
    parser.add_argument('folder', type=str, help='Path to folder containing videos')
    parser.add_argument('--dry-run', action='store_true', help='Only print actions without deleting files')
    return parser.parse_args()

def main():
    args = parse_args()
    find_duplicate_videos(args.folder, args.dry_run)

if __name__ == "__main__":
    main()