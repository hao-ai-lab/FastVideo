import json
import os
from pathlib import Path
import cv2
import argparse
from tqdm import tqdm

def combine_json_files(json_paths):
    """Combine multiple JSON files into one."""
    combined_data = {
        "generation_info": {},
        "captions": {}
    }
    
    for json_path in json_paths:
        with open(json_path, 'r') as f:
            data = json.load(f)
            
            # Merge generation_info (keep the latest one)
            combined_data["generation_info"].update(data["generation_info"])
            
            # Merge captions
            combined_data["captions"].update(data["captions"])
    
    return combined_data

def get_video_metadata(video_path):
    """Get video metadata using OpenCV."""
    try:
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise ValueError(f"Could not open video file: {video_path}")
            
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = frame_count / fps if fps > 0 else 0
        
        cap.release()
        
        size = os.path.getsize(video_path)
        
        return {
            "resolution": {"width": width, "height": height},
            "size": size,
            "fps": fps,
            "duration": duration
        }
    except Exception as e:
        print(f"Error processing video {video_path}: {str(e)}")
        return None

def cleanup_unused_videos(video_folder, captions_data, dry_run=False):
    """Delete videos that exist in folder but not in captions."""
    video_files = set(f.name for f in Path(video_folder).glob('*') if f.is_file())
    caption_files = set(captions_data["captions"].keys())
    
    # Find videos that are in folder but not in captions
    videos_to_delete = video_files - caption_files
    
    if videos_to_delete:
        print(f"\nFound {len(videos_to_delete)} videos without captions to delete:")
        total_size = 0
        
        for video in videos_to_delete:
            video_path = Path(video_folder) / video
            size = os.path.getsize(video_path)
            total_size += size
            print(f"- {video} ({size / (1024*1024):.2f} MB)")
            
        print(f"\nTotal space to be freed: {total_size / (1024*1024):.2f} MB")
        
        if not dry_run:
            if input("\nProceed with deletion? (y/N): ").lower() == 'y':
                for video in videos_to_delete:
                    video_path = Path(video_folder) / video
                    try:
                        os.remove(video_path)
                        print(f"Deleted: {video}")
                    except Exception as e:
                        print(f"Error deleting {video}: {str(e)}")
                print("\nCleanup completed.")
            else:
                print("\nDeletion cancelled.")
        else:
            print("\nDry run - no files were deleted.")
    else:
        print("\nNo unused videos found.")

def get_available_videos(video_folder, captions_data):
    """Get list of videos that exist both in folder and captions."""
    video_files = set(f.name for f in Path(video_folder).glob('*') if f.is_file())
    caption_files = set(captions_data["captions"].keys())
    
    # Find intersection of both sets
    valid_videos = video_files.intersection(caption_files)
    
    if len(valid_videos) < len(video_files) or len(valid_videos) < len(caption_files):
        print(f"\nFound {len(valid_videos)} videos with both files and captions")
        print(f"Videos in folder: {len(video_files)}")
        print(f"Videos in captions: {len(caption_files)}")
        print(f"Skipped {len(video_files) - len(valid_videos)} videos without captions")
        print(f"Skipped {len(caption_files) - len(valid_videos)} captions without video files")
    
    return valid_videos

def convert_json_format(data, video_folder):
    """Convert the JSON format and add video metadata."""
    new_format = []
    
    # Get videos that exist both in folder and captions
    valid_videos = get_available_videos(video_folder, data)
    
    for video_filename in tqdm(valid_videos, desc="Processing videos"):
        video_path = Path(video_folder) / video_filename
        video_info = data["captions"][video_filename]
        
        metadata = get_video_metadata(video_path)
        if metadata is None:
            continue
            
        new_entry = {
            "path": video_filename,
            **metadata,
            "cap": [video_info["caption"]]
        }
        
        new_format.append(new_entry)
    
    return new_format

def save_json(data, output_path):
    """Save the converted data to a JSON file."""
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Combine JSON files and convert video caption format with metadata.', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-i', '--inputs', nargs='+', required=True, help='Paths to input JSON files to combine')
    parser.add_argument('-v', '--video-folder', required=True, help='Path to folder containing video files')
    parser.add_argument('-o', '--output', default='converted_output.json', help='Path to save the converted JSON output')
    parser.add_argument('--save-combined', default='combined.json', help='Path to save the combined JSON before conversion')
    parser.add_argument('--verbose', action='store_true', help='Enable verbose output')
    parser.add_argument('--cleanup', action='store_true', help='Delete videos without captions')
    parser.add_argument('--dry-run', action='store_true', help='Show what would be deleted without actually deleting')
    return parser.parse_args()

def main():
    args = parse_arguments()
    
    # Validate input files
    for input_file in args.inputs:
        if not os.path.exists(input_file):
            raise FileNotFoundError(f"Input JSON file not found: {input_file}")
    
    if not os.path.exists(args.video_folder):
        raise FileNotFoundError(f"Video folder not found: {args.video_folder}")
    
    if args.verbose:
        print(f"Combining input files: {', '.join(args.inputs)}")
        print(f"Video folder: {args.video_folder}")
        print(f"Output will be saved to: {args.output}")
    
    # Combine JSON files
    combined_data = combine_json_files(args.inputs)
    
    # Cleanup unused videos if requested
    if args.cleanup:
        cleanup_unused_videos(args.video_folder, combined_data, args.dry_run)
    
    # Save combined JSON if requested
    if args.save_combined:
        save_json(combined_data, args.save_combined)
        if args.verbose:
            print(f"Combined JSON saved to: {args.save_combined}")
    
    # Convert the format
    converted_data = convert_json_format(combined_data, args.video_folder)
    
    # Save the final result
    save_json(converted_data, args.output)
    print(f"\nConversion completed. Output saved to: {args.output}")
    
    if args.verbose:
        print(f"Processed {len(converted_data)} videos")

if __name__ == "__main__":
    main()