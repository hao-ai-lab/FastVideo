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

def convert_json_format(data, video_folder):
    """Convert the JSON format and add video metadata."""
    new_format = []
    
    for video_filename, video_info in tqdm(data["captions"].items(), desc="Processing videos"):
        video_path = Path(video_folder) / video_filename
        
        if not video_path.exists():
            print(f"Warning: Video file not found: {video_path}")
            continue
            
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