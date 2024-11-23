import json
import os
import shutil
from pathlib import Path
import math
import argparse

def split_dataset(input_json_path, source_video_folder, output_base_path, chunk_size=100):
    """
    Split a large video dataset into smaller chunks with corresponding JSON files.
    
    Args:
        input_json_path (str): Path to the input JSON file
        source_video_folder (str): Path to the folder containing source videos
        output_base_path (str): Base path for output folders
        chunk_size (int): Number of videos per chunk
    """
    print(f"Starting dataset split with parameters:")
    print(f"Input JSON: {input_json_path}")
    print(f"Source folder: {source_video_folder}")
    print(f"Output path: {output_base_path}")
    print(f"Chunk size: {chunk_size}")
    
    # Check if paths exist
    if not os.path.exists(input_json_path):
        raise FileNotFoundError(f"Input JSON file not found: {input_json_path}")
    if not os.path.exists(source_video_folder):
        raise FileNotFoundError(f"Source video folder not found: {source_video_folder}")
    
    # Create output directory if it doesn't exist
    os.makedirs(output_base_path, exist_ok=True)
    
    # Read the input JSON file
    print("Reading input JSON file...")
    with open(input_json_path, 'r') as f:
        data = json.load(f)
    
    # Get all video entries
    video_entries = list(data['captions'].items())
    total_videos = len(video_entries)
    print(f"Found {total_videos} videos in the input JSON")
    
    # Calculate number of chunks
    num_chunks = math.ceil(total_videos / chunk_size)
    print(f"Will create {num_chunks} chunks with {chunk_size} videos each")
    
    # Track statistics
    successful_copies = 0
    failed_copies = 0
    
    # Process each chunk
    for chunk_idx in range(num_chunks):
        start_idx = chunk_idx * chunk_size
        end_idx = min((chunk_idx + 1) * chunk_size, total_videos)
        
        # Create chunk folder names
        folder_name = f"HSH_{start_idx}_{end_idx}"
        chunk_folder = os.path.join(output_base_path, folder_name)
        videos_folder = os.path.join(chunk_folder, "videos")
        
        print(f"\nProcessing chunk {chunk_idx + 1}/{num_chunks}: {folder_name}")
        
        # Create directories
        os.makedirs(chunk_folder, exist_ok=True)
        os.makedirs(videos_folder, exist_ok=True)
        
        # Create chunk data dictionary
        chunk_data = {
            "generation_info": data["generation_info"].copy(),
            "captions": {}
        }
        
        # Update generation info for the chunk
        chunk_data["generation_info"]["total_videos"] = end_idx - start_idx
        chunk_data["generation_info"]["chunk_info"] = {
            "chunk_index": chunk_idx,
            "start_index": start_idx,
            "end_index": end_idx,
            "total_chunks": num_chunks
        }
        
        # Process videos in this chunk
        chunk_successful = 0
        chunk_failed = 0
        
        for video_name, video_info in video_entries[start_idx:end_idx]:
            # Add to chunk JSON
            chunk_data["captions"][video_name] = video_info
            
            # Copy video file
            src_video_path = os.path.join(source_video_folder, video_name)
            dst_video_path = os.path.join(videos_folder, video_name)
            
            try:
                if os.path.exists(src_video_path):
                    shutil.copy2(src_video_path, dst_video_path)
                    chunk_successful += 1
                else:
                    print(f"Warning: Source video not found: {src_video_path}")
                    chunk_failed += 1
            except Exception as e:
                print(f"Error copying {video_name}: {str(e)}")
                chunk_failed += 1
        
        successful_copies += chunk_successful
        failed_copies += chunk_failed
        
        # Save chunk JSON file
        json_output_path = os.path.join(chunk_folder, f"{folder_name}_captions.json")
        with open(json_output_path, 'w') as f:
            json.dump(chunk_data, f, indent=2)
        
        print(f"Chunk {chunk_idx + 1} complete:")
        print(f"- Successfully copied: {chunk_successful} videos")
        print(f"- Failed to copy: {chunk_failed} videos")
    
    # Print final statistics
    print("\nDataset split complete!")
    print(f"Total statistics:")
    print(f"- Total videos processed: {total_videos}")
    print(f"- Successfully copied: {successful_copies} videos")
    print(f"- Failed to copy: {failed_copies} videos")
    print(f"- Created {num_chunks} chunks in: {output_base_path}")

def main():
    parser = argparse.ArgumentParser(description="Split video dataset into chunks and organize files")
    parser.add_argument("-i", "--input", required=True, help="Input JSON file path")
    parser.add_argument("-s", "--source", required=True, help="Source video folder path")
    parser.add_argument("-o", "--output", required=True, help="Output base directory path")
    parser.add_argument("-c", "--chunk-size", type=int, default=100, help="Number of videos per chunk (default: 100)")
    parser.add_argument("-v", "--verbose", action="store_true", help="Print detailed progress information")
    
    args = parser.parse_args()
    
    try:
        split_dataset(args.input, args.source, args.output, args.chunk_size)
    except Exception as e:
        print(f"Error: {str(e)}")
        exit(1)

if __name__ == "__main__":
    main()