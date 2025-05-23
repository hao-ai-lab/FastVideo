import json
from pathlib import Path
import csv
import cv2


def get_video_info(video_path, metadata):
    """Extract video information using OpenCV and corresponding metadata"""
    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return None

    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = frame_count / fps if fps > 0 else 0

    cap.release()

    return {
        "path": video_path.name,
        "title": metadata.get("Video Title", ""),
        "description": metadata.get("Video Description", ""),
        "video_url": metadata.get("Video URL", ""),
        "download_url": metadata.get("Download URL", ""),
        "resolution": {
            "width": width,
            "height": height
        },
        "fps": fps,
        "duration": duration,
        "cap": [metadata.get("Video Description", "")]
    }


def read_csv_file(csv_path):
    """Read and return the content of a CSV file"""
    try:
        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            return list(reader)
    except Exception as e:
        print(f"Error reading CSV file {csv_path}: {e}")
        return None


def process_videos_from_csv(video_dir_path, csv_path, verbose=False):
    """Process videos using metadata from CSV file
    
    Args:
        video_dir_path (str): Path to directory containing video files
        csv_path (str): Path to CSV file containing video metadata
        verbose (bool): Whether to print verbose processing information
    """
    video_dir = Path(video_dir_path)
    csv_data = read_csv_file(csv_path)
    processed_data = []

    # Ensure directories exist
    if not video_dir.exists():
        print(f"Error: Video directory does not exist: {video_dir}")
        return []

    if csv_data is None:
        return []

    # Process each video file
    for row in csv_data:
        video_filename = row.get("Filename")
        if not video_filename:
            continue

        video_file = video_dir / video_filename

        # Check if video file exists
        if not video_file.exists():
            print(f"Warning: Video file not found: {video_filename}")
            continue

        # Process video and add to results
        video_info = get_video_info(video_file, row)
        if video_info:
            processed_data.append(video_info)

    return processed_data


def save_results(processed_data, output_path):
    """Save processed data to JSON file
    
    Args:
        processed_data (list): List of processed video information
        output_path (str): Full path for output JSON file
    """
    output_path = Path(output_path)

    # Create parent directories if they don't exist
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(processed_data, f, indent=2, ensure_ascii=False)

    return output_path


def parse_args():
    """Parse command line arguments"""
    import argparse

    parser = argparse.ArgumentParser(description='Process videos using metadata from CSV file')
    parser.add_argument('--video_dir', '-v', required=True, help='Directory containing video files')
    parser.add_argument('--csv_path', '-c', required=True, help='Path to CSV file containing video metadata')
    parser.add_argument('--output_path',
                        '-o',
                        required=True,
                        help='Full path for output JSON file (e.g., /path/to/output/videos2caption.json)')
    parser.add_argument('--verbose', action='store_true', help='Print verbose processing information')

    return parser.parse_args()


if __name__ == "__main__":
    # Parse command line arguments
    args = parse_args()

    # Process videos from CSV
    processed_videos = process_videos_from_csv(args.video_dir, args.csv_path, args.verbose)

    if processed_videos:
        # Save results
        output_path = save_results(processed_videos, args.output_path)

        print(f"\nProcessed {len(processed_videos)} videos")
        print(f"Results saved to: {output_path}")

        # Print example of processed data
        print("\nExample of processed video info:")
        print(json.dumps(processed_videos[0], indent=2))
    else:
        print("No videos were processed successfully")
