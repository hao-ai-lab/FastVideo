import json
from pathlib import Path

import cv2


def get_video_info(video_path, prompt_text):
    """Extract video information using OpenCV and corresponding prompt text"""
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
        "resolution": {
            "width": width,
            "height": height
        },
        "fps": fps,
        "duration": duration,
        "cap": [prompt_text]
    }


def read_prompt_file(prompt_path):
    """Read and return the content of a prompt file"""
    try:
        with open(prompt_path, 'r', encoding='utf-8') as f:
            return f.read().strip()
    except Exception as e:
        print(f"Error reading prompt file {prompt_path}: {e}")
        return None


def process_videos_and_prompts(video_dir_path, prompt_dir_path, verbose=False):
    """Process videos and their corresponding prompt files
    
    Args:
        video_dir_path (str): Path to directory containing video files
        prompt_dir_path (str): Path to directory containing prompt files
        verbose (bool): Whether to print verbose processing information
    """
    video_dir = Path(video_dir_path)
    prompt_dir = Path(prompt_dir_path)
    processed_data = []

    # Ensure directories exist
    if not video_dir.exists() or not prompt_dir.exists():
        print(
            f"Error: One or both directories do not exist:\nVideos: {video_dir}\nPrompts: {prompt_dir}"
        )
        return []

    # Process each video file
    for video_file in video_dir.glob('*.mp4'):
        video_name = video_file.stem
        prompt_file = prompt_dir / f"{video_name}.txt"

        # Check if corresponding prompt file exists
        if not prompt_file.exists():
            print(f"Warning: No prompt file found for video {video_name}")
            continue

        # Read prompt content
        prompt_text = read_prompt_file(prompt_file)
        if prompt_text is None:
            continue

        # Process video and add to results
        video_info = get_video_info(video_file, prompt_text)
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

    parser = argparse.ArgumentParser(
        description='Process videos and their corresponding prompt files')
    parser.add_argument('--video_dir',
                        '-v',
                        required=True,
                        help='Directory containing video files')
    parser.add_argument('--prompt_dir',
                        '-p',
                        required=True,
                        help='Directory containing prompt text files')
    parser.add_argument(
        '--output_path',
        '-o',
        required=True,
        help=
        'Full path for output JSON file (e.g., /path/to/output/videos2caption.json)'
    )
    parser.add_argument('--verbose',
                        action='store_true',
                        help='Print verbose processing information')

    return parser.parse_args()


if __name__ == "__main__":
    # Parse command line arguments
    args = parse_args()

    # Process videos and prompts
    processed_videos = process_videos_and_prompts(args.video_dir,
                                                  args.prompt_dir,
                                                  args.verbose)

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
