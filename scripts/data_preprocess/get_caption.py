import os
import base64
import json
from pathlib import Path
from typing import List
import cv2
from PIL import Image
import io
from tqdm import tqdm
import argparse
from openai import OpenAI
import random

def extract_frames(video_path: str, fps: int = 3) -> List[Image.Image]:
    """
    Extract frames at specified FPS from video using OpenCV.
    
    Args:
        video_path: Path to the video file
        fps: Frames per second to extract (default: 3 fps = 18 total frames for 6s video)
        
    Returns:
        List of PIL Image objects
    """
    cap = cv2.VideoCapture(video_path)
    
    # Get video properties
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    
    # Calculate frame indices to extract
    frame_interval = int(video_fps / fps)
    frames_to_extract = range(0, total_frames, frame_interval)
    
    frames = []
    for frame_idx in frames_to_extract:
        # Set frame position
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if ret:
            # Convert BGR to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # Convert to PIL Image
            pil_image = Image.fromarray(frame_rgb)
            frames.append(pil_image)
    
    cap.release()
    return frames

def get_gpt4_caption(frames: List[Image.Image], api_key: str) -> str:
    """
    Get caption from GPT-4 Vision API for the frames.
    """
    client = OpenAI(api_key=api_key)
    num_sentences = random.randint(4, 9)
    # Prepare content with text and images
    content = [{
        "type": "text",
        "text": f"Describe this video and its style in a very detailed manner. Pay attention to all objects in the video. The description should be useful for AI to re-generate the video. The description should be around {num_sentences} sentences."
    }]
    
    # Add each frame to the content
    for frame in frames:
        buffer = io.BytesIO()
        frame.save(buffer, format="JPEG")
        base64_image = base64.b64encode(buffer.getvalue()).decode('utf-8')
        
        content.append({
            "type": "image_url",
            "image_url": {
                "url": f"data:image/jpeg;base64,{base64_image}"
            }
        })
    
    try:
        response = client.chat.completions.create(
            model="gpt-4-vision-preview",
            messages=[{
                "role": "user",
                "content": content
            }],
            max_tokens=256
        )
        
        return response.choices[0].message.content
        
    except Exception as e:
        print(f"Error calling GPT-4 Vision API: {str(e)}")
        raise

def process_videos(video_folder: str, output_file: str, api_key: str, fps: int = 3):
    """
    Process all videos in the folder and generate captions.
    """
    video_folder = Path(video_folder)
    captions = {}
    
    # Load existing captions if file exists
    if os.path.exists(output_file):
        with open(output_file, 'r') as f:
            captions = json.load(f)
    
    # Get list of video files
    video_files = list(video_folder.glob("*.mp4"))
    if not video_files:
        print(f"No .mp4 files found in {video_folder}")
        return
    
    for video_file in tqdm(video_files, desc="Processing videos"):
        if video_file.name in captions:
            print(f"\nSkipping {video_file.name} - already processed")
            continue
            
        print(f"\nProcessing {video_file.name}")
        
        # Extract frames
        frames = extract_frames(str(video_file), fps)
        print(f"Extracted {len(frames)} frames")
        
        # Get caption from GPT-4
        caption = get_gpt4_caption(frames, api_key)
        
        # Store caption
        captions[video_file.name] = {
            "caption": caption,
            "metadata": {
                "frames_analyzed": len(frames),
                "sampling_fps": fps
            }
        }
        
        # Save progress after each video
        with open(output_file, 'w') as f:
            json.dump(captions, f, indent=4)
    
    print(f"\nProcessing complete. Captions saved to {output_file}")

def parse_args():
    """
    Parse command line arguments.
    """
    parser = argparse.ArgumentParser(description='Caption videos using GPT-4 Vision.')
    parser.add_argument('--video_folder', type=str, required=True, help='Path to the folder containing .mp4 videos')
    parser.add_argument('--output_file', type=str, required=True, help='Path to save the JSON file with captions')
    parser.add_argument('--fps', type=int, default=3, help='Frames per second to sample (default: 3)')
    
    return parser.parse_args()

def main():
    # Parse command line arguments
    args = parse_args()
    
    # Get API key from environment
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY not found in environment variables")

    # Process videos
    process_videos(
        video_folder=args.video_folder,
        output_file=args.output_file,
        api_key=api_key,
        fps=args.fps
    )

if __name__ == "__main__":
    main()