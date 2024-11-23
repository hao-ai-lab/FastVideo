import os
import base64
import json
from pathlib import Path
from typing import List, Tuple, Dict, Optional
import cv2
from PIL import Image
import io
from tqdm import tqdm
import argparse
from openai import OpenAI
import random
from datetime import datetime

def load_existing_captions(json_path: str) -> Dict:
    """
    Load existing captions from a JSON file if it exists.
    
    Args:
        json_path: Path to the JSON file
        
    Returns:
        Dictionary containing existing captions and generation info
    """
    if Path(json_path).exists():
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            return data.get('captions', {}), data.get('generation_info', {})
    return {}, {}

def extract_frames(video_path: str, fps: int = 3) -> Optional[List[Image.Image]]:
    """
    Extract frames at specified FPS from video using OpenCV.
    Handles potential errors during frame extraction.
    
    Args:
        video_path: Path to the video file
        fps: Frames per second to extract (default: 3 fps = 18 total frames for 6s video)
        
    Returns:
        List of PIL Image objects or None if extraction fails
    """
    try:
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            print(f"Error: Could not open video file {video_path}")
            return None
        
        # Get video properties
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        video_fps = cap.get(cv2.CAP_PROP_FPS)
        
        if total_frames <= 0 or video_fps <= 0:
            print(f"Error: Invalid video properties for {video_path}")
            cap.release()
            return None
        
        # Calculate frame indices to extract
        frame_interval = max(1, int(video_fps / fps))
        frames_to_extract = range(0, total_frames, frame_interval)
        
        frames = []
        for frame_idx in frames_to_extract:
            try:
                # Set frame position
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ret, frame = cap.read()
                if not ret or frame is None:
                    print(f"Warning: Could not read frame {frame_idx} from {video_path}")
                    continue
                
                # Convert BGR to RGB
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                # Convert to PIL Image
                pil_image = Image.fromarray(frame_rgb)
                frames.append(pil_image)
                
            except Exception as e:
                print(f"Warning: Error processing frame {frame_idx} from {video_path}: {str(e)}")
                continue
        
        cap.release()
        
        if not frames:
            print(f"Warning: No frames were successfully extracted from {video_path}")
            return None
            
        return frames
        
    except Exception as e:
        print(f"Error extracting frames from {video_path}: {str(e)}")
        if 'cap' in locals():
            cap.release()
        return None

def get_gpt4_caption(frames: List[Image.Image], api_key: str) -> Tuple[str, int]:
    """
    Get caption from GPT-4 Vision API for the frames.
    
    Returns:
        Tuple containing (caption_text, number_of_sentences_requested)
    """
    client = OpenAI(api_key=api_key)
    num_sentences = random.randint(4, 9)
    
    content = [{
        "type": "text",
        "text": f"""
            Write a long caption for the given video, please directly start with what is happening, focusing on human figures present 
            If there is no human figure, then just describe the most prominent object or scene, depict the scene in a way that would be easy for an AI to recreate.
            Structure your description in {num_sentences} sentences, emphasizing both character dynamics and distinctive visual characteristics that would enable precise AI recreation.
        """
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
            model="gpt-4o",
            messages=[{
                "role": "user",
                "content": content
            }],
            max_tokens=256
        )
        
        return response.choices[0].message.content, num_sentences
        
    except Exception as e:
        print(f"Error calling GPT-4 Vision API: {str(e)}")
        raise

def get_output_path(output_dir: str, resume_file: str = None) -> Path:
    """
    Generate output JSON file path with timestamp or use existing file for resume.
    
    Args:
        output_dir: Directory to save output
        resume_file: Path to existing JSON file to resume from
        
    Returns:
        Path object for output file
    """
    if resume_file and Path(resume_file).exists():
        return Path(resume_file)
        
    # Create output directory if it doesn't exist
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate timestamp-based filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = output_dir / f"captions_{timestamp}.json"
    
    return output_file

def process_videos(video_folder: str, output_dir: str, api_key: str, fps: int = 3, resume_file: str = None):
    """
    Process all videos in the folder and generate captions.
    
    Args:
        video_folder: Path to folder containing videos
        output_dir: Directory to save output
        api_key: OpenAI API key
        fps: Frames per second to sample
        resume_file: Path to existing JSON file to resume from
    """
    video_folder = Path(video_folder)
    output_file = get_output_path(output_dir, resume_file)
    
    # Load existing captions if resuming
    captions, generation_info = load_existing_captions(str(output_file)) if resume_file else ({}, {})
    
    # Track failed videos
    failed_videos = []
    
    # Get list of video files
    video_files = list(video_folder.glob("*.mp4"))
    if not video_files:
        print(f"No .mp4 files found in {video_folder}")
        return
    
    print(f"Will save captions to: {output_file}")
    print(f"Found {len(captions)} existing captions" if captions else "Starting fresh captioning")
    
    for video_file in tqdm(video_files, desc="Processing videos"):
        if video_file.name in captions:
            print(f"\nSkipping {video_file.name} - already processed")
            continue
            
        print(f"\nProcessing {video_file.name}")
        
        try:
            # Extract frames
            frames = extract_frames(str(video_file), fps)
            if frames is None:
                print(f"Skipping {video_file.name} due to frame extraction failure")
                failed_videos.append((video_file.name, "Frame extraction failed"))
                continue
                
            print(f"Extracted {len(frames)} frames")
            
            # Get caption from GPT-4
            caption_text, num_sentences = get_gpt4_caption(frames, api_key)
            
            # Store caption
            captions[video_file.name] = {
                "caption": caption_text,
                "metadata": {
                    "frames_analyzed": len(frames),
                    "sampling_fps": fps,
                    "processed_time": datetime.now().isoformat(),
                    "requested_sentences": num_sentences,
                    "video_path": str(video_file)
                }
            }
            
            # Save progress after each video
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump({
                    "generation_info": {
                        "timestamp": datetime.now().isoformat(),
                        "video_folder": str(video_folder),
                        "sampling_fps": fps,
                        "total_videos": len(video_files),
                        "resumed_from": resume_file if resume_file else None,
                        "failed_videos": failed_videos
                    },
                    "captions": captions
                }, f, indent=4, ensure_ascii=False)
                
        except Exception as e:
            print(f"\nError processing {video_file.name}: {str(e)}")
            failed_videos.append((video_file.name, str(e)))
            print("Saving progress and continuing with next video...")
            continue
    
    print(f"\nProcessing complete. Captions saved to {output_file}")
    
    # Save a summary
    summary_file = output_file.parent / f"summary_{output_file.stem}.txt"
    with open(summary_file, 'w', encoding='utf-8') as f:
        f.write(f"Video Captioning Summary\n")
        f.write(f"=====================\n")
        f.write(f"Generated on: {datetime.now().isoformat()}\n")
        f.write(f"Videos processed: {len(captions)}\n")
        f.write(f"Videos failed: {len(failed_videos)}\n")
        f.write(f"Sampling rate: {fps} fps\n")
        if resume_file:
            f.write(f"Resumed from: {resume_file}\n")
        f.write("\n")
        
        if failed_videos:
            f.write("\nFailed Videos:\n")
            f.write("-------------\n")
            for video_name, error in failed_videos:
                f.write(f"{video_name}: {error}\n")
            f.write("\n")
        
        f.write("\nSuccessful Captions:\n")
        f.write("-----------------\n")
        for video_name, data in captions.items():
            f.write(f"\nVideo: {video_name}\n")
            f.write(f"Frames: {data['metadata']['frames_analyzed']}\n")
            f.write(f"Requested sentences: {data['metadata']['requested_sentences']}\n")
            f.write(f"Caption: {data['caption']}\n")
            f.write("-" * 80 + "\n")
    
    print(f"Summary saved to {summary_file}")

def parse_args():
    """
    Parse command line arguments.
    """
    parser = argparse.ArgumentParser(description='Caption videos using GPT-4 Vision.')
    parser.add_argument('--video_folder', type=str, required=True, help='Path to the folder containing .mp4 videos')
    parser.add_argument('--output_dir', type=str, required=True, help='Directory to save caption files')
    parser.add_argument('--fps', type=int, default=3, help='Frames per second to sample (default: 3)')
    parser.add_argument('--resume', type=str, help='Path to existing JSON file to resume from')
    
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
        output_dir=args.output_dir,
        api_key=api_key,
        fps=args.fps,
        resume_file=args.resume
    )

if __name__ == "__main__":
    main()