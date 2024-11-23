import cv2
import numpy as np
from pathlib import Path
import os
import argparse
import subprocess

def ensure_directory(path):
    """Create directory if it doesn't exist"""
    os.makedirs(path, exist_ok=True)
    return path

def split_video(input_path, first_half_path, second_half_path):
    """Split video into two 3-second parts"""
    # Split first half (0-3s)
    subprocess.run([
        'ffmpeg', '-i', input_path,
        '-t', '3',  # Duration of 3 seconds
        '-c', 'copy',  # Copy without re-encoding
        '-y', first_half_path
    ], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    
    # Split second half (3-6s)
    subprocess.run([
        'ffmpeg', '-i', input_path,
        '-ss', '3',  # Start from 3 seconds
        '-t', '3',  # Duration of 3 seconds
        '-c', 'copy',  # Copy without re-encoding
        '-y', second_half_path
    ], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

def interpolate_frames(input_path, output_path, target_duration=6):
    """
    Use frame interpolation to extend video duration to target_duration
    while maintaining smooth motion
    """
    print(f"Applying frame interpolation to {Path(input_path).name}...")
    
    # First get original video FPS
    probe_cmd = [
        'ffprobe', 
        '-v', 'error',
        '-select_streams', 'v:0',
        '-show_entries', 'stream=r_frame_rate',
        '-of', 'default=noprint_wrappers=1:nokey=1',
        input_path
    ]
    fps_str = subprocess.check_output(probe_cmd).decode().strip()
    fps_num, fps_den = map(int, fps_str.split('/'))
    original_fps = fps_num / fps_den
    
    # Calculate target FPS to achieve 6 seconds duration
    target_fps = original_fps * 2  # Double the frame rate
    
    # Use minterpolate filter for motion-compensated frame interpolation
    subprocess.run([
        'ffmpeg', '-i', input_path,
        '-filter:v', f'minterpolate=\'fps={target_fps}:mi_mode=mci:mc_mode=aobmc:me_mode=bidir:vsbmc=1\'',
        '-c:v', 'libx264', 
        '-preset', 'slow',  # Better quality
        '-crf', '18',      # High quality
        '-y', output_path
    ], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    
    # Verify duration
    duration_cmd = [
        'ffprobe', 
        '-v', 'error',
        '-show_entries', 'format=duration',
        '-of', 'default=noprint_wrappers=1:nokey=1',
        output_path
    ]
    duration = float(subprocess.check_output(duration_cmd).decode().strip())
    print(f"Output duration: {duration:.2f} seconds")

def process_video(input_path, output_dir):
    """Process a single video"""
    try:
        ensure_directory(output_dir)
        filename = Path(input_path).stem
        temp_dir = ensure_directory(os.path.join(output_dir, 'temp'))
        
        # Define paths
        temp_first = os.path.join(temp_dir, f"{filename}_part1_temp.mp4")
        temp_second = os.path.join(temp_dir, f"{filename}_part2_temp.mp4")
        final_first = os.path.join(output_dir, f"{filename}_part1_slow.mp4")
        final_second = os.path.join(output_dir, f"{filename}_part2_slow.mp4")
        
        print(f"Processing {filename}:")
        print("1. Splitting video into 3-second segments...")
        split_video(input_path, temp_first, temp_second)
        
        print("2. Applying frame interpolation to first half...")
        interpolate_frames(temp_first, final_first)
        
        print("3. Applying frame interpolation to second half...")
        interpolate_frames(temp_second, final_second)
        
        # Clean up
        os.remove(temp_first)
        os.remove(temp_second)
        os.rmdir(temp_dir)
        
        print(f"Successfully created:\n{final_first}\n{final_second}")
        return final_first, final_second
        
    except subprocess.CalledProcessError as e:
        print(f"FFmpeg error: {e.stderr.decode() if e.stderr else str(e)}")
        raise
    except Exception as e:
        print(f"Error: {str(e)}")
        raise

def main():
    parser = argparse.ArgumentParser(
        description='Split videos into 3s segments and use frame interpolation to create 6s outputs.'
    )
    parser.add_argument('--input', help='Input folder containing videos')
    parser.add_argument('--output', help='Output folder for processed videos')
    
    args = parser.parse_args()
    
    # Verify input folder exists
    if not os.path.exists(args.input):
        print(f"Error: Input folder '{args.input}' does not exist!")
        return
    
    # Check FFmpeg installation and version
    try:
        version = subprocess.check_output(['ffmpeg', '-version']).decode()
        if 'ffmpeg version' not in version:
            raise Exception("FFmpeg not found")
    except:
        print("Error: FFmpeg is not installed or not found in system PATH!")
        return
    
    input_folder = os.path.abspath(args.input)
    output_folder = ensure_directory(args.output)
    
    print(f"Processing videos from: {input_folder}")
    print(f"Saving to: {output_folder}")
    
    # Process all videos
    videos = list(Path(input_folder).glob('*.mp4'))
    total = len(videos)
    
    for i, video_file in enumerate(videos, 1):
        try:
            print(f"\nProcessing video {i}/{total}: {video_file.name}")
            process_video(str(video_file), output_folder)
        except Exception as e:
            print(f"Failed to process {video_file.name}: {str(e)}")
            continue

if __name__ == "__main__":
    main()