import os
from pathlib import Path
import cv2
from tqdm import tqdm
import argparse
import shutil
from moviepy.editor import VideoFileClip

def filter_videos(src_folder: str, 
                  valid_folder: str,  # For 6-60s videos
                  clips_folder: str = None,  # For 6s clips
                  min_duration: float = 6.0, 
                  max_duration: float = 60.0, 
                  clip_duration: float = 6.0,
                  only_save_clips: bool = None  # None: do both, True: only clips from valid
                  ):
    """
    Process videos with three modes:
    1. only_save_clips=None: Copy 6-60s videos to valid_folder AND create 6s clips
    2. only_save_clips=True: Only create 6s clips from existing valid_folder
    """
    src_folder = Path(src_folder)
    valid_folder = Path(valid_folder)
    clips_folder = Path(clips_folder) if clips_folder else None
    
    if not only_save_clips:  # If doing full processing or both
        valid_folder.mkdir(parents=True, exist_ok=True)
    if clips_folder:
        clips_folder.mkdir(parents=True, exist_ok=True)
    
    # Determine which folder to scan based on mode
    scan_folder = valid_folder if only_save_clips else src_folder
    videos = list(scan_folder.glob("*.mp4"))
    
    if not videos:
        print(f"No .mp4 files found in {scan_folder}")
        return {}
    
    print(f"Found {len(videos)} videos in {scan_folder}")
    print(f"Mode: {'Only creating clips from valid folder' if only_save_clips else 'Full processing'}")
    if not only_save_clips:
        print(f"- Saving {min_duration}-{max_duration}s videos to: {valid_folder}")
    if clips_folder:
        print(f"- Creating {clip_duration}s clips to: {clips_folder}")
    
    stats = {
        "total_videos": len(videos),
        "valid_videos": 0,
        "too_short": 0,
        "too_long": 0,
        "clips_generated": 0,
        "existing_valid_files": 0,
        "existing_clips": 0,
        "clips_info": [],
        "valid_durations": []
    }
    
    # Progress bars
    pbar = tqdm(total=100, desc="Overall progress", position=0)
    video_pbar = tqdm(total=len(videos), desc="Processing videos", position=1, leave=True)
    
    for i, video_path in enumerate(videos):
        progress = (i + 1) / len(videos) * 100
        pbar.n = int(progress)
        pbar.refresh()
        
        # Get video duration
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            tqdm.write(f"\nCould not open {video_path.name}")
            video_pbar.update(1)
            continue
            
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = frame_count / fps
        cap.release()
        
        # Process based on mode
        if only_save_clips:
            # When only creating clips, all videos in valid_folder are considered valid
            valid = True
        else:
            # In full processing mode, check duration range
            valid = min_duration <= duration <= max_duration
        
        if valid:
            # In full processing mode, copy to valid folder
            if not only_save_clips:
                dst_path = valid_folder / video_path.name
                if not dst_path.exists():
                    shutil.copy2(video_path, dst_path)
                    stats["valid_videos"] += 1
                    stats["valid_durations"].append((video_path.name, duration))
                else:
                    stats["existing_valid_files"] += 1
            
            # Create clips if clips_folder is specified
            if clips_folder:
                try:
                    video = VideoFileClip(str(video_path))
                    num_clips = int(duration // clip_duration)
                    base_name = video_path.stem
                    
                    for clip_idx in range(num_clips):
                        start_time = clip_idx * clip_duration
                        output_name = f"{base_name}-{clip_idx+1}.mp4"
                        output_path = clips_folder / output_name
                        
                        if not output_path.exists():
                            clip = video.subclip(start_time, start_time + clip_duration)
                            clip.write_videofile(
                                str(output_path),
                                codec='libx264',
                                audio_codec='aac',
                                verbose=False,
                                logger=None
                            )
                            clip.close()
                            
                            stats["clips_generated"] += 1
                            stats["clips_info"].append({
                                "original_video": video_path.name,
                                "clip_name": output_name,
                                "start_time": start_time,
                                "duration": clip_duration
                            })
                            tqdm.write(f"Created clip: {output_name}")
                        else:
                            stats["existing_clips"] += 1
                            continue
                    
                    video.close()
                except Exception as e:
                    tqdm.write(f"Error creating clips for {video_path.name}: {str(e)}")
        else:
            if duration < min_duration:
                stats["too_short"] += 1
            else:
                stats["too_long"] += 1
        
        video_pbar.update(1)
    
    pbar.close()
    video_pbar.close()
    
    # Print report
    print("\nVideo Processing Report:")
    print("=" * 50)
    print(f"Total videos processed: {stats['total_videos']}")
    if not only_save_clips:
        print(f"Valid videos saved: {stats['valid_videos']}")
        print(f"Existing valid files skipped: {stats['existing_valid_files']}")
        print(f"Videos too short (<{min_duration}s): {stats['too_short']}")
        print(f"Videos too long (>{max_duration}s): {stats['too_long']}")
    if clips_folder:
        print(f"New clips generated: {stats['clips_generated']}")
        print(f"Existing clips skipped: {stats['existing_clips']}")
    
    if stats["valid_durations"]:
        print(f"\nSample of valid videos:")
        for name, duration in sorted(stats["valid_durations"], key=lambda x: x[1])[:5]:
            print(f"  {name}: {duration:.2f}s")
    
    if clips_folder and stats["clips_info"]:
        print("\nSample of generated clips:")
        for clip in stats["clips_info"][:5]:
            print(f"  {clip['clip_name']} (from {clip['original_video']}, start: {clip['start_time']}s)")
    
    return stats

def parse_args():
    """
    Parse command line arguments.
    """
    parser = argparse.ArgumentParser(description='Process videos and create clips.')
    
    parser.add_argument('--src_video_folder', type=str, required=True, help='Path to the folder containing source .mp4 videos')
    parser.add_argument('--valid_video_folder', type=str, required=True, help='Path to save valid-length videos')
    parser.add_argument('--clips_folder', type=str, required=True, help='Path to save generated clips')
    parser.add_argument('--only_save_clips', action='store_true', help='Only create 6s clips from valid videos')
    
    return parser.parse_args()

def main():
    # Parse command line arguments
    args = parse_args()
    
    # Process videos
    filter_videos(
        args.src_video_folder,
        args.valid_video_folder,
        args.clips_folder,
        min_duration=6,
        max_duration=60,
        clip_duration=6,
        only_save_clips=args.only_save_clips
    )

if __name__ == "__main__":
    main()