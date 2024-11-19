import os
from pathlib import Path
import cv2
from tqdm import tqdm
import argparse
import shutil
from moviepy.editor import VideoFileClip
from collections import defaultdict
import random
import math

def is_16_9_ratio(width: int, height: int, tolerance: float = 0.01) -> bool:
    """
    Check if the video dimensions match 16:9 aspect ratio within tolerance.
    
    Args:
        width: Video width in pixels
        height: Video height in pixels
        tolerance: Acceptable deviation from exact 16:9 ratio (default 1%)
    
    Returns:
        bool: True if aspect ratio is within tolerance of 16:9
    """
    target_ratio = 16 / 9
    actual_ratio = width / height
    return abs(actual_ratio - target_ratio) <= (target_ratio * tolerance)

def get_video_type(video_path: Path) -> str:
    """Extract video type from filename (type_xxx.mp4 -> type)"""
    return video_path.stem.split('_')[0]

def get_balanced_subset(videos_by_type: dict, total_desired: int) -> list:
    """
    Calculate and select a balanced subset of videos from each type
    to create a dataset of desired total size.
    """
    total_types = len(videos_by_type)
    if total_types == 0:
        return []
        
    base_per_type = total_desired // total_types
    remaining = total_desired - (base_per_type * total_types)
    
    balanced_videos = []
    types_needing_extra = random.sample(list(videos_by_type.keys()), remaining)
    
    for video_type, videos in videos_by_type.items():
        num_videos = base_per_type + (1 if video_type in types_needing_extra else 0)
        num_available = len(videos)
        
        if num_available <= num_videos:
            selected = videos
        else:
            selected = random.sample(videos, num_videos)
        
        balanced_videos.extend(selected)
    
    random.shuffle(balanced_videos)
    return balanced_videos

def filter_videos(src_folder: str, 
                  valid_folder: str,
                  clips_folder: str = None,
                  min_duration: float = 6.0, 
                  max_duration: float = 60.0, 
                  clip_duration: float = 6.0,
                  only_save_clips: bool = None,
                  total_desired: int = None,
                  aspect_ratio_tolerance: float = 0.01,
                  video_type_filter: str = None  # New parameter
                  ):
    """
    Process videos with filtering options:
    - Optional video type filter (e.g., 'celebv')
    - 16:9 aspect ratio filtering with configurable tolerance
    - Duration limits
    
    Modes:
    1. only_save_clips=None: Copy 6-60s videos to valid_folder AND create 6s clips
    2. only_save_clips=True: Only create 6s clips from existing valid_folder
    """
    src_folder = Path(src_folder)
    valid_folder = Path(valid_folder)
    clips_folder = Path(clips_folder) if clips_folder else None
    
    if not only_save_clips:
        valid_folder.mkdir(parents=True, exist_ok=True)
    if clips_folder:
        clips_folder.mkdir(parents=True, exist_ok=True)
    
    scan_folder = valid_folder if only_save_clips else src_folder
    all_videos = list(scan_folder.glob("*.mp4"))
    
    # Apply video type filter if specified
    if video_type_filter:
        all_videos = [v for v in all_videos if get_video_type(v) == video_type_filter]
        print(f"\nFiltered for video type '{video_type_filter}': {len(all_videos)} videos found")
    
    if not all_videos:
        print(f"No matching .mp4 files found in {scan_folder}")
        return {}
    
    videos_by_type = defaultdict(list)
    for video_path in all_videos:
        video_type = get_video_type(video_path)
        videos_by_type[video_type].append(video_path)
    
    print("\nInitial distribution of videos:")
    for video_type, videos in videos_by_type.items():
        print(f"- {video_type}: {len(videos)} videos")
    
    if total_desired:
        videos = get_balanced_subset(videos_by_type, total_desired)
        print(f"\nCreating balanced subset of {total_desired} videos:")
        
        selected_by_type = defaultdict(int)
        for video in videos:
            selected_by_type[get_video_type(video)] += 1
        
        for video_type, count in selected_by_type.items():
            print(f"- {video_type}: {count} videos")
    else:
        videos = all_videos
    
    print(f"\nProcessing total of {len(videos)} videos")
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
        "wrong_aspect_ratio": 0,
        "clips_generated": 0,
        "existing_valid_files": 0,
        "existing_clips": 0,
        "clips_info": [],
        "valid_durations": [],
        "videos_per_type": dict(videos_by_type),
        "processed_per_type": defaultdict(int),
        "aspect_ratios": []
    }
    
    pbar = tqdm(total=100, desc="Overall progress", position=0)
    video_pbar = tqdm(total=len(videos), desc="Processing videos", position=1, leave=True)
    
    for i, video_path in enumerate(videos):
        progress = (i + 1) / len(videos) * 100
        pbar.n = int(progress)
        pbar.refresh()
        
        video_type = get_video_type(video_path)
        
        # Get video properties
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            tqdm.write(f"\nCould not open {video_path.name}")
            video_pbar.update(1)
            continue
            
        # Check aspect ratio
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        aspect_ratio = width / height if height != 0 else 0
        stats["aspect_ratios"].append((video_path.name, aspect_ratio))
        
        if not is_16_9_ratio(width, height, aspect_ratio_tolerance):
            stats["wrong_aspect_ratio"] += 1
            cap.release()
            video_pbar.update(1)
            continue
            
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = frame_count / fps
        cap.release()
        
        if only_save_clips:
            valid = True
        else:
            valid = min_duration <= duration <= max_duration
        
        if valid:
            stats["processed_per_type"][video_type] += 1
            
            if not only_save_clips:
                dst_path = valid_folder / video_path.name
                if not dst_path.exists():
                    shutil.copy2(video_path, dst_path)
                    stats["valid_videos"] += 1
                    stats["valid_durations"].append((video_path.name, duration))
                else:
                    stats["existing_valid_files"] += 1
            
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
                                "duration": clip_duration,
                                "type": video_type
                            })
                            tqdm.write(f"Created clip: {output_name} (Type: {video_type})")
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
    
    print("\nVideo Processing Report:")
    print("=" * 50)
    print(f"Total videos processed: {stats['total_videos']}")
    
    print("\nProcessed videos by type:")
    for video_type, count in stats["processed_per_type"].items():
        print(f"- {video_type}: {count} videos")
    
    if not only_save_clips:
        print(f"\nValid videos saved: {stats['valid_videos']}")
        print(f"Existing valid files skipped: {stats['existing_valid_files']}")
        print(f"Videos too short (<{min_duration}s): {stats['too_short']}")
        print(f"Videos too long (>{max_duration}s): {stats['too_long']}")
        print(f"Videos with wrong aspect ratio: {stats['wrong_aspect_ratio']}")
    
    if clips_folder:
        print(f"\nNew clips generated: {stats['clips_generated']}")
        print(f"Existing clips skipped: {stats['existing_clips']}")
    
    if stats["aspect_ratios"]:
        print(f"\nSample of video aspect ratios:")
        for name, ratio in sorted(stats["aspect_ratios"], key=lambda x: x[1])[:5]:
            print(f"  {name}: {ratio:.3f}")
    
    if stats["valid_durations"]:
        print(f"\nSample of valid videos:")
        for name, duration in sorted(stats["valid_durations"], key=lambda x: x[1])[:5]:
            print(f"  {name}: {duration:.2f}s")
    
    if clips_folder and stats["clips_info"]:
        print("\nSample of generated clips by type:")
        clips_by_type = defaultdict(list)
        for clip in stats["clips_info"]:
            clips_by_type[clip["type"]].append(clip)
        
        for video_type, clips in clips_by_type.items():
            print(f"\n{video_type}:")
            for clip in clips[:3]:
                print(f"  {clip['clip_name']} (start: {clip['start_time']}s)")
    
    return stats

def parse_args():
    parser = argparse.ArgumentParser(description='Process videos and create clips.')
    
    parser.add_argument('--src_video_folder', type=str, required=True, help='Path to the folder containing source .mp4 videos')
    parser.add_argument('--valid_video_folder', type=str, required=True, help='Path to save valid-length videos')
    parser.add_argument('--clips_folder', type=str, required=True, help='Path to save generated clips')
    parser.add_argument('--only_save_clips', action='store_true', help='Only create 6s clips from valid videos')
    parser.add_argument('--total_desired', type=int, help='Total number of videos desired in the balanced subset')
    parser.add_argument('--aspect_ratio_tolerance', type=float, default=0.01, help='Tolerance for 16:9 aspect ratio check (default: 0.01)')
    parser.add_argument('--video_type', type=str, default=None, help='Filter for specific video type (e.g., "celebv")')  # New argument
    
    return parser.parse_args()

def main():
    args = parse_args()
    
    filter_videos(
        args.src_video_folder,
        args.valid_video_folder,
        args.clips_folder,
        min_duration=6,
        max_duration=60,
        clip_duration=6,
        only_save_clips=args.only_save_clips,
        total_desired=args.total_desired,
        aspect_ratio_tolerance=args.aspect_ratio_tolerance,
        video_type_filter=args.video_type  # Added parameter
    )

if __name__ == "__main__":
    main()
