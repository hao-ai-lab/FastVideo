import os
import glob
import cv2
import json
import numpy as np
import argparse
from tqdm import tqdm
from collections import defaultdict
from multiprocessing import Pool, cpu_count
import functools

def parse_filename(fname):
    parts = fname.replace(".png", "").split("_")
    return {
        "episode": int(parts[0]),
        "step": int(parts[1]),
        "action": int(parts[2]),
        "fname": fname,
        "path": os.path.join(args.image_dir, fname)
    }

def process_episode(episode_data):
    episode_id, frames_info = episode_data
    
    frames_info.sort(key=lambda x: x['step'])
    if not frames_info:
        return None

    last_step = frames_info[-1]['step']
    start_idx = len(frames_info) - 1
    
    # check continuity
    segment_start_idx = 0 
    for i in range(len(frames_info) - 2, -1, -1):
        if frames_info[i+1]['step'] != frames_info[i]['step'] + 1:
            segment_start_idx = i + 1
            break
            
    valid_frames = frames_info[segment_start_idx:]
    if len(valid_frames) < 10:
        return None

    output_video_path = os.path.join(args.output_dir, "videos", f"episode_{episode_id:03d}.mp4")
    output_action_path = os.path.join(args.output_dir, "videos", f"episode_{episode_id:03d}_actions.npy")
    actions = np.array([f['action'] for f in valid_frames], dtype=np.int8)
    np.save(output_action_path, actions)

    first_img = cv2.imread(valid_frames[0]['path'])
    height, width, layers = first_img.shape
    fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
    video = cv2.VideoWriter(output_video_path, fourcc, args.fps, (width, height))
    
    for f_info in valid_frames:
        img = cv2.imread(f_info['path'])
        video.write(img)        
    video.release()

    return {
        "video_path": os.path.relpath(output_video_path, args.output_dir),
        "action_path": os.path.relpath(output_action_path, args.output_dir),
        "num_frames": len(valid_frames),
        "width": width,
        "height": height,
        "episode_id": episode_id
    }

def main():
    os.makedirs(os.path.join(args.output_dir, "videos"), exist_ok=True)
    print("Scanning files...")
    files = glob.glob(os.path.join(args.image_dir, "*.png"))
    
    episodes = defaultdict(list)
    print("Parsing filenames...")
    for f in tqdm(files):
        fname = os.path.basename(f)
        info = parse_filename(fname)
        episodes[info['episode']].append(info)
        
    print(f"Found {len(episodes)} episodes.")
    episodes_list = sorted(episodes.items())
    global process_episode_worker
    process_episode_worker = functools.partial(process_episode)
    
    metadata = []
    num_workers = max(1, cpu_count() - 2)
    print(f"Processing with {num_workers} workers...")
    
    with Pool(num_workers) as p:
        results = list(tqdm(p.imap(process_episode, episodes_list), total=len(episodes_list)))
    metadata = [r for r in results if r is not None]

    metadata_path = os.path.join(args.output_dir, "metadata.json")
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)        
    print(f"Done! Processed {len(metadata)} valid episodes.")
    print(f"Metadata saved to {metadata_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_dir", type=str, required=True, help="Directory containing PNG images")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory for videos and metadata")
    parser.add_argument("--fps", type=int, default=12, help="Frame rate for output videos")
    
    args = parser.parse_args()
    main()
