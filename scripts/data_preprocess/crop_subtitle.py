import argparse
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import time
import logging
from tqdm import tqdm
from moviepy.editor import VideoFileClip
import numpy as np
from skimage.transform import resize

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', handlers=[logging.FileHandler('video_processing.log'), logging.StreamHandler()])

def resize_to_original(frame, original_height):
    """Resize frame back to original height while maintaining width."""
    current_height = frame.shape[0]
    current_width = frame.shape[1]
    
    # Calculate width to maintain aspect ratio
    target_width = current_width
    
    resized = resize(frame, (original_height, target_width), 
                    preserve_range=True, anti_aliasing=True)
    return np.array(resized, dtype=np.uint8)

def crop_video(input_file: Path, output_dir: Path, args):
    video = None
    cropped = None
    try:
        output_file = output_dir / f"cropped_{input_file.name}"
        
        if output_file.exists():
            output_file.unlink()
        
        video = VideoFileClip(str(input_file))
        original_height = video.h
        original_width = video.w
        
        def process_frame(frame):
            # Crop the bottom portion
            cropped = frame[0:original_height-args.crop_bottom, :]
            
            # Resize back to original height if resize flag is True
            if args.resize:
                return resize_to_original(cropped, original_height)
            return cropped
        
        cropped = video.fl_image(process_frame)
        
        logging.info(f"Processing: {input_file.name}")
        logging.info(f"Original size: {original_width}x{original_height}")
        if not args.resize:
            new_height = original_height - args.crop_bottom
            logging.info(f"Output size: {original_width}x{new_height}")
        else:
            logging.info(f"Output size: {original_width}x{original_height} (restored to original height)")
        
        cropped.write_videofile(
            str(output_file),
            codec='libx264',
            audio_codec='aac',
            temp_audiofile='temp-audio.m4a',
            remove_temp=True,
            verbose=False,
            logger=None
        )
        
        logging.info(f"Successfully processed: {input_file.name}")
        return True
        
    except Exception as e:
        logging.error(f"Error processing {input_file.name}: {str(e)}")
        return False
    finally:
        try:
            if video is not None:
                video.close()
            if cropped is not None:
                cropped.close()
        except:
            pass

def process_folder(args):
    input_path = Path(args.input_dir)
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    video_extensions = {'.mp4', '.avi', '.mov', '.mkv', '.webm'}
    video_files = [f for f in input_path.iterdir() if f.is_file() and f.suffix.lower() in video_extensions]
    
    if not video_files:
        logging.error(f"No video files found in {args.input_dir}")
        return
    
    existing_files = [f for f in video_files if (output_path / f"cropped_{f.name}").exists()]
    if existing_files:
        logging.info(f"Following files will be overwritten: {['cropped_' + f.name for f in existing_files]}")
    
    logging.info(f"Found {len(video_files)} video files to process")
    logging.info(f"Resize mode: {'enabled - will restore to original height' if args.resize else 'disabled'}")
    
    if args.max_workers == 1:
        for video_file in video_files:
            crop_video(video_file, output_path, args)
    else:
        with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
            futures = [executor.submit(crop_video, video_file, output_path, args) 
                      for video_file in video_files]
            for future in futures:
                future.result()

def parse_args():
    parser = argparse.ArgumentParser(description='Batch process videos to crop subtitles and maintain aspect ratio')
    parser.add_argument('--input_dir', required=True, help='Input directory containing video files')
    parser.add_argument('--output_dir', required=True, help='Output directory for processed videos')
    parser.add_argument('--crop-bottom', type=int, default=40, help='Number of pixels to crop from bottom (default: 40)')
    parser.add_argument('--max-workers', type=int, default=4, help='Maximum number of concurrent processes (default: 4)')
    parser.add_argument('--log-level', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'], default='INFO', help='Set the logging level (default: INFO)')
    parser.add_argument('--resize', action='store_true', help='Enable resizing back to original height after cropping')
    return parser.parse_args()

def main():
    args = parse_args()
    logging.getLogger().setLevel(getattr(logging, args.log_level))
    
    if not Path(args.input_dir).exists():
        logging.error(f"Input directory not found: {args.input_dir}")
        return
    
    start_time = time.time()
    process_folder(args)
    duration = time.time() - start_time
    logging.info(f"Batch processing completed in {duration:.2f} seconds")

if __name__ == "__main__":
    main()