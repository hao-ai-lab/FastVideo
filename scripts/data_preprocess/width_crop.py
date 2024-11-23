import argparse
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import time
import logging
from moviepy.editor import VideoFileClip
import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('video_processing.log'),
        logging.StreamHandler()
    ]
)

def crop_video(input_file: Path, output_dir: Path, args):
    """
    Crop video width from both sides
    """
    video = None
    cropped = None
    try:
        output_file = output_dir / f"{input_file.name}"
        
        if output_file.exists():
            output_file.unlink()
        
        video = VideoFileClip(str(input_file))
        original_height = video.h
        original_width = video.w
        
        def process_frame(frame):
            # Crop 2 pixels from both sides
            crop_size = args.crop_sides
            return frame[:, crop_size:original_width-crop_size]
        
        cropped = video.fl_image(process_frame)
        
        logging.info(f"Processing: {input_file.name}")
        logging.info(f"Original size: {original_width}x{original_height}")
        new_width = original_width - (args.crop_sides * 2)
        logging.info(f"Output size: {new_width}x{original_height}")
        
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
    logging.info(f"Will crop {args.crop_sides} pixels from each side")
    
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
    parser = argparse.ArgumentParser(description='Batch process videos to crop width while maintaining height')
    parser.add_argument('--input_dir', required=True, help='Input directory containing video files')
    parser.add_argument('--output_dir', required=True, help='Output directory for processed videos')
    parser.add_argument('--crop-sides', type=int, default=2, help='Number of pixels to crop from each side (default: 2)')
    parser.add_argument('--max-workers', type=int, default=4, help='Maximum number of concurrent processes (default: 4)')
    parser.add_argument('--log-level', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'], default='INFO', 
                       help='Set the logging level (default: INFO)')
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