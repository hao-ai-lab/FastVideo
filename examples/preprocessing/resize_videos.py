import argparse
import logging
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
from moviepy.video.io.VideoFileClip import VideoFileClip
from skimage.transform import resize
from tqdm import tqdm

# Configure logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    handlers=[logging.FileHandler('video_processing.log')])


def is_16_9_ratio(width: int, height: int, tolerance: float = 0.1) -> bool:
    target_ratio = 16 / 9
    actual_ratio = width / height
    return abs(actual_ratio - target_ratio) <= (target_ratio * tolerance)


def resize_video(args_tuple):
    """
    Resize a single video file.
    args_tuple: (input_file, output_dir, width, height, fps)
    """
    input_file, output_dir, width, height, fps = args_tuple
    video = None
    resized = None
    output_file = output_dir / f"{input_file.name}"

    if output_file.exists():
        output_file.unlink()

    video = VideoFileClip(str(input_file))

    if not is_16_9_ratio(video.w, video.h):
        return (input_file.name, "skipped", "Not 16:9")

    def process_frame(frame):
        frame_float = frame.astype(float) / 255.0
        resized = resize(frame_float, (height, width, 3), mode='reflect', anti_aliasing=True, preserve_range=True)
        return (resized * 255).astype(np.uint8)

    resized = video.fl_image(process_frame)
    resized = resized.set_fps(fps)

    resized.write_videofile(str(output_file),
                            codec='libx264',
                            audio_codec='aac',
                            temp_audiofile=f'temp-audio-{input_file.stem}.m4a',
                            remove_temp=True,
                            verbose=False,
                            logger=None,
                            fps=fps)

    return (input_file.name, "success", None)


def process_folder(args):
    input_path = Path(args.input_dir)
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    video_extensions = {'.mp4', '.avi', '.mov', '.mkv', '.webm'}
    video_files = [f for f in input_path.iterdir() if f.is_file() and f.suffix.lower() in video_extensions]

    if not video_files:
        print(f"No video files found in {args.input_dir}")
        return

    print(f"Found {len(video_files)} videos")
    print(f"Target: {args.width}x{args.height} at {args.fps}fps")

    # Prepare arguments for parallel processing
    process_args = [(video_file, output_path, args.width, args.height, args.fps) for video_file in video_files]

    successful = 0
    skipped = 0
    failed = []

    # Use ProcessPoolExecutor instead of ThreadPoolExecutor
    with tqdm(total=len(video_files), desc="Converting videos", dynamic_ncols=True) as pbar:
        # Use max_workers as specified or default to CPU count
        max_workers = args.max_workers
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks
            future_to_file = {executor.submit(resize_video, arg): arg[0] for arg in process_args}

            # Process completed tasks
            for future in as_completed(future_to_file):
                filename, status, message = future.result()
                if status == "success":
                    successful += 1
                elif status == "skipped":
                    skipped += 1
                else:
                    failed.append((filename, message))
                pbar.update(1)

    # Print final summary
    print(f"\nDone! Processed: {successful}, Skipped: {skipped}, Failed: {len(failed)}")
    if failed:
        print("Failed files:")
        for fname, error in failed:
            print(f"- {fname}: {error}")


def parse_args():
    parser = argparse.ArgumentParser(description='Batch resize videos to specified resolution and FPS (16:9 only)')
    parser.add_argument('--input_dir', required=True, help='Input directory containing video files')
    parser.add_argument('--output_dir', required=True, help='Output directory for processed videos')
    parser.add_argument('--width', type=int, default=1280, help='Target width in pixels (default: 848)')
    parser.add_argument('--height', type=int, default=720, help='Target height in pixels (default: 480)')
    parser.add_argument('--fps', type=int, default=30, help='Target frames per second (default: 30)')
    parser.add_argument('--max_workers',
                        type=int,
                        default=4,
                        help='Maximum number of concurrent processes (default: 4)')
    parser.add_argument('--log-level',
                        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                        default='INFO',
                        help='Set the logging level (default: INFO)')
    return parser.parse_args()


def main():
    args = parse_args()
    logging.getLogger().setLevel(getattr(logging, args.log_level))

    if not Path(args.input_dir).exists():
        logging.error(f"Input directory not found: {args.input_dir}")
        return

    start_time = time.perf_counter()
    process_folder(args)
    duration = time.perf_counter() - start_time
    logging.info(f"Batch processing completed in {duration:.2f} seconds")


if __name__ == "__main__":
    main()
