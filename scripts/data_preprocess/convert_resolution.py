import cv2
import numpy as np
import os
from tqdm import tqdm

def convert_video(input_path, output_path, target_height=480, target_fps=30):
    """
    Convert video resolution and frame rate using OpenCV with progress bar.
    
    Args:
        input_path (str): Path to input video file
        output_path (str): Path to save converted video
        target_height (int): Target height in pixels (default: 480p)
        target_fps (int): Target frames per second (default: 30fps)
    """
    try:
        # Open the input video
        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            raise Exception("Error: Could not open input video")

        # Get original video properties
        original_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        original_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        original_fps = int(cap.get(cv2.CAP_PROP_FPS))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Calculate new width to maintain aspect ratio
        aspect_ratio = original_width / original_height
        target_width = int(target_height * aspect_ratio)
        target_width = target_width - (target_width % 2)  # Ensure even width

        # Print conversion details
        print(f"\nOriginal: {original_width}x{original_height} at {original_fps}fps")
        print(f"Target: {target_width}x{target_height} at {target_fps}fps")
        print(f"Total frames to process: {total_frames}")

        # Create video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # or 'XVID' for AVI
        out = cv2.VideoWriter(
            output_path,
            fourcc,
            target_fps,
            (target_width, target_height)
        )

        if not out.isOpened():
            raise Exception("Error: Could not create output video file")

        # Calculate frame sampling rate for FPS conversion
        frame_sampling = original_fps / target_fps

        # Calculate expected output frames
        expected_frames = int(total_frames * (target_fps / original_fps))
        
        # Initialize tqdm progress bar
        pbar = tqdm(total=expected_frames, 
                   desc="Converting video",
                   unit="frames",
                   dynamic_ncols=True,
                   bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]")

        # Process the video
        frame_count = 0
        processed_count = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Only process frames according to target FPS
            if frame_count % frame_sampling < 1:
                # Resize frame
                resized_frame = cv2.resize(
                    frame,
                    (target_width, target_height),
                    interpolation=cv2.INTER_AREA
                )
                
                # Write the frame
                out.write(resized_frame)
                processed_count += 1
                pbar.update(1)

            frame_count += 1

        # Clean up
        pbar.close()
        cap.release()
        out.release()

        # Calculate compression ratio
        input_size = os.path.getsize(input_path) / (1024 * 1024)  # MB
        output_size = os.path.getsize(output_path) / (1024 * 1024)  # MB
        compression_ratio = (1 - (output_size / input_size)) * 100

        # Print completion message
        print(f"\nConversion completed successfully!")
        print(f"Input size: {input_size:.1f}MB")
        print(f"Output size: {output_size:.1f}MB")
        print(f"Compression ratio: {compression_ratio:.1f}%")
        print(f"Processed frames: {processed_count}")
        print(f"Output saved to: {output_path}")

    except Exception as e:
        print(f"\nAn error occurred: {str(e)}")
        # Clean up in case of error
        if 'cap' in locals():
            cap.release()
        if 'out' in locals():
            out.release()
        if 'pbar' in locals():
            pbar.close()

# Example usage
if __name__ == "__main__":
    input_video = "./videos/hsh.mp4"
    output_video = "output_480p_30fps.mp4"
    
    convert_video(input_video, output_video)

