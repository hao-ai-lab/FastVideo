import os
import cv2


def main():
    input_video = "footsies-dataset/videos/episode_000_part_000.mp4"
    output_dir = "footsies-dataset/validate"
    output_video = os.path.join(output_dir, "episode_000_part_000_cropped.mp4")
    output_image = os.path.join(output_dir, "episode_000_part_000_first_frame.png")

    os.makedirs(output_dir, exist_ok=True)
    cap = cv2.VideoCapture(input_video)
    if not cap.isOpened():
        print(f"Error: Cannot open video {input_video}")
        return

    orig_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"Original video: {orig_width}x{orig_height}, {fps} fps, {total_frames} frames")

    target_width = 640
    target_height = 352

    target_aspect = target_width / target_height
    orig_aspect = orig_width / orig_height

    if orig_aspect > target_aspect:
        scale_height = target_height
        scale_width = int(orig_width * (target_height / orig_height))
        crop_x = (scale_width - target_width) // 2
        crop_y = 0
    else:
        scale_width = target_width
        scale_height = int(orig_height * (target_width / orig_width))
        crop_x = 0
        crop_y = (scale_height - target_height) // 2

    print(f"Scale to: {scale_width}x{scale_height}, then crop at ({crop_x}, {crop_y})")
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video, fourcc, fps, (target_width, target_height))
    first_frame_saved = False
    frame_count = 0

    print("Processing video...")
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        scaled = cv2.resize(frame, (scale_width, scale_height))
        cropped = scaled[crop_y:crop_y + target_height, crop_x:crop_x + target_width]

        out.write(cropped)
        if not first_frame_saved:
            cv2.imwrite(output_image, cropped)
            first_frame_saved = True
            print(f"First frame saved to: {output_image}")

        frame_count += 1
        if frame_count % 100 == 0:
            print(f"Processed {frame_count}/{total_frames} frames...")

    cap.release()
    out.release()

    print(f"\nDone! Processed {frame_count} frames.")
    print(f"  Video: {output_video}")
    print(f"  Image: {output_image}")


if __name__ == "__main__":
    main()
