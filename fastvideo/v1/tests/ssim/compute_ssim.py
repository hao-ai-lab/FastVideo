# SPDX-License-Identifier: Apache-2.0
import argparse
import os

import numpy as np
import torch
from pytorch_msssim import ms_ssim, ssim
from torchvision.io import read_video


def compute_video_ssim_torchvision(video1_path, video2_path, use_ms_ssim=True):
    print(f"Computing SSIM between {video1_path} and {video2_path}...")

    frames1, _, _ = read_video(video1_path,
                               pts_unit='sec',
                               output_format="TCHW")
    frames2, _, _ = read_video(video2_path,
                               pts_unit='sec',
                               output_format="TCHW")

    # Ensure same number of frames
    min_frames = min(frames1.shape[0], frames2.shape[0])
    frames1 = frames1[:min_frames]
    frames2 = frames2[:min_frames]

    frames1 = frames1.float() / 255.0
    frames2 = frames2.float() / 255.0

    if torch.cuda.is_available():
        frames1 = frames1.cuda()
        frames2 = frames2.cuda()

    ssim_values = []

    # Process each frame individually
    for i in range(min_frames):
        img1 = frames1[i:i + 1]
        img2 = frames2[i:i + 1]

        with torch.no_grad():
            if use_ms_ssim:
                value = ms_ssim(img1, img2, data_range=1.0)
            else:
                value = ssim(img1, img2, data_range=1.0)

            ssim_values.append(value.item())

    if ssim_values:
        mean_ssim = np.mean(ssim_values)
        min_ssim = np.min(ssim_values)
        max_ssim = np.max(ssim_values)
        min_frame_idx = np.argmin(ssim_values)
        max_frame_idx = np.argmax(ssim_values)

        print(f"Mean SSIM: {mean_ssim:.4f}")
        print(f"Min SSIM: {min_ssim:.4f} (at frame {min_frame_idx})")
        print(f"Max SSIM: {max_ssim:.4f} (at frame {max_frame_idx})")

        return mean_ssim, min_ssim, max_ssim
    else:
        print('No SSIM values calculated')
        return 0, 0, 0


def compare_folders(reference_folder, generated_folder, use_ms_ssim=True):
    """
    Compare videos with the same filename between reference_folder and generated_folder
    """
    reference_videos = [
        f for f in os.listdir(reference_folder) if f.endswith('.mp4')
    ]

    results = {}

    for video_name in reference_videos:
        ref_path = os.path.join(reference_folder, video_name)
        gen_path = os.path.join(generated_folder, video_name)

        if os.path.exists(gen_path):
            print(f"\nComparing {video_name}...")
            try:
                ssim_value = compute_video_ssim_torchvision(
                    ref_path, gen_path, use_ms_ssim)
                results[video_name] = ssim_value
            except Exception as e:
                print(f"Error comparing {video_name}: {e}")
                results[video_name] = None
        else:
            print(
                f"\nSkipping {video_name} - no matching file in generated folder"
            )

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Compare videos using SSIM/MS-SSIM metrics')
    parser.add_argument('--reference',
                        '-r',
                        type=str,
                        help='Path to reference videos directory')
    parser.add_argument('--generated',
                        '-g',
                        type=str,
                        help='Path to generated videos directory')
    parser.add_argument('--use-ms-ssim',
                        action='store_true',
                        help='Use MS-SSIM instead of SSIM')
    args = parser.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))

    reference_folder = args.reference if args.reference else os.path.join(
        script_dir, 'reference_videos')
    generated_folder = args.generated if args.generated else os.path.join(
        script_dir, 'generated_videos')

    if not os.path.exists(reference_folder):
        print(f"ERROR: Reference folder {reference_folder} does not exist!")
        exit(1)

    if not os.path.exists(generated_folder):
        print(f"ERROR: Generated folder {generated_folder} does not exist!")
        exit(1)

    print(f"Comparing videos between {reference_folder} and {generated_folder}")
    results = compare_folders(reference_folder, generated_folder,
                              args.use_ms_ssim)

    print("\n===== SSIM Results Summary =====")
    for video_name, ssim_value in results.items():
        if ssim_value is not None:
            print(
                f"{video_name}: {ssim_value[0]:.4f}, Min SSIM: {ssim_value[1]:.4f}, Max SSIM: {ssim_value[2]:.4f}"
            )
        else:
            print(f"{video_name}: Error during comparison")

    valid_ssims = [v for v in results.values() if v is not None]
    if valid_ssims:
        avg_ssim = np.mean([v[0] for v in valid_ssims])
        print(f"\nAverage SSIM across all videos: {avg_ssim:.4f}")
    else:
        print("\nNo valid SSIM values to average")
