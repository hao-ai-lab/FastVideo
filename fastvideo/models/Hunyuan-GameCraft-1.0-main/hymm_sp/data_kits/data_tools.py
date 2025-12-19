import os
import cv2
import torch
import numpy as np
import imageio
import torchvision
from einops import rearrange


def save_videos_grid(videos: torch.Tensor, path: str, rescale=False, n_rows=6, fps=8, quality=8):
    """
    Saves a batch of videos as a grid animation in GIF or video format.
    
    Args:
        videos (torch.Tensor): Input video tensor with shape (batch, channels, time, height, width)
        path (str): Output file path (e.g., "output/videos.gif")
        rescale (bool): If True, rescales video values from [-1, 1] to [0, 1]
        n_rows (int): Number of rows in the grid layout
        fps (int): Frames per second for the output animation
        quality (int): Quality parameter for imageio (1-10, higher = better quality)
    
    Process:
        1. Rearranges tensor dimensions to (time, batch, channels, height, width)
        2. For each frame in time:
            a. Creates a grid of videos using torchvision.utils.make_grid
            b. Adjusts dimensions to (height, width, channels)
            c. Rescales values if needed
            d. Converts to 8-bit uint8 format (0-255)
        3. Saves frames as an animated GIF/video using imageio
    """
    # Rearrange dimensions to (time, batch, channels, height, width) for frame-wise processing
    videos = rearrange(videos, "b c t h w -> t b c h w")
    outputs = []  # Stores processed frames for animation
    
    for frame in videos:
        # Create a grid of videos with n_rows rows
        grid = torchvision.utils.make_grid(frame, nrow=n_rows)
        
        # Convert from (channels, height, width) to (height, width, channels)
        grid = grid.transpose(0, 1).transpose(1, 2).squeeze(-1)
        
        # Rescale from [-1, 1] to [0, 1] if needed (common in GAN outputs)
        if rescale:
            grid = (grid + 1.0) / 2.0
        
        # Clamp values to valid range [0, 1] and convert to 8-bit uint8 (0-255)
        grid = torch.clamp(grid, 0, 1)
        grid_np = (grid * 255).numpy().astype(np.uint8)
        
        outputs.append(grid_np)
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(path), exist_ok=True)
    
    # Save frames as an animated GIF/video
    imageio.mimsave(path, outputs, fps=fps, quality=quality)


def pad_image(crop_img, size, color=(255, 255, 255), resize_ratio=1):
    """
    Resizes and pads an image to fit a target size while preserving aspect ratio.
    
    Args:
        crop_img (np.ndarray): Input image (shape: [height, width, channels])
        size (tuple): Target size in (width, height) format
        color (tuple): RGB color for padding (default: white)
        resize_ratio (float): Scaling factor for resizing before padding (0-1)
    
    Returns:
        np.ndarray: Padded image with shape (target_height, target_width, channels)
    
    Process:
        1. Calculates scaling factors to fit image within target size
        2. Resizes image while preserving aspect ratio
        3. Adds padding to reach exact target size, centering the resized image
    """
    # Get input image dimensions
    crop_h, crop_w = crop_img.shape[:2]
    target_w, target_h = size  # Target dimensions (width, height)
    
    # Calculate scaling factors to fit image within target size
    scale_h = target_h / crop_h  # Scale needed to fit height
    scale_w = target_w / crop_w  # Scale needed to fit width
    
    # Choose the smaller scale to avoid exceeding target dimensions
    if scale_w > scale_h:
        # Height is the limiting factor: resize based on height
        resize_h = int(target_h * resize_ratio)
        resize_w = int(crop_w / crop_h * resize_h)  # Preserve aspect ratio
    else:
        # Width is the limiting factor: resize based on width
        resize_w = int(target_w * resize_ratio)
        resize_h = int(crop_h / crop_w * resize_w)  # Preserve aspect ratio
    
    # Resize the image using OpenCV
    resized_img = cv2.resize(crop_img, (resize_w, resize_h))
    
    # Calculate padding needed to reach target size (centered)
    pad_left = (target_w - resize_w) // 2
    pad_top = (target_h - resize_h) // 2
    pad_right = target_w - resize_w - pad_left  # Ensure total width matches target
    pad_bottom = target_h - resize_h - pad_top  # Ensure total height matches target
    
    # Add padding with the specified color
    padded_img = cv2.copyMakeBorder(
        resized_img,
        top=pad_top,
        bottom=pad_bottom,
        left=pad_left,
        right=pad_right,
        borderType=cv2.BORDER_CONSTANT,
        value=color
    )
    
    return padded_img
