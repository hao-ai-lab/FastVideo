import numpy as np
from PIL import Image
import requests
from io import BytesIO

from fastvideo.models.dits.hyworld.data_utils import generate_crop_size_list

# Target resolution configs (matching HY-WorldPlay)
TARGET_SIZE_CONFIG = {
    "360p": {"bucket_hw_base_size": 480, "bucket_hw_bucket_stride": 16},
    "480p": {"bucket_hw_base_size": 640, "bucket_hw_bucket_stride": 16},
    "720p": {"bucket_hw_base_size": 960, "bucket_hw_bucket_stride": 16},
    "1080p": {"bucket_hw_base_size": 1440, "bucket_hw_bucket_stride": 16},
}


def get_closest_resolution(image_height, image_width, target_resolution="480p"):
    """
    Get closest supported resolution for given image dimensions.
    
    Args:
        image_height: Height of input image
        image_width: Width of input image  
        target_resolution: Target resolution string (e.g., "480p", "720p")
        
    Returns:
        tuple[int, int]: (height, width) of closest supported resolution
    """
    config = TARGET_SIZE_CONFIG[target_resolution]
    bucket_hw_base_size = config["bucket_hw_base_size"]
    bucket_hw_bucket_stride = config["bucket_hw_bucket_stride"]
    
    crop_size_list = generate_crop_size_list(bucket_hw_base_size, bucket_hw_bucket_stride)
    aspect_ratios = np.array([round(float(h) / float(w), 5) for h, w in crop_size_list])
    
    # Find closest aspect ratio
    image_ratio = float(image_height) / float(image_width)
    closest_idx = np.abs(aspect_ratios - image_ratio).argmin()
    closest_size = crop_size_list[closest_idx]
    
    return closest_size[0], closest_size[1]  # (height, width)


def get_resolution_from_image(image_path, target_resolution="480p"):
    """
    Automatically determine resolution from input image.
    
    Args:
        image_path: Path or URL to input image
        target_resolution: Target resolution tier ("480p", "720p", etc.)
        
    Returns:
        tuple[int, int]: (height, width) matching HY-WorldPlay's bucket selection
    """
    # Handle URL inputs
    if isinstance(image_path, str) and image_path.startswith(('http://', 'https://')):
        response = requests.get(image_path)
        response.raise_for_status()
        img = Image.open(BytesIO(response.content))
    else:
        img = Image.open(image_path)
    img_width, img_height = img.size
    return get_closest_resolution(img_height, img_width, target_resolution)

