import os
import cv2
import torch
import json
import numpy as np
import pandas as pd
from PIL import Image
import torchvision.transforms as transforms
from torch.utils.data import Dataset
import csv


def fix_nulls(s):
    """
    Helper generator to remove null characters from input lines.
    Prevents parsing errors caused by invalid null bytes in CSV/JSON files.
    
    Args:
        s: Input iterable containing strings with potential null characters
        
    Yields:
        Strings with null characters replaced by spaces
    """
    for line in s:
        yield line.replace('\0', ' ')

def get_closest_ratio(height: float, width: float, ratios: list, buckets: list):
    """
    Find the closest aspect ratio from predefined buckets
    
    Args:
        height: Image height
        width: Image width
        ratios: List of predefined aspect ratios to match against
        buckets: List of size tuples corresponding to ratios
        
    Returns:
        Tuple containing:
            - Closest matching size bucket
            - Closest ratio value
    """
    aspect_ratio = float(height) / float(width)
    closest_ratio_id = np.abs(ratios - aspect_ratio).argmin()
    closest_ratio = min(ratios, key=lambda ratio: abs(float(ratio) - aspect_ratio))
    return buckets[closest_ratio_id], float(closest_ratio)


def generate_crop_size_list(base_size=256, patch_size=16, max_ratio=4.0):
    """
    Generate valid crop sizes that maintain compatible dimensions with model patches
    
    Args:
        base_size: Base dimension for calculating patch count
        patch_size: Size of model's input patches
        max_ratio: Maximum allowed aspect ratio (height/width)
        
    Returns:
        List of (width, height) tuples representing valid crop sizes
    """
    # Calculate total number of patches from base size
    num_patches = round((base_size / patch_size) ** 2)
    assert max_ratio >= 1.0, "Maximum ratio must be at least 1.0"
    
    crop_size_list = []
    wp, hp = num_patches, 1  # Initialize with maximum width patches
    
    # Generate valid patch combinations
    while wp > 0:
        # Only add sizes that maintain acceptable aspect ratio
        if max(wp, hp) / min(wp, hp) <= max_ratio:
            crop_size_list.append((wp * patch_size, hp * patch_size))
        
        # Move to next valid patch configuration
        if (hp + 1) * wp <= num_patches:
            hp += 1
        else:
            wp -= 1
    return crop_size_list


class VideoCSVDataset(Dataset):
    """
    Dataset class for loading video generation data from CSV files
    
    Handles:
        - CSV parsing with null character handling
        - Loading prompt and metadata
        - Supporting multiple task types (image-to-video, etc.)
    """
    def __init__(self, csv_path, col_name='prompt', task_type=''):
        """
        Args:
            csv_path: Path to CSV file containing dataset metadata
            col_name: Column name containing generation prompts
            task_type: Type of task (e.g., "i2v" for image-to-video)
        """
        # Read CSV with null character handling
        with open(csv_path, 'r', newline="\n", encoding='utf-8-sig') as csvfile:
            self.dataset = list(csv.DictReader(fix_nulls(csvfile), delimiter=';'))
        
        self.col_name = col_name
        self.task_type = task_type

    def __len__(self):
        """Return total number of samples in dataset"""
        return len(self.dataset)

    def __getitem__(self, idx):
        """
        Get dataset item by index
        
        Args:
            idx: Index of sample to retrieve
            
        Returns:
            Dictionary containing:
                - Prompt and metadata
                - Paths to auxiliary files (npy, video, poses)
                - Index for tracking outputs
        """
        example = {}
        example["prompt"] = self.dataset[idx][self.col_name]
        example['seed'] = int(self.dataset[idx]['seed'])
        example['index'] = self.dataset[idx]['index']
        
        # Add optional auxiliary paths if present in CSV
        if "npy_path" in self.dataset[idx]:
            example['npy_path'] = self.dataset[idx]['npy_path']
        if "video_path" in self.dataset[idx]:
            example['video_path'] = self.dataset[idx]['video_path']
        if "monst3r_poses" in self.dataset[idx]:
            example['monst3r_poses'] = self.dataset[idx]['monst3r_poses']
            
        # Add image reference path for image-to-video tasks
        if self.task_type == "i2v":
            example['ref_image'] = self.dataset[idx]['ref_image_path']
            
        return example


class JsonDataset(object):
    """
    Dataset class for loading data from JSON files and image sequences
    
    Handles:
        - Reading image data from multiple formats
        - Preprocessing for model compatibility
        - Generating conditional and unconditional inputs
    """
    def __init__(self, args):
        """
        Args:
            args: Command-line arguments containing configuration
        """
        self.args = args
        self.data_list = args.input
        self.pad_color = (255, 255, 255)  # White padding
        self.llava_size = (336, 336)      # Standard size for LLaVA model
        self.ref_size = (args.video_size[1], args.video_size[0])  # Reference output size
        
        # Get list of data paths from input list or single file
        if self.data_list.endswith('.list'):
            self.data_paths = [line.strip() for line in open(self.data_list, 'r')] if self.data_list else []
        else:
            self.data_paths = [self.data_list]
        
        # Transformation pipeline for LLaVA model input
        self.llava_transform = transforms.Compose(
            [
                transforms.Resize(self.llava_size, interpolation=transforms.InterpolationMode.BILINEAR),
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.48145466, 0.4578275, 0.4082107),
                    (0.26862954, 0.26130258, 0.27577711)
                ),
            ]
        )
        
    def __len__(self):
        """Return total number of data items"""
        return len(self.data_paths)
    
    def read_image(self, image_path):
        """
        Read image from path with fallback handling
        
        Args:
            image_path: Path to image file or dictionary containing path
            
        Returns:
            Tuple of (LLaVA-formatted image, reference-sized image)
        """
        # Extract path from dictionary if needed
        if isinstance(image_path, dict):
            image_path = image_path['seg_item_image_path']

        try:
            # Primary method: OpenCV for faster reading
            face_image_masked = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
        except:
            # Fallback: PIL for special formats
            face_image_masked = Image.open(image_path).convert('RGB')

        # Prepare images for different processing stages
        cat_face_image = pad_image(face_image_masked.copy(), self.ref_size)
        llava_face_image = pad_image(face_image_masked.copy(), self.llava_size)
        return llava_face_image, cat_face_image

    def __getitem__(self, idx):
        """
        Get preprocessed data item by index
        
        Args:
            idx: Index of item to retrieve
            
        Returns:
            Dictionary containing:
                - Preprocessed tensors for model input
                - Metadata (prompt, index, paths)
        """
        data_path = self.data_paths[idx]
        data_name = os.path.basename(os.path.splitext(data_path)[0])
        
        # Load data from JSON or use default parameters
        if data_path.endswith('.json'):
            data = json.load(open(data_path, 'r'))
            llava_item_image, cat_item_image = self.read_image(data)
            item_prompt = data['item_prompt']
            seed = data['seed']
            prompt = data['prompt']
            negative_prompt = data.get('negative_prompt', '')  # Default to empty string
        else:
            # Handle non-JSON data (direct image files)
            llava_item_image, cat_item_image = self.read_image(data_path)
            item_prompt = 'object'
            seed = self.args.seed
            prompt = self.args.pos_prompt
            negative_prompt = self.args.neg_prompt
            
        # Convert to tensors with appropriate transformations
        llava_item_tensor = self.llava_transform(Image.fromarray(llava_item_image.astype(np.uint8)))
        cat_item_tensor = torch.from_numpy(cat_item_image.copy()).permute((2, 0, 1)) / 255.0  # Normalize to [0,1]

        # Create unconditional input (white background)
        uncond_llava_item_image = np.ones_like(llava_item_image) * 255
        uncond_llava_item_tensor = self.llava_transform(Image.fromarray(uncond_llava_item_image))
        
        # Assemble final batch dictionary
        return {
            "pixel_value_llava": llava_item_tensor,
            "uncond_pixel_value_llava": uncond_llava_item_tensor,
            "pixel_value_ref": cat_item_tensor,
            "prompt": prompt,
            "negative_prompt": negative_prompt,
            "seed": seed,
            "name": item_prompt,
            'data_name': data_name,
            'index': [idx]  # Index for output tracking
        }
