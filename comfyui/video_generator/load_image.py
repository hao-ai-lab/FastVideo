import hashlib
import os

import folder_paths
import numpy as np
import torch
from PIL import Image, ImageOps, ImageSequence

from .node_helpers import pillow


class LoadImagePath:

    @classmethod
    def INPUT_TYPES(s):
        input_dir = folder_paths.get_input_directory()
        files = [
            f for f in os.listdir(input_dir)
            if os.path.isfile(os.path.join(input_dir, f))
        ]
        files = folder_paths.filter_files_content_types(files, ["image"])
        return {
            "required": {
                "image": (sorted(files), {
                    "image_upload": True
                })
            },
        }

    CATEGORY = "fastvideo"

    RETURN_TYPES = ("STRING", "IMAGE", "MASK")
    RETURN_NAMES = ("image_path", "IMAGE", "MASK")
    FUNCTION = "load_image"

    def load_image(self, image):
        image_path = folder_paths.get_annotated_filepath(image)

        img = pillow(Image.open, image_path)

        output_images: list[torch.Tensor] = []
        output_masks: list[torch.Tensor] = []
        w, h = None, None

        excluded_formats = ['MPO']

        for i in ImageSequence.Iterator(img):
            processed_image = pillow(ImageOps.exif_transpose, i)
            if processed_image is None:
                continue

            if processed_image.mode == 'I':
                processed_image = processed_image.point(lambda i: i * (1 / 255))
            image = processed_image.convert("RGB")

            if len(output_images) == 0:
                w = image.size[0]
                h = image.size[1]

            if image.size[0] != w or image.size[1] != h:
                continue

            image = np.array(image).astype(np.float32) / 255.0
            image = torch.from_numpy(image)[
                None,
            ]
            if 'A' in processed_image.getbands():
                mask = np.array(processed_image.getchannel('A')).astype(
                    np.float32) / 255.0
                mask = 1. - torch.from_numpy(mask)
            elif processed_image.mode == 'P' and 'transparency' in processed_image.info:
                mask = np.array(
                    processed_image.convert('RGBA').getchannel('A')).astype(
                        np.float32) / 255.0
                mask = 1. - torch.from_numpy(mask)
            else:
                mask = torch.zeros((64, 64), dtype=torch.float32, device="cpu")
            output_images.append(image)
            output_masks.append(mask.unsqueeze(0))

        if len(output_images) > 1 and img.format not in excluded_formats:
            output_image = torch.cat(output_images, dim=0)
            output_mask = torch.cat(output_masks, dim=0)
        else:
            output_image = output_images[0]
            output_mask = output_masks[0]

        return (image_path, output_image, output_mask)

    @classmethod
    def IS_CHANGED(s, image):
        image_path = folder_paths.get_annotated_filepath(image)
        m = hashlib.sha256()
        with open(image_path, 'rb') as f:
            m.update(f.read())
        return m.digest().hex()

    @classmethod
    def VALIDATE_INPUTS(s, image):
        if not folder_paths.exists_annotated_filepath(image):
            return "Invalid image file: {}".format(image)

        return True
