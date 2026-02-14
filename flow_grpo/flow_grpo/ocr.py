from paddleocr import PaddleOCR
import torch
import numpy as np
from Levenshtein import distance
from typing import List, Union, Tuple
from PIL import Image

class OcrScorer:
    def __init__(self, use_gpu: bool = False):
        """
        OCR reward calculator
        :param use_gpu: Whether to use GPU acceleration for PaddleOCR
        """
        self.ocr = PaddleOCR(
            use_angle_cls=False,
            lang="en",
            use_gpu=use_gpu,
            show_log=False  # Disable unnecessary log output
        )

    @torch.no_grad()
    def __call__(self, 
                images: Union[List[Image.Image], List[np.ndarray]], 
                prompts: List[str]) -> torch.Tensor:
        """
        Calculate OCR reward
        :param images: List of input images (PIL or numpy format)
        :param prompts: Corresponding target text list
        :return: Reward tensor (CPU)
        """
        prompts = [prompt.split('"')[1] for prompt in prompts]
        rewards = []
        # Ensure input lengths are consistent
        assert len(images) == len(prompts), "Images and prompts must have the same length"
        for img, prompt in zip(images, prompts):
            # Convert image format
            if isinstance(img, Image.Image):
                img = np.array(img)
            
            try:
                # OCR recognition
                result = self.ocr.ocr(img, cls=False)
                # Extract recognized text (handle possible multi-line results)
                recognized_text = ''.join([res[1][0] if res[1][1] > 0 else '' for res in result[0]]) if result[0] else ''
                
                recognized_text = recognized_text.replace(' ', '').lower()
                prompt = prompt.replace(' ', '').lower()
                if prompt in recognized_text:
                    dist = 0
                else:
                    dist = distance(recognized_text, prompt)
                # Recognized many unrelated characters, only add one character penalty
                if dist > len(prompt):
                    dist = len(prompt)
                
            except Exception as e:
                # Error handling (e.g., OCR parsing failure)
                print(f"OCR processing failed: {str(e)}")
                dist = len(prompt)  # Maximum penalty
            reward = 1-dist/(len(prompt))
            rewards.append(reward)

        return rewards

class OcrScorer_video_or_image:
    def __init__(self, use_gpu: bool = False):
        """
        OCR reward calculator
        :param use_gpu: Whether to use GPU acceleration for PaddleOCR
        """
        self.ocr = PaddleOCR(
            use_angle_cls=False,
            lang="en",
            use_gpu=use_gpu,
            show_log=False  # Disable unnecessary log output
        )
        self.frame_interval = 4

    @torch.no_grad()
    def __call__(self, images: Union[List[Image.Image], List[np.ndarray]], prompts: List[str]) -> Tuple[List[float], torch.Tensor]:
        """
        :param images: List of images or videos (each video as np.ndarray of shape [F, H, W, C])
        :param prompts: List of prompts containing target text
        :return: (List of OCR rewards, Tensor of attention regions)
        """
        # prompts = [prompt.split('"')[1] for prompt in prompts]
        # assert len(images) == len(prompts), "Mismatch between images and prompts."

        rewards = []
        for img, prompt in zip(images, prompts):
            prompt = prompt.replace(' ', '').lower()
            frame_rewards = []

            # Handle video: shape (F, H, W, C)
            if isinstance(img, np.ndarray) and img.ndim == 4:
                sampled_frames = img[::self.frame_interval]
            else:
                sampled_frames = [img]

            for frame in sampled_frames:
                region = None
                if isinstance(frame, Image.Image):
                    frame = np.array(frame)
                try:
                    result = self.ocr.ocr(frame, cls=False)
                    text = ''.join([res[1][0] if res[1][1] > 0 else '' for res in result[0]]) if result[0] else ''
                    text = text.replace(' ', '').lower()

                    dist = distance(text, prompt)
                    dist = min(dist, len(prompt))
    
                except Exception as e:
                    print(f"OCR failed on frame: {e}")
                    dist = len(prompt)

                reward = 1 - dist / len(prompt)
                if reward > 0:
                    frame_rewards.append(reward)

            if frame_rewards:
                rewards.append(sum(frame_rewards) / len(frame_rewards))
            else:
                rewards.append(0.0)

        return rewards

if __name__ == "__main__":
    example_image_path = "media_images_eval_images_499_ef42de47b8ec98892954.jpg"
    example_image = Image.open(example_image_path)
    example_prompt = 'New York Skyline with "Hello World" written with fireworks on the sky'
    # Instantiate scorer
    scorer = OcrScorer(use_gpu=False)

    # Call scorer and print result
    reward = scorer([example_image], [example_prompt])
    print(f"OCR Reward: {reward}")