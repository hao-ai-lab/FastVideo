from paddleocr import PaddleOCR
import torch
import numpy as np
from Levenshtein import distance
from typing import Any, Union, List
from PIL import Image

from fastvideo.training.rl.rewards.base import BaseRewardModel
from fastvideo.logger import init_logger

logger = init_logger(__name__)

class OcrScorerVideo(BaseRewardModel):
    """
    OCR reward model for multi-frame video OCR evaluation.

    This model evaluates multiple frames across the video sequence,
    sampling frames at a specified interval and averaging the OCR scores.
    """

    def __init__(
        self,
        model_path: str | None = None,
        device: str = "cuda",
        frame_interval: int = 4
    ):
        """
        OCR reward calculator for videos

        Args:
            model_path: Not used for PaddleOCR (kept for BaseRewardModel compatibility)
            device: Device string (used to determine use_gpu if not explicitly set)
            frame_interval: Sample every Nth frame (default: 4)
        """
        super().__init__(model_path=model_path, device=device)

        self.frame_interval = frame_interval
        self.ocr = PaddleOCR(
            use_angle_cls=False,
            lang="en",
            device=device,
        )

        logger.info(
            "Initialized OcrScorerVideo (device=%s, frame_interval=%d)",
            device,
            frame_interval
        )

    def _process_single_video(self, frames: Union[np.ndarray | List[Image.Image]], prompt: str) -> float:
        """
        Process a single video (list of frames) and return its OCR reward.

        Args:
            frames: List of frames (numpy arrays or PIL Images)
            prompt: Text prompt containing target OCR text in quotes

        Returns:
            Average reward across positive-scoring frames
        """
        # Extract target text from prompt
        try:
            target_text = prompt.split('"')[1].replace(' ', '').lower()
        except IndexError:
            logger.warning("Failed to extract quoted text from prompt: %s", prompt)
            target_text = prompt.replace(' ', '').lower()

        if not target_text:
            return 0.0

        num_frames = len(frames)
        frame_rewards = []

        # Sample frames at specified interval
        for frame_idx in range(0, num_frames, self.frame_interval):
            frame = frames[frame_idx]
            if isinstance(frame, Image.Image):
                frame = np.array(frame.convert("RGB"))

            if frame.dtype != np.uint8:
                frame = (frame * 255).astype(np.uint8) if frame.max() <= 1.0 else frame.astype(np.uint8)

            # Handle frame shape: transpose [C, H, W] to [H, W, C]
            if frame.ndim == 3 and frame.shape[0] == 3:
                frame = np.transpose(frame, (1, 2, 0))

            # Run OCR
            try:
                result = self.ocr.predict(frame)
                page = result[0] if result else {}
                recognized_text = "".join(
                    text if score > 0 else ""
                    for text, score in zip(page["rec_texts"], page["rec_scores"])
                )
            except Exception as e:
                logger.debug("OCR failed on frame %d: %s", frame_idx, str(e))
                recognized_text = ''

            recognized_text = recognized_text.replace(' ', '').lower()
            dist = min(distance(recognized_text, target_text), len(target_text))
            reward = 1.0 - dist / len(target_text)
            if reward > 0:
                frame_rewards.append(reward)

        return sum(frame_rewards) / len(frame_rewards) if frame_rewards else 0.0

    @torch.no_grad()
    def compute_reward(
        self,
        videos: Union[List[np.ndarray] | List[List[Image.Image]]],
        prompts: List[str],
        **kwargs: Any
    ) -> torch.Tensor:
        """
        Calculate OCR reward by evaluating sampled frames across the video.

        Args:
            videos: List of videos, where each video is a list of frames (numpy arrays or PIL Images)
            prompts: List of text prompts containing target OCR text in quotes
            **kwargs: Additional arguments

        Returns:
            Reward tensor [B] with averaged OCR similarity scores across frames
        """
        assert len(videos) == len(prompts), f"Number of videos ({len(videos)}) must match number of prompts ({len(prompts)})"

        rewards = [self._process_single_video(video, prompt) for video, prompt in zip(videos, prompts)]

        # Convert to tensor
        return torch.tensor(rewards, dtype=torch.float32, device=self.device)
