from paddleocr import PaddleOCR
import torch
import numpy as np
from Levenshtein import distance
from typing import Any
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

    def __init__(self,
                 model_path: str | None = None,
                 device: str = "cpu",
                 frame_interval: int = 4):
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
            use_gpu=False,
            show_log=False  # Disable unnecessary log output
        )

        logger.info("Initialized OcrScorerVideo (device=%s, frame_interval=%d)",
                    device, frame_interval)

    def _process_single_video(self, video_tensor: torch.Tensor,
                              prompt: str) -> float:
        """
        Process a single video tensor and return its OCR reward.

        Args:
            video_tensor: Video tensor of shape [C, T, H, W]
            prompt: Text prompt containing target OCR text in quotes

        Returns:
            Average reward across positive-scoring frames
        """
        # Extract target text from prompt
        try:
            target_text = prompt.split('"')[1].replace(' ', '').lower()
        except IndexError:
            logger.warning("Failed to extract quoted text from prompt: %s",
                           prompt)
            target_text = prompt.replace(' ', '').lower()
        
        if not target_text:
            return 0.0

        # video_tensor is [C, T, H, W]
        C, T, H, W = video_tensor.shape

        # Convert to numpy and move to CPU if needed
        video_np = video_tensor.detach().cpu().numpy()

        # Convert from [C, T, H, W] to [T, H, W, C] for easier frame extraction
        video_np = np.transpose(video_np, (1, 2, 3, 0))  # [T, H, W, C]

        # Normalize to [0, 255] uint8 if needed
        if video_np.max() <= 1.0:
            video_np = (video_np * 255).astype(np.uint8)
        else:
            video_np = video_np.astype(np.uint8)

        frame_rewards = []

        # Sample frames at specified interval
        for frame_idx in range(0, T, self.frame_interval):
            frame = video_np[frame_idx]  # [H, W, C]
            # Run OCR
            try:
                result = self.ocr.ocr(frame, cls=False)
                if result and result[0]:
                    recognized_text = "".join(
                        [line[1][0] for line in result[0] if line[1][1] > 0])
                else:
                    recognized_text = ""
            except Exception as e:
                logger.info("OCR failed on frame %d: %s", frame_idx, str(e))
                recognized_text = ''


            recognized_text = recognized_text.replace(' ', '').lower()
            if target_text in recognized_text:
                dist = 0
            else:
                dist = distance(recognized_text, target_text)
            dist = min(dist, len(target_text))
            reward = 1.0 - dist / len(target_text)

            if reward > 0:
                frame_rewards.append(reward)


        return sum([reward / len(frame_rewards)
                    for reward in frame_rewards]) if frame_rewards else 0.0

    @torch.no_grad()
    def compute_reward(self, videos: torch.Tensor, prompts: list[str],
                       **kwargs: Any) -> torch.Tensor:
        """
        Calculate OCR reward by evaluating sampled frames across the video.

        Args:
            videos: Video tensor of shape [B, C, T, H, W]
                   B = batch size
                   C = channels (typically 3 for RGB)
                   T = number of frames (temporal dimension)
                   H, W = height, width
            prompts: List of text prompts containing target OCR text in quotes (length B)
            **kwargs: Additional arguments

        Returns:
            Reward tensor [B] with averaged OCR similarity scores across frames
        """
        # Ensure videos is a torch tensor with correct shape
        assert isinstance(
            videos,
            torch.Tensor), f"videos must be torch.Tensor, got {type(videos)}"
        assert videos.ndim == 5, f"videos must have 5 dimensions [B, C, T, H, W], got shape {videos.shape}"


        B, C, T, H, W = videos.shape
        assert len(
            prompts
        ) == B, f"Number of prompts ({len(prompts)}) must match batch size ({B})"

        rewards = []
        for b in range(B):
            # Extract single video: [C, T, H, W]
            video = videos[b]
            reward = self._process_single_video(video, prompts[b])
            rewards.append(reward)

        rewards = torch.tensor(rewards, dtype=torch.float32, device=self.device)


        # Check for NaN or Inf values
        if torch.isnan(rewards).any() or torch.isinf(rewards).any():
            logger.warning(
                "NaN or Inf detected in OCR rewards, returning zero tensor")
            return torch.zeros_like(rewards)

        return rewards


if __name__ == "__main__":
    example_image_path = "flowgrpo_cmd.png"
    example_image = Image.open(example_image_path)
    example_prompt = '/f1ow_grpo$'

    # Convert image to RGB if needed
    if example_image.mode != 'RGB':
        example_image = example_image.convert('RGB')

    # Convert PIL Image to numpy array [H, W, C]
    image_np = np.array(example_image)

    # Normalize to [0, 1] range and convert to float32
    image_np = image_np.astype(np.float32) / 255.0

    # Convert to torch tensor and reshape: [H, W, C] -> [C, H, W]
    image_tensor = torch.from_numpy(image_np).permute(2, 0, 1)

    # Add temporal dimension: [C, H, W] -> [C, T, H, W] where T=1
    video_tensor = image_tensor.unsqueeze(1)  # [C, 1, H, W]

    # Add batch dimension: [C, T, H, W] -> [B, C, T, H, W] where B=1
    video_tensor = video_tensor.unsqueeze(0)  # [1, C, 1, H, W]

    # Instantiate scorer
    scorer = OcrScorerVideo(device="cpu")

    # Call compute_reward method with video tensor
    reward = scorer.compute_reward(video_tensor, [example_prompt])
