from paddleocr import PaddleOCR
import torch
import numpy as np
from Levenshtein import distance
from typing import Any

from fastvideo.training.rl.rewards.base import BaseRewardModel
from fastvideo.logger import init_logger

logger = init_logger(__name__)


class OcrScorerVideo(BaseRewardModel):
    """
    OCR reward model for multi-frame video OCR evaluation.

    This model evaluates multiple frames across the video sequence,
    sampling frames at a specified interval and averaging the OCR scores.
    """

    def __init__(self, model_path: str | None = None, device: str = "cpu", frame_interval: int = 4):
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

        logger.info("Initialized OcrScorerVideo (device=%s, frame_interval=%d)", device, frame_interval)

    def _process_single_video(self, video_tensor: torch.Tensor, prompt: str) -> float:
        """
        Process a single video tensor and return its OCR reward.

        Args:
            video_tensor: Video tensor of shape [C, T, H, W]
            prompt: Text prompt containing target OCR text in quotes

        Returns:
            Average reward across positive-scoring frames
        """
        prompt = prompt.replace(' ', '').lower()

        # video_tensor is [C, T, H, W]
        C, T, H, W = video_tensor.shape
        video_np = video_tensor.detach().float().cpu().numpy()
        video_np = np.transpose(video_np, (1, 2, 3, 0))  # [C, T, H, W] --> [T, H, W, C]
        video_np = np.clip(np.round(video_np * 255.0), 0.0, 255.0).astype(np.uint8)

        rewards = []

        # Sample frames at specified interval
        for frame_idx in range(0, T, self.frame_interval):
            frame = video_np[frame_idx]  # [H, W, C]
            # Run OCR
            try:
                result = self.ocr.ocr(frame, cls=False)
                # Same text extraction as flow_grpo OcrScorer_video_or_image
                recognized_text = (''.join([res[1][0] if res[1][1] > 0 else ''
                                            for res in result[0]]) if result[0] else '')
                recognized_text = recognized_text.replace(' ', '').lower()

                dist = distance(recognized_text, prompt)
                dist = min(dist, len(prompt))
            except Exception:
                dist = len(prompt)

            reward = 1.0 - dist / len(prompt)
            rewards.append(reward)

        return (sum(rewards) / len(rewards)) if rewards else 0.0

    @torch.no_grad()
    def compute_reward(self, videos: torch.Tensor, prompts: list[str], **kwargs: Any) -> torch.Tensor:
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
        prompts = [prompt.split('"')[1] for prompt in prompts]
        assert len(videos) == len(prompts), "Mismatch between images and prompts."

        # Ensure videos is a torch tensor with correct shape
        assert isinstance(videos, torch.Tensor), f"videos must be torch.Tensor, got {type(videos)}"
        assert videos.ndim == 5, f"videos must have 5 dimensions [B, C, T, H, W], got shape {videos.shape}"

        B, C, T, H, W = videos.shape
        assert len(prompts) == B, f"Number of prompts ({len(prompts)}) must match batch size ({B})"

        rewards = []
        for b in range(B):
            # Extract single video: [C, T, H, W]
            video = videos[b]
            reward = self._process_single_video(video, prompts[b])
            rewards.append(reward)

        rewards = torch.tensor(rewards, dtype=torch.float32, device=self.device)

        # Check for NaN or Inf values
        if torch.isnan(rewards).any() or torch.isinf(rewards).any():
            logger.warning("NaN or Inf detected in OCR rewards, returning zero tensor")
            return torch.zeros_like(rewards)

        return rewards


if __name__ == "__main__":
    # # Original unit test using a single image as video
    # example_image_path = "flowgrpo_cmd.png"
    # example_image = Image.open(example_image_path)
    # example_prompt = '/f1ow_grpo$'
    # if example_image.mode != 'RGB':
    #     example_image = example_image.convert('RGB')
    # image_np = np.array(example_image)
    # image_np = image_np.astype(np.float32) / 255.0
    # image_tensor = torch.from_numpy(image_np).permute(2, 0, 1)
    # video_tensor = image_tensor.unsqueeze(1).unsqueeze(0)
    # scorer = OcrScorerVideo(device="cpu")
    # reward = scorer.compute_reward(video_tensor, [example_prompt])

    # Unit test: read decoded videos from a flow_grpo safetensor and print rewards
    import os
    from safetensors.torch import load_file as load_safetensors

    SAFETENSOR_PATH = "/mnt/fast-disks/hao_lab/shijie/FastVideo/align_logs/fv_logs/decoded_videos/batch_0_rank_0.safetensors"  # flow_grpo saved: decoded_videos [B, C, T, H, W]
    if not os.path.isfile(SAFETENSOR_PATH):
        print(f"Missing {SAFETENSOR_PATH}; run flow_grpo first to generate decoded_videos, or copy a file here.")
        exit(1)
    data = load_safetensors(SAFETENSOR_PATH)
    # Same load and normalization as flow_grpo: permute(0,1,3,4,2) then *255 round clamp
    raw = data["decoded_videos"]  # [B, T, C, H, W]
    videos_np = (raw.permute(0, 1, 3, 4, 2) * 255).round().clamp(0, 255).to(torch.uint8).cpu().numpy()
    # Convert [B, T, H, W, C] to [B, C, T, H, W] in [0,1] so _process_single_video *255 matches flow_grpo
    videos = torch.from_numpy(videos_np.transpose(0, 4, 1, 2, 3)).float() / 255.0
    B = videos.shape[0]

    # Same prompt parsing as flow_grpo: all lines, first B
    with open("prompts.txt", encoding="utf-8") as f:
        lines = [line.strip() for line in f]
    prompts = lines[:B]

    # Use CPU to match flow_grpo OcrScorer_video_or_image(use_gpu=False)
    scorer = OcrScorerVideo(device="cpu")
    rewards = scorer.compute_reward(videos, prompts)
    print("OCR rewards:\n", rewards.tolist())
