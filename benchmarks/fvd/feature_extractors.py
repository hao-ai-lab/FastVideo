"""
Pluggable Feature Extractors for FVD Computation.
Supports I3D (standard), CLIP, and VideoMAE via a common interface.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from abc import ABC, abstractmethod
from huggingface_hub import hf_hub_download
from tqdm import tqdm

try:
    from transformers import CLIPModel, CLIPProcessor, VideoMAEModel
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False


class BaseFeatureExtractor(ABC, nn.Module):
    """Abstract base class for all video feature extractors."""

    def __init__(self, device: str = 'cuda'):
        super().__init__()
        self.device = torch.device(
            device if torch.cuda.is_available() else 'cpu')

    @property
    @abstractmethod
    def feature_dim(self) -> int:
        """Dimension of the output feature vector."""
        pass

    @abstractmethod
    def preprocess(self, videos: torch.Tensor) -> torch.Tensor:
        """
        Args:
            videos: [B, T, C, H, W] in [0, 255] range.
        Returns:
            Preprocessed tensor ready for the model.
        """
        pass

    @abstractmethod
    def extract_features_batch(self, videos: torch.Tensor) -> torch.Tensor:
        """
        Extract features for a single batch.
        Args:
            videos: [B, T, C, H, W] (raw input)
        Returns:
            Features: [B, feature_dim]
        """
        pass

    @torch.no_grad()
    def extract_features(self,
                         videos: torch.Tensor,
                         batch_size: int = 32,
                         verbose: bool = True) -> torch.Tensor:
        """
        Extract features for a large tensor of videos by batching.
        """
        N = len(videos)
        all_features = []

        iterator = range(0, N, batch_size)
        if verbose:
            iterator = tqdm(
                iterator,
                desc=f"Extracting features ({self.__class__.__name__})")

        for i in iterator:
            batch = videos[i:i + batch_size].to(self.device)
            features = self.extract_features_batch(batch)
            all_features.append(features.cpu())

        return torch.cat(all_features, dim=0)


# 1. I3D Extractor (The Standard FVD Metric)
class I3DFeatureExtractor(BaseFeatureExtractor):
    REPO_ID = 'flateon/FVD-I3D-torchscript'
    MODEL_FILENAME = 'i3d_torchscript.pt'

    def __init__(self, device: str = 'cuda', cache_dir: str | None = None):
        super().__init__(device)
        self.cache_dir = cache_dir
        self.model = self._load_model()
        self.model.eval()
        self.model.to(self.device)

    @property
    def feature_dim(self) -> int:
        return 400

    def _load_model(self) -> torch.nn.Module:
        try:
            model_path = hf_hub_download(repo_id=self.REPO_ID,
                                         filename=self.MODEL_FILENAME,
                                         cache_dir=self.cache_dir)
            return torch.jit.load(model_path, map_location=self.device)
        except Exception as e:
            raise RuntimeError(f"Failed to load I3D model: {e}") from e

    def preprocess(self, videos: torch.Tensor) -> torch.Tensor:
        """Standard I3D preprocessing: Resize to 224, Norm to [-1, 1]."""
        B, T, C, H, W = videos.shape

        if T < 10:
            raise ValueError(f"I3D requires at least 10 frames, got {T}")

        # Normalize to [0, 1]
        if videos.max() > 1.0:
            videos = videos / 255.0

        # Scale to [-1, 1]
        videos = videos * 2.0 - 1.0

        # Resize to 224x224
        if H != 224 or W != 224:
            videos = videos.reshape(B * T, C, H, W)
            videos = F.interpolate(videos,
                                   size=(224, 224),
                                   mode='bilinear',
                                   align_corners=False)
            videos = videos.reshape(B, T, C, 224, 224)

        # [B, T, C, H, W] -> [B, C, T, H, W]
        return videos.permute(0, 2, 1, 3, 4).contiguous()

    def extract_features_batch(self, videos: torch.Tensor) -> torch.Tensor:
        batch = self.preprocess(videos)
        # TorchScript I3D returns raw logits when return_features=True
        return self.model(batch,
                          rescale=False,
                          resize=False,
                          return_features=True)


# 2. CLIP Extractor (Semantic/Content Quality)
class CLIPFeatureExtractor(BaseFeatureExtractor):

    def __init__(self,
                 device: str = 'cuda',
                 model_name: str = "openai/clip-vit-base-patch32"):
        if not TRANSFORMERS_AVAILABLE:
            raise ImportError(
                "Please install transformers: pip install transformers")
        super().__init__(device)
        self.processor = CLIPProcessor.from_pretrained(model_name)
        self.model = CLIPModel.from_pretrained(model_name).to(self.device)
        self.model.eval()
        self._feature_dim = self.model.config.projection_dim

    @property
    def feature_dim(self) -> int:
        return self._feature_dim

    def preprocess(self, videos: torch.Tensor) -> torch.Tensor:
        # Ensure values are [0, 255]
        if videos.max() <= 1.0:
            videos = videos * 255.0

        return videos.to(torch.uint8)

    def extract_features_batch(self, videos: torch.Tensor) -> torch.Tensor:
        # Input: [B, T, C, H, W]
        B, T, C, H, W = videos.shape
        videos = self.preprocess(videos)

        # Flatten B*T to treat frames as images
        images = videos.view(B * T, C, H, W)

        # HF Processor
        inputs = self.processor(images=images,
                                return_tensors="pt",
                                padding=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # Extract features [B*T, Dim]
        outputs = self.model.get_image_features(**inputs)

        # Reshape [B, T, Dim] and Average Pooling over time
        outputs = outputs.view(B, T, -1)
        return outputs.mean(dim=1)


# 3. VideoMAE Extractor (Structure/Motion Quality)
class VideoMAEFeatureExtractor(BaseFeatureExtractor):

    def __init__(self,
                 device: str = 'cuda',
                 model_name: str = "MCG-NJU/videomae-base"):
        if not TRANSFORMERS_AVAILABLE:
            raise ImportError(
                "Please install transformers: pip install transformers")
        super().__init__(device)
        self.model = VideoMAEModel.from_pretrained(model_name).to(self.device)
        self.model.eval()

        self.register_buffer(
            'mean',
            torch.tensor([0.485, 0.456, 0.406],
                         device=self.device).view(1, 1, 3, 1, 1))
        self.register_buffer(
            'std',
            torch.tensor([0.229, 0.224, 0.225],
                         device=self.device).view(1, 1, 3, 1, 1))

    @property
    def feature_dim(self) -> int:
        return self.model.config.hidden_size

    def preprocess(self, videos: torch.Tensor) -> torch.Tensor:
        """
        Efficient GPU-based preprocessing.
        Input: [B, T, C, H, W] in range [0, 255]
        """
        B, T, C, H, W = videos.shape

        # 1. Resize to 224x224
        if H != 224 or W != 224:
            videos = videos.view(B * T, C, H, W)
            videos = F.interpolate(videos,
                                   size=(224, 224),
                                   mode='bilinear',
                                   align_corners=False)
            videos = videos.view(B, T, C, 224, 224)

        # 2. Normalize to [0, 1]
        if videos.dtype != torch.float32:
            videos = videos.float()

        if videos.max() > 1.0:
            videos = videos / 255.0

        # 3. Apply ImageNet Mean/Std
        return (videos - self.mean) / self.std

    def extract_features_batch(self, videos: torch.Tensor) -> torch.Tensor:
        # Input: [B, T, C, H, W]

        # Fast GPU Preprocessing
        pixel_values = self.preprocess(videos)

        # Forward pass
        outputs = self.model(pixel_values)

        # Global Average Pooling of last hidden state [B, T_patches, 768] -> [B, 768]
        return outputs.last_hidden_state.mean(dim=1)


# Factory
def load_extractor(name: str, device: str = 'cuda') -> BaseFeatureExtractor:
    name = name.lower()
    if name == 'i3d':
        return I3DFeatureExtractor(device)
    elif name == 'clip':
        return CLIPFeatureExtractor(device)
    elif name == 'videomae':
        return VideoMAEFeatureExtractor(device)
    else:
        raise ValueError(
            f"Unknown extractor: {name}. Options: i3d, clip, videomae")
