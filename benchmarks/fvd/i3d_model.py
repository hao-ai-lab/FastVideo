"""I3D Feature Extractor for FVD Computation"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from huggingface_hub import hf_hub_download
from tqdm import tqdm
from contextlib import suppress


class I3DFeatureExtractor(nn.Module):
    """
    I3D feature extractor for FVD computation.
    Extracts 400-dimensional features from videos using I3D model
    trained on Kinetics-400.
    """

    REPO_ID = 'flateon/FVD-I3D-torchscript'
    MODEL_FILENAME = 'i3d_torchscript.pt'

    def __init__(self,
                 device: str = 'cuda',
                 cache_dir: str | Path | None = None):
        super().__init__()

        self.device_str = device
        if device == 'cuda' and not torch.cuda.is_available():
            print(
                "Warning: CUDA requested but not available – falling back to CPU"
            )
            self.device = torch.device('cpu')
        else:
            self.device = torch.device(device)

        # Set cache directory for HuggingFace Hub
        if cache_dir is not None:
            self.cache_dir = str(Path(cache_dir).resolve())
        else:
            self.cache_dir = None  # Use HF default cache

        self.model = self._load_model()
        self.model.eval()

        with suppress(Exception):
            self.model.to(self.device)

    def _load_model(self) -> torch.nn.Module:
        """Download and load I3D TorchScript model from Hugging Face Hub."""
        print(f"Loading I3D model from Hugging Face Hub ({self.REPO_ID})...")
        
        try:
            # Download model from Hugging Face Hub
            model_path = hf_hub_download(
                repo_id=self.REPO_ID,
                filename=self.MODEL_FILENAME,
                cache_dir=self.cache_dir
            )
            
            # Load directly to chosen device
            model = torch.jit.load(model_path, map_location=self.device)
            print("I3D model loaded successfully")
            return model
            
        except Exception as e:
            raise RuntimeError(
                f"Failed to load I3D model from Hugging Face Hub. Error: {e}\n"
                f"Ensure you have internet connection and huggingface_hub installed:\n"
                f"pip install huggingface_hub"
            ) from e

    def preprocess(self, videos: torch.Tensor) -> torch.Tensor:
        """
        Preprocess videos for I3D.
        
        Args:
            videos: [B, T, C, H, W], values in [0, 255]
        
        Returns:
            Preprocessed videos [B, C, T, 224, 224] (normalized and resized)
        """
        B, T, C, H, W = videos.shape

        if T < 10:
            raise ValueError(f"I3D requires at least 10 frames, got {T}")

        # Normalize to [0, 1] if needed
        if videos.max() > 1.0:
            videos = videos / 255.0

        # Resize to 224x224 if needed
        if H != 224 or W != 224:
            videos = videos.reshape(B * T, C, H, W)
            videos = F.interpolate(videos, size=(224, 224), mode='bilinear', align_corners=False)
            videos = videos.reshape(B, T, C, 224, 224)

        # Convert to [B, C, T, H, W] format
        videos = videos.permute(0, 2, 1, 3, 4).contiguous()

        return videos

    @torch.no_grad()
    def extract_features(self,
                         videos: torch.Tensor,
                         batch_size: int = 32,
                         verbose: bool = True) -> torch.Tensor:
        """
        Extract I3D features
        
        Args:
            videos: [N, T, C, H, W], values in [0, 255]
            batch_size: Batch size for processing
            verbose: Show progress bar
        
        Returns:
            Features [N, 400]
        """
        N = len(videos)
        all_features = []

        iterator = range(0, N, batch_size)
        if verbose:
            iterator = tqdm(iterator, desc="Extracting I3D features")

        for i in iterator:
            batch = videos[i:i + batch_size].to(self.device)
            batch = self.preprocess(batch)  # Now returns [B, C, T, H, W]

            # Use the HF model without rescale/resize (we handle it in preprocess)
            features = self.model(batch, rescale=False, resize=False, return_features=True)

            all_features.append(features.cpu())

        return torch.cat(all_features, dim=0)

    def __call__(self,
                 videos: torch.Tensor,
                 batch_size: int = 32) -> torch.Tensor:
        return self.extract_features(videos, batch_size=batch_size)