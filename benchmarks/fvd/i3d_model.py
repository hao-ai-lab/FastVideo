"""I3D Feature Extractor for FVD Computation"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
import requests
from tqdm import tqdm
from contextlib import suppress


class I3DFeatureExtractor(nn.Module):
    """
    I3D feature extractor for FVD computation.
    Extracts 400-dimensional features from videos using I3D model
    trained on Kinetics-400.
    """

    MODEL_URL = 'https://www.dropbox.com/s/ge9e5ujwgetktms/i3d_torchscript.pt?dl=1'

    def __init__(self,
                 device: str = 'cuda',
                 cache_dir: str | Path | None = None):
        super().__init__()

        self.device_str = device
        if device == 'cuda' and not torch.cuda.is_available():
            print(
                "Warning: CUDA requested but not available — falling back to CPU"
            )
            self.device = torch.device('cpu')
        else:
            self.device = torch.device(device)

        if cache_dir is None:
            self.cache_dir = Path(
                __file__).resolve().parent / 'cache_dir' / 'i3d'  # Changed
        else:
            self.cache_dir = Path(cache_dir)  # Changed

        self.cache_dir.mkdir(parents=True, exist_ok=True)

        self.model = self._load_model()
        self.model.eval()

        with suppress(Exception):
            self.model.to(self.device)

    def _load_model(self) -> torch.nn.Module:
        """Download and load I3D TorchScript model."""
        model_path = self.cache_dir / 'i3d_torchscript.pt'

        if not model_path.exists():
            print(f"Downloading I3D model to {model_path}...")
            self._download_model(model_path)
        else:
            print(f"Loading I3D model from {model_path}")

        try:
            # load directly to chosen device if possible
            model = torch.jit.load(str(model_path), map_location=self.device)
            print("I3D model loaded successfully")
            return model
        except Exception as e:
            print(f"Error loading model: {e}. Re-downloading...")
            with suppress(Exception):
                model_path.unlink(missing_ok=True)
            self._download_model(model_path)
            return torch.jit.load(str(model_path), map_location=self.device)

    def _download_model(self, save_path: Path):
        """Download I3D model from Dropbox"""
        try:
            response = requests.get(self.MODEL_URL, stream=True, timeout=60)
            response.raise_for_status()

            total_size = int(response.headers.get('content-length', 0))

            with open(save_path, 'wb') as f, tqdm(
                    desc="Downloading I3D",
                    total=total_size,
                    unit='B',
                    unit_scale=True,
                    unit_divisor=1024,
            ) as pbar:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
                    pbar.update(len(chunk))

            print(f"Model downloaded to {save_path}")
        except Exception as e:
            raise RuntimeError(
                f"Failed to download I3D model. Error: {e}\n"
                f"Download manually from {self.MODEL_URL} and save to {save_path}"
            ) from e

    def preprocess(self, videos: torch.Tensor) -> torch.Tensor:
        """
        Preprocess videos for I3D.
        
        Args:
            videos: [B, T, C, H, W], values in [0, 255]
        
        Returns:
            Preprocessed videos [B, T, C, 224, 224]
        """
        B, T, C, H, W = videos.shape

        if T < 10:
            raise ValueError(f"I3D requires at least 10 frames, got {T}")

        # Resize to 224x224
        if H != 224 or W != 224:
            videos = videos.view(B * T, C, H, W)
            videos = F.interpolate(videos,
                                   size=(224, 224),
                                   mode='bilinear',
                                   align_corners=False)
            videos = videos.view(B, T, C, 224, 224)

        # Normalize to [0, 1]
        if videos.max() > 1.0:
            videos = videos / 255.0

        # Normalize to [-1, 1]
        videos = videos * 2.0 - 1.0

        return videos

    @torch.no_grad()
    def extract_features(self,
                         videos: torch.Tensor,
                         batch_size: int = 32,
                         verbose: bool = True) -> torch.Tensor:
        """
        Extract I3D features
        
        Args:
            videos: [N, T, C, H, W]
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
            batch = self.preprocess(batch)

            # I3D expects [B, C, T, H, W]
            batch = batch.permute(0, 2, 1, 3,
                                  4)  # [B, T, C, H, W] to [B, C, T, H, W]

            features = self.model(batch)

            features = torch.log(
                features + 1e-10)  # Convert to log space from probabilities
            all_features.append(features.cpu())

        return torch.cat(all_features, dim=0)

    def __call__(self,
                 videos: torch.Tensor,
                 batch_size: int = 32) -> torch.Tensor:
        return self.extract_features(videos, batch_size=batch_size)
