"""Lightweight aesthetic prediction head on top of CLIP embeddings."""

import os
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn as nn

class AestheticHead(nn.Module):
    """A single-layer aesthetic predictor on top of CLIP embeddings."""

    def __init__(self, embed_dim=512):
        """
        Initialize linear projection head.

        Args:
            embed_dim (int): CLIP embedding dimension.
        """
        super().__init__()
        # Linear scoring layer on top of CLIP embedding
        self.linear = nn.Linear(embed_dim, 1)
        
        # Initialize with mock weights for testing (reproducible seed)
        torch.manual_seed(42)
        nn.init.normal_(self.linear.weight, mean=0.0, std=0.02)
        nn.init.constant_(self.linear.bias, 0.0)

    def forward(self, x):
        """
        Predict normalized aesthetic score.

        Args:
            x (torch.Tensor): CLIP embeddings with shape (B, 512).

        Returns:
            torch.Tensor: Scores in range [0, 1].
        """
        # Formula: score = w · embedding + b
        score = self.linear(x)
        
        # Standard aesthetic predictors (AVA/LAION) output scores in range [1, 10].
        # We scale by 10 to get a [0, 1] range for our UI.
        normalized_score = score / 10.0
        
        # Ensure result is strictly within [0, 1]
        return torch.clamp(normalized_score, 0.0, 1.0)

# Global instance for the helper function
_head = None

def _resolve_weight_path() -> Optional[Path]:
    """Resolve pretrained aesthetic weight path from known locations."""
    base_dir = Path(__file__).resolve().parent
    env_path = os.environ.get("AESTHETIC_HEAD_WEIGHTS", "").strip()
    candidates = [
        Path(env_path) if env_path else None,
        base_dir / "aesthetic_weights.pth",
        base_dir / "aesthetic_head.pth",
        base_dir / "ava+logos-l14-linearMSE.pth",
    ]
    for candidate in candidates:
        if candidate and candidate.exists():
            return candidate
    return None


def _extract_state_dict(checkpoint: Dict) -> Dict:
    """Extract a compatible state_dict from a checkpoint object."""
    if "state_dict" in checkpoint and isinstance(checkpoint["state_dict"], dict):
        checkpoint = checkpoint["state_dict"]
    if "model_state_dict" in checkpoint and isinstance(checkpoint["model_state_dict"], dict):
        checkpoint = checkpoint["model_state_dict"]
    
    # Handle case where keys are just 'weight', 'bias' instead of 'linear.weight', 'linear.bias'
    new_state_dict = {}
    for k, v in checkpoint.items():
        if k == "weight":
            new_state_dict["linear.weight"] = v
        elif k == "bias":
            new_state_dict["linear.bias"] = v
        else:
            new_state_dict[k] = v
    return new_state_dict


def get_aesthetic_head(device="cpu"):
    """Return singleton head, loading pretrained weights when available."""
    global _head
    if _head is None:
        _head = AestheticHead()

        weight_path = _resolve_weight_path()
        if weight_path is not None:
            try:
                checkpoint = torch.load(weight_path, map_location="cpu")
                state_dict = _extract_state_dict(checkpoint)
                _head.load_state_dict(state_dict, strict=False)
                print(f"[AestheticHead] Loaded pretrained weights from: {weight_path}")
            except Exception as exc:
                print(f"[AestheticHead] Failed to load pretrained weights ({exc}). Using fallback init.")
        else:
            print("[AestheticHead] No pretrained weights found. Using fallback init.")

        _head.eval()
        _head.to(device)
    return _head

def predict_score(embedding: np.ndarray) -> float:
    """Helper function for testing a single numpy embedding."""
    head = get_aesthetic_head()
    # Convert numpy array to tensor (add batch dimension)
    tensor_emb = torch.tensor(embedding, dtype=torch.float32).unsqueeze(0)
    
    with torch.no_grad():
        score = head(tensor_emb).item()
        
    return float(score)
