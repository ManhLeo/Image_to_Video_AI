"""Pretrained aesthetic predictor wrapper with safe fallback behavior.

This module wraps the pip package `aesthetic-predictor` and explicitly
uses its `small` head (512-dim input), which is compatible with CLIP
ViT-B/32 embeddings used by this project.
"""

from __future__ import annotations

from typing import Optional

import torch


class PretrainedAestheticModel:
    """Wrapper for external pretrained aesthetic prediction."""

    def __init__(self, device: str = "cpu") -> None:
        """
        Initialize predictor wrapper.

        Args:
            device (str): Runtime device ("cpu" or "cuda").
        """
        self.device = device
        self._predictor: Optional[torch.nn.Module] = None
        self._load_error: Optional[str] = None
        self._load_predictor()

    @property
    def available(self) -> bool:
        """Return True when pretrained predictor is ready."""
        return self._predictor is not None

    @property
    def load_error(self) -> Optional[str]:
        """Return load error message if initialization failed."""
        return self._load_error

    def _load_predictor(self) -> None:
        """Attempt to load predictor from external dependency."""
        try:
            # Deferred import keeps app boot robust when dependency is missing.
            from aesthetic_predictor import get_aesthetic_model  # type: ignore

            predictor = get_aesthetic_model("small")
            predictor.eval()
            predictor.to(self.device)
            self._predictor = predictor
            print("[PretrainedAesthetic] Loaded aesthetic-predictor (small, 512-dim).")
        except Exception as exc:
            self._predictor = None
            self._load_error = str(exc)
            print(f"[PretrainedAesthetic] Unavailable: {exc}")

    def predict_from_embeddings(self, embeddings: torch.Tensor) -> torch.Tensor:
        """
        Predict aesthetic score from normalized CLIP embeddings.

        Args:
            embeddings (torch.Tensor): Tensor shape (B, 512).

        Returns:
            torch.Tensor: Scores in range [0, 1], shape (B, 1).

        Raises:
            RuntimeError: If predictor is not available.
        """
        if self._predictor is None:
            raise RuntimeError("Pretrained aesthetic predictor is not available.")

        with torch.no_grad():
            raw_scores = self._predictor(embeddings.to(self.device))
            # Convert raw linear output to stable [0, 1] range.
            scores = torch.sigmoid(raw_scores)
            if scores.ndim == 1:
                scores = scores.unsqueeze(1)
            return scores
