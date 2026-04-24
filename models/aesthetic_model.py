"""CLIP-based aesthetic inference module for the GPU phase."""

from typing import List, Tuple

import open_clip
import torch
from PIL import Image

import config
from models.aesthetic_head import get_aesthetic_head
from models.pretrained_aesthetic import PretrainedAestheticModel

class AestheticModel:
    """Wrap CLIP encoder and aesthetic head for batch scoring."""

    def __init__(self):
        """Initialize CLIP model, preprocessing, and aesthetic head."""
        self.device = config.CLIP_DEVICE
        self.model_name = config.CLIP_MODEL
        self.pretrained = config.CLIP_PRETRAINED
        
        print(f"[AestheticModel] Loading CLIP model: {self.model_name} on {self.device}...")
        
        # Load model and transforms
        model, _, preprocess = open_clip.create_model_and_transforms(
            self.model_name,
            pretrained=self.pretrained
        )
        
        self.model = model.to(self.device)
        self.preprocess = preprocess
        
        # Load fallback head first (always available).
        self.aesthetic_head = get_aesthetic_head(self.device)
        self.pretrained_aesthetic = PretrainedAestheticModel(self.device)
        
        # Optimize for Low VRAM
        if config.CLIP_FP16 and self.device == "cuda":
            print("[AestheticModel] Enabling FP16 mode for low VRAM.")
            self.model = self.model.half()
            # Aesthetic head is tiny, but let's keep it float32 for stable scoring
            
        self.model.eval()
        self.scene_labels: List[str] = ["portrait", "group photo", "outdoor", "low light"]
        self.scene_text_features: torch.Tensor | None = None
        self._init_scene_prompts()

    def _init_scene_prompts(self) -> None:
        """Initialize CLIP text embeddings for lightweight scene classification."""
        if not getattr(config, "ENABLE_SCENE_AWARE_SCORING", False):
            return
        try:
            text_tokens = open_clip.tokenize(self.scene_labels).to(self.device)
            with torch.no_grad():
                text_features = self.model.encode_text(text_tokens)
                text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            self.scene_text_features = text_features.to(torch.float32).cpu()
            print("[AestheticModel] Scene prompts initialized.")
        except Exception as exc:
            self.scene_text_features = None
            print(f"[AestheticModel] Scene prompt init failed: {exc}")


    def encode_batch(self, images_tensor: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encodes a batch of images and returns normalized embeddings and aesthetic scores.
        Input: torch.Tensor of shape (B, 3, 224, 224)
        Output: tuple(embeddings (B, 512), scores (B, 1))
        """
        images_tensor = images_tensor.to(self.device)
        
        if config.CLIP_FP16 and self.device == "cuda":
            images_tensor = images_tensor.half()
            
        with torch.no_grad():
            image_features = self.model.encode_image(images_tensor)
            # Normalize embeddings
            image_features /= image_features.norm(dim=-1, keepdim=True)
            
            # Prefer pretrained aesthetic predictor (512-dim compatible).
            # Fallback to local head if dependency is missing or runtime fails.
            features_fp32 = image_features.to(torch.float32)
            if self.pretrained_aesthetic.available:
                try:
                    scores = self.pretrained_aesthetic.predict_from_embeddings(features_fp32)
                except Exception as exc:
                    print(f"[AestheticModel] Pretrained aesthetic failed, fallback activated: {exc}")
                    scores = self.aesthetic_head(features_fp32)
            else:
                scores = self.aesthetic_head(features_fp32)

            # Defensive normalization: keep all downstream scores in [0, 1].
            scores = torch.clamp(scores, 0.0, 1.0)
            
            # Memory safety: Ensure all intermediate tensors are moved to CPU
            # and delete GPU references to free VRAM immediately
            if self.device == "cuda":
                del features_fp32
                torch.cuda.empty_cache()
            
        return image_features.cpu(), scores.cpu()


    def preprocess_image(self, pil_img: Image.Image) -> torch.Tensor:
        """Preprocesses a single PIL image for CLIP."""
        return self.preprocess(pil_img)

    def predict_scene_from_embeddings(self, embeddings: torch.Tensor) -> Tuple[List[str], torch.Tensor]:
        """Predict scene labels from normalized CLIP embeddings."""
        batch_size = embeddings.shape[0]
        if self.scene_text_features is None:
            return ["unknown"] * batch_size, torch.zeros(batch_size, dtype=torch.float32)

        emb = embeddings.to(torch.float32)
        emb = emb / (emb.norm(dim=-1, keepdim=True) + 1e-8)
        sims = torch.matmul(emb, self.scene_text_features.T)
        probs = torch.softmax(sims, dim=-1)
        confidence, idx = torch.max(probs, dim=-1)
        labels = [self.scene_labels[i] for i in idx.tolist()]
        return labels, confidence
