"""Core photo-level score aggregation and selection policy module."""

import os
from models.face_analyzer import FaceAnalyzer
from models.quality_analyzer import QualityAnalyzer
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Tuple
import numpy as np
import config


@dataclass
class PhotoScore:
    # Input info
    image_path: str = ""
    filename: str = ""
    
    # Raw scores (0–100 each)
    smile_score: float = 0.0
    eye_open_score: float = 0.0
    sharpness_score: float = 0.0
    exposure_score: float = 0.0
    contrast_score: float = 0.0
    noise_score: float = 0.0
    composition_score: float = 0.0
    
    # Face info
    face_count: int = 0
    has_closed_eyes: bool = False
    face_ratio: float = 0.0
    center_distance: float = 0.0
    
    # Quality flags
    is_blurry: bool = False
    is_overexposed: bool = False
    is_underexposed: bool = False
    
    # Final results
    aesthetic_score: float = 0.0
    embedding: Optional[np.ndarray] = field(default=None, repr=False)
    overall_score: float = 0.0
    auto_selected: bool = False
    rejection_reason: str = ""  # why photo was rejected
    score_label: str = ""       # "Xuất sắc" / "Tốt" / "Khá" / "Loại bỏ"
    brief_note: str = ""        # Vietnamese note about the photo
    scene_label: str = "unknown"
    scene_confidence: float = 0.0


class PhotoScorer:
    @staticmethod
    def _normalize_sensitivity(sensitivity: int) -> int:
        """
        Normalize sensitivity into the supported UI-safe range.

        Args:
            sensitivity (int): Raw user value.

        Returns:
            int: Clamped sensitivity in range [5, 10].
        """
        try:
            value = int(sensitivity)
        except (TypeError, ValueError):
            value = 7
        return max(5, min(10, value))

    def __init__(self):
        self.face_analyzer = FaceAnalyzer()
        self.quality_analyzer = QualityAnalyzer()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def score_photo(self, image_path: str, user_config: Dict[str, Any]) -> PhotoScore:
        """Main scoring function that combines face and quality analysis."""
        filename = os.path.basename(image_path)
        
        # 1. Face Analysis
        face_res = self.face_analyzer.analyze(image_path)
        
        # 2. Quality & Composition Analysis
        qual_res = self.quality_analyzer.analyze(image_path, face_positions=face_res.face_positions)
        
        # 3. Build PhotoScore object
        photo = PhotoScore(
            image_path=image_path,
            filename=filename,
            smile_score=face_res.smile_score,
            eye_open_score=face_res.eye_open_score,
            sharpness_score=qual_res.sharpness_score,
            exposure_score=qual_res.exposure_score,
            contrast_score=qual_res.contrast_score,
            noise_score=qual_res.noise_score,
            composition_score=qual_res.composition_score,
            face_count=face_res.face_count,
            has_closed_eyes=face_res.has_closed_eyes,
            face_ratio=face_res.face_ratio,
            center_distance=face_res.center_distance,
            is_blurry=qual_res.is_blurry,
            is_overexposed=qual_res.is_overexposed,
            is_underexposed=qual_res.is_underexposed
        )
        
        # 4. Calculate weighted overall_score
        priorities = user_config.get("priorities", ["smile", "quality", "lighting", "aesthetic"])
        photo.overall_score = self._calculate_weighted_score(photo, priorities)
        
        # 5. Determine auto_selected and label
        sensitivity = user_config.get("sensitivity", 6)
        photo.auto_selected, photo.rejection_reason = self._should_auto_select(photo, sensitivity)
        photo.score_label = self._get_score_label(photo.overall_score)
        
        # 6. Generate Vietnamese brief_note
        photo.brief_note = self._generate_brief_note(photo)
        
        return photo

    def _calculate_weighted_score(
        self,
        photo: PhotoScore,
        priorities: List[str],
        theme: str = "family",
        scene_label: Optional[str] = None
    ) -> float:
        """Calculates a weighted score based on user priorities, theme, and geometry."""
        # Standard base weights
        weights = {
            "smile": 0.25,
            "sharpness": 0.20,
            "exposure": 0.20,
            "aesthetic": 0.20
        }
        
        # Theme-based boost
        if theme == "individual":
            weights["aesthetic"] += 0.05
        elif theme in ["family", "group"]:
            weights["smile"] += 0.05

        # Scene-aware dynamic weighting from CLIP prompt classification.
        if getattr(config, "ENABLE_SCENE_AWARE_SCORING", False):
            scene = (scene_label or photo.scene_label or "unknown").lower()
            if scene == "portrait":
                weights["smile"] += 0.05
                weights["aesthetic"] += 0.05
            elif scene == "group photo":
                weights["smile"] += 0.08
                weights["sharpness"] += 0.03
            elif scene == "outdoor":
                weights["exposure"] += 0.08
                weights["sharpness"] += 0.02
            elif scene == "low light":
                weights["exposure"] += 0.10
                weights["sharpness"] += 0.03
            
        # Priority boost (explicit user selection)
        if priorities:
            boost = 0.10
            for p in priorities:
                if p == "smile" and "smile" in weights: weights["smile"] += boost
                if p == "quality" and "sharpness" in weights: weights["sharpness"] += boost
                if p == "lighting" and "exposure" in weights: weights["exposure"] += boost
                if p == "aesthetic" and "aesthetic" in weights: weights["aesthetic"] += boost

            # Re-normalize base weights to sum to 0.85 (reserving 0.15 for geometry)
            total_w = sum(weights.values())
            weights = {k: (v/total_w) * 0.85 for k, v in weights.items()}
        
        # Apply Step 5 formula strictly
        score = (
            weights["smile"] * photo.smile_score +
            weights["sharpness"] * photo.sharpness_score +
            weights["exposure"] * photo.exposure_score +
            weights["aesthetic"] * photo.aesthetic_score +
            0.10 * (photo.face_ratio * 100.0) -
            0.05 * (photo.center_distance * 100.0)
        )
        
        # Hard penalties
        if photo.is_blurry:         score *= 0.35
        if photo.has_closed_eyes:   score *= 0.50
        if photo.face_count == 0 and "smile" in priorities:   score *= 0.70
        
        return min(100.0, max(0.0, score))


    def _should_auto_select(self, photo: PhotoScore, sensitivity: int) -> Tuple[bool, str]:
        """Determines if the photo should be automatically selected."""
        normalized_sensitivity = self._normalize_sensitivity(sensitivity)
        min_threshold = normalized_sensitivity * 10
        rejection_reason = ""

        if photo.is_blurry:
            rejection_reason = "Ảnh bị mờ"
            return False, rejection_reason
            
        if photo.has_closed_eyes:
            rejection_reason = "Có người nhắm mắt"
            return False, rejection_reason
            
        if photo.is_overexposed and photo.overall_score < 40:
            rejection_reason = "Ảnh cháy sáng"
            return False, rejection_reason
            
        if photo.overall_score < min_threshold:
            rejection_reason = f"Điểm {photo.overall_score:.0f} < ngưỡng {min_threshold}"
            return False, rejection_reason

        return True, ""

    def _generate_brief_note(self, photo: PhotoScore) -> str:
        """Generates a short Vietnamese note about the photo quality."""
        if photo.is_blurry: return "Ảnh bị mờ, không rõ nét"
        if photo.has_closed_eyes: return "Có người nhắm mắt"
        
        if photo.overall_score >= 85:
            if photo.smile_score >= 80: return "Nụ cười rất đẹp và tự nhiên, ảnh xuất sắc!"
            return "Ảnh kỹ thuật tốt, ánh sáng và bố cục ấn tượng"
            
        if photo.overall_score >= 70:
            if photo.smile_score >= 70: return "Nụ cười tốt, chất lượng ảnh đạt yêu cầu"
            return "Ảnh đủ tiêu chuẩn, chất lượng khá"
            
        if photo.overall_score >= 50:
            notes = []
            if photo.sharpness_score < 60: notes.append("hơi mờ")
            if photo.exposure_score < 60: notes.append("ánh sáng chưa tốt")
            return "Ảnh mức khá" + (", " + ", ".join(notes) if notes else "")
            
        return "Ảnh chưa đạt yêu cầu"

    def _get_score_label(self, score: float) -> str:
        """Returns a string label for the score."""
        if score >= 85: return "Xuất sắc ⭐"
        if score >= 70: return "Tốt ✓"
        if score >= 50: return "Khá"
        return "Loại bỏ ✗"

    def close(self):
        self.face_analyzer.close()
