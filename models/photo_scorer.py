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
    """Dataclass to hold all metrics and final decision for a single image."""
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
    rejection_reason: str = ""
    score_label: str = ""       # "Xuất sắc" / "Tốt" / "Khá" / "Loại bỏ"
    brief_note: str = ""        # Vietnamese note about the photo
    scene_label: str = "unknown"
    scene_confidence: float = 0.0

    # Structured Output (Production Ready)
    analysis_result: Dict[str, Any] = field(default_factory=dict)


class SelectionFilter:
    """Stage 1: Hard Filtering logic to reject extreme low-quality or invalid images."""
    
    @staticmethod
    def evaluate(photo: PhotoScore) -> Tuple[bool, str]:
        """
        Evaluate hard filters.
        Returns: (accepted: bool, reason: str)
        """
        # 1. Extreme Blur Filter
        if photo.sharpness_score < config.MIN_SHARPNESS:
            return False, f"Ảnh quá mờ (Sharpness {photo.sharpness_score:.1f} < {config.MIN_SHARPNESS})"

        # 2. Closed Eyes Filter (Hard threshold)
        if photo.eye_open_score < config.MIN_EYE_OPEN:
            return False, f"Mắt nhắm hẳn (Eye score {photo.eye_open_score:.1f} < {config.MIN_EYE_OPEN})"

        # 3. Face Presence & Rescue Logic
        if photo.face_count == 0:
            # Rescue: If aesthetic is very high, accept as artistic/landscape
            if photo.aesthetic_score >= config.RESCUE_AESTHETIC_THRESHOLD:
                return True, "no_face_but_high_aesthetic"
            else:
                return False, f"Không tìm thấy khuôn mặt (Aesthetic {photo.aesthetic_score:.1f} < {config.RESCUE_AESTHETIC_THRESHOLD})"

        return True, ""


class QualityRanker:
    """Stage 2: Soft Scoring and Ranking based on weighted quality signals."""
    
    @staticmethod
    def calculate(photo: PhotoScore) -> float:
        """
        Calculates final score using weighted quality metrics + refinements.
        Formula: 0.5*Aesthetic + 0.25*Sharpness + 0.25*Lighting
        """
        # Base quality score (Normalized signals)
        base_score = (
            config.W_AESTHETIC * photo.aesthetic_score +
            config.W_SHARPNESS * photo.sharpness_score +
            config.W_LIGHTING * photo.exposure_score
        )

        # Refinement 1: Soft Eye Penalty (Mắt hơi nhắm)
        if config.MIN_EYE_OPEN <= photo.eye_open_score < 50:
            base_score -= config.PENALTY_EYES
            
        # Refinement 2: Smile Bonus (Cộng điểm nụ cười)
        if photo.smile_score > 80:
            base_score += config.BONUS_SMILE

        return float(min(100.0, max(0.0, base_score)))


class PhotoScorer:
    """Orchestrates the two-stage pipeline: Filtering then Ranking."""

    def __init__(self):
        self.face_analyzer = FaceAnalyzer()
        self.quality_analyzer = QualityAnalyzer()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def score_photo(self, image_path: str, user_config: Dict[str, Any]) -> PhotoScore:
        """Main entry point for scoring a single photo."""
        filename = os.path.basename(image_path)
        
        # 1. Feature Extraction (Face + Quality)
        face_res = self.face_analyzer.analyze(image_path)
        qual_res = self.quality_analyzer.analyze(image_path, face_positions=face_res.face_positions)
        
        # 2. Construct Data Object
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
        
        # Note: aesthetic_score is filled later by BatchProcessor (GPU phase)
        # For initial CPU pass, we just return the object with extracted features.
        return photo

    def finalize_score(self, photo: PhotoScore) -> PhotoScore:
        """
        Performs the 2-stage decision once all features (including Aesthetic) are ready.
        """
        # Stage 1: Filtering
        accepted, reason = SelectionFilter.evaluate(photo)
        
        # Stage 2: Scoring (Even if rejected, we calculate score for analytics/debug)
        final_score = QualityRanker.calculate(photo)
        
        # Update photo state
        photo.overall_score = final_score
        photo.auto_selected = accepted
        photo.rejection_reason = reason if not accepted else ""
        if accepted and reason == "no_face_but_high_aesthetic":
             photo.rejection_reason = "Artistic Shot (High Aesthetic)"
             
        photo.score_label = self._get_score_label(photo.overall_score)
        if not accepted:
            photo.score_label = "Loại bỏ ✗"
            
        photo.brief_note = self._generate_brief_note(photo)
        
        # Structured Output
        photo.analysis_result = {
            "accepted": accepted,
            "reason": photo.rejection_reason,
            "scores": {
                "aesthetic": round(photo.aesthetic_score, 1),
                "sharpness": round(photo.sharpness_score, 1),
                "lighting": round(photo.exposure_score, 1),
                "smile": round(photo.smile_score, 1),
                "eyes": round(photo.eye_open_score, 1)
            },
            "final_score": round(photo.overall_score, 1)
        }
        
        return photo

    def _get_score_label(self, score: float) -> str:
        if score >= 85: return "Xuất sắc ⭐"
        if score >= 70: return "Tốt ✓"
        if score >= 50: return "Khá"
        return "Trung bình"

    def _generate_brief_note(self, photo: PhotoScore) -> str:
        if photo.rejection_reason:
            return photo.rejection_reason
        
        if photo.overall_score >= 85: return "Ảnh xuất sắc, chất lượng nghệ thuật cao"
        if photo.overall_score >= 70: return "Ảnh tốt, đáp ứng các tiêu chuẩn kỹ thuật"
        if photo.overall_score >= 50: return "Ảnh đạt yêu cầu"
        return "Chất lượng chưa cao"

    def close(self):
        self.face_analyzer.close()
