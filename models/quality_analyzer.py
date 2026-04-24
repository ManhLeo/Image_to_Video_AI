"""Technical image quality analysis utilities used in CPU phase."""

import cv2
import numpy as np
from dataclasses import dataclass
from typing import List, Optional, Tuple
import math
import config

@dataclass
class QualityResult:
    sharpness_score: float = 0.0      # 0–100 (Laplacian variance)
    exposure_score: float = 0.0       # 0–100 (histogram analysis)
    contrast_score: float = 0.0       # 0–100 (RMS contrast)
    noise_score: float = 0.0          # 0–100 (lower noise = higher score)
    composition_score: float = 0.0    # 0–100 (rule of thirds)
    is_blurry: bool = False
    is_overexposed: bool = False
    is_underexposed: bool = False
    mean_brightness: float = 0.0
    laplacian_variance: float = 0.0
    overall_quality: float = 0.0

class QualityAnalyzer:
    def __init__(self):
        self.blur_threshold = getattr(config, "BLUR_THRESHOLD", 80.0)

    def analyze(self, image_path: str, face_positions: Optional[List[dict]] = None) -> QualityResult:
        """Main function to analyze technical photo quality."""
        img = cv2.imread(image_path)
        if img is None:
            return QualityResult()

        # Run individual analyses
        sharp_score, lap_var, is_blurry = self._analyze_sharpness(img)
        exp_score, mean_bright, is_over, is_under = self._analyze_exposure(img)
        contrast_score = self._analyze_contrast(img)
        noise_score = self._analyze_noise(img)
        comp_score = self._analyze_composition(img, face_positions or [])

        result = QualityResult(
            sharpness_score=sharp_score,
            exposure_score=exp_score,
            contrast_score=contrast_score,
            noise_score=noise_score,
            composition_score=comp_score,
            is_blurry=is_blurry,
            is_overexposed=is_over,
            is_underexposed=is_under,
            mean_brightness=mean_bright,
            laplacian_variance=lap_var
        )

        result.overall_quality = self._calculate_overall_quality(result)
        return result

    def _analyze_sharpness(self, img: np.ndarray) -> Tuple[float, float, bool]:
        """Calculates sharpness score using Laplacian variance on center crop."""
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Center region analysis (50% crop)
        h, w = gray.shape
        center = gray[h//4:3*h//4, w//4:3*w//4]
        center_var = cv2.Laplacian(center, cv2.CV_64F).var()
        
        is_blurry = center_var < self.blur_threshold
        
        # Score mapping (center_var -> score)
        if center_var <= 50:
            score = (center_var / 50.0) * 30.0
        elif center_var <= 100:
            score = 30.0 + ((center_var - 50.0) / 50.0) * 30.0
        elif center_var <= 300:
            score = 60.0 + ((center_var - 100.0) / 200.0) * 25.0
        else:
            score = 85.0 + min(15.0, ((center_var - 300.0) / 700.0) * 15.0)
            
        return float(score), float(center_var), is_blurry

    def _analyze_exposure(self, img: np.ndarray) -> Tuple[float, float, bool, bool]:
        """Calculates exposure score using histogram analysis."""
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
        hist_norm = hist.ravel() / (hist.sum() + 1e-6)
        
        mean_brightness = np.average(np.arange(256), weights=hist_norm)
        
        # Check over/underexposed (top/bottom 10%)
        bright_pixels = hist_norm[230:].sum()
        dark_pixels = hist_norm[:25].sum()
        
        is_overexposed = bright_pixels > 0.15
        is_underexposed = dark_pixels > 0.15
        
        if 100 <= mean_brightness <= 160 and not is_overexposed and not is_underexposed:
            exposure_score = 90 + (1 - abs(mean_brightness - 130) / 30.0) * 10
        elif is_overexposed or is_underexposed:
            exposure_score = max(0, 50 - abs(mean_brightness - 128) * 0.5)
        else:
            exposure_score = max(0, 70 - abs(mean_brightness - 128) * 0.3)
            
        return float(min(100, exposure_score)), float(mean_brightness), is_overexposed, is_underexposed

    def _analyze_contrast(self, img: np.ndarray) -> float:
        """Calculates RMS Contrast score."""
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(float)
        rms_contrast = gray.std()
        
        if 40 <= rms_contrast <= 80:
            score = 80 + (1 - abs(rms_contrast - 60) / 20.0) * 20
        elif rms_contrast < 20:
            score = rms_contrast * 2.0
        else:
            score = max(40, 100 - (rms_contrast - 80) * 0.8)
            
        return float(min(100, score))

    def _analyze_noise(self, img: np.ndarray) -> float:
        """Estimates noise level using Gaussian blur difference."""
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(float)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0).astype(float)
        noise_level = np.abs(gray - blurred).mean()
        
        score = max(0, 100 - noise_level * 8.0)
        return float(min(100, score))

    def _analyze_composition(self, img: np.ndarray, face_positions: List[dict]) -> float:
        """Calculates composition score based on Rule of Thirds."""
        if not face_positions:
            return 50.0
            
        h, w = img.shape[:2]
        thirds_x = [w/3.0, 2*w/3.0]
        thirds_y = [h/3.0, 2*h/3.0]
        intersections = [(x, y) for x in thirds_x for y in thirds_y]
        
        # Scoring primary face (largest)
        # Find largest face by area
        primary_face = max(face_positions, key=lambda f: f["width"] * f["height"])
        
        # Convert relative to absolute
        fx = primary_face["xmin"] * w
        fy = primary_face["ymin"] * h
        fw = primary_face["width"] * w
        fh = primary_face["height"] * h
        
        face_center = (fx + fw/2.0, fy + fh/2.0)
        
        min_dist = min([math.sqrt((face_center[0]-ix)**2 + (face_center[1]-iy)**2) for ix, iy in intersections])
        max_possible = math.sqrt(w**2 + h**2) / 4.0
        
        composition_score = max(0, 100 - (min_dist / max_possible) * 100.0)
        return float(composition_score)

    def _calculate_overall_quality(self, result: QualityResult) -> float:
        """Combines individual metrics into a single quality score."""
        score = (result.sharpness_score * 0.35 +
                 result.exposure_score * 0.30 +
                 result.contrast_score * 0.20 +
                 result.noise_score * 0.15)
        
        if result.is_blurry: score *= 0.4
        if result.is_overexposed: score *= 0.7
        if result.is_underexposed: score *= 0.7
        
        return float(min(100, score))
