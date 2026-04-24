"""Face landmark analysis for smile, eye-open, and geometry features."""

import cv2
import mediapipe as mp
import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple
import math
import os

from mediapipe.tasks import python
from mediapipe.tasks.python import vision

@dataclass
class FaceResult:
    face_count: int = 0
    smile_score: float = 0.0        # 0–100
    eye_open_score: float = 0.0     # 0–100
    has_closed_eyes: bool = False
    dominant_emotion: str = "neutral"
    face_sizes: List[float] = field(default_factory=list)  # relative sizes
    face_positions: List[dict] = field(default_factory=list)
    landmarks_detected: bool = False
    face_ratio: float = 0.0         # Add face area ratio
    center_distance: float = 0.0    # Add distance from center

class FaceAnalyzer:
    def __init__(self):
        """Initializes FaceLandmarker using the modern MediaPipe Tasks API."""
        model_path = os.path.join(os.path.dirname(__file__), "face_landmarker.task")
        
        if not os.path.exists(model_path):
            print(f"Warning: MediaPipe model not found at {model_path}. Face analysis will be disabled.")
            self.landmarker = None
            return

        try:
            print("[FaceAnalyzer] Initializing MediaPipe FaceLandmarker (Tasks API).")
            base_options = python.BaseOptions(model_asset_path=model_path)
            options = vision.FaceLandmarkerOptions(
                base_options=base_options,
                output_face_blendshapes=True,
                output_facial_transformation_matrixes=True,
                num_faces=10
            )
            self.landmarker = vision.FaceLandmarker.create_from_options(options)
        except Exception as e:
            print(f"Error initializing MediaPipe Tasks: {e}")
            self.landmarker = None

    def _get_dist(self, p1, p2):
        return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

    def analyze(self, image_path: str) -> FaceResult:
        """Analyzes faces using MediaPipe Tasks API."""
        if self.landmarker is None:
            return FaceResult()

        img = cv2.imread(image_path)
        if img is None:
            return FaceResult()

        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w = img.shape[:2]
        
        # Convert to MediaPipe Image
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        
        # Run inference
        result = self.landmarker.detect(mp_image)
        
        face_count = len(result.face_landmarks) if result.face_landmarks else 0
        if face_count == 0:
            return FaceResult()

        per_face_results = []
        face_positions = []
        face_sizes = []

        for i, face_landmarks in enumerate(result.face_landmarks):
            # Landmarks in Tasks API are objects with x, y, z
            coords = [(lm.x, lm.y, lm.z) for lm in face_landmarks]
            
            # Calculate metrics
            smile_score = self._calculate_smile_score(coords, (h, w))
            eye_score, has_closed = self._calculate_eye_open_score(coords)
            
            # Calculate approximate bbox from landmarks (normalized coordinates)
            xs = [c[0] for c in coords]
            ys = [c[1] for c in coords]
            xmin, xmax = min(xs), max(xs)
            ymin, ymax = min(ys), max(ys)
            
            # Geometry Features
            face_ratio = (xmax - xmin) * (ymax - ymin)
            face_center_x = (xmin + xmax) / 2.0
            face_center_y = (ymin + ymax) / 2.0
            # Distance to image center (0.5, 0.5)
            center_dist = math.sqrt((face_center_x - 0.5)**2 + (face_center_y - 0.5)**2)
            
            per_face_results.append({
                "smile_score": smile_score,
                "eye_open_score": eye_score,
                "has_closed_eyes": has_closed,
                "face_ratio": face_ratio,
                "center_dist": center_dist
            })
            
            face_positions.append({
                "xmin": xmin, "ymin": ymin, 
                "width": xmax - xmin, "height": ymax - ymin
            })
            face_sizes.append(face_ratio)

        # Aggregate
        final_scores = self.aggregate_scores(per_face_results)
        return FaceResult(
            face_count=face_count,
            smile_score=final_scores["smile_score"],
            eye_open_score=final_scores["eye_open_score"],
            has_closed_eyes=final_scores["has_closed_eyes"],
            face_sizes=face_sizes,
            face_positions=face_positions,
            landmarks_detected=True,
            face_ratio=final_scores["face_ratio"],
            center_distance=final_scores["center_dist"]
        )

    def _calculate_smile_score(self, landmarks, img_shape) -> float:
        # Indices are same for 468/478 landmark model
        c_l = landmarks[61]
        c_r = landmarks[291]
        f_l = landmarks[234]
        f_r = landmarks[454]
        face_width = self._get_dist(f_l, f_r)
        
        if face_width == 0: return 0.0
        
        mouth_width = self._get_dist(c_l, c_r)
        mouth_width_ratio = mouth_width / face_width
        
        # lift of mouth corners relative to center
        lip_center_y = (landmarks[13][1] + landmarks[14][1]) / 2
        lift_l = lip_center_y - c_l[1]
        lift_r = lip_center_y - c_r[1]
        avg_lift_ratio = (lift_l + lift_r) / (face_width + 1e-6) * 100
        
        base_score = 0
        if mouth_width_ratio > 0.45: base_score += 40
        elif mouth_width_ratio > 0.38: base_score += 20
        
        if avg_lift_ratio > 2.0: base_score += 30
        elif avg_lift_ratio > 0.5: base_score += 10
        
        return float(min(100, base_score))

    def _calculate_eye_open_score(self, landmarks) -> Tuple[float, bool]:
        def get_ear(p1_idx, p2_idx, p3_idx, p4_idx, p5_idx, p6_idx):
            p1, p2, p3, p4, p5, p6 = [landmarks[i] for i in [p1_idx, p2_idx, p3_idx, p4_idx, p5_idx, p6_idx]]
            v_dist1 = self._get_dist(p2, p6)
            v_dist2 = self._get_dist(p3, p5)
            h_dist = self._get_dist(p1, p4)
            if h_dist == 0: return 0.0
            return (v_dist1 + v_dist2) / (2.0 * h_dist)

        ear_l = get_ear(362, 385, 387, 263, 373, 380)
        ear_r = get_ear(33, 160, 158, 133, 153, 144)
        
        avg_ear = (ear_l + ear_r) / 2.0
        has_closed = (ear_l < 0.15 or ear_r < 0.15)
        
        score = (avg_ear / 0.30) * 85
        score = min(100.0, score)
            
        return float(score), has_closed

    def aggregate_scores(self, per_face_results: List[dict]) -> dict:
        if not per_face_results:
            return {"smile_score": 0.0, "eye_open_score": 0.0, "has_closed_eyes": False, "face_ratio": 0.0, "center_dist": 0.0}
            
        per_face_results.sort(key=lambda x: x.get("face_ratio", 0), reverse=True)
        primary_face = per_face_results[0]
        
        smile_scores = [f["smile_score"] for f in per_face_results]
        eye_scores = [f["eye_open_score"] for f in per_face_results]
        closed_eyes = [f["has_closed_eyes"] for f in per_face_results]
        return {
            "smile_score": float(np.min(smile_scores)),
            "eye_open_score": float(np.min(eye_scores)),
            "has_closed_eyes": any(closed_eyes),
            "face_ratio": primary_face.get("face_ratio", 0.0),
            "center_dist": primary_face.get("center_dist", 0.0)
        }

    def close(self):
        if self.landmarker:
            self.landmarker.close()
