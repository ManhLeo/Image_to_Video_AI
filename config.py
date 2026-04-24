"""Central configuration for paths, scoring, and low-VRAM runtime settings."""

import os
import torch

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ASSETS_DIR = os.path.join(BASE_DIR, "assets")
BGM_DIR = os.path.join(ASSETS_DIR, "bgm")
OUTPUT_DIR = os.path.join(BASE_DIR, "output")
TEMP_DIR = os.path.join(BASE_DIR, ".temp")

# Create dirs if not exist
for d in [ASSETS_DIR, BGM_DIR, OUTPUT_DIR, TEMP_DIR]:
    os.makedirs(d, exist_ok=True)

# Device
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Image processing
MAX_IMAGE_SIZE = 1024       # px longest side for analysis
ANALYSIS_IMAGE_SIZE = 640  # px for face detection

# Scoring weights (default)
DEFAULT_WEIGHTS = {
    "smile": 0.35,
    "eye_open": 0.20,
    "sharpness": 0.20,
    "exposure": 0.15,
    "composition": 0.10,
}

# --- Stage 1: Filtering (Hard Gates) ---
MIN_SHARPNESS = 30             # Loại extreme blur
MIN_EYE_OPEN = 30              # Loại mắt nhắm hẳn
RESCUE_AESTHETIC_THRESHOLD = 60 # Chấp nhận ảnh nếu aesthetic > 60

# --- Stage 2: Ranking (Soft Scoring) ---
W_AESTHETIC = 0.50
W_SHARPNESS = 0.25
W_LIGHTING = 0.25
PENALTY_EYES = 5.0             # Trừ điểm nếu 30 <= eyes < 50
BONUS_SMILE = 1.5              # Cộng điểm nếu smile > 80

# Legacy thresholds (for backward compatibility if needed)
MIN_SCORE_THRESHOLD = 60
BLUR_THRESHOLD = 30.0
DARK_THRESHOLD = 40
BRIGHT_THRESHOLD = 215

# Video settings
VIDEO_FPS = 25
VIDEO_WIDTH = 1920
VIDEO_HEIGHT = 1080
DEFAULT_SLIDE_DURATION = 3  # seconds per photo
FADE_DURATION = 0.5         # seconds for crossfade

# BGM tracks
BGM_TRACKS = {
    "warm_memories": {"name": "Warm Memories (Nhẹ nhàng)", "file": "warm_memories.mp3"},
    "happy_steps":   {"name": "Happy Steps (Vui tươi)",   "file": "happy_steps.mp3"},
    "gentle_breeze": {"name": "Gentle Breeze (Tình cảm)", "file": "gentle_breeze.mp3"},
    "none":          {"name": "Không có nhạc",             "file": None},
}

# CLIP settings (aesthetic scoring)
USE_CLIP = True
CLIP_MODEL = "ViT-B-32"
CLIP_PRETRAINED = "openai"

# Low VRAM Optimization Settings
LOW_VRAM_MODE = True
CLIP_BATCH_SIZE = 8
CLIP_FP16 = True
CLIP_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
FORCE_CPU_VIDEO = True
ENABLE_TORCH_CLEANUP = True

# Diversity selection (CPU-only, embedding based)
ENABLE_DIVERSITY_FILTER = True
MAX_SELECTED_OUTPUT = 12
MMR_LAMBDA = 0.72
CLUSTER_RATIO = 0.4

# Scene-aware scoring (uses existing CLIP embeddings, CPU scoring logic)
ENABLE_SCENE_AWARE_SCORING = True

