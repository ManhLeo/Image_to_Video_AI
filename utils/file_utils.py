"""Filesystem helpers for temporary and output artifact management."""

import os
import shutil
from datetime import datetime
import config

def clean_temp_dir():
    """Removes all files in config.TEMP_DIR and recreates it."""
    try:
        if os.path.exists(config.TEMP_DIR):
            shutil.rmtree(config.TEMP_DIR, ignore_errors=True)
        os.makedirs(config.TEMP_DIR, exist_ok=True)
    except Exception as e:
        print(f"Error cleaning temp directory: {e}")

def get_output_filename(prefix: str = "video") -> str:
    """Generates a timestamped filename for output video."""
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{prefix}_{ts}.mp4"

def copy_to_output(src: str) -> str:
    """Copies a file to the config.OUTPUT_DIR and returns the new path."""
    try:
        os.makedirs(config.OUTPUT_DIR, exist_ok=True)
        dest = os.path.join(config.OUTPUT_DIR, os.path.basename(src))
        shutil.copy2(src, dest)
        return dest
    except Exception as e:
        print(f"Error copying to output: {e}")
        return src
