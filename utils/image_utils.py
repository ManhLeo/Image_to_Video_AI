"""Image IO and preprocessing helpers shared across the pipeline."""

import os
from PIL import Image
from typing import List, Tuple
import config

def preprocess_image(image_path: str, max_size: int = 1024) -> str:
    """
    Resizes image for AI analysis and saves it to the temporary directory.
    Returns the new path.
    """
    try:
        with Image.open(image_path) as raw_image:
            img = raw_image.convert("RGB")
        w, h = img.size
        
        if max(w, h) > max_size:
            if w > h:
                new_w = max_size
                new_h = int(h * (max_size / w))
            else:
                new_h = max_size
                new_w = int(w * (max_size / h))
            
            img = img.resize((new_w, new_h), Image.Resampling.LANCZOS)
            
        filename = os.path.basename(image_path)
        temp_path = os.path.join(config.TEMP_DIR, f"proc_{filename}")
        img.save(temp_path, quality=85)
        return temp_path
    except Exception as e:
        print(f"Error preprocessing {image_path}: {e}")
        return image_path

def get_image_dimensions(image_path: str) -> Tuple[int, int]:
    """Returns (width, height) of the image."""
    with Image.open(image_path) as img:
        return img.size

def get_file_size_str(image_path: str) -> str:
    """Returns a human-readable file size string."""
    size = os.path.getsize(image_path)
    if size > 1024*1024:
        return f"{size/1024/1024:.1f} MB"
    return f"{size/1024:.0f} KB"

def load_images_from_folder(folder_path: str) -> List[str]:
    """Returns a sorted list of image paths from the folder."""
    exts = {'.jpg', '.jpeg', '.png', '.webp', '.bmp', '.tiff'}
    paths = []
    if not os.path.exists(folder_path):
        return []
        
    for f in os.listdir(folder_path):
        if os.path.splitext(f)[1].lower() in exts:
            paths.append(os.path.join(folder_path, f))
    return sorted(paths)

def create_thumbnail(image_path: str, size: Tuple[int, int] = (200, 150)) -> str:
    """Creates a thumbnail for Gradio display and returns the path."""
    try:
        with Image.open(image_path) as raw_image:
            img = raw_image.convert("RGB")
        img.thumbnail(size, Image.Resampling.LANCZOS)
        
        filename = os.path.basename(image_path)
        thumb_path = os.path.join(config.TEMP_DIR, f"thumb_{filename}")
        img.save(thumb_path)
        return thumb_path
    except Exception as e:
        print(f"Error creating thumbnail for {image_path}: {e}")
        return image_path
