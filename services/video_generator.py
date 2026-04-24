"""Video rendering service for slideshow export using MoviePy."""

import os
from dataclasses import dataclass
from typing import Callable, List, Optional

import numpy as np
from PIL import Image, ImageEnhance
from moviepy.editor import (
    AudioFileClip,
    ImageClip,
    concatenate_audioclips,
    concatenate_videoclips,
)

import config

@dataclass
class VideoConfig:
    slide_duration: float = 3.0           # seconds per photo
    transition: str = "fade"              # "fade" | "slide" | "zoom" | "none"
    mood: str = "classic"                 # "classic" | "pop" | "cinematic"
    bgm_key: str = "none"                 # key in config.BGM_TRACKS
    bgm_volume: float = 0.8               # 0.0–1.0
    output_filename: str = "output.mp4"
    width: int = config.VIDEO_WIDTH
    height: int = config.VIDEO_HEIGHT
    fps: int = config.VIDEO_FPS

class VideoGenerator:
    """Generate slideshow videos with transitions and optional BGM."""

    def __init__(self):
        self.width = config.VIDEO_WIDTH
        self.height = config.VIDEO_HEIGHT
        self.fps = config.VIDEO_FPS

    def generate(
        self,
        image_paths: List[str],
        video_config: VideoConfig,
        progress_callback: Optional[Callable] = None
    ) -> str:
        """Generates an MP4 video from images with filters, transitions, and audio."""
        if not image_paths:
            return ""
        if len(image_paths) < 2 and progress_callback:
            progress_callback(0.05, "Tập ảnh nhỏ, đang tạo video từ ít ảnh...")

        if progress_callback: progress_callback(0.05, "Đang chuẩn bị ảnh...")
        
        clips = self._create_clips(
            image_paths, 
            video_config.mood,
            video_config.slide_duration, 
            video_config.transition
        )
        
        if progress_callback: progress_callback(0.40, "Đang ghép video...")
        
        # Method "compose" is needed for transitions like crossfade
        if video_config.transition in ["fade", "zoom", "slide"]:
            # For crossfade to work, we need a slight overlap.
            # However, concatenate_videoclips with method="compose" and crossfadein/out
            # handles the compositing.
            # Usually, you'd use padding=-fade_duration if you want true crossfade overlap.
            # But here we follow the prompt's implied logic.
            final_video = concatenate_videoclips(clips, method="compose", padding=-config.FADE_DURATION if video_config.transition != "none" else 0)
        else:
            final_video = concatenate_videoclips(clips)
        
        if progress_callback: progress_callback(0.60, "Đang thêm nhạc nền...")
        
        final_video, audio_clip = self._add_audio(final_video, video_config.bgm_key, video_config.bgm_volume)
        
        if progress_callback: progress_callback(0.75, "Đang xuất video (có thể mất vài phút)...")
        
        output_path = os.path.join(config.OUTPUT_DIR, video_config.output_filename)
        
        # Ensure output dir exists
        os.makedirs(config.OUTPUT_DIR, exist_ok=True)
        
        final_video.write_videofile(
            output_path,
            fps=video_config.fps,
            codec="libx264",
            audio_codec="aac",
            temp_audiofile=os.path.join(config.TEMP_DIR, "temp_audio.m4a"),
            remove_temp=True,
            logger=None,  # suppress moviepy logs
            threads=2,
        )

        
        final_video.close()
        if audio_clip is not None:
            audio_clip.close()
        for clip in clips:
            clip.close()
        
        if progress_callback: progress_callback(1.0, "Hoàn tất!")
        
        return output_path

    def _prepare_image(self, image_path: str, mood: str) -> np.ndarray:
        """Loads, resizes (letterbox), and applies mood filters to an image."""
        with Image.open(image_path) as raw_image:
            img = raw_image.convert("RGB")
        
        # Resize to fit (letterbox)
        img.thumbnail((self.width, self.height), Image.Resampling.LANCZOS)
        
        # Center on black canvas
        background = Image.new("RGB", (self.width, self.height), (0, 0, 0))
        offset = ((self.width - img.width) // 2, (self.height - img.height) // 2)
        background.paste(img, offset)
        img = background

        # Apply mood filter (PIL enhancements)
        if mood == "classic":
            img = ImageEnhance.Color(img).enhance(0.95)
            # Add subtle warm tone overlay is better done in color grading or by blending a color
        elif mood == "pop":
            img = ImageEnhance.Color(img).enhance(1.4)
            img = ImageEnhance.Brightness(img).enhance(1.1)
        elif mood == "cinematic":
            img = ImageEnhance.Contrast(img).enhance(1.2)
            img = ImageEnhance.Color(img).enhance(0.75)
            
        img_array = np.array(img)
        
        # Fine-grained color grading
        return self._apply_mood_color_grade(img_array, mood)

    def _apply_mood_color_grade(self, img_array: np.ndarray, mood: str) -> np.ndarray:
        """Performs pixel-level color grading for the specified mood."""
        img_array = img_array.astype(float)
        
        if mood == "classic":
            # Slightly warm shadows
            shadow_mask = (img_array.mean(axis=2) < 80)
            img_array[shadow_mask, 0] = np.clip(img_array[shadow_mask, 0] + 10, 0, 255) # R
            img_array[shadow_mask, 2] = np.clip(img_array[shadow_mask, 2] - 5, 0, 255)  # B
            
        elif mood == "pop":
            # Already boosted in PIL, can add more here if needed
            pass
            
        elif mood == "cinematic":
            # Teal shadows, orange highlights
            shadow_mask = (img_array.mean(axis=2) < 100)
            highlight_mask = (img_array.mean(axis=2) > 160)
            
            # Teal shadows (Blue up, Red down)
            img_array[shadow_mask, 2] = np.clip(img_array[shadow_mask, 2] + 15, 0, 255)
            img_array[shadow_mask, 0] = np.clip(img_array[shadow_mask, 0] - 10, 0, 255)
            
            # Orange highlights (Red+Green up, Blue down)
            img_array[highlight_mask, 0] = np.clip(img_array[highlight_mask, 0] + 15, 0, 255)
            img_array[highlight_mask, 1] = np.clip(img_array[highlight_mask, 1] + 10, 0, 255)
            img_array[highlight_mask, 2] = np.clip(img_array[highlight_mask, 2] - 10, 0, 255)
            
            img_array *= 0.92 # Darken overall
            
        return np.clip(img_array, 0, 255).astype(np.uint8)

    def _create_clips(self, image_paths: List[str], mood: str, slide_duration: float, transition: str) -> List[ImageClip]:
        """Creates a list of ImageClips with transitions applied."""
        clips = []
        total = len(image_paths)
        fade_duration = config.FADE_DURATION
        
        for i, path in enumerate(image_paths):
            img_array = self._prepare_image(path, mood)
            
            clip = ImageClip(img_array, duration=slide_duration)
            
            # Apply Ken Burns / Zoom effect
            if transition == "zoom":
                # Slow Ken Burns zoom effect
                # We need a function for resize: t -> scale
                def make_zoom(t):
                    return 1.0 + 0.05 * (t / slide_duration)
                clip = clip.resize(make_zoom)
            
            # Apply Transitions (fade/zoom/slide all use crossfade here for simplicity)
            if transition in ["fade", "zoom", "slide"]:
                if i > 0:
                    clip = clip.crossfadein(fade_duration)
                # Note: crossfadeout is usually redundant if the next clip crossfadein
                # and concatenate(method='compose', padding=-fade) is used.
            
            clips.append(clip)
            
        return clips

    def _add_audio(self, video_clip, bgm_key: str, bgm_volume: float):
        """Adds background music with looping and fading."""
        bgm_info = config.BGM_TRACKS.get(bgm_key)
        if not bgm_info or not bgm_info["file"]:
            return video_clip, None
            
        bgm_path = os.path.join(config.BGM_DIR, bgm_info["file"])
        if not os.path.exists(bgm_path):
            print(f"Warning: BGM file not found: {bgm_path}")
            return video_clip, None
            
        audio = AudioFileClip(bgm_path).volumex(bgm_volume)
        
        video_duration = video_clip.duration
        if audio.duration < video_duration:
            # Loop audio
            loops_needed = int(video_duration / audio.duration) + 1
            audio = concatenate_audioclips([audio] * loops_needed)
            
        audio = audio.set_duration(video_duration)
        audio = audio.audio_fadeout(2.0)
        
        return video_clip.set_audio(audio), audio
