"""Batch orchestration for CPU->GPU->CPU photo scoring pipeline."""

import os
from collections import Counter
from typing import Any, Callable, Dict, List, Optional

import numpy as np
import torch
from PIL import Image

from models.photo_scorer import PhotoScorer, PhotoScore
from models.aesthetic_model import AestheticModel
from utils.image_utils import preprocess_image
import config

class BatchProcessor:
    """Run full-image batch processing with low-VRAM safeguards."""

    def __init__(self):
        self.scorer = PhotoScorer()
        self.results: List[PhotoScore] = []

    def process_all(
        self,
        image_paths: List[str],
        user_config: Dict[str, Any],
        progress_callback: Optional[Callable] = None
    ) -> List[PhotoScore]:
        """
        Process all images following the strict sequential architecture:
        1. CPU Phase: Face + Quality
        2. GPU Phase: CLIP (Aesthetic + Embedding)
        3. CPU Phase: Scoring + Selection
        """
        self.results = []
        if not image_paths:
            return self.results

        total = len(image_paths)
        
        # --- PHASE 1: CPU Stage (Face Detection + Quality Analysis) ---
        for i, path in enumerate(image_paths):
            try:
                if progress_callback:
                    progress_callback(i + 1, total, None, (i + 0.1) / total * 0.4) # First 40%
                
                processed_path = preprocess_image(path, max_size=config.ANALYSIS_IMAGE_SIZE)
                try:
                    score = self.scorer.score_photo(processed_path, user_config)
                    score.image_path = path
                    score.filename = os.path.basename(path)
                    self.results.append(score)
                finally:
                    if processed_path != path and os.path.exists(processed_path):
                        os.unlink(processed_path)
                    
            except Exception as e:
                error_score = PhotoScore(image_path=path, filename=os.path.basename(path), brief_note=f"Error: {str(e)}")
                self.results.append(error_score)

        # --- PHASE 2: GPU Stage (CLIP ONLY) ---
        if config.USE_CLIP:
            self.run_clip_phase(image_paths, progress_callback)

        # --- PHASE 3: CPU Stage (Final Scoring + Diversity) ---
        self.run_scoring_phase(user_config)
        
        # Sort by overall_score descending
        self.results.sort(key=lambda x: x.overall_score, reverse=True)
        return self.results

    def run_clip_phase(self, image_paths: List[str], progress_callback: Optional[Callable] = None):
        """Runs batch CLIP inference for aesthetic scoring and embeddings."""
        if not self.results:
            return
        
        model = None
        try:
            # 1. Load CLIP (FP16)
            model = AestheticModel()
            batch_size = config.CLIP_BATCH_SIZE
            total = len(self.results)
            
            i = 0
            while i < total:
                # Use current batch_size, but ensure we don't go out of bounds
                current_batch_size = min(batch_size, total - i)
                batch_results = self.results[i : i + current_batch_size]
                batch_images = []
                
                for r in batch_results:
                    with Image.open(r.image_path) as raw_image:
                        pil_img = raw_image.convert("RGB").resize((224, 224), Image.Resampling.LANCZOS)
                    tensor = model.preprocess_image(pil_img)
                    batch_images.append(tensor)
                
                # Stack to batch tensor
                batch_tensor = torch.stack(batch_images)
                
                try:
                    # Batch inference
                    embeddings, scores = model.encode_batch(batch_tensor)
                    scene_labels, scene_confidence = model.predict_scene_from_embeddings(embeddings)
                    
                    # Store results
                    for j, r in enumerate(batch_results):
                        emb = embeddings[j].numpy()
                        r.embedding = emb
                        # Aesthetic Head outputs [0,1], scale to 0-100
                        r.aesthetic_score = float(scores[j].item() * 100)
                        r.scene_label = scene_labels[j]
                        r.scene_confidence = float(scene_confidence[j].item())
                    
                    if progress_callback:
                        progress_pct = 0.4 + (min(i + current_batch_size, total) / total * 0.4) # Middle 40%
                        progress_callback(min(i + current_batch_size, total), total, batch_results[-1], progress_pct)
                        
                    # Move to next batch
                    i += current_batch_size
                    
                except RuntimeError as e:
                    if "out of memory" in str(e).lower() and batch_size > 1:
                        new_batch_size = max(1, batch_size // 2)
                        print(f"[BatchProcessor] OOM at batch size {batch_size}. Reducing to {new_batch_size}.")
                        batch_size = new_batch_size
                        if config.ENABLE_TORCH_CLEANUP and torch.cuda.is_available():
                            torch.cuda.empty_cache()
                            torch.cuda.ipc_collect()
                        # Retry same images with the reduced batch size.
                    else:
                        raise e

        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                print("[BatchProcessor] GPU OOM detected. Falling back to CPU CLIP.")
                # Cleanup GPU memory before fallback
                if model:
                    del model
                if config.ENABLE_TORCH_CLEANUP and torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    torch.cuda.ipc_collect()
                
                # Fallback to CPU execution
                self._run_clip_cpu_fallback(progress_callback)
            else:
                print(f"RuntimeError in CLIP phase: {e}")
        except Exception as e:
            print(f"Error in CLIP phase: {e}")
        finally:
            # 2. UNLOAD and CLEAR VRAM
            if 'model' in locals() and model is not None:
                del model
            if config.ENABLE_TORCH_CLEANUP and torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
            print("[BatchProcessor] CLIP phase complete, VRAM cleanup executed.")

    def _run_clip_cpu_fallback(self, progress_callback: Optional[Callable] = None):
        """Fallback to CPU if GPU OOMs."""
        original_device = config.CLIP_DEVICE
        model = None
        
        try:
            config.CLIP_DEVICE = "cpu"
            print("[BatchProcessor] Running CLIP fallback on CPU...")
            model = AestheticModel()
            total = len(self.results)
            batch_size = max(1, config.CLIP_BATCH_SIZE // 2)
            
            for i in range(0, total, batch_size):
                batch_results = self.results[i : i + batch_size]
                batch_images = []
                for r in batch_results:
                    with Image.open(r.image_path) as raw_image:
                        pil_img = raw_image.convert("RGB").resize((224, 224), Image.Resampling.LANCZOS)
                    batch_images.append(model.preprocess_image(pil_img))
                
                batch_tensor = torch.stack(batch_images)
                embeddings, scores = model.encode_batch(batch_tensor)
                scene_labels, scene_confidence = model.predict_scene_from_embeddings(embeddings)
                
                for j, r in enumerate(batch_results):
                    r.embedding = embeddings[j].numpy()
                    r.aesthetic_score = float(scores[j].item() * 100)
                    r.scene_label = scene_labels[j]
                    r.scene_confidence = float(scene_confidence[j].item())
                    
                if progress_callback:
                    progress_pct = 0.4 + (min(i + batch_size, total) / total * 0.4)
                    progress_callback(min(i + batch_size, total), total, batch_results[-1], progress_pct)
        except Exception as e:
            print(f"Error in CLIP CPU Fallback: {e}")
        finally:
            config.CLIP_DEVICE = original_device
            if model is not None:
                del model


    def run_scoring_phase(self, user_config: Dict[str, Any]):
        """Final CPU scoring, ranking and diversity filtering."""
        print(f"\n[BatchProcessor] Stage 2: Finalizing scores for {len(self.results)} images...")
        
        for r in self.results:
            # Performs Stage 1 (Filter) and Stage 2 (Ranking)
            self.scorer.finalize_score(r)
            
            # Detailed Logging for Production Debugging
            status = "✅ ACCEPTED" if r.auto_selected else "❌ REJECTED"
            reason = f"({r.rejection_reason})" if r.rejection_reason else ""
            print(f"  {status} {r.filename:<30} | Score: {r.overall_score:>5.1f} | {reason}")
            if r.auto_selected:
                res = r.analysis_result["scores"]
                print(f"    └─ [Metrics] Aesthetic: {res['aesthetic']} | Sharp: {res['sharpness']} | Light: {res['lighting']} | Smile: {res['smile']} | Eyes: {res['eyes']}")

        if getattr(config, "ENABLE_DIVERSITY_FILTER", False):
            self._apply_diversity_filter()

    def _apply_diversity_filter(self) -> None:
        """Apply MMR + lightweight clustering on currently selected photos."""
        selected = [r for r in self.results if r.auto_selected]
        if len(selected) <= 1:
            return

        max_selected = max(1, int(getattr(config, "MAX_SELECTED_OUTPUT", 12)))
        if len(selected) <= max_selected:
            return

        emb_selected = [r for r in selected if r.embedding is not None]
        if len(emb_selected) < 2:
            return

        embeddings = np.stack([r.embedding for r in emb_selected], axis=0)
        labels = self._simple_kmeans(
            embeddings,
            cluster_count=min(
                len(emb_selected),
                max(2, int(round(max_selected * float(getattr(config, "CLUSTER_RATIO", 0.4)))))
            )
        )

        clusters: Dict[int, List[PhotoScore]] = {}
        for idx, photo in enumerate(emb_selected):
            clusters.setdefault(int(labels[idx]), []).append(photo)

        # Sort each cluster by quality descending.
        for cluster_items in clusters.values():
            cluster_items.sort(key=lambda item: item.overall_score, reverse=True)

        chosen: List[PhotoScore] = []
        active_cluster_ids = sorted(clusters.keys())
        lambda_val = float(getattr(config, "MMR_LAMBDA", 0.72))
        max_score = max((r.overall_score for r in emb_selected), default=1.0) or 1.0

        # Round-robin cluster sampling with MMR tie-break for diversity.
        while active_cluster_ids and len(chosen) < max_selected:
            next_round: List[int] = []
            for cluster_id in active_cluster_ids:
                cluster_candidates = clusters.get(cluster_id, [])
                if not cluster_candidates:
                    continue
                candidate = self._pick_best_mmr_candidate(
                    cluster_candidates,
                    chosen,
                    max_score=max_score,
                    lambda_val=lambda_val
                )
                if candidate is not None:
                    chosen.append(candidate)
                    cluster_candidates.remove(candidate)
                if cluster_candidates:
                    next_round.append(cluster_id)
                if len(chosen) >= max_selected:
                    break
            active_cluster_ids = next_round

        chosen_paths = {photo.image_path for photo in chosen}
        for photo in selected:
            if photo.image_path not in chosen_paths:
                photo.auto_selected = False
                photo.rejection_reason = "Lọc đa dạng (MMR + Clustering)"

    def _pick_best_mmr_candidate(
        self,
        candidates: List[PhotoScore],
        selected: List[PhotoScore],
        max_score: float,
        lambda_val: float
    ) -> Optional[PhotoScore]:
        """Pick one candidate maximizing MMR criterion."""
        best_candidate: Optional[PhotoScore] = None
        best_value = -float("inf")

        for candidate in candidates:
            relevance = float(candidate.overall_score) / max_score
            if not selected or candidate.embedding is None:
                mmr_value = relevance
            else:
                max_similarity = 0.0
                for picked in selected:
                    if picked.embedding is None:
                        continue
                    max_similarity = max(max_similarity, float(np.dot(candidate.embedding, picked.embedding)))
                mmr_value = lambda_val * relevance - (1.0 - lambda_val) * max_similarity

            if mmr_value > best_value:
                best_value = mmr_value
                best_candidate = candidate

        return best_candidate

    def _simple_kmeans(self, embeddings: np.ndarray, cluster_count: int, iterations: int = 12) -> np.ndarray:
        """Run lightweight NumPy k-means for embedding clustering."""
        n_samples = embeddings.shape[0]
        if cluster_count >= n_samples:
            return np.arange(n_samples)

        # Initialize by top-score spread to keep deterministic behavior.
        centroids = embeddings[np.linspace(0, n_samples - 1, cluster_count, dtype=int)].copy()
        labels = np.zeros(n_samples, dtype=np.int32)

        for _ in range(iterations):
            # Assign labels by cosine-like similarity since embeddings are normalized.
            sims = np.matmul(embeddings, centroids.T)
            labels = np.argmax(sims, axis=1)

            # Update centroids.
            for cluster_idx in range(cluster_count):
                members = embeddings[labels == cluster_idx]
                if len(members) == 0:
                    continue
                centroid = np.mean(members, axis=0)
                norm = np.linalg.norm(centroid) + 1e-8
                centroids[cluster_idx] = centroid / norm

        return labels

    def mmr_selection(self, results: List[PhotoScore], lambda_val: float = 0.7) -> List[PhotoScore]:
        """Maximal Marginal Relevance (MMR) for diverse and high-scoring image selection."""
        if not results: return []
        
        # Normalize scores to [0, 1] for MMR formula
        max_score = max((r.overall_score for r in results), default=1.0)
        if max_score == 0: max_score = 1.0
        
        unselected = list(results)
        selected = []
        
        while unselected:
            best_idx = -1
            best_mmr = -float('inf')
            
            for i, candidate in enumerate(unselected):
                relevance = candidate.overall_score / max_score
                
                if not selected:
                    mmr_score = relevance
                else:
                    max_sim = 0.0
                    if candidate.embedding is not None:
                        for s in selected:
                            if s.embedding is not None:
                                sim = np.dot(candidate.embedding, s.embedding)
                                max_sim = max(max_sim, sim)
                    
                    mmr_score = lambda_val * relevance - (1.0 - lambda_val) * max_sim
                    candidate._temp_max_sim = max_sim
                
                if mmr_score > best_mmr:
                    best_mmr = mmr_score
                    best_idx = i
                    
            chosen = unselected.pop(best_idx)
            
            # Drop duplicates if similarity is extremely high
            if selected and getattr(chosen, '_temp_max_sim', 0.0) > 0.9:
                chosen.auto_selected = False
                chosen.rejection_reason = "Ảnh trùng lặp (MMR Similarity > 0.9)"
            else:
                selected.append(chosen)
                
        return selected


    def get_summary(self) -> Dict[str, Any]:
        """Return statistics about the processed batch."""
        if not self.results:
            return {}

        total = len(self.results)
        selected = [r for r in self.results if r.auto_selected]
        rejected = [r for r in self.results if not r.auto_selected]
        
        all_scores = [r.overall_score for r in self.results]
        smile_scores = [r.smile_score for r in self.results]
        
        rejection_reasons = Counter([r.rejection_reason for r in self.results if r.rejection_reason])
        
        return {
            "total": total,
            "selected_count": len(selected),
            "rejected_count": len(rejected),
            "avg_score": float(np.mean(all_scores)) if all_scores else 0.0,
            "avg_smile": float(np.mean(smile_scores)) if smile_scores else 0.0,
            "top_photo": self.results[0] if self.results else None,
            "rejection_reasons": dict(rejection_reasons)
        }

    def filter_results(self, min_score: float = 0, only_selected: bool = False) -> List[PhotoScore]:
        """Filter results by criteria."""
        filtered = self.results
        if min_score > 0:
            filtered = [r for r in filtered if r.overall_score >= min_score]
        if only_selected:
            filtered = [r for r in filtered if r.auto_selected]
        return filtered

    def close(self):
        """Release resources."""
        self.scorer.close()
