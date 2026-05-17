"""
face_reference.py — Face reference system for identity-preserving restoration.

Extracts face embeddings from reference images (expectation.png, user dataset)
and uses them to guide face restoration with identity matching.

Usage:
    from utils.face_reference import FaceReference
    ref = FaceReference("expectation.png", dataset_dir="photos/")
    ref.extract_reference()
    
    # During restoration
    restored = ref.restore_with_identity(blurred_face, target_embedding)
"""
import numpy as np
from pathlib import Path
from typing import Optional, List, Tuple
import cv2

from utils.config import load_config
from utils.logger import get_logger

cfg = load_config()
log = get_logger("face_ref", cfg["logging"]["log_file"], cfg["logging"]["level"])

# Face detection and embedding
try:
    import face_recognition
    HAS_FACE_REC = True
except ImportError:
    HAS_FACE_REC = False

try:
    from deepface import DeepFace
    HAS_DEEPFACE = True
except ImportError:
    HAS_DEEPFACE = False


class FaceReference:
    """
    Face reference system for identity-preserving restoration.
    
    Extracts face embeddings from reference images and guides
    restoration to preserve identity and match quality.
    """
    
    def __init__(
        self,
        reference_image: str = "expectation.png",
        dataset_dir: str = "photos/",
        model: str = "opencv",  # opencv, vgg-face, facenet, arcface
    ):
        self.reference_image = Path(reference_image)
        self.dataset_dir = Path(dataset_dir)
        self.model = model
        self.reference_embedding = None
        self.reference_landmarks = None
        self.dataset_embeddings = []
        self.reference_face = None
        
    def extract_reference(self) -> bool:
        """Extract face embedding from reference image."""
        if not self.reference_image.exists():
            log.warning("Reference image not found: %s", self.reference_image)
            return False
            
        img = cv2.imread(str(self.reference_image))
        if img is None:
            log.warning("Cannot read reference image: %s", self.reference_image)
            return False
            
        # Extract face
        face = self._extract_face(img)
        if face is None:
            log.warning("No face found in reference image")
            return False
            
        self.reference_face = face
        
        # Extract embedding
        embedding = self._get_embedding(face)
        if embedding is None:
            log.warning("Cannot extract embedding from reference")
            return False
            
        self.reference_embedding = embedding
        
        # Extract landmarks for quality reference
        self.reference_landmarks = self._get_landmarks(face)
        
        log.info("Reference face extracted from %s", self.reference_image)
        return True
        
    def load_dataset(self) -> int:
        """Load face embeddings from dataset directory."""
        if not self.dataset_dir.exists():
            log.warning("Dataset directory not found: %s", self.dataset_dir)
            return 0
            
        count = 0
        for ext in ("jpg", "jpeg", "png", "webp"):
            for img_path in self.dataset_dir.glob(f"*.{ext}"):
                try:
                    img = cv2.imread(str(img_path))
                    if img is None:
                        continue
                        
                    face = self._extract_face(img)
                    if face is None:
                        continue
                        
                    embedding = self._get_embedding(face)
                    if embedding is not None:
                        self.dataset_embeddings.append({
                            "path": str(img_path),
                            "embedding": embedding,
                            "face": face,
                        })
                        count += 1
                except Exception as e:
                    log.debug("Failed to process %s: %s", img_path, e)
                
        log.info("Loaded %d face embeddings from dataset", count)
        return count
        
    def find_best_match(
        self,
        face: np.ndarray,
        threshold: float = 0.6,
    ) -> Optional[dict]:
        """Find best matching face from dataset."""
        if not self.dataset_embeddings:
            return None
            
        embedding = self._get_embedding(face)
        if embedding is None:
            return None
            
        best_match = None
        best_similarity = -1
        
        for entry in self.dataset_embeddings:
            similarity = self._compute_similarity(embedding, entry["embedding"])
            if similarity > best_similarity:
                best_similarity = similarity
                best_match = entry
                
        if best_similarity >= threshold:
            log.debug("Best match: similarity=%.3f", best_similarity)
            return best_match
            
        return None
        
    def restore_with_identity(
        self,
        face: np.ndarray,
        strength: float = 0.7,
        target_size: Tuple[int, int] = (512, 512),
    ) -> np.ndarray:
        """
        Restore face while preserving identity.
        
        Args:
            face: Input face image (BGR)
            strength: Restoration strength (0.0-1.0)
            target_size: Output size
            
        Returns:
            Restored face image
        """
        if self.reference_face is None:
            log.warning("No reference face loaded, using basic restoration")
            return self._basic_restore(face, target_size)
            
        # Get input face embedding
        input_embedding = self._get_embedding(face)
        if input_embedding is None:
            return self._basic_restore(face, target_size)
            
        # Compute identity distance
        identity_distance = self._compute_distance(
            input_embedding, self.reference_embedding
        )
        
        # Adaptive strength based on identity preservation
        # If face is already similar to reference, use less restoration
        adaptive_strength = strength * (1.0 - identity_distance * 0.5)
        
        # Restore face
        restored = self._guided_restore(
            face, 
            self.reference_face,
            adaptive_strength,
            target_size,
        )
        
        # Verify identity is preserved
        restored_embedding = self._get_embedding(restored)
        if restored_embedding is not None:
            preservation_score = self._compute_similarity(
                restored_embedding, self.reference_embedding
            )
            log.debug(
                "Identity preservation: %.3f (distance=%.3f, strength=%.2f)",
                preservation_score, identity_distance, adaptive_strength,
            )
            
        return restored
        
    def match_color_profile(
        self,
        face: np.ndarray,
        target: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Match color profile to reference or target.
        
        Args:
            face: Input face image
            target: Optional target image (uses reference if None)
            
        Returns:
            Color-matched face image
        """
        if target is None:
            target = self.reference_face
            
        if target is None:
            return face
            
        # Convert to LAB color space
        lab_face = cv2.cvtColor(face, cv2.COLOR_BGR2LAB)
        lab_target = cv2.cvtColor(target, cv2.COLOR_BGR2LAB)
        
        # Compute mean and std for each channel
        for i in range(3):
            face_mean, face_std = lab_face[:,:,i].mean(), lab_face[:,:,i].std()
            target_mean, target_std = lab_target[:,:,i].mean(), lab_target[:,:,i].std()
            
            # Normalize and shift
            lab_face[:,:,i] = ((lab_face[:,:,i] - face_mean) * 
                              (target_std / (face_std + 1e-6)) + 
                              target_mean)
            
        # Clip and convert back
        lab_face = np.clip(lab_face, 0, 255).astype(np.uint8)
        return cv2.cvtColor(lab_face, cv2.COLOR_LAB2BGR)
        
    def enhance_detail(
        self,
        face: np.ndarray,
        sharpness: float = 1.5,
        contrast: float = 1.2,
        saturation: float = 1.1,
    ) -> np.ndarray:
        """
        Enhance face details to match reference quality.
        
        Args:
            face: Input face image
            sharpness: Sharpness multiplier
            contrast: Contrast multiplier
            saturation: Saturation multiplier
            
        Returns:
            Enhanced face image
        """
        # Sharpen
        kernel = np.array([[-1,-1,-1],
                          [-1, 9,-1],
                          [-1,-1,-1]]) * sharpness / 1.5
        kernel[1,1] = kernel[1,1] + 1 - sharpness / 1.5
        sharpened = cv2.filter2D(face, -1, kernel)
        
        # Contrast
        enhanced = cv2.convertScaleAbs(sharpened, alpha=contrast, beta=0)
        
        # Saturation
        hsv = cv2.cvtColor(enhanced, cv2.COLOR_BGR2HSV).astype(np.float32)
        hsv[:,:,1] = hsv[:,:,1] * saturation
        hsv[:,:,1] = np.clip(hsv[:,:,1], 0, 255)
        enhanced = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
        
        return enhanced
        
    def _extract_face(
        self,
        img: np.ndarray,
        target_size: Tuple[int, int] = (512, 512),
    ) -> Optional[np.ndarray]:
        """Extract largest face from image."""
        if HAS_FACE_REC:
            face_locations = face_recognition.face_locations(img)
            if not face_locations:
                return None
                
            # Get largest face
            top, right, bottom, left = max(
                face_locations,
                key=lambda r: (r[2]-r[0]) * (r[1]-r[3])
            )
            
            # Add padding
            h, w = img.shape[:2]
            pad = int(max(bottom-top, right-left) * 0.2)
            top = max(0, top - pad)
            bottom = min(h, bottom + pad)
            left = max(0, left - pad)
            right = min(w, right + pad)
            
            face = img[top:bottom, left:right]
            return cv2.resize(face, target_size)
            
        elif HAS_DEEPFACE:
            try:
                faces = DeepFace.extract_faces(img, enforce_detection=False)
                if faces:
                    face_obj = faces[0]
                    x, y, w, h = (face_obj['facial_area']['x'],
                                  face_obj['facial_area']['y'],
                                  face_obj['facial_area']['w'],
                                  face_obj['facial_area']['h'])
                    # Add padding
                    pad = int(max(w, h) * 0.2)
                    y1 = max(0, y - pad)
                    y2 = min(img.shape[0], y + h + pad)
                    x1 = max(0, x - pad)
                    x2 = min(img.shape[1], x + w + pad)
                    face = img[y1:y2, x1:x2]
                    return cv2.resize(face, target_size)
            except Exception as e:
                log.debug("DeepFace extraction failed: %s", e)
                
        return None
        
    def _get_embedding(self, face: np.ndarray) -> Optional[np.ndarray]:
        """Get face embedding."""
        if HAS_DEEPFACE:
            try:
                embedding = DeepFace.represent(
                    face, 
                    model_name=self.model,
                    enforce_detection=False,
                )
                if embedding:
                    return np.array(embedding[0]["embedding"])
            except Exception as e:
                log.debug("DeepFace embedding failed: %s", e)
                
        if HAS_FACE_REC:
            try:
                # Resize to expected size
                face_rgb = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
                face_resized = cv2.resize(face_rgb, (150, 150))
                encodings = face_recognition.face_encodings(face_resized)
                if encodings:
                    return encodings[0]
            except Exception as e:
                log.debug("face_recognition embedding failed: %s", e)
                
        return None
        
    def _get_landmarks(self, face: np.ndarray) -> Optional[np.ndarray]:
        """Get face landmarks."""
        if HAS_FACE_REC:
            face_rgb = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
            landmarks = face_recognition.face_landmarks(face_rgb)
            if landmarks:
                # Convert to numpy array
                points = []
                for feature in landmarks[0].values():
                    points.extend(feature)
                return np.array(points)
        return None
        
    def _compute_similarity(
        self,
        emb1: np.ndarray,
        emb2: np.ndarray,
    ) -> float:
        """Compute cosine similarity between embeddings."""
        # Normalize
        emb1 = emb1 / (np.linalg.norm(emb1) + 1e-6)
        emb2 = emb2 / (np.linalg.norm(emb2) + 1e-6)
        return float(np.dot(emb1, emb2))
        
    def _compute_distance(
        self,
        emb1: np.ndarray,
        emb2: np.ndarray,
    ) -> float:
        """Compute Euclidean distance between embeddings."""
        return float(np.linalg.norm(emb1 - emb2))
        
    def _basic_restore(
        self,
        face: np.ndarray,
        target_size: Tuple[int, int],
    ) -> np.ndarray:
        """Basic face restoration without identity guidance."""
        # Use GFPGAN as fallback
        try:
            from gfpgan import GFPGANer
            restorer = GFPGANer(
                model_path="experiments/pretrained_models/GFPGANv1.4.pth",
                upscale=1,
                arch="clean",
                channel_multiplier=2,
                bg_upsampler=None,
            )
            _, _, output = restorer.enhance(
                face,
                has_aligned=False,
                only_center_face=False,
                paste_back=True,
            )
            return cv2.resize(output, target_size)
        except Exception as e:
            log.debug("GFPGAN fallback failed: %s", e)
            return cv2.resize(face, target_size)
            
    def _guided_restore(
        self,
        face: np.ndarray,
        reference: np.ndarray,
        strength: float,
        target_size: Tuple[int, int],
    ) -> np.ndarray:
        """Guided face restoration using reference."""
        # Try CodeFormer first (better quality)
        try:
            from gfpgan import GFPGANer
            restorer = GFPGANer(
                model_path="experiments/pretrained_models/GFPGANv1.4.pth",
                upscale=1,
                arch="clean",
                channel_multiplier=2,
                bg_upsampler=None,
            )
            
            # Enhance with GFPGAN
            _, _, enhanced = restorer.enhance(
                face,
                has_aligned=False,
                only_center_face=False,
                paste_back=True,
            )
            
            # Blend with reference based on strength
            # Use face landmarks to align
            output = self._align_and_blend(
                enhanced, reference, strength
            )
            
            return cv2.resize(output, target_size)
            
        except Exception as e:
            log.debug("Guided restore failed: %s", e)
            return self._basic_restore(face, target_size)
            
    def _align_and_blend(
        self,
        face: np.ndarray,
        reference: np.ndarray,
        strength: float,
    ) -> np.ndarray:
        """Align and blend face with reference."""
        # Simple alpha blending for now
        # TODO: Add landmark-based alignment
        if face.shape != reference.shape:
            reference = cv2.resize(reference, face.shape[:2])
            
        # Alpha blend
        alpha = strength
        blended = cv2.addWeighted(face, 1-alpha, reference, alpha, 0)
        
        return blended


# Convenience function
def create_face_reference(
    reference_image: str = "expectation.png",
    dataset_dir: str = "photos/",
) -> FaceReference:
    """Create and initialize face reference system."""
    ref = FaceReference(reference_image, dataset_dir)
    ref.extract_reference()
    ref.load_dataset()
    return ref
