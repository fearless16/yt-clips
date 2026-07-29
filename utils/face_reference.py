import numpy as np
from pathlib import Path
from typing import Optional, List, Tuple
import cv2

from utils.config import load_config
from utils.logger import get_logger
from utils.face_detect import detect_faces, detect_face

cfg = load_config()
log = get_logger("face_ref", cfg["logging"]["log_file"], cfg["logging"]["level"])

try:
    from gfpgan import GFPGANer
    HAS_GFPGAN = True
except ImportError:
    HAS_GFPGAN = False


def _extract_face(img: np.ndarray, target_size: Tuple[int, int] = (512, 512)) -> Optional[np.ndarray]:
    face = detect_face(img, score_threshold=0.5)
    if face is None:
        return None
    x, y, w, h = face
    h_img, w_img = img.shape[:2]
    pad = int(max(w, h) * 0.2)
    top = max(0, y - pad)
    bottom = min(h_img, y + h + pad)
    left = max(0, x - pad)
    right = min(w_img, x + w + pad)
    face_crop = img[top:bottom, left:right]
    return cv2.resize(face_crop, target_size)


def _get_embedding(face: np.ndarray) -> Optional[np.ndarray]:
    face_resized = cv2.resize(face, (112, 112))
    lab = cv2.cvtColor(face_resized, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(l)
    hist_l = cv2.calcHist([l], [0], None, [32], [0, 256]).flatten()
    hist_a = cv2.calcHist([a], [0], None, [32], [0, 256]).flatten()
    hist_b = cv2.calcHist([b], [0], None, [32], [0, 256]).flatten()
    hist = np.concatenate([hist_l, hist_a, hist_b])
    hist = hist / max(hist.sum(), 1)
    return hist


def _compute_similarity(emb1: np.ndarray, emb2: np.ndarray) -> float:
    emb1 = emb1 / (np.linalg.norm(emb1) + 1e-6)
    emb2 = emb2 / (np.linalg.norm(emb2) + 1e-6)
    return float(np.dot(emb1, emb2))


class FaceReference:
    def __init__(
        self,
        reference_image: str = "expectation.png",
        dataset_dir: str = "photos/",
        model: str = "opencv",
    ):
        self.reference_image = Path(reference_image)
        self.dataset_dir = Path(dataset_dir)
        self.model = model
        self.reference_embedding = None
        self.reference_landmarks = None
        self.dataset_embeddings = []
        self.reference_face = None

    def extract_reference(self) -> bool:
        if not self.reference_image.exists():
            log.warning("Reference image not found: %s", self.reference_image)
            return False

        img = cv2.imread(str(self.reference_image))
        if img is None:
            log.warning("Cannot read reference image: %s", self.reference_image)
            return False

        face = _extract_face(img)
        if face is None:
            log.warning("No face found in reference image")
            return False

        self.reference_face = face

        embedding = _get_embedding(face)
        if embedding is None:
            log.warning("Cannot extract embedding from reference")
            return False

        self.reference_embedding = embedding

        log.info("Reference face extracted from %s", self.reference_image)
        return True

    def load_dataset(self) -> int:
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

                    face = _extract_face(img)
                    if face is None:
                        continue

                    embedding = _get_embedding(face)
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
        if not self.dataset_embeddings:
            return None

        embedding = _get_embedding(face)
        if embedding is None:
            return None

        best_match = None
        best_similarity = -1

        for entry in self.dataset_embeddings:
            similarity = _compute_similarity(embedding, entry["embedding"])
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
        if self.reference_face is None:
            log.warning("No reference face loaded, using basic restoration")
            return self._basic_restore(face, target_size)

        input_embedding = _get_embedding(face)
        if input_embedding is None:
            return self._basic_restore(face, target_size)

        identity_distance = self._compute_distance(
            input_embedding, self.reference_embedding
        )

        adaptive_strength = strength * (1.0 - identity_distance * 0.5)

        restored = self._guided_restore(
            face,
            self.reference_face,
            adaptive_strength,
            target_size,
        )

        restored_embedding = _get_embedding(restored)
        if restored_embedding is not None:
            preservation_score = _compute_similarity(
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
        if target is None:
            target = self.reference_face

        if target is None:
            return face

        lab_face = cv2.cvtColor(face, cv2.COLOR_BGR2LAB)
        lab_target = cv2.cvtColor(target, cv2.COLOR_BGR2LAB)

        for i in range(3):
            face_mean, face_std = lab_face[:,:,i].mean(), lab_face[:,:,i].std()
            target_mean, target_std = lab_target[:,:,i].mean(), lab_target[:,:,i].std()
            lab_face[:,:,i] = ((lab_face[:,:,i] - face_mean) *
                              (target_std / (face_std + 1e-6)) +
                              target_mean)

        lab_face = np.clip(lab_face, 0, 255).astype(np.uint8)
        return cv2.cvtColor(lab_face, cv2.COLOR_LAB2BGR)

    def enhance_detail(
        self,
        face: np.ndarray,
        sharpness: float = 1.5,
        contrast: float = 1.2,
        saturation: float = 1.1,
    ) -> np.ndarray:
        kernel = np.array([[-1,-1,-1],
                          [-1, 9,-1],
                          [-1,-1,-1]]) * sharpness / 1.5
        kernel[1,1] = kernel[1,1] + 1 - sharpness / 1.5
        sharpened = cv2.filter2D(face, -1, kernel)

        enhanced = cv2.convertScaleAbs(sharpened, alpha=contrast, beta=0)

        hsv = cv2.cvtColor(enhanced, cv2.COLOR_BGR2HSV).astype(np.float32)
        hsv[:,:,1] = hsv[:,:,1] * saturation
        hsv[:,:,1] = np.clip(hsv[:,:,1], 0, 255)
        enhanced = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

        return enhanced

    def _compute_similarity(self, emb1: np.ndarray, emb2: np.ndarray) -> float:
        emb1 = emb1 / (np.linalg.norm(emb1) + 1e-6)
        emb2 = emb2 / (np.linalg.norm(emb2) + 1e-6)
        return float(np.dot(emb1, emb2))

    def _compute_distance(self, emb1: np.ndarray, emb2: np.ndarray) -> float:
        return float(np.linalg.norm(emb1 - emb2))

    def _basic_restore(self, face: np.ndarray, target_size: Tuple[int, int]) -> np.ndarray:
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
        try:
            from gfpgan import GFPGANer
            restorer = GFPGANer(
                model_path="experiments/pretrained_models/GFPGANv1.4.pth",
                upscale=1,
                arch="clean",
                channel_multiplier=2,
                bg_upsampler=None,
            )
            _, _, enhanced = restorer.enhance(
                face,
                has_aligned=False,
                only_center_face=False,
                paste_back=True,
            )
            output = self._align_and_blend(enhanced, reference, strength)
            return cv2.resize(output, target_size)
        except Exception as e:
            log.debug("Guided restore failed: %s", e)
            return self._basic_restore(face, target_size)

    def _align_and_blend(self, face: np.ndarray, reference: np.ndarray, strength: float) -> np.ndarray:
        if face.shape != reference.shape:
            reference = cv2.resize(reference, face.shape[:2])
        alpha = strength
        blended = cv2.addWeighted(face, 1 - alpha, reference, alpha, 0)
        return blended


def create_face_reference(
    reference_image: str = "expectation.png",
    dataset_dir: str = "photos/",
) -> FaceReference:
    ref = FaceReference(reference_image, dataset_dir)
    ref.extract_reference()
    ref.load_dataset()
    return ref
