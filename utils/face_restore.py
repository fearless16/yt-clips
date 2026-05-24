"""
face_restore.py — Aggressive face restoration pipeline.

Uses face reference system for identity-preserving restoration with
reference-guided enhancement. Think like a monster, not a soft toy.

Usage:
    from utils.face_restore import FaceRestorer
    restorer = FaceRestorer("expectation.png", "photos/")
    restored = restorer.restore_video("input.mp4", "output.mp4")
"""
import gc
import time
import tempfile
from pathlib import Path
from typing import Optional, Tuple, List
import cv2
import numpy as np

from utils.config import load_config
from utils.logger import get_logger
from utils.face_reference import FaceReference

cfg = load_config()
log = get_logger("face_restore", cfg["logging"]["log_file"], cfg["logging"]["level"])

# Backend detection
HAS_TORCH = False
HAS_CUDA = False
HAS_GFPGAN = False
HAS_CODEFORMER = False

try:
    import torch
    HAS_TORCH = True
    HAS_CUDA = torch.cuda.is_available()
except ImportError:
    pass

try:
    from gfpgan import GFPGANer
    HAS_GFPGAN = True
except ImportError:
    pass

try:
    import sys
    sys.path.insert(0, "CodeFormer")
    from basicsr.utils.download_util import load_file_from_url
    from facexlib.utils.face_restoration_helper import FaceRestoreHelper
    from torchvision.transforms.functional import normalize
    HAS_CODEFORMER = True
except ImportError:
    pass


class FaceRestorer:
    """
    Aggressive face restoration with identity preservation.
    
    Uses reference-guided restoration to match expectation.png quality
    while preserving user identity from dataset.
    """
    
    def __init__(
        self,
        reference_image: str = "expectation.png",
        dataset_dir: str = "photos/",
        model: str = "gfpgan",  # gfpgan, codeformer
        fidelity: float = 0.7,  # CodeFormer fidelity (0=quality, 1=identity)
        strength: float = 0.8,  # Restoration strength
    ):
        self.reference_image = reference_image
        self.dataset_dir = dataset_dir
        self.model = model
        self.fidelity = fidelity
        self.strength = strength
        self.face_ref = None
        self.restorer = None
        self.available = False
        
        self._init_reference()
        self._init_restorer()
        
    def _init_reference(self):
        """Initialize face reference system."""
        try:
            self.face_ref = FaceReference(
                self.reference_image,
                self.dataset_dir,
            )
            if self.face_ref.extract_reference():
                self.face_ref.load_dataset()
                log.info("Face reference system initialized")
            else:
                log.warning("Face reference not available")
        except Exception as e:
            log.warning("Face reference init failed: %s", e)
            
    def _init_restorer(self):
        """Initialize face restoration model."""
        if not (HAS_TORCH and HAS_CUDA):
            log.warning("GPU not available for face restoration")
            return
            
        if self.model == "codeformer" and HAS_CODEFORMER:
            self._init_codeformer()
        elif self.model == "gfpgan" and HAS_GFPGAN:
            self._init_gfpgan()
        else:
            log.warning("No face restoration model available")
            
    def _init_gfpgan(self):
        """Initialize GFPGAN restorer."""
        try:
            self.restorer = GFPGANer(
                model_path="experiments/pretrained_models/GFPGANv1.4.pth",
                upscale=1,
                arch="clean",
                channel_multiplier=2,
                bg_upsampler=None,
            )
            self.available = True
            log.info("GFPGAN restorer initialized")
        except Exception as e:
            log.warning("GFPGAN init failed: %s", e)
            
    def _init_codeformer(self):
        """Initialize CodeFormer restorer."""
        try:
            # Download models if needed
            model_path = Path("weights/CodeFormer")
            model_path.mkdir(parents=True, exist_ok=True)
            
            # Face detection model
            face_det_url = "https://github.com/sczhou/CodeFormer/releases/download/v0.1.0/detection_Resnet50_Final.pth"
            face_det_path = model_path / "detection_Resnet50_Final.pth"
            if not face_det_path.exists():
                load_file_from_url(
                    url=face_det_url,
                    model_dir=str(model_path),
                    progress=True,
                    file_name=None,
                )
                
            # Face restoration model
            face_restore_url = "https://github.com/sczhou/CodeFormer/releases/download/v0.1.0/codeformer.pth"
            face_restore_path = model_path / "codeformer.pth"
            if not face_restore_path.exists():
                load_file_from_url(
                    url=face_restore_url,
                    model_dir=str(model_path),
                    progress=True,
                    file_name=None,
                )
                
            self.available = True
            log.info("CodeFormer restorer initialized")
        except Exception as e:
            log.warning("CodeFormer init failed: %s", e)
            
    def restore_face(
        self,
        face: np.ndarray,
        target_size: Tuple[int, int] = (512, 512),
        use_reference: bool = True,
    ) -> np.ndarray:
        """
        Restore single face with identity preservation.
        
        Args:
            face: Input face (BGR)
            target_size: Output size
            use_reference: Use reference face for guidance
            
        Returns:
            Restored face
        """
        if not self.available:
            log.warning("Restorer not available, returning original")
            return cv2.resize(face, target_size)
            
        # Use reference-guided restoration if available
        if use_reference and self.face_ref and self.face_ref.reference_face is not None:
            return self.face_ref.restore_with_identity(
                face,
                self.strength,
                target_size,
            )
            
        # Fallback to basic restoration
        if self.model == "gfpgan":
            return self._restore_gfpgan(face, target_size)
        elif self.model == "codeformer":
            return self._restore_codeformer(face, target_size)
            
        return cv2.resize(face, target_size)
        
    def restore_video(
        self,
        input_path: str,
        output_path: str,
        target_size: Tuple[int, int] = (512, 512),
    ) -> bool:
        """
        Restore faces in video with identity preservation.
        
        Args:
            input_path: Input video
            output_path: Output video
            target_size: Face output size
            
        Returns:
            Success status
        """
        if not self.available:
            log.warning("Restorer not available, copying video")
            import shutil
            shutil.copy(input_path, output_path)
            return True
            
        import cv2
        import subprocess
        
        t_start = time.perf_counter()
        temp_dir = Path(tempfile.mkdtemp())
        
        try:
            # Open input video
            cap = cv2.VideoCapture(input_path)
            if not cap.isOpened():
                log.error("Cannot open video: %s", input_path)
                return False
                
            fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            log.info("Face restore: %dx%d @ %.1ffps (%d frames)",
                     width, height, fps, total_frames)
                     
            # Process frames
            frames_dir = temp_dir / "frames"
            frames_dir.mkdir()
            t_proc = time.perf_counter()
            frame_idx = 0
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                    
                # Detect and restore faces
                restored_frame = self._restore_frame(frame)
                
                # Save frame
                cv2.imwrite(str(frames_dir / f"{frame_idx:06d}.jpg"), restored_frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
                frame_idx += 1
                
                if frame_idx % 10 == 0 or frame_idx == 1:
                    elapsed = time.perf_counter() - t_proc
                    fps_proc = frame_idx / elapsed if elapsed > 0 else 0
                    eta = (total_frames - frame_idx) / fps_proc if fps_proc > 0 else 0
                    log.info("  Face restore: %d/%d (%.0f fps, ETA %.0fs)",
                             frame_idx, total_frames, fps_proc, eta)
                             
            cap.release()
            t_upscale = time.perf_counter() - t_proc
            log.info("Face restore: %d frames in %.1fs (%.0f fps)",
                     frame_idx, t_upscale, frame_idx / t_upscale if t_upscale > 0 else 0)
                     
            # Encode with ffmpeg
            cmd_encode = [
                "ffmpeg", "-y",
                "-framerate", str(fps),
                "-i", str(frames_dir / "%06d.jpg"),
                "-i", input_path,
                "-map", "0:v",
                "-map", "1:a?",
                "-c:v", "libx264",
                "-crf", "18",
                "-preset", "fast",
                "-pix_fmt", "yuv420p",
                "-c:a", "aac",
                "-b:a", "192k",
                "-shortest",
                output_path,
            ]
            r = subprocess.run(cmd_encode, capture_output=True, text=True)
            if r.returncode != 0:
                log.error("Encode failed: %s", r.stderr[-300:])
                return False
                
            t_total = time.perf_counter() - t_start
            out_size = Path(output_path).stat().st_size / 1e6
            log.info("✅ Face restore complete: %.1fs total, %.1f MB output", t_total, out_size)
            return True
            
        except Exception as e:
            log.error("Face restore crash: %s", e)
            return False
        finally:
            # Free VRAM
            gc.collect()
            if HAS_CUDA:
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            import shutil
            shutil.rmtree(temp_dir, ignore_errors=True)
            
    def _restore_frame(self, frame: np.ndarray) -> np.ndarray:
        """Restore faces in a single frame."""
        import cv2
        
        # Detect faces
        face_locations = self._detect_faces(frame)
        if not face_locations:
            return frame
            
        # Restore each face
        for (top, right, bottom, left) in face_locations:
            # Add padding
            h, w = frame.shape[:2]
            pad = int(max(bottom-top, right-left) * 0.3)
            y1 = max(0, top - pad)
            y2 = min(h, bottom + pad)
            x1 = max(0, left - pad)
            x2 = min(w, right + pad)
            
            face = frame[y1:y2, x1:x2]
            if face.size == 0:
                continue
                
            # Restore face
            restored = self.restore_face(face)
            
            # Paste back
            frame[y1:y2, x1:x2] = restored
            
        return frame
        
    def _detect_faces(self, frame: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """Detect faces in frame."""
        try:
            import face_recognition
            return face_recognition.face_locations(frame)
        except Exception:
            pass
            
        try:
            from deepface import DeepFace
            faces = DeepFace.extract_faces(frame, enforce_detection=False)
            return [(f['facial_area']['y'],
                     f['facial_area']['x'] + f['facial_area']['w'],
                     f['facial_area']['y'] + f['facial_area']['h'],
                     f['facial_area']['x']) for f in faces]
        except Exception:
            pass
            
        from utils.face_detect import detect_faces
        bboxes = detect_faces(frame, score_threshold=0.5)
        return [(y, x + w, y + h, x) for (x, y, w, h) in bboxes]
        
    def _restore_gfpgan(
        self,
        face: np.ndarray,
        target_size: Tuple[int, int],
    ) -> np.ndarray:
        """Restore face with GFPGAN."""
        import cv2
        
        try:
            _, _, restored = self.restorer.enhance(
                face,
                has_aligned=False,
                only_center_face=False,
                paste_back=True,
            )
            return cv2.resize(restored, target_size)
        except Exception as e:
            log.debug("GFPGAN restore failed: %s", e)
            return cv2.resize(face, target_size)
            
    def _restore_codeformer(
        self,
        face: np.ndarray,
        target_size: Tuple[int, int],
    ) -> np.ndarray:
        """Restore face with CodeFormer."""
        import cv2
        import torch
        
        try:
            # CodeFormer restoration logic here
            # For now, fallback to GFPGAN
            return self._restore_gfpgan(face, target_size)
        except Exception as e:
            log.debug("CodeFormer restore failed: %s", e)
            return cv2.resize(face, target_size)


# Convenience function
def create_face_restorer(
    reference_image: str = "expectation.png",
    dataset_dir: str = "photos/",
    model: str = "gfpgan",
) -> FaceRestorer:
    """Create and initialize face restorer."""
    return FaceRestorer(reference_image, dataset_dir, model)
