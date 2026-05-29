"""Tests for utils/face_detect.py — OpenCV DNN face detector with sampled video frames."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import cv2
import numpy as np

INPUT_VIDEO = Path("input/video.mp4")
SAMPLES = 8
MIN_FACE_AREA = 30 * 30


def _sample_frames(video_path: str = None, n: int = SAMPLES):
    """Yield n evenly-spaced BGR frames from the video."""
    path = video_path or str(INPUT_VIDEO)
    cap = cv2.VideoCapture(path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total < n:
        n = total
    step = max(total // n, 1)
    for i in range(0, total, step):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret, frame = cap.read()
        if ret:
            yield i, frame
    cap.release()


class TestMediaPipeModel:
    """Verify the MediaPipe model initialises correctly."""

    def test_model_initializes(self):
        """Test that _get_mp_detector() returns a valid detector instance and the model file exists."""
        from utils.face_detect import _get_mp_detector
        detector = _get_mp_detector()
        assert detector is not None
        model_path = Path(__file__).resolve().parent.parent / "face_detector.tflite"
        assert model_path.exists(), "face_detector.tflite not found"


class TestDetectFace:
    """Single and multi-face detection tests on sampled video frames."""

    def test_detect_face_returns_valid_bbox(self):
        """detect_face should return (x, y, w, h) with sane dimensions."""
        from utils.face_detect import detect_face
        detected = False
        for idx, frame in _sample_frames():
            bbox = detect_face(frame)
            if bbox is None:
                continue
            detected = True
            x, y, w, h = bbox
            assert w > 0 and h > 0, f"frame {idx}: non-positive dims ({w}x{h})"
            assert w * h >= MIN_FACE_AREA, f"frame {idx}: face too small ({w}x{h})"
            assert x >= 0 and y >= 0, f"frame {idx}: negative origin ({x},{y})"
            assert x + w <= frame.shape[1], f"frame {idx}: bbox exceeds frame width"
            assert y + h <= frame.shape[0], f"frame {idx}: bbox exceeds frame height"
        assert detected, "no face found in any sampled frame"

    def test_detect_face_returns_none_on_blank(self):
        """A blank (all-black) frame should return None."""
        from utils.face_detect import detect_face
        blank = np.zeros((480, 640, 3), dtype=np.uint8)
        assert detect_face(blank) is None

    def test_detect_face_returns_none_on_noise(self):
        """Uniform random noise should return None with score_threshold=0.7."""
        from utils.face_detect import detect_face
        noise = np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)
        assert detect_face(noise, score_threshold=0.7) is None

    def test_detect_faces_returns_list(self):
        """detect_faces should return a list of (x, y, w, h) tuples within frame bounds."""
        from utils.face_detect import detect_faces
        for idx, frame in _sample_frames():
            faces = detect_faces(frame)
            assert isinstance(faces, list)
            for bbox in faces:
                x, y, w, h = bbox
                assert w > 0 and h > 0
                assert x + w <= frame.shape[1]
                assert y + h <= frame.shape[0]

    def test_detect_face_is_largest_in_detect_faces(self):
        """The single face from detect_face should match the largest face from detect_faces."""
        from utils.face_detect import detect_face, detect_faces
        for _, frame in _sample_frames():
            single = detect_face(frame)
            all_faces = detect_faces(frame)
            if single is None:
                assert len(all_faces) == 0, \
                    "detect_face returned None but detect_faces found faces"
                continue
            assert len(all_faces) > 0, \
                "detect_face found a face but detect_faces returned empty"
            largest = max(all_faces, key=lambda r: r[2] * r[3])
            assert single == largest, (
                f"detect_face returned {single} but largest in "
                f"detect_faces is {largest}"
            )

    def test_detect_face_consistent(self):
        """Running detect_face twice on the same frame should return the same bbox."""
        from utils.face_detect import detect_face
        for _, frame in _sample_frames():
            bbox1 = detect_face(frame)
            bbox2 = detect_face(frame)
            if bbox1 is None and bbox2 is None:
                continue
            assert bbox1 == bbox2, f"inconsistent results: {bbox1} vs {bbox2}"

    def test_lower_threshold_increases_recall(self):
        """Lowering score_threshold should not decrease the number of detections."""
        from utils.face_detect import detect_faces
        high, low = 0, 0
        for _, frame in _sample_frames():
            high += len(detect_faces(frame, score_threshold=0.9))
            low += len(detect_faces(frame, score_threshold=0.5))
        assert low >= high, "lower threshold should not reduce detections"


class TestCropFace:
    """crop_face_with_padding should return sane crops."""

    def test_crop_returns_smaller_region(self):
        """Crop should be smaller than the original frame."""
        from utils.face_detect import crop_face_with_padding
        frame = np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)
        bbox = (200, 100, 100, 150)
        crop, (x1, y1, x2, y2) = crop_face_with_padding(frame, bbox)
        assert crop.shape[0] <= frame.shape[0]
        assert crop.shape[1] <= frame.shape[1]
        assert x2 > x1 and y2 > y1
        assert x1 >= 0 and y1 >= 0
        assert x2 <= frame.shape[1] and y2 <= frame.shape[0]

    def test_crop_does_not_exceed_frame_bounds(self):
        """Bbox near the edge with padding should stay within frame bounds."""
        from utils.face_detect import crop_face_with_padding
        frame = np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)
        bbox = (0, 0, 50, 50)
        crop, (x1, y1, x2, y2) = crop_face_with_padding(frame, bbox, padding=0.5)
        assert x1 >= 0 and y1 >= 0

    def test_crop_on_detected_bbox(self):
        """Crop a detected face from a video frame end-to-end."""
        from utils.face_detect import detect_face, crop_face_with_padding
        for _, frame in _sample_frames():
            bbox = detect_face(frame)
            if bbox is None:
                continue
            crop, _ = crop_face_with_padding(frame, bbox)
            assert crop.size > 0
            break


class TestEdgeCases:
    """Edge cases and robustness checks."""

    def test_detect_face_handles_non_contiguous_array(self):
        """detect_face should handle non-contiguous memory array (sliced frame)."""
        from utils.face_detect import detect_face
        for _, frame in _sample_frames():
            non_contiguous = frame[::2, ::2]
            assert not non_contiguous.flags["C_CONTIGUOUS"], \
                "expected non-contiguous array but got contiguous"
            bbox = detect_face(non_contiguous)
            if bbox is not None:
                x, y, w, h = bbox
                assert w > 0 and h > 0

    def test_video_has_consistent_detection(self):
        """At least 50% of sampled frames should have a face detected."""
        from utils.face_detect import detect_face
        detected = 0
        total = 0
        for _, frame in _sample_frames():
            total += 1
            if detect_face(frame) is not None:
                detected += 1
        ratio = detected / max(total, 1)
        assert ratio >= 0.5, (
            f"only {detected}/{total} frames ({ratio:.0%}) had a face detected, "
            f"expected at least 50%"
        )
