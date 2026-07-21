"""Tests for utils/face_detect.py — SCRFD (GPU) + YuNet (CPU) face detection on sampled video frames."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import cv2
import numpy as np

INPUT_VIDEO = Path("input/video.mp4")
SAMPLES = 8
MIN_FACE_AREA = 15 * 15  # many faces in this video are ~15-30px


def _sample_frames(video_path: str = None, n: int = SAMPLES):
    """Yield n evenly-spaced BGR frames from the video."""
    path = video_path or str(INPUT_VIDEO)
    if not Path(path).exists():
        return
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


class TestDetectorInit:
    def test_scrfd_backend_loads(self):
        from utils.face_detect import _get_onnx_session
        det = _get_onnx_session()
        assert det is not None

    def test_yunet_backend_loads(self):
        from utils.face_detect import _get_yunet
        det = _get_yunet((320, 320))
        assert det is not None


class TestDetectFace:
    def test_detect_face_returns_valid_bbox(self):
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
        from utils.face_detect import detect_face
        blank = np.zeros((480, 640, 3), dtype=np.uint8)
        assert detect_face(blank) is None

    def test_detect_face_returns_none_on_noise(self):
        from utils.face_detect import detect_face
        noise = np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)
        assert detect_face(noise, score_threshold=0.7) is None

    def test_detect_faces_returns_list(self):
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
        from utils.face_detect import detect_face
        for _, frame in _sample_frames():
            bbox1 = detect_face(frame)
            bbox2 = detect_face(frame)
            if bbox1 is None and bbox2 is None:
                continue
            assert bbox1 == bbox2, f"inconsistent results: {bbox1} vs {bbox2}"

    def test_lower_threshold_increases_recall(self):
        from utils.face_detect import detect_faces
        high, low = 0, 0
        for _, frame in _sample_frames():
            high += len(detect_faces(frame, score_threshold=0.9))
            low += len(detect_faces(frame, score_threshold=0.5))
        assert low >= high, "lower threshold should not reduce detections"


class TestBackendAuto:
    def test_backend_auto_returns_faces(self):
        from utils.face_detect import detect_faces
        for _, frame in _sample_frames():
            faces = detect_faces(frame, backend='auto')
            if len(faces) > 0:
                for bbox in faces:
                    x, y, w, h = bbox
                    assert w > 0 and h > 0

    def test_backend_onnx_returns_faces(self):
        from utils.face_detect import detect_faces
        for _, frame in _sample_frames():
            faces = detect_faces(frame, backend='onnx')
            if len(faces) > 0:
                for bbox in faces:
                    x, y, w, h = bbox
                    assert w > 0 and h > 0

    def test_backend_yunet_returns_faces(self):
        from utils.face_detect import detect_faces
        for _, frame in _sample_frames():
            faces = detect_faces(frame, backend='yunet')
            if len(faces) > 0:
                for bbox in faces:
                    x, y, w, h = bbox
                    assert w > 0 and h > 0


class TestCropFace:
    def test_crop_returns_smaller_region(self):
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
        from utils.face_detect import crop_face_with_padding
        frame = np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)
        bbox = (0, 0, 50, 50)
        crop, (x1, y1, x2, y2) = crop_face_with_padding(frame, bbox, padding=0.5)
        assert x1 >= 0 and y1 >= 0

    def test_crop_on_detected_bbox(self):
        from utils.face_detect import detect_face, crop_face_with_padding
        for _, frame in _sample_frames():
            bbox = detect_face(frame)
            if bbox is None:
                continue
            crop, _ = crop_face_with_padding(frame, bbox)
            assert crop.size > 0
            break


class TestEdgeCases:
    def test_detect_face_handles_non_contiguous_array(self):
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
        from utils.face_detect import detect_face
        detected = 0
        total = 0
        for _, frame in _sample_frames():
            total += 1
            if detect_face(frame) is not None:
                detected += 1
        if total == 0:
            return
        ratio = detected / total
        assert ratio >= 0.5, (
            f"only {detected}/{total} frames ({ratio:.0%}) had a face detected, "
            f"expected at least 50%"
        )

    def test_onnx_and_yunet_both_detect_faces(self):
        from utils.face_detect import detect_faces
        onnx_any, yunet_any = False, False
        for _, frame in _sample_frames():
            if detect_faces(frame, backend='onnx'):
                onnx_any = True
            if detect_faces(frame, backend='yunet'):
                yunet_any = True
            if onnx_any and yunet_any:
                break
        assert onnx_any, "ONNX (SCRFD) detected no faces in any frame"
        assert yunet_any, "YuNet detected no faces in any frame"
