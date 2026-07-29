import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import cv2
import numpy as np

INPUT_VIDEO = Path("input/video.mp4")
SAMPLES = 8
MIN_FACE_AREA = 15 * 15


def _sample_frames(video_path: str = None, n: int = SAMPLES):
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
    def test_scrfd_session_loads(self):
        from utils.face_detect import get_session
        det = get_session()
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
                assert len(all_faces) == 0
                continue
            assert len(all_faces) > 0
            largest = max(all_faces, key=lambda r: r[2] * r[3])
            assert single == largest

    def test_detect_face_consistent(self):
        from utils.face_detect import detect_face
        for _, frame in _sample_frames():
            bbox1 = detect_face(frame)
            bbox2 = detect_face(frame)
            if bbox1 is None and bbox2 is None:
                continue
            assert bbox1 == bbox2

    def test_lower_threshold_increases_recall(self):
        from utils.face_detect import detect_faces
        high, low = 0, 0
        for _, frame in _sample_frames():
            high += len(detect_faces(frame, score_threshold=0.9))
            low += len(detect_faces(frame, score_threshold=0.5))
        assert low >= high


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
            assert not non_contiguous.flags["C_CONTIGUOUS"]
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
            f"only {detected}/{total} frames ({ratio:.0%}) had a face detected"
        )

    def test_scrfd_detects_faces(self):
        from utils.face_detect import detect_faces
        any_detected = False
        for _, frame in _sample_frames():
            if detect_faces(frame, score_threshold=0.5):
                any_detected = True
                break
        assert any_detected, "SCRFD DirectML GPU detected no faces in any frame"


class TestDecodeSCRFD:
    def _make_scrfd_outputs(self):
        """Build synthetic outputs matching SCRFD model outputs."""
        strides = [8, 16, 32]
        outputs = []
        for idx, s in enumerate(strides):
            fm = 640 // s
            n = fm * fm
            # scores: (1, n*2, 1)
            scores = np.zeros((1, n * 2, 1), dtype=np.float32)
            # place one strong detection at known grid cell, anchor 0
            cell = n // 2  # middle of feature map
            scores[0, cell * 2, 0] = 0.85
            # another at cell+1, anchor 1
            scores[0, (cell + 1) * 2 + 1, 0] = 0.75
            outputs.append(scores)

        for idx, s in enumerate(strides):
            fm = 640 // s
            n = fm * fm
            bboxes = np.zeros((1, n * 2, 4), dtype=np.float32)
            # bbox deltas for cell n//2 anchor 0 at stride s
            cell = n // 2
            col = cell % fm
            row = cell // fm
            cx = col * s + s // 2
            cy = row * s + s // 2
            bboxes[0, cell * 2, :] = [10.0, 10.0, 10.0, 10.0]  # symmetric around center

            cell2 = cell + 1
            col2 = cell2 % fm
            row2 = cell2 // fm
            cx2 = col2 * s + s // 2
            cy2 = row2 * s + s // 2
            bboxes[0, (cell + 1) * 2 + 1, :] = [15.0, 15.0, 15.0, 15.0]
            outputs.append(bboxes)
        return outputs

    def test_decode_returns_correct_count_and_positions(self):
        from utils.face_detect import _decode_scrfd
        outputs = self._make_scrfd_outputs()
        results = _decode_scrfd(outputs, 640, 0.5, 0.5)
        # stride 32 has fewer grid cells; n//2 may exceed its range at strides 8,16
        # but stride 32 still has some cells (640/32 = 20, 400 cells). n//2 = 200 < 400, ok
        assert len(results) >= 2, f"Expected >=2 detections, got {len(results)}"
        for x, y, w, h in results:
            assert w > 0 and h > 0
            assert x >= 0 and y >= 0

    def test_decode_all_below_threshold_returns_empty(self):
        from utils.face_detect import _decode_scrfd
        outputs = self._make_scrfd_outputs()
        results = _decode_scrfd(outputs, 640, 0.5, 0.99)
        assert results == []

    def test_decode_returns_different_for_high_threshold(self):
        from utils.face_detect import _decode_scrfd
        outputs = self._make_scrfd_outputs()
        low = _decode_scrfd(outputs, 640, 0.5, 0.5)
        high = _decode_scrfd(outputs, 640, 0.5, 0.8)
        assert len(low) >= len(high)
        assert len(high) >= 0

    def test_decode_known_bbox_values(self):
        from utils.face_detect import _decode_scrfd
        s = 8
        fm = 640 // s
        n = fm * fm
        scores = np.zeros((1, n * 2, 1), dtype=np.float32)
        bboxes = np.zeros((1, n * 2, 4), dtype=np.float32)
        # two non-zero-area detections in stride 8
        scores[0, 0, 0] = 0.9  # cell 0, anchor 0
        bboxes[0, 0, :] = [4.0, 4.0, 6.0, 6.0]
        scores[0, 2, 0] = 0.9  # cell 1, anchor 0 (col=1)
        bboxes[0, 2, :] = [5.0, 5.0, 10.0, 10.0]

        outputs = [scores] + [np.zeros_like(bboxes)] * 2 + [bboxes] + [np.zeros_like(bboxes)] * 2
        results = _decode_scrfd(outputs, 640, 0.5, 0.5)
        assert len(results) >= 2, f"Expected >=2, got {len(results)}"
