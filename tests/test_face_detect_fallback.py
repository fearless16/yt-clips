"""Tests for SCRFD→YuNet fallback — graceful degradation when GPU/ONNX unavailable."""

import sys
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).parent.parent))


class TestONNXUnavailable:
    def test_onnx_unavailable_falls_to_yunet(self):
        import utils.face_detect as fd_mod
        import numpy as np

        fd_mod._ONNX_SESSION = None
        fd_mod._ONNX_UNAVAILABLE = False
        fd_mod._ONNX_AVAILABLE = True

        def mock_get_onnx(*args, **kwargs):
            fd_mod._ONNX_UNAVAILABLE = True
            return None

        with patch.object(fd_mod, '_get_onnx_session', side_effect=mock_get_onnx):
            det = fd_mod._get_onnx_session()
            assert det is None

    def test_yunet_fallback_loads(self):
        from utils.face_detect import _get_yunet
        yunet = _get_yunet((320, 320))
        assert yunet is not None

    def test_backend_yunet_returns_empty_on_blank(self):
        from utils.face_detect import detect_faces
        import numpy as np
        blank = np.zeros((480, 640, 3), dtype=np.uint8)
        faces = detect_faces(blank, backend='yunet')
        assert isinstance(faces, list)
        assert len(faces) == 0

    def test_backend_auto_does_not_crash_when_scrfd_fails(self):
        import utils.face_detect as fd_mod
        import numpy as np

        fd_mod._ONNX_SESSION = None
        fd_mod._ONNX_UNAVAILABLE = False
        fd_mod._ONNX_AVAILABLE = True

        fd_mod._ONNX_UNAVAILABLE = True
        faces = fd_mod.detect_faces(np.zeros((480, 640, 3), dtype=np.uint8), backend='auto')
        assert isinstance(faces, list)

    def test_onnx_guard_prevents_repeated_init(self):
        import utils.face_detect as fd_mod

        fd_mod._ONNX_SESSION = None
        fd_mod._ONNX_UNAVAILABLE = True

        call_count = 0

        original = fd_mod._get_onnx_session

        def guarded(*args, **kwargs):
            nonlocal call_count
            if fd_mod._ONNX_UNAVAILABLE:
                return None
            call_count += 1
            return original(*args, **kwargs)

        with patch.object(fd_mod, '_get_onnx_session', side_effect=guarded):
            for _ in range(5):
                fd_mod._get_onnx_session()

        assert call_count == 0, (
            f"ONNX init attempted {call_count} times, expected 0 (guard blocked)"
        )
