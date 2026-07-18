"""Tests for mediapipe fallback — graceful degradation when mediapipe is missing."""

import sys
from pathlib import Path
from unittest.mock import patch, MagicMock

sys.path.insert(0, str(Path(__file__).parent.parent))


class TestMediapipeMissing:
    """When mediapipe is not installed, face_detect should degrade gracefully."""

    def test_mediapipe_unavailable_single_warning(self):
        """Module should log exactly ONE warning when mediapipe is missing, not per-frame."""
        import importlib
        import utils.face_detect as fd_mod
        from unittest.mock import patch as _patch

        # Force re-import with mediapipe unavailable
        real_import = __builtins__.__import__ if hasattr(__builtins__, '__import__') else __import__

        def mock_import(name, *args, **kwargs):
            if name == 'mediapipe':
                raise ImportError("No module named 'mediapipe'")
            return real_import(name, *args, **kwargs)

        # Reset singleton so _get_mp runs fresh
        fd_mod._MP_INSTANCE = None
        fd_mod._MP_UNAVAILABLE = False

        with _patch('builtins.__import__', side_effect=mock_import):
            with _patch.object(fd_mod.log, 'warning') as mock_warn:
                result = fd_mod.detect_faces(MagicMock())

        # Should have exactly ONE warning about mediapipe missing
        mediapipe_warnings = [c for c in mock_warn.call_args_list
                              if 'mediapipe' in str(c).lower()]
        assert len(mediapipe_warnings) == 1, (
            f"Expected 1 mediapipe warning, got {len(mediapipe_warnings)}: "
            f"{mediapipe_warnings}"
        )

    def test_mediapipe_unavailable_returns_empty_list(self):
        """detect_faces should return [] when mediapipe is not installed."""
        import utils.face_detect as fd_mod
        import numpy as np

        fd_mod._MP_INSTANCE = None
        fd_mod._MP_UNAVAILABLE = False

        real_import = __builtins__.__import__ if hasattr(__builtins__, '__import__') else __import__

        def mock_import(name, *args, **kwargs):
            if name == 'mediapipe':
                raise ImportError("No module named 'mediapipe'")
            return real_import(name, *args, **kwargs)

        frame = np.zeros((480, 640, 3), dtype=np.uint8)

        with patch('builtins.__import__', side_effect=mock_import):
            fd_mod._MP_INSTANCE = None
            fd_mod._MP_UNAVAILABLE = False
            result = fd_mod.detect_faces(frame)

        assert result == []

    def test_mediapipe_unavailable_returns_none_for_detect_face(self):
        """detect_face should return None when mediapipe is not installed."""
        import utils.face_detect as fd_mod
        import numpy as np

        fd_mod._MP_INSTANCE = None
        fd_mod._MP_UNAVAILABLE = False

        real_import = __builtins__.__import__ if hasattr(__builtins__, '__import__') else __import__

        def mock_import(name, *args, **kwargs):
            if name == 'mediapipe':
                raise ImportError("No module named 'mediapipe'")
            return real_import(name, *args, **kwargs)

        frame = np.zeros((480, 640, 3), dtype=np.uint8)

        with patch('builtins.__import__', side_effect=mock_import):
            fd_mod._MP_INSTANCE = None
            fd_mod._MP_UNAVAILABLE = False
            result = fd_mod.detect_face(frame)

        assert result is None

    def test_no_repeated_import_attempts(self):
        """After first failure, _get_mp should not try importing mediapipe again."""
        import utils.face_detect as fd_mod
        import numpy as np

        fd_mod._MP_INSTANCE = None
        fd_mod._MP_UNAVAILABLE = False

        call_count = 0
        real_import = __builtins__.__import__ if hasattr(__builtins__, '__import__') else __import__

        def mock_import(name, *args, **kwargs):
            nonlocal call_count
            if name == 'mediapipe':
                call_count += 1
                raise ImportError("No module named 'mediapipe'")
            return real_import(name, *args, **kwargs)

        frame = np.zeros((480, 640, 3), dtype=np.uint8)

        with patch('builtins.__import__', side_effect=mock_import):
            fd_mod._MP_INSTANCE = None
            fd_mod._MP_UNAVAILABLE = False
            # Call 5 times
            for _ in range(5):
                fd_mod.detect_faces(frame)

        # mediapipe import should only be attempted ONCE
        assert call_count == 1, (
            f"mediapipe import attempted {call_count} times, expected 1"
        )
