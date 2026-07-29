import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))


class TestNoFallback:
    def test_directml_gpu_required(self):
        import onnxruntime as ort
        assert "DmlExecutionProvider" in ort.get_available_providers(), \
            "DirectML GPU provider is required — install onnxruntime-directml"

    def test_scrfd_session_initializes(self):
        from utils.face_detect import get_session
        session = get_session()
        assert session is not None, "SCRFD GPU session must initialize"
