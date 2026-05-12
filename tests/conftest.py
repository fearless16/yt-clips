import pytest
import subprocess
from pathlib import Path


def _ffmpeg(cmd: list, description: str, timeout: int = 60) -> None:
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    if result.returncode != 0:
        raise RuntimeError(f"FFmpeg {description} failed:\n{result.stderr[:1000]}")


def _base_cmd(out: Path, duration: int, size: str, rate: int = 30) -> tuple:
    cmd = [
        "ffmpeg", "-y",
        "-f", "lavfi", "-i", f"color=c=0x2d5a27:s={size}:d={duration}:r={rate}",
        "-f", "lavfi", "-i", f"sine=frequency=440:duration={duration}",
    ]
    filter_parts = []
    return cmd, filter_parts


# ─── Solo layout ───────────────────────────────────────────────────────────

@pytest.fixture(scope="session")
def test_video_solo(tmp_path_factory):
    """16:9 640x360 — synthetic face on left, green bg."""
    out = tmp_path_factory.mktemp("videos") / "solo.mp4"
    cmd = [
        "ffmpeg", "-y",
        "-f", "lavfi", "-i", "color=c=0x2d5a27:s=640x360:d=4:r=30",
        "-f", "lavfi", "-i", "sine=frequency=440:duration=4",
        "-filter_complex",
        "[0:v]drawbox=x=40:y=80:w=120:h=160:color=0xffcc99:t=fill[face]",
        "-map", "[face]", "-map", "1:a",
        "-c:v", "libx264", "-c:a", "aac", "-shortest", "-pix_fmt", "yuv420p",
        str(out),
    ]
    _ffmpeg(cmd, "solo video")
    return out


# ─── Dual split-screen layout ──────────────────────────────────────────────

@pytest.fixture(scope="session")
def test_video_dual(tmp_path_factory):
    """Two face regions with a vertical divider — simulates guest layout."""
    out = tmp_path_factory.mktemp("videos") / "dual.mp4"
    cmd = [
        "ffmpeg", "-y",
        "-f", "lavfi", "-i", "color=c=0x1a1a2e:s=640x360:d=4:r=30",
        "-f", "lavfi", "-i", "sine=frequency=440:duration=4",
        "-filter_complex",
        "[0:v]"
        "drawbox=x=20:y=80:w=120:h=160:color=0xffcc99:t=fill,"
        "drawbox=x=500:y=80:w=120:h=160:color=0xcc9966:t=fill,"
        "drawbox=x=318:y=0:w=4:h=360:color=0xffffff:t=fill[split]",
        "-map", "[split]", "-map", "1:a",
        "-c:v", "libx264", "-c:a", "aac", "-shortest", "-pix_fmt", "yuv420p",
        str(out),
    ]
    _ffmpeg(cmd, "dual video")
    return out


# ─── Black panel on right ──────────────────────────────────────────────────

@pytest.fixture(scope="session")
def test_video_black_panel(tmp_path_factory):
    """Right half is black — guest cam off scenario."""
    out = tmp_path_factory.mktemp("videos") / "black_panel.mp4"
    cmd = [
        "ffmpeg", "-y",
        "-f", "lavfi", "-i", "color=c=0x2d5a27:s=640x360:d=4:r=30",
        "-f", "lavfi", "-i", "color=c=black:s=640x360:d=4:r=30",
        "-f", "lavfi", "-i", "sine=frequency=440:duration=4",
        "-filter_complex",
        "[0:v]crop=320:360:0:0[left];"
        "[1:v]crop=320:360:0:0[right];"
        "[left][right]hstack=inputs=2,drawbox=x=40:y=80:w=120:h=160:color=0xffcc99:t=fill[pic]",
        "-map", "[pic]", "-map", "2:a",
        "-c:v", "libx264", "-c:a", "aac", "-shortest", "-pix_fmt", "yuv420p",
        str(out),
    ]
    _ffmpeg(cmd, "black panel video")
    return out


# ─── Screen-share layout ───────────────────────────────────────────────────

@pytest.fixture(scope="session")
def test_video_screen_share(tmp_path_factory):
    """High edge-density frame — simulates screen share with small face PIP."""
    out = tmp_path_factory.mktemp("videos") / "screen_share.mp4"
    # Generate a checkerboard pattern (high edge density) + small face region
    cmd = [
        "ffmpeg", "-y",
        "-f", "lavfi", "-i", "color=c=0x1a1a2e:s=640x360:d=4:r=30",
        "-f", "lavfi", "-i", "sine=frequency=440:duration=4",
        "-filter_complex",
        "[0:v]"
        "drawgrid=w=40:h=40:color=0xffffff@0.3:t=2,"
        "drawbox=x=40:y=240:w=100:h=100:color=0xffcc99:t=fill[share]",
        "-map", "[share]", "-map", "1:a",
        "-c:v", "libx264", "-c:a", "aac", "-shortest", "-pix_fmt", "yuv420p",
        str(out),
    ]
    _ffmpeg(cmd, "screen share video")
    return out


# ─── Chat overlay layout ───────────────────────────────────────────────────

@pytest.fixture(scope="session")
def test_video_chat(tmp_path_factory):
    """Face on left, bright text-like region on right — chat overlay."""
    out = tmp_path_factory.mktemp("videos") / "chat.mp4"
    cmd = [
        "ffmpeg", "-y",
        "-f", "lavfi", "-i", "color=c=0x2d5a27:s=640x360:d=4:r=30",
        "-f", "lavfi", "-i", "sine=frequency=440:duration=4",
        "-filter_complex",
        "[0:v]"
        "drawbox=x=40:y=80:w=120:h=160:color=0xffcc99:t=fill,"
        "drawbox=x=500:y=0:w=140:h=360:color=0x333333@0.8:t=fill,"
        "drawbox=x=510:y=20:w=120:h=30:color=0xffffff:t=fill,"
        "drawbox=x=510:y=60:w=120:h=30:color=0xcccccc:t=fill[chat]",
        "-map", "[chat]", "-map", "1:a",
        "-c:v", "libx264", "-c:a", "aac", "-shortest", "-pix_fmt", "yuv420p",
        str(out),
    ]
    _ffmpeg(cmd, "chat overlay video")
    return out


# ─── Blank / no-face layout ────────────────────────────────────────────────

@pytest.fixture(scope="session")
def test_video_blank(tmp_path_factory):
    """Solid green, no face — should be dropped by pipeline."""
    out = tmp_path_factory.mktemp("videos") / "blank.mp4"
    cmd = [
        "ffmpeg", "-y",
        "-f", "lavfi", "-i", "color=c=0x2d5a27:s=640x360:d=4:r=30",
        "-f", "lavfi", "-i", "sine=frequency=440:duration=4",
        "-c:v", "libx264", "-c:a", "aac", "-shortest", "-pix_fmt", "yuv420p",
        str(out),
    ]
    _ffmpeg(cmd, "blank video")
    return out


# ─── Black screen ─────────────────────────────────────────────────────────

@pytest.fixture(scope="session")
def test_video_black(tmp_path_factory):
    """Full black frames + silence — transition/blank segment."""
    out = tmp_path_factory.mktemp("videos") / "black.mp4"
    cmd = [
        "ffmpeg", "-y",
        "-f", "lavfi", "-i", "color=c=black:s=640x360:d=4:r=30",
        "-f", "lavfi", "-i", "sine=frequency=50:duration=4:volume=0.01",
        "-c:v", "libx264", "-c:a", "aac", "-shortest", "-pix_fmt", "yuv420p",
        str(out),
    ]
    _ffmpeg(cmd, "black video")
    return out


# ─── Logo fixture ─────────────────────────────────────────────────────────

@pytest.fixture(scope="session")
def test_logo(tmp_path_factory):
    """Simple red square logo."""
    out = tmp_path_factory.mktemp("assets") / "logo.png"
    cmd = [
        "ffmpeg", "-y", "-f", "lavfi", "-i", "color=c=red:s=100x100:d=1",
        "-frames:v", "1", str(out),
    ]
    _ffmpeg(cmd, "logo")
    return out


# ─── All-in-one fixture for parameterized tests ───────────────────────────

@pytest.fixture(
    scope="session",
    params=[
        "test_video_solo",
        "test_video_dual",
        "test_video_black_panel",
        "test_video_screen_share",
        "test_video_chat",
        "test_video_blank",
        "test_video_black",
    ],
    ids=["solo", "dual", "black_panel", "screen_share", "chat", "blank", "black"],
)
def any_video(request):
    """Parametrized fixture — runs test against every layout type."""
    return request.getfixturevalue(request.param)


# ─── Backward compatibility aliases (old test names) ──────────────────────

@pytest.fixture(scope="session")
def test_video(test_video_solo):
    """Alias: old name for solo video."""
    return test_video_solo
