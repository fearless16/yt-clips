"""colab_server.py — Face OS compute server for Google Colab.

Run this on Colab to offload pipeline execution via HTTP.
Exposes REST endpoints for enrollment, processing, and telemetry.

Usage on Colab:
    !pip install flask pyngrok
    !python face_os/colab_server.py

Endpoints:
    POST /enroll       — enroll identity from reference images
    POST /process      — process video clip
    GET  /telemetry    — get last run telemetry
    GET  /health       — health check
    GET  /gpu          — GPU info
"""

import os
import sys
import json
import time
import tempfile
import traceback
from pathlib import Path

# Ensure project root is in sys.path
_project_root = str(Path(__file__).resolve().parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from flask import Flask, request, jsonify

app = Flask(__name__)

# Global pipeline state
_pipeline = None
_last_telemetry = None
_last_frame_telemetry = None


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "engine": "face_os", "timestamp": time.time()})


@app.route("/gpu", methods=["GET"])
def gpu_info():
    """Return GPU information."""
    try:
        import torch
        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            return jsonify({
                "available": True,
                "name": torch.cuda.get_device_name(0),
                "memory_total": props.total_memory / 1e9,
                "memory_free": (props.total_memory - torch.cuda.memory_allocated(0)) / 1e9,
            })
        return jsonify({"available": False, "name": "CUDA not available"})
    except Exception as e:
        return jsonify({"available": False, "name": "CPU only", "detail": str(e)})


@app.route("/enroll", methods=["POST"])
def enroll():
    """Enroll identity from reference images.

    Form data:
        reference: reference image file OR drive path to reference
        photos: directory of photo files (optional)
        drive_path: path on Colab Drive to read photos from (optional)

    Returns:
        JSON with enrollment status
    """
    global _pipeline

    try:
        from face_os.pipeline import FaceOSPipeline

        # Get reference image
        ref_file = request.files.get("reference")
        drive_path = request.form.get("drive_path")

        if ref_file:
            ref_path = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
            ref_file.save(ref_path.name)
        elif drive_path:
            ref_path = drive_path
        else:
            return jsonify({"error": "No reference image provided"}), 400

        # Get photos — from upload, drive path, or auto-detect
        photos_dir = None
        uploaded_photos = request.files.getlist("photos")
        if uploaded_photos:
            photos_dir = tempfile.mkdtemp()
            for f in uploaded_photos:
                f.save(os.path.join(photos_dir, f.filename))
        elif drive_path:
            # Look for photos/ subfolder relative to drive_path
            candidate = os.path.join(os.path.dirname(drive_path), "photos")
            if os.path.isdir(candidate):
                photos_dir = candidate
            else:
                photos_dir = tempfile.mkdtemp()

        # Create pipeline and enroll
        _pipeline = FaceOSPipeline(use_bidirectional=False)
        success = _pipeline.enroll(str(ref_path), photos_dir)

        return jsonify({
            "success": success,
            "message": "Enrollment complete" if success else "Enrollment failed",
        })

    except Exception as e:
        return jsonify({"error": str(e), "trace": traceback.format_exc()}), 500


@app.route("/process", methods=["POST"])
def process():
    """Process a video clip.

    Form data:
        video: video file OR drive_path to video
        max_frames: max frames to process (default: 30)
        drive_path: path on Colab Drive to read video from (optional)

    Returns:
        JSON with processing results and telemetry
    """
    global _pipeline, _last_telemetry, _last_frame_telemetry

    if _pipeline is None:
        return jsonify({"error": "Not enrolled. Call /enroll first."}), 400

    try:
        video_file = request.files.get("video")
        drive_path = request.form.get("drive_path")
        max_frames = int(request.form.get("max_frames", 30))

        if video_file:
            video_path = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
            video_file.save(video_path.name)
        elif drive_path and os.path.exists(drive_path):
            video_path = drive_path
        else:
            return jsonify({"error": "No video file provided"}), 400

        # Output path
        output_path = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False).name

        # Process
        t0 = time.time()
        result = _pipeline.process(video_path.name, output_path, max_frames=max_frames)
        elapsed = time.time() - t0

        # Collect telemetry
        _last_telemetry = _pipeline.get_telemetry_report()
        _last_frame_telemetry = _pipeline.get_frame_telemetry()

        # Read output video
        output_data = None
        if result and os.path.exists(result):
            with open(result, "rb") as f:
                output_data = f.read()

        return jsonify({
            "success": result is not None,
            "output_path": result,
            "wall_time_s": round(elapsed, 1),
            "frames_processed": len(_last_frame_telemetry),
            "fps": round(len(_last_frame_telemetry) / elapsed, 2) if elapsed > 0 else 0,
            "telemetry": _last_telemetry,
            "per_frame_count": len(_last_frame_telemetry),
            "output_size_bytes": len(output_data) if output_data else 0,
        })

    except Exception as e:
        return jsonify({"error": str(e), "trace": traceback.format_exc()}), 500


@app.route("/telemetry", methods=["GET"])
def telemetry():
    """Get telemetry from last run."""
    if _last_telemetry is None:
        return jsonify({"error": "No telemetry available. Run /process first."}), 400
    return jsonify({
        "aggregate": _last_telemetry,
        "per_frame_count": len(_last_frame_telemetry) if _last_frame_telemetry else 0,
    })


@app.route("/telemetry/frames", methods=["GET"])
def frame_telemetry():
    """Get per-frame telemetry from last run."""
    if _last_frame_telemetry is None:
        return jsonify({"error": "No frame telemetry available."}), 400
    start = int(request.args.get("start", 0))
    limit = int(request.args.get("limit", 100))
    return jsonify({
        "frames": _last_frame_telemetry[start:start+limit],
        "total": len(_last_frame_telemetry),
    })


@app.route("/reset", methods=["POST"])
def reset():
    """Reset pipeline state."""
    global _pipeline, _last_telemetry, _last_frame_telemetry
    _pipeline = None
    _last_telemetry = None
    _last_frame_telemetry = None
    return jsonify({"status": "reset"})


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "5000"))
    print(f"Face OS Colab Server starting on port {port}")
    print(f"Endpoints: /health, /gpu, /enroll, /process, /telemetry, /telemetry/frames, /reset")
    app.run(host="0.0.0.0", port=port, debug=False)
