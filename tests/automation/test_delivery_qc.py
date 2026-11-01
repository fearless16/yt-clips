import json
import subprocess

import pytest

from automation.delivery_qc import check_delivery


@pytest.mark.parametrize("integrated,peak,expected", [("-16.0", "-1.5", "PASS"), ("-12", "-0.5", "FAIL"), ("-inf", "-inf", "FAIL")])
def test_qc_uses_encoded_input_measurements_not_hypothetical_output(tmp_path, monkeypatch, integrated, peak, expected):
    clip = tmp_path / "clip.mp4"
    clip.write_bytes(b"fixture")
    probe = {"streams": [{"codec_type": "video", "width": 1080, "height": 1920, "start_time": "0", "duration": "10"},
                         {"codec_type": "audio", "start_time": "0", "duration": "10"}], "format": {"duration": "10.0"}}
    calls = []
    def run(cmd, **kwargs):
        calls.append(cmd)
        if cmd[0] == "ffprobe":
            return subprocess.CompletedProcess(cmd, 0, json.dumps(probe), "")
        return subprocess.CompletedProcess(cmd, 0, "", json.dumps({"input_i": integrated, "input_tp": peak, "output_i": "-16", "output_tp": "-2"}))
    monkeypatch.setattr("automation.delivery_qc.subprocess.run", run)
    result = check_delivery(clip, 10)
    assert result["status"] == expected
    assert result["listening_review"] == "NOT CHECKED"
    assert "-xerror" in calls[1]
    assert json.loads(clip.with_name("clip_qc.json").read_text())["status"] == expected


def test_failed_decode_cannot_pass_qc(tmp_path, monkeypatch):
    clip = tmp_path / "clip.mp4"
    clip.write_bytes(b"fixture")
    def run(cmd, **kwargs):
        raise subprocess.CalledProcessError(1, cmd, stderr="decode error")
    monkeypatch.setattr("automation.delivery_qc.subprocess.run", run)
    assert check_delivery(clip)["status"] == "FAIL"


def test_equal_duration_but_offset_audio_fails_delivery_qc(tmp_path, monkeypatch):
    clip = tmp_path / "clip.mp4"
    clip.write_bytes(b"fixture")
    probe = {"streams": [{"codec_type": "video", "width": 1080, "height": 1920, "start_time": "0", "duration": "10"},
                         {"codec_type": "audio", "start_time": "0.5", "duration": "10"}], "format": {"duration": "10.5"}}
    def run(cmd, **kwargs):
        if cmd[0] == "ffprobe":
            return subprocess.CompletedProcess(cmd, 0, json.dumps(probe), "")
        return subprocess.CompletedProcess(cmd, 0, "", json.dumps({"input_i": "-16", "input_tp": "-2"}))
    monkeypatch.setattr("automation.delivery_qc.subprocess.run", run)
    result = check_delivery(clip)
    assert result["status"] == "FAIL"
    assert result["av_timing"]["status"] == "FAIL"
    assert any("start offset" in error for error in result["errors"])
