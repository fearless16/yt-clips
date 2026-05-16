"""
Tests for push_code.py — smart code sync to Google Drive.
"""
from pathlib import Path
from unittest.mock import patch, MagicMock
import pytest


class TestPushCodeVerifySource:
    """
    Verify the ACTUAL push_code.py source includes these files.
    This catches if someone edits push_code.py and forgets the job file.
    """

    def test_push_code_contains_remote_job_json(self):
        with open("push_code.py") as f:
            content = f.read()
        assert "remote_job.json" in content, (
            "push_code.py does not reference remote_job.json anywhere!"
        )

    def test_push_code_contains_colab_url_txt(self):
        with open("push_code.py") as f:
            content = f.read()
        assert "colab_url.txt" in content, (
            "push_code.py does not reference colab_url.txt anywhere!"
        )
