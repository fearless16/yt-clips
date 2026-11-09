import os
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_removed_learning_implementations_cannot_return():
    removed = {
        "automation/seo/seo_learner.py",
        "self_learner",
        "automation/learner",
        "clip_learner.py",
        "weight_learner.py",
        "docs/IMPROVEMENT_PLAN.md",
    }
    assert not [path for path in removed if (ROOT / path).exists()]


def test_production_python_has_no_legacy_learning_database_reference():
    offenders = []
    for directory, folders, files in os.walk(ROOT):
        # Production code only: never read thousands of dependency/generated
        # files under .venv/temp or traverse downloaded media/model trees.
        folders[:] = [name for name in folders if not name.startswith(".") and name not in {
            "tests", "face_os", "node_modules", "temp", "input", "shorts", "logs",
            "transcripts", "highlights", "graphify-out", "__pycache__"}]
        for name in files:
            if not name.endswith(".py"):
                continue
            path = Path(directory) / name
            text = path.read_text(encoding="utf-8", errors="ignore")
            if "seo_performance.json" in text or "automation.seo.seo_learner" in text:
                offenders.append(path.relative_to(ROOT).as_posix())
    assert offenders == []
