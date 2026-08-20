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
    for path in ROOT.rglob("*.py"):
        relative = path.relative_to(ROOT).as_posix()
        if relative.startswith(("tests/", "face_os/")):
            continue
        text = path.read_text(encoding="utf-8", errors="ignore")
        if "seo_performance.json" in text or "automation.seo.seo_learner" in text:
            offenders.append(relative)
    assert offenders == []
