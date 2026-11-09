"""Source-range-bound semantic review, separate from timing or release approval."""

import math
from typing import Any


def completeness_status(item: dict[str, Any]) -> str:
    """Missing evidence is unknown, never a pass. Keep explicit legacy booleans."""
    if "completeness" not in item:
        value = item.get("complete_thought")
        return "PASS" if value is True else "FAIL" if value is False else "UNVERIFIED"
    review = item["completeness"]
    if not isinstance(review, dict):
        return "NEEDS_REVIEW"
    status = review.get("status")
    if status in ("UNVERIFIED", "NEEDS_REVIEW"):
        return status
    if status not in ("PASS", "FAIL"):
        return "NEEDS_REVIEW"
    expected = status == "PASS"
    if item.get("complete_thought") is not expected:
        return "NEEDS_REVIEW"
    if review.get("method") not in ("text_only", "owner_review"):
        return "NEEDS_REVIEW"
    if not isinstance(review.get("reason"), str) or not review["reason"].strip():
        return "NEEDS_REVIEW"
    try:
        start, end = item["start"], item["end"]
        if not all(isinstance(t, (int, float)) and not isinstance(t, bool)
                   and math.isfinite(t) for t in (start, end)) or start < 0 or end <= start:
            return "NEEDS_REVIEW"
        if review.get("source_range") != [start, end]:
            return "NEEDS_REVIEW"
    except KeyError:
        return "NEEDS_REVIEW"
    return status


def completeness_value(item: dict[str, Any]) -> bool | None:
    status = completeness_status(item)
    return True if status == "PASS" else False if status == "FAIL" else None


def set_completeness(item: dict[str, Any], status: str, *,
                     reason: str = "", method: str = "not_reviewed") -> None:
    if status not in ("UNVERIFIED", "PASS", "FAIL", "NEEDS_REVIEW"):
        raise ValueError("Invalid completeness status")
    item["complete_thought"] = True if status == "PASS" else False if status == "FAIL" else None
    item["completeness"] = {
        "status": status, "method": method, "reason": reason,
        "source_range": [item["start"], item["end"]],
    }
