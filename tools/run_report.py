#!/usr/bin/env python3
"""
run_report.py — One-shot face detection report pipeline.

Usage:
  python tools/run_report.py                    # run everything
  python tools/run_report.py --skip-sampling    # reuse existing per_frame_results
  python tools/run_report.py --open             # open report in browser when done
  python tools/run_report.py --validate-only    # only check existing outputs
"""

import json
import sys
import subprocess
import os
import time
from pathlib import Path

TOOLS = Path(__file__).resolve().parent
ROOT = TOOLS.parent
sys.path.insert(0, str(ROOT))

REQUIRED_FILES = {
    "expectation.png": None,
    "input/video.mp4": None,
}

REPORTS = ROOT / "reports" / "face_detection"

STEP_OUTPUTS = {
    "expectation":    [str(REPORTS / "expectation_metrics.json"), str(REPORTS / "expectation_face_roi.png")],
    "sampling":       [str(REPORTS / "per_frame_results.json"), str(REPORTS / "detection_summary.json")],
    "crop":           [str(REPORTS / "cropped_metrics.json"), str(REPORTS / "cropped_summary.json")],
    "report":         [str(REPORTS / "face_detection_report.html")],
}

PASS = "✓"
FAIL = "✗"
SKIP = "–"

def green(s):  return f"\033[92m{s}\033[0m"
def red(s):    return f"\033[91m{s}\033[0m"
def yellow(s): return f"\033[93m{s}\033[0m"
def blue(s):   return f"\033[94m{s}\033[0m"
def bold(s):   return f"\033[1m{s}\033[0m"

def header(title):
    w = 68
    print(f"\n  {blue('━' * w)}")
    print(f"  {blue('┃')}  {bold(title):<62s} {blue('┃')}")
    print(f"  {blue('━' * w)}")

def _resolve(p):
    return p if isinstance(p, Path) and p.is_absolute() else ROOT / p

def check_outputs(step_name):
    files = STEP_OUTPUTS[step_name]
    missing = [f for f in files if not _resolve(f).exists()]
    return missing, len(missing) == 0

def run_step(script_name, step_name, skip_flag=False, python_args=None):
    header(f"Step: {step_name}")
    if skip_flag:
        missing, ok = check_outputs(step_name)
        if ok:
            print(f"  {SKIP} Skipping — all outputs already exist")
            for f in STEP_OUTPUTS[step_name]:
                p = _resolve(f)
                print(f"     {green('✔')} {f} ({p.stat().st_size/1024:.0f} KB)")
            return True
        else:
            print(f"  {yellow('⟳')} Missing outputs, running anyway")

    script = TOOLS / script_name
    if not script.exists():
        print(f"  {red(FAIL)} Script not found: {script}")
        return False

    print(f"  Running: python3 tools/{script_name}")
    t0 = time.time()
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT)

    result = subprocess.run(
        [sys.executable, str(script)] + (python_args or []),
        capture_output=True, text=True, env=env, cwd=str(ROOT)
    )
    elapsed = time.time() - t0

    if result.returncode != 0:
        print(f"  {red(FAIL)} Failed (exit code {result.returncode}, {elapsed:.1f}s)")
        if result.stdout.strip():
            for line in result.stdout.strip().splitlines()[:10]:
                print(f"    │ {line}")
        if result.stderr.strip():
            print(f"  {red('Stderr:')}")
            for line in result.stderr.strip().splitlines()[:15]:
                print(f"    │ {line}")
        return False

    missing, ok = check_outputs(step_name)
    if not ok:
        print(f"  {red(FAIL)} Missing outputs: {', '.join(missing)}")
        return False

    print(f"  {green(PASS)} Done in {elapsed:.1f}s")
    for f in STEP_OUTPUTS[step_name]:
        p = _resolve(f)
        print(f"     {green('✔')} {f} ({p.stat().st_size/1024:.0f} KB)" if p.exists()
              else f"     {red('✘')} {f} (missing)")
    return True

def validate():
    header("Validation")
    all_ok = True

    # 1. Required input files
    print(f"  {bold('Input files:')}")
    for name, _ in REQUIRED_FILES.items():
        p = ROOT / name
        ok = p.exists()
        if not ok:
            print(f"    {red(FAIL)} {name} — NOT FOUND")
            all_ok = False
        else:
            print(f"    {green(PASS)} {name} ({p.stat().st_size/1024:.0f} KB)")

    # 2. Step outputs
    for step_name in STEP_OUTPUTS:
        missing, ok = check_outputs(step_name)
        status = green(PASS) if ok else red(FAIL)
        print(f"  {status} {step_name}")
        if not ok:
            all_ok = False

    # 3. JSON validity
    print(f"  {bold('JSON integrity:')}")
    for step_name in STEP_OUTPUTS:
        for relpath in STEP_OUTPUTS[step_name]:
            if not relpath.endswith(".json"):
                continue
            p = ROOT / relpath
            if not p.exists():
                print(f"    {red(FAIL)} {relpath} — missing")
                all_ok = False
                continue
            try:
                with open(p) as f:
                    json.load(f)
                print(f"    {green(PASS)} {relpath}")
            except json.JSONDecodeError as e:
                print(f"    {red(FAIL)} {relpath} — invalid JSON: {e}")
                all_ok = False

    # 4. Data sanity
    print(f"  {bold('Data sanity:')}")
    summary_path = REPORTS / "detection_summary.json"
    if summary_path.exists():
        with open(summary_path) as f:
            s = json.load(f)
        checks = [
            ("detection_rate > 0", s.get("detection_rate", 0) > 0),
            ("samples == 30", s.get("samples") == 30),
            ("avg_confidence > 0.5", s.get("avg_confidence", 0) > 0.5),
            ("frame_width > 0", s.get("frame_width", 0) > 0),
            ("frame_height > 0", s.get("frame_height", 0) > 0),
        ]
        for label, ok in checks:
            sym = green(PASS) if ok else red(FAIL)
            print(f"    {sym} {label}")

    # 5. Report HTML
    report = REPORTS / "face_detection_report.html"
    if report.exists():
        with open(report) as f:
            html = f.read()
        content_checks = [
            ("Contains 'After Cropping'", "After Cropping" in html),
            ("Contains face ROI images", "data:image" in html),
            ("Contains 'Sharpness Drop'", "Sharpness Drop" in html),
            ("Contains 'Face Position Consistency'", "Face Position Consistency" in html),
            ("Size > 100 KB", len(html) > 100 * 1024),
        ]
        for label, ok in content_checks:
            sym = green(PASS) if ok else red(FAIL)
            print(f"    {sym} {label}")

    print(f"  {bold('RAM:')}")
    try:
        import psutil
        m = psutil.virtual_memory()
        free_gb = m.available / 1024**3
        sym = green(PASS) if free_gb > 1.0 else yellow("⚠")
        print(f"    {sym} {free_gb:.2f} GB free / {m.total/1024**3:.1f} GB total")
    except ImportError:
        print(f"    {SKIP} psutil not available")

    print()
    if all_ok:
        print(f"  {green('All checks passed.')}")
    else:
        print(f"  {red('Some checks failed.')}")
    return all_ok

def open_report():
    report = ROOT / "tools/face_detection_report.html"
    if not report.exists():
        print(f"  {red(FAIL)} Report not found at {report}")
        return False
    import subprocess, sys
    if sys.platform == "darwin":
        subprocess.run(["open", str(report)], check=True)
    elif sys.platform == "linux":
        subprocess.run(["xdg-open", str(report)], check=True)
    print(f"  {green(PASS)} Opened in browser")
    return True

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Face Detection Report Pipeline")
    parser.add_argument("--skip-sampling", action="store_true",
                        help="Reuse existing per_frame_results.json (skip frame sampling)")
    parser.add_argument("--open", action="store_true",
                        help="Open report in browser when done")
    parser.add_argument("--validate-only", action="store_true",
                        help="Only validate existing outputs, skip all processing")
    args = parser.parse_args()

    REPORTS.mkdir(parents=True, exist_ok=True)
    (REPORTS / "frames").mkdir(parents=True, exist_ok=True)
    (REPORTS / "cropped_faces").mkdir(parents=True, exist_ok=True)

    print(f"\n  {bold('Face Detection Report Pipeline')}")
    print(f"  {'─' * 40}")
    print(f"  Root: {ROOT}")
    print(f"  Reports: {REPORTS}")
    print(f"  Mode: {'validate only' if args.validate_only else 'full pipeline'}")

    if args.validate_only:
        ok = validate()
        sys.exit(0 if ok else 1)

    # Step 1: Expectation analysis
    if not run_step("expectation_analyzer.py", "expectation"):
        print(f"\n  {red('Pipeline aborted at expectation analysis')}")
        sys.exit(1)

    # Step 2: Frame sampling
    if not run_step("frame_sampler.py", "sampling", skip_flag=args.skip_sampling):
        print(f"\n  {red('Pipeline aborted at frame sampling')}")
        sys.exit(1)

    # Step 3: Crop analysis
    if not run_step("crop_analyzer.py", "crop"):
        print(f"\n  {red('Pipeline aborted at crop analysis')}")
        sys.exit(1)

    # Step 4: Report generation
    if not run_step("report_generator.py", "report"):
        print(f"\n  {red('Pipeline aborted at report generation')}")
        sys.exit(1)

    # Validate
    ok = validate()

    # Open browser
    if args.open:
        open_report()

    print(f"\n  {green('Done.')}")
    sys.exit(0 if ok else 1)

if __name__ == "__main__":
    main()
