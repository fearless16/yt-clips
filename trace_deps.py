#!/usr/bin/env python3
"""
Diagnostic script to trace dependency failures in watcher.py and automation module.
Runs in one shot to identify import/dependency issues.
"""

import sys
import os
import traceback
from pathlib import Path

# Add project root to path
project_root = str(Path(__file__).resolve().parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

print("=" * 70)
print("DEPENDENCY TRACE SCRIPT")
print("=" * 70)
print(f"Project root: {project_root}")
print(f"Python: {sys.version}")
print()

def test_import(module_name, submodule=None):
    """Test importing a module and report success/failure."""
    full_name = f"{module_name}.{submodule}" if submodule else module_name
    try:
        if submodule:
            # Handle nested submodules like "seo.seo"
            parts = submodule.split(".")
            obj = __import__(module_name, fromlist=[parts[0]])
            for part in parts:
                obj = getattr(obj, part)
        else:
            __import__(module_name)
        print(f"✓ {full_name}")
        return True
    except Exception as e:
        print(f"✗ {full_name}: {type(e).__name__}: {e}")
        return False

def test_watcher_imports():
    """Test imports used by watcher.py."""
    print("\n" + "=" * 70)
    print("TESTING WATCHER IMPORTS")
    print("=" * 70)

    # Standard library imports (should always work)
    stdlib_modules = [
        "json", "os", "subprocess", "sys", "time", "threading",
        "http.server", "pathlib", "urllib.parse"
    ]
    for mod in stdlib_modules:
        test_import(mod)

    # Project-specific imports
    print("\n--- Project Imports ---")
    test_import("automation", "cli")

def test_automation_module():
    """Test automation module structure."""
    print("\n" + "=" * 70)
    print("TESTING AUTOMATION MODULE STRUCTURE")
    print("=" * 70)

    # Test __init__.py
    print("\n--- automation/__init__.py ---")
    try:
        import automation
        print(f"✓ automation package loaded (version: {getattr(automation, 'VERSION', 'unknown')})")
    except Exception as e:
        print(f"✗ automation package: {type(e).__name__}: {e}")
        traceback.print_exc()

    # Test submodules
    print("\n--- automation submodules ---")
    submodules = [
        "_cache", "config", "env", "memory", "transcript", "scoring",
        "watcher", "tunnel", "worker", "orchestrator", "cli"
    ]
    for sub in submodules:
        test_import("automation", sub)

def test_orchestrator_dependencies():
    """Test orchestrator.py specific dependencies."""
    print("\n" + "=" * 70)
    print("TESTING ORCHESTRATOR DEPENDENCIES")
    print("=" * 70)

    # Orchestrator imports these root-level modules
    root_modules = [
        "download", "transcribe", "highlight", "export", "sync",
        "upload", "scheduler", "thumbnail", "ref_grade", "face_mapper"
    ]

    print("\n--- Root-level modules (expected in project root) ---")
    for mod in root_modules:
        test_import(mod)

    # SEO subpackage
    print("\n--- SEO subpackage ---")
    test_import("automation", "seo.seo")
    test_import("automation", "seo.analytics")

def check_file_structure():
    """Check expected file structure."""
    print("\n" + "=" * 70)
    print("CHECKING FILE STRUCTURE")
    print("=" * 70)

    expected_dirs = [
        "automation",
        "automation/seo",
        "tests",
    ]

    expected_files = [
        "automation/__init__.py",
        "automation/orchestrator.py",
        "automation/watcher.py",
        "automation/cli.py",
        "automation/seo/__init__.py",
        "automation/seo/seo.py",
        "automation/seo/analytics.py",
        "tests/test_automation.py",
    ]

    print("\n--- Directories ---")
    for d in expected_dirs:
        path = Path(project_root) / d
        if path.exists() and path.is_dir():
            print(f"✓ {d}/")
        else:
            print(f"✗ {d}/ (missing or not a directory)")

    print("\n--- Files ---")
    for f in expected_files:
        path = Path(project_root) / f
        if path.exists() and path.is_file():
            print(f"✓ {f}")
        else:
            print(f"✗ {f} (missing)")

def check_sys_path():
    """Check sys.path configuration."""
    print("\n" + "=" * 70)
    print("CHECKING SYS.PATH")
    print("=" * 70)
    for i, p in enumerate(sys.path):
        print(f"{i:2d}: {p}")

def run_pytest():
    """Run pytest to see actual failures."""
    print("\n" + "=" * 70)
    print("RUNNING PYTEST")
    print("=" * 70)
    import subprocess
    try:
        result = subprocess.run(
            [sys.executable, "-m", "pytest", "tests/test_automation.py", "-v", "--tb=short"],
            cwd=project_root,
            capture_output=True,
            text=True,
            timeout=60
        )
        print("STDOUT:")
        print(result.stdout)
        if result.stderr:
            print("\nSTDERR:")
            print(result.stderr)
        print(f"\nReturn code: {result.returncode}")
    except Exception as e:
        print(f"Failed to run pytest: {e}")

def main():
    """Run all diagnostics."""
    check_sys_path()
    check_file_structure()
    test_watcher_imports()
    test_automation_module()
    test_orchestrator_dependencies()
    run_pytest()

    print("\n" + "=" * 70)
    print("DIAGNOSTIC COMPLETE")
    print("=" * 70)

if __name__ == "__main__":
    main()
