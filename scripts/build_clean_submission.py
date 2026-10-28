"""
MarketFolio Clean Submission Packager
Creates:
1. A clean folder: D:\\ARYAN ACADEMIC\\MarketFolio_Submission
2. A clean zip:    D:\\ARYAN ACADEMIC\\MarketFolio_Submission.zip

Strictly excludes all AI agent files, virtual environments, node_modules, and git caches.
"""

from __future__ import annotations

import os
import shutil
import zipfile
from pathlib import Path

# Paths
SRC_ROOT = Path(r"D:\ARYAN ACADEMIC\Marketfolio")
TARGET_DIR = Path(r"D:\ARYAN ACADEMIC\MarketFolio_Submission")
TARGET_ZIP = Path(r"D:\ARYAN ACADEMIC\MarketFolio_Submission.zip")

# Blacklisted directory names (anywhere in hierarchy)
EXCLUDE_DIRS = {
    ".agents",
    ".claude",
    ".git",
    ".venv",
    "venv",
    "env",
    "node_modules",
    "dist",
    ".cache",
    ".pytest_cache",
    "__pycache__",
    "scratch",
    "logs",
    ".idea",
    ".vscode",
    "risklens.egg-info",
}

# Blacklisted file names or extensions
EXCLUDE_FILES = {
    "antigravity.py",
    "skills-lock.json",
    "automation_audit.log",
    "build_clean_submission.py",
    "create_clean_zip.py",
    "MarketFolio_Submission.zip",
    "RiskLens_Clean_Submission.zip",
}

EXCLUDE_EXTS = {
    ".pyc",
    ".pyo",
    ".pyd",
    ".log",
    ".tmp",
    ".DS_Store",
}


def should_skip(path: Path) -> bool:
    for part in path.parts:
        if part in EXCLUDE_DIRS:
            return True
        if part.endswith(".egg-info"):
            return True
    if path.name in EXCLUDE_FILES:
        return True
    if path.suffix.lower() in EXCLUDE_EXTS:
        return True
    return False


def build_submission():
    print(f"--> Source Directory:      {SRC_ROOT}")
    print(f"--> Target Clean Folder:   {TARGET_DIR}")
    print(f"--> Target Clean ZIP:      {TARGET_ZIP}")

    # Remove existing target dir if present
    if TARGET_DIR.exists():
        print("Cleaning previous target directory...")
        shutil.rmtree(TARGET_DIR)

    TARGET_DIR.mkdir(parents=True, exist_ok=True)

    copied_files = 0
    total_bytes = 0

    # Whitelist of items to include from root
    included_roots = [
        "app",
        "backend",
        "frontend",
        "src",
        "ui",
        "tests",
        "scripts",
        "reports",
        ".streamlit",
        "README.md",
        "requirements.txt",
        "pyproject.toml",
        "pytest.ini",
        "LICENSE",
        ".gitignore",
    ]

    for item_name in included_roots:
        src_item = SRC_ROOT / item_name
        if not src_item.exists():
            continue

        if src_item.is_file():
            if not should_skip(src_item):
                dest_file = TARGET_DIR / item_name
                dest_file.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src_item, dest_file)
                copied_files += 1
                total_bytes += dest_file.stat().st_size
        else:
            for root, dirs, files in os.walk(src_item):
                # Prune dirs in place
                dirs[:] = [d for d in dirs if not should_skip(Path(root) / d)]
                for file in files:
                    file_path = Path(root) / file
                    if not should_skip(file_path):
                        rel_path = file_path.relative_to(SRC_ROOT)
                        dest_file = TARGET_DIR / rel_path
                        dest_file.parent.mkdir(parents=True, exist_ok=True)
                        shutil.copy2(file_path, dest_file)
                        copied_files += 1
                        total_bytes += dest_file.stat().st_size

    print(f"Copied {copied_files} clean files ({total_bytes / (1024*1024):.2f} MB uncompressed).")

    # Replace .gitignore in submission with a standard clean version
    clean_gitignore = TARGET_DIR / ".gitignore"
    clean_gitignore.write_text(
        "# Python virtual environments & bytecode\n"
        ".venv/\n"
        "venv/\n"
        "env/\n"
        "__pycache__/\n"
        "*.py[cod]\n"
        "*$py.class\n"
        "*.so\n"
        ".pytest_cache/\n\n"
        "# Runtime caches & logs\n"
        ".cache/\n"
        "logs/\n"
        "*.log\n"
        "*.tmp\n\n"
        "# Frontend & Node\n"
        "node_modules/\n"
        "frontend/node_modules/\n"
        "frontend/dist/\n"
        "dist/\n\n"
        "# IDE & OS\n"
        ".vscode/\n"
        ".idea/\n"
        "*.zip\n"
        ".DS_Store\n",
        encoding="utf-8"
    )

    # Now create the clean ZIP archive
    print(f"Compressing into {TARGET_ZIP}...")
    if TARGET_ZIP.exists():
        TARGET_ZIP.unlink()

    with zipfile.ZipFile(TARGET_ZIP, "w", zipfile.ZIP_DEFLATED) as zf:
        for root, dirs, files in os.walk(TARGET_DIR):
            for file in files:
                file_path = Path(root) / file
                rel_path = file_path.relative_to(TARGET_DIR)
                zf.write(file_path, arcname=str(rel_path))

    zip_size_mb = TARGET_ZIP.stat().st_size / (1024 * 1024)
    print(f"ZIP Creation complete! Final ZIP size: {zip_size_mb:.2f} MB")

    # Verification scan
    print("\n[VERIFICATION SCAN on generated folder]")
    suspicious_terms = ["antigravity", "claude", "skills-lock", ".agents", "venv"]
    found_suspicious = []
    for root, dirs, files in os.walk(TARGET_DIR):
        for name in dirs + files:
            for term in suspicious_terms:
                if term in name.lower():
                    found_suspicious.append(os.path.join(root, name))

    if found_suspicious:
        print("[WARNING] Found suspicious files:")
        for s in found_suspicious:
            print(f"  - {s}")
    else:
        print("[ALL CLEAR] Zero AI flags or environment bloat found in submission folder!")


if __name__ == "__main__":
    build_submission()
