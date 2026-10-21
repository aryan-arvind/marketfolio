"""
Clean ZIP packager for RiskLens project.
Excludes .venv, .git, caches, and bytecode to produce a clean, lightweight zip (< 2 MB) in seconds.
"""

from __future__ import annotations

import os
import time
import zipfile
from pathlib import Path

IGNORE_DIRS = {
    ".venv",
    "venv",
    "env",
    ".git",
    ".cache",
    ".pytest_cache",
    "__pycache__",
    ".idea",
    ".vscode",
}

IGNORE_EXTENSIONS = {
    ".pyc",
    ".pyo",
    ".pyd",
    ".log",
    ".tmp",
    ".DS_Store",
}


def package_project(root_dir: Path | None = None, output_zip: Path | None = None) -> Path:
    if root_dir is None:
        root_dir = Path(__file__).resolve().parent.parent

    if output_zip is None:
        output_zip = root_dir / "RiskLens_Clean_Submission.zip"

    start_time = time.time()
    file_count = 0
    total_uncompressed = 0

    print(f"Scanning directory: {root_dir}")
    print(f"Target archive:    {output_zip.name}")
    print("Excluding: .venv, .git, caches, bytecode...")

    with zipfile.ZipFile(output_zip, "w", zipfile.ZIP_DEFLATED) as zf:
        for dirpath, dirnames, filenames in os.walk(root_dir):
            # Prune ignored directories in-place so os.walk does not descend into them
            dirnames[:] = [d for d in dirnames if d not in IGNORE_DIRS and not d.startswith(".")]

            rel_dir = Path(dirpath).relative_to(root_dir)

            for fname in filenames:
                if fname == output_zip.name:
                    continue
                file_path = Path(dirpath) / fname
                if file_path.suffix.lower() in IGNORE_EXTENSIONS:
                    continue

                rel_file = rel_dir / fname
                zf.write(file_path, arcname=str(rel_file))
                file_count += 1
                total_uncompressed += file_path.stat().st_size

    elapsed = time.time() - start_time
    zip_size_kb = output_zip.stat().st_size / 1024
    uncompressed_mb = total_uncompressed / (1024 * 1024)

    print("\n--- Packaging Complete ---")
    print(f"Files Packaged:      {file_count}")
    print(f"Uncompressed Size:   {uncompressed_mb:.2f} MB")
    print(f"Compressed ZIP Size: {zip_size_kb:.1f} KB")
    print(f"Time Taken:          {elapsed:.2f} seconds (not 19 hours!)")
    print(f"Saved to:            {output_zip}")

    return output_zip


if __name__ == "__main__":
    package_project()
