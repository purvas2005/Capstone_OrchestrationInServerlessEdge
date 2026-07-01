#!/usr/bin/env python3

from pathlib import Path
import subprocess
import sys
import re

PROJECT_ROOT = Path(__file__).resolve().parent.parent

LINKS_FILE = PROJECT_ROOT / "config" / "dataset_links.txt"
DOWNLOAD_DIR = PROJECT_ROOT / "data" / "raw"

DOWNLOAD_DIR.mkdir(parents=True, exist_ok=True)

with open(LINKS_FILE) as f:
    entries = [x.strip() for x in f if x.strip()]

print(f"Found {len(entries)} files.\n")

success = 0
failed = []

for i, entry in enumerate(entries, start=1):

    m = re.search(r"/d/([A-Za-z0-9_-]+)", entry)

    if m:
        file_id = m.group(1)
    else:
        file_id = entry

    print("="*70)
    print(f"[{i}/{len(entries)}] {file_id}")
    print("="*70)

    cmd = [
        sys.executable,
        "-m",
        "gdown",
        "--continue",
        "-O",
        str(DOWNLOAD_DIR),
        file_id
    ]

    result = subprocess.run(cmd)

    if result.returncode == 0:
        success += 1
    else:
        failed.append(file_id)

print("\nDownload complete.")
print(f"Successful: {success}")
print(f"Failed: {len(failed)}")
