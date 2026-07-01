#!/usr/bin/env python3

from pathlib import Path
import zipfile

PROJECT_ROOT = Path(__file__).resolve().parent.parent

RAW = PROJECT_ROOT / "data" / "raw"
EXTRACTED = PROJECT_ROOT / "data" / "extracted"

EXTRACTED.mkdir(parents=True, exist_ok=True)

zip_files = sorted(RAW.glob("*.zip"))

print(f"Found {len(zip_files)} ZIP files.\n")

for i, archive in enumerate(zip_files, start=1):

    destination = EXTRACTED / archive.stem

    if destination.exists():
        print(f"[{i}/{len(zip_files)}] Skipping {archive.name}")
        continue

    print(f"[{i}/{len(zip_files)}] Extracting {archive.name}")

    destination.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(archive) as z:
        z.extractall(destination)

print("\nExtraction complete.")
