"""
Download the SESR-Eval benchmark dataset from Zenodo (record 16408882).

The dataset contains title-abstract screening data for evaluating LLMs in
systematic literature reviews.

Usage:
    python download_data.py
    python download_data.py -o data/
    python download_data.py --no-extract
"""

import argparse
import csv
import hashlib
import re
import shutil
import sys
import zipfile
from pathlib import Path

import requests
from tqdm import tqdm

ZENODO_URL = "https://zenodo.org/api/records/16408882/files/package.zip/content"
FILE_SIZE = 67503360  # bytes
MD5_CHECKSUM = "b1348de05e968435906f32a750f079f7"
FILENAME = "package.zip"


def verify_md5(path: Path, expected: str) -> bool:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest() == expected


def download(url: str, dest: Path) -> None:
    with requests.get(url, stream=True, timeout=120) as r:
        r.raise_for_status()
        with tqdm(total=FILE_SIZE, unit="B", unit_scale=True, desc=FILENAME) as pbar:
            with open(dest, "wb") as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
                    pbar.update(len(chunk))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download the SESR-Eval benchmark dataset from Zenodo."
    )
    parser.add_argument(
        "-o", "--output-dir",
        default="data",
        help="Directory to save the downloaded data (default: data/)",
    )
    parser.add_argument(
        "--no-extract",
        action="store_true",
        help="Keep package.zip without extracting",
    )
    parser.add_argument(
        "--no-verify",
        action="store_true",
        help="Skip MD5 checksum verification",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    zip_path = output_dir / FILENAME

    if zip_path.exists():
        print(f"Found existing {zip_path}, skipping download.")
    else:
        print(f"Downloading {FILENAME} ({FILE_SIZE / 1_048_576:.1f} MB) ...")
        try:
            download(ZENODO_URL, zip_path)
        except Exception as e:
            print(f"Download failed: {e}", file=sys.stderr)
            sys.exit(1)
        print(f"Saved to {zip_path}")

    if not args.no_verify:
        print("Verifying MD5 checksum ...", end=" ", flush=True)
        if verify_md5(zip_path, MD5_CHECKSUM):
            print("OK")
        else:
            print("FAILED", file=sys.stderr)
            print(
                "Checksum mismatch. The file may be corrupted. "
                "Delete it and re-run to try again.",
                file=sys.stderr,
            )
            sys.exit(1)

    if not args.no_extract:
        extract_dir = output_dir / "sesr"
        prefix = "data/3-processed-data/"
        print(f"Extracting {prefix} contents to {extract_dir}/ ...")
        extract_dir.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(zip_path, "r") as zf:
            members = [m for m in zf.infolist() if m.filename.startswith(prefix) and m.filename != prefix]
            for member in members:
                parts = Path(member.filename[len(prefix):]).parts
                # strip the intermediate 'output/' folder: 17/output/file -> 17/file
                if len(parts) >= 2 and parts[1] == "output":
                    parts = (parts[0],) + parts[2:]
                if not parts:
                    continue
                dest = extract_dir / Path(*parts)
                if member.is_dir():
                    dest.mkdir(parents=True, exist_ok=True)
                else:
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    with zf.open(member) as src, open(dest, "wb") as dst:
                        dst.write(src.read())
        # Rename numbered folders using first 25 chars of Title from secondary_study_data.csv
        numbered = sorted(p for p in extract_dir.iterdir() if p.is_dir() and p.name.isdigit())
        for folder in numbered:
            secondary_csv = folder / "secondary_study_data.csv"
            if not secondary_csv.exists():
                continue
            with open(secondary_csv, newline="", encoding="utf-8") as f:
                row = next(csv.DictReader(f), None)
            row = row or {}
            title = row.get("Title", "").strip()[:25].strip()
            safe = re.sub(r'[\\/:*?"<>|]', "_", title).replace(" ", "_")
            if safe and safe != folder.name:
                folder = folder.rename(extract_dir / safe)
            criteria_text = row.get("Inclusion / Exclusion criteria", "").strip()
            instructions_text = row.get("Prompt instruction", "").strip()
            if criteria_text:
                (folder / "criteria.conf").write_text(criteria_text, encoding="utf-8")
            if instructions_text:
                (folder / "instructions.txt").write_text(instructions_text, encoding="utf-8")
            for unwanted in ("scripts", "backup"):
                p = folder / unwanted
                if p.is_dir():
                    shutil.rmtree(p)

        top_level = sorted(p.name for p in extract_dir.iterdir())
        print("Extracted:")
        for entry in top_level:
            print(f"  {extract_dir}/{entry}")

    print("Done.")


if __name__ == "__main__":
    main()
