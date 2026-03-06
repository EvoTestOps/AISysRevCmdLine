"""
Download the CSR benchmark dataset from Mendeley.

The archive contains two datasets:
  20240827_Systematic_Review_Data          -> data/csr-large/
  20240827_Systematic_review_manual_search_data -> data/csr-22rerun/

Usage:
    python download_csr.py
    python download_csr.py -o data/
    python download_csr.py --no-extract
    python download_csr.py --refresh
"""

import argparse
import csv
import io
import re
import sys
import tempfile
import zipfile
from pathlib import Path

import requests
from tqdm import tqdm

MENDELEY_URL = "https://data.mendeley.com/public-api/zip/7sgmg89zb6/download/1"
FILENAME = "csr.zip"

# Top-level folder inside the outer zip
OUTER_PREFIX = "Title and Abstract Screening Data for Systematic Review/"

# Maps inner zip filename -> output folder name
INNER_ZIP_MAP = {
    "20240827_Systematic_Review_Data.zip": "csr-large",
    "20240827_Systematic_review_manual_search_data.zip": "csr-22rerun",
}

# Tags expected in Review_data and Title_Abstract fields
REVIEW_TAGS = ("RTI", "BG", "OBJ", "SEL")

# Maps csr-large CSV filenames to short split names (order matters for 'splits' column)
CSR_LARGE_SPLITS = {
    "20240827_dev_set.csv": "dev",
    "20240827_val_set.csv": "val",
    "20240827_random_test_set.csv": "random_test",
    "20240827_heart_test_set.csv": "heart_test",
    "20240827_HIV_test_set.csv": "hiv_test",
}


def _extract_tag(text: str, tag: str) -> str:
    """Return the text between [TAG] and the next tag (or end of string)."""
    m = re.search(rf"\[{tag}\]\s*(.*?)(?=\s*\[|$)", text, re.DOTALL)
    return m.group(1).strip() if m else ""


def download(url: str, dest: Path) -> None:
    with requests.get(url, stream=True, timeout=120) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0)) or None
        with tqdm(total=total, unit="B", unit_scale=True, desc=FILENAME) as pbar:
            with open(dest, "wb") as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
                    pbar.update(len(chunk))


def extract_inner_zip(inner_bytes: bytes, dest_dir: Path) -> int:
    """Extract an inner zip (given as bytes) into dest_dir, stripping any common top-level folder."""
    files_written = 0
    with zipfile.ZipFile(io.BytesIO(inner_bytes)) as inner_zf:
        names = inner_zf.namelist()
        # Determine if all entries share a single top-level folder to strip
        top_levels = {Path(n).parts[0] for n in names if Path(n).parts}
        strip_prefix = top_levels.pop() + "/" if len(top_levels) == 1 else ""

        for member in inner_zf.infolist():
            name = member.filename
            if strip_prefix and name.startswith(strip_prefix):
                name = name[len(strip_prefix):]
            if not name:
                continue
            dest = dest_dir / name
            if member.is_dir():
                dest.mkdir(parents=True, exist_ok=True)
            else:
                dest.parent.mkdir(parents=True, exist_ok=True)
                with inner_zf.open(member) as src, open(dest, "wb") as dst:
                    dst.write(src.read())
                files_written += 1
    return files_written


def extract(zip_path: Path, output_dir: Path) -> int:
    """Extract outer zip, finding and extracting inner zips. Returns number of files written."""
    files_written = 0
    matched: set[str] = set()
    found_inner_names: set[str] = set()

    with zipfile.ZipFile(zip_path, "r") as zf:
        for member in zf.infolist():
            name = member.filename
            inner_name = name[len(OUTER_PREFIX):] if name.startswith(OUTER_PREFIX) else name
            if not inner_name or member.is_dir():
                continue
            found_inner_names.add(inner_name)
            dest_folder = INNER_ZIP_MAP.get(inner_name)
            if dest_folder is None:
                continue
            matched.add(inner_name)
            print(f"  Extracting {inner_name} -> {output_dir / dest_folder}/ ...")
            inner_bytes = zf.read(member)
            dest_dir = output_dir / dest_folder
            dest_dir.mkdir(parents=True, exist_ok=True)
            files_written += extract_inner_zip(inner_bytes, dest_dir)

    missing = set(INNER_ZIP_MAP.keys()) - matched
    if missing:
        print(
            f"WARNING: Expected inner zips not found: {sorted(missing)}\n"
            f"  Inner files seen in outer zip: {sorted(found_inner_names)}\n"
            f"  Run --list to inspect zip contents.",
            file=sys.stderr,
        )
    unmatched = found_inner_names - set(INNER_ZIP_MAP.keys())
    if unmatched:
        print(f"NOTE: Ignored unrecognized inner files: {sorted(unmatched)}", file=sys.stderr)
    return files_written


def _build_primary_correct(original: Path, primary: Path) -> tuple[int, int, int]:
    """
    Write primary_correct.csv from original.csv:
      - Drop Review_data column
      - Split Title_Abstract ([TIT]/[ABS]) into separate title and abstract columns

    Returns (rows_written, missing_title, missing_abstract).
    """
    missing_title = 0
    missing_abstract = 0
    rows_written = 0

    with open(original, newline="", encoding="utf-8") as fin, \
         open(primary, "w", newline="", encoding="utf-8") as fout:
        reader = csv.DictReader(fin)
        src_fields = reader.fieldnames or []

        if "Title_Abstract" not in src_fields:
            print(f"  WARNING: 'Title_Abstract' column not found in {original.name}; "
                  f"columns present: {src_fields}", file=sys.stderr)

        out_fields = []
        for f in src_fields:
            if f == "Title_Abstract":
                out_fields += ["title", "abstract"]
            elif f != "Review_data":
                out_fields.append(f)

        writer = csv.DictWriter(fout, fieldnames=out_fields, extrasaction="ignore")
        writer.writeheader()

        for i, src_row in enumerate(reader, start=1):
            ta = src_row.get("Title_Abstract", "")
            m_tit = re.search(r"\[TIT\]\s*(.*?)(?=\s*\[ABS\]|$)", ta, re.DOTALL)
            m_abs = re.search(r"\[ABS\]\s*(.*?)$", ta, re.DOTALL)
            title = m_tit.group(1).strip() if m_tit else ""
            abstract = m_abs.group(1).strip() if m_abs else ""
            if not title:
                missing_title += 1
            if not abstract:
                missing_abstract += 1
            out_row = {k: v for k, v in src_row.items() if k not in ("Title_Abstract", "Review_data")}
            out_row["title"] = title
            out_row["abstract"] = abstract
            writer.writerow(out_row)
            rows_written += 1

    return rows_written, missing_title, missing_abstract


def organize_csr_large(base_dir: Path) -> None:
    """Build secondary_studies.csv from the csr-large split CSVs.

    Reads all split files, deduplicates by Review_URL, and writes one row per
    secondary study with metadata and a comma-separated list of which splits it
    appears in.
    """
    # study_url -> {"review_title": ..., "background": ..., "objective": ...,
    #               "selection_criteria": ..., "splits": [split_name, ...]}
    studies: dict[str, dict] = {}

    for filename, split_name in CSR_LARGE_SPLITS.items():
        csv_path = base_dir / filename
        if not csv_path.exists():
            continue
        print(f"  Reading {filename} ...")
        with open(csv_path, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                url = row.get("Review_URL", "")
                if not url:
                    continue
                if url not in studies:
                    studies[url] = {
                        "review_title": row.get("Review_Title", ""),
                        "background": row.get("Background", ""),
                        "objective": row.get("Objective", ""),
                        "selection_criteria": row.get("Selection_criteria", ""),
                        "splits": [],
                        "n_papers": 0,
                        "n_included": 0,
                        "n_fulltext_excluded": 0,
                        "n_tiab_excluded": 0,
                    }
                if split_name not in studies[url]["splits"]:
                    studies[url]["splits"].append(split_name)
                studies[url]["n_papers"] += 1
                try:
                    label = float(row.get("label", ""))
                except (ValueError, TypeError):
                    label = None
                if label == 1.0:
                    studies[url]["n_included"] += 1
                elif label == 0.5:
                    studies[url]["n_fulltext_excluded"] += 1
                elif label == 0.0:
                    studies[url]["n_tiab_excluded"] += 1

    if not studies:
        print(f"  No csr-large split files found in {base_dir}/ — skipping.", file=sys.stderr)
        return

    out_path = base_dir / "secondary_studies.csv"
    fields = ["review_url", "review_title", "background", "objective", "selection_criteria", "splits",
              "n_papers", "n_included", "n_fulltext_excluded", "n_tiab_excluded"]
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for url in sorted(studies):
            entry = studies[url]
            writer.writerow({
                "review_url": url,
                "review_title": entry["review_title"],
                "background": entry["background"],
                "objective": entry["objective"],
                "selection_criteria": entry["selection_criteria"],
                "splits": ",".join(entry["splits"]),
                "n_papers": entry["n_papers"],
                "n_included": entry["n_included"],
                "n_fulltext_excluded": entry["n_fulltext_excluded"],
                "n_tiab_excluded": entry["n_tiab_excluded"],
            })

    print(f"  Written {len(studies)} secondary studies -> {out_path}")


def organize_22rerun(base_dir: Path) -> None:
    """Move each flat CSV in csr-22rerun/ into a subfolder named from its RTI field."""
    csv_files = sorted(base_dir.glob("*.csv"))
    if not csv_files:
        print(f"  No flat CSV files found in {base_dir}/ — already organized or empty.")
        return

    print(f"Organizing {len(csv_files)} CSV files in {base_dir}/ ...")
    warnings = 0

    for csv_path in csv_files:
        # Read first row to extract review-level metadata from Review_data
        try:
            with open(csv_path, newline="", encoding="utf-8") as f:
                row = next(csv.DictReader(f), None)
        except Exception as e:
            print(f"  ERROR: Could not read {csv_path.name}: {e}", file=sys.stderr)
            warnings += 1
            continue

        if row is None:
            print(f"  WARNING: {csv_path.name} is empty (no data rows), skipping.", file=sys.stderr)
            warnings += 1
            continue

        if "Review_data" not in row:
            print(f"  WARNING: {csv_path.name} has no 'Review_data' column; "
                  f"columns: {list(row.keys())}", file=sys.stderr)
            warnings += 1

        review_data = row.get("Review_data", "")

        rti = _extract_tag(review_data, "RTI")
        if not rti:
            print(f"  WARNING: {csv_path.name}: [RTI] tag not found in Review_data; "
                  f"falling back to filename.", file=sys.stderr)
            warnings += 1
            rti = csv_path.stem

        sel = _extract_tag(review_data, "SEL")
        bg  = _extract_tag(review_data, "BG")
        obj = _extract_tag(review_data, "OBJ")

        for tag, label in (("SEL", "criteria.conf"), ("BG", "background.txt"), ("OBJ", "objectives.txt")):
            if not _extract_tag(review_data, tag):
                print(f"  WARNING: {csv_path.name}: [{tag}] tag not found in Review_data "
                      f"— {label} will not be written.", file=sys.stderr)
                warnings += 1

        safe = re.sub(r'[\\/:*?"<>| ]', "_", rti[:25]).strip("_")
        if not safe:
            safe = csv_path.stem

        folder = base_dir / safe
        folder.mkdir(parents=True, exist_ok=True)

        try:
            csv_path.rename(folder / "original.csv")
        except Exception as e:
            print(f"  ERROR: Could not move {csv_path.name} -> {safe}/original.csv: {e}", file=sys.stderr)
            warnings += 1
            continue

        if sel:
            (folder / "criteria.conf").write_text(sel, encoding="utf-8")
        if bg:
            (folder / "background.txt").write_text(bg, encoding="utf-8")
        if obj:
            (folder / "objectives.txt").write_text(obj, encoding="utf-8")

        original = folder / "original.csv"
        primary  = folder / "primary_correct.csv"
        try:
            rows, missing_t, missing_a = _build_primary_correct(original, primary)
        except Exception as e:
            print(f"  ERROR: Could not build primary_correct.csv for {safe}: {e}", file=sys.stderr)
            warnings += 1
            continue

        issues = []
        if missing_t:
            issues.append(f"{missing_t}/{rows} rows missing title")
        if missing_a:
            issues.append(f"{missing_a}/{rows} rows missing abstract")
        status = f"  WARNING: {'; '.join(issues)}" if issues else ""

        print(f"  {csv_path.name} -> {safe}/ ({rows} rows){status}")
        if issues:
            warnings += 1

    if warnings:
        print(f"  {warnings} warning(s) during organization — review output above.", file=sys.stderr)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download the CSR benchmark dataset from Mendeley."
    )
    parser.add_argument(
        "-o", "--output-dir",
        default="data",
        help="Directory to save the extracted data (default: data/)",
    )
    parser.add_argument(
        "--no-extract",
        action="store_true",
        help="Download the zip but skip extraction",
    )
    parser.add_argument(
        "--refresh",
        action="store_true",
        help="Re-download and re-extract even if output folders exist",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="Download and list zip contents without extracting",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    dest_folders = [output_dir / name for name in INNER_ZIP_MAP.values()]

    if not args.refresh and not args.no_extract and not args.list and all(p.exists() for p in dest_folders):
        print("CSR datasets already exist:")
        for p in dest_folders:
            print(f"  {p}/")
        print("Use --refresh to re-download.")
        return

    with tempfile.NamedTemporaryFile(suffix=".zip", delete=False) as tmp:
        zip_path = Path(tmp.name)

    print(f"Downloading {FILENAME} from Mendeley ...")
    try:
        download(MENDELEY_URL, zip_path)
    except Exception as e:
        print(f"ERROR: Download failed: {e}", file=sys.stderr)
        zip_path.unlink(missing_ok=True)
        sys.exit(1)
    print("Download complete.")

    if args.list:
        print("Zip contents:")
        with zipfile.ZipFile(zip_path, "r") as zf:
            for name in sorted(zf.namelist()):
                print(f"  {name}")
        zip_path.unlink(missing_ok=True)
        return

    if not args.no_extract:
        print(f"Extracting to {output_dir}/ ...")
        try:
            files_written = extract(zip_path, output_dir)
        except Exception as e:
            print(f"ERROR: Extraction failed: {e}", file=sys.stderr)
            zip_path.unlink(missing_ok=True)
            sys.exit(1)
        if files_written == 0:
            print(
                "ERROR: No files were extracted. The zip structure may not match expectations.\n"
                "  Run with --list to inspect zip contents.",
                file=sys.stderr,
            )
            zip_path.unlink(missing_ok=True)
            sys.exit(1)
        print(f"Extracted {files_written} files:")
        for p in dest_folders:
            print(f"  {p}/")
        organize_22rerun(output_dir / "csr-22rerun")
        organize_csr_large(output_dir / "csr-large")

    zip_path.unlink(missing_ok=True)
    print("Done.")


if __name__ == "__main__":
    main()
