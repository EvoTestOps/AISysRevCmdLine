"""
Convert bibliographic files to CSV format compatible with the screening pipeline.

Supported input formats:
  - BibTeX (.bib): Web of Science and Scopus exports
  - MEDLINE/PubMed tagged format (.txt): PubMed exports

Output CSV columns: title, abstract, doi, year, authors, journal, keywords,
                    source_db, entry_key

Usage:
    python bib2csv.py <file1> [file2 ...] [-o output.csv]
"""

import argparse
import csv
import logging
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

from helpers import get_unique_filename

# ---------------------------------------------------------------------------
# Logging setup (same pattern as all other scripts in this project)
# ---------------------------------------------------------------------------
logging.getLogger().handlers.clear()
root = logging.getLogger()
root.setLevel(logging.INFO)
fmt = logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
file_handler = logging.FileHandler("app.log", mode="w", encoding="utf-8")
file_handler.setLevel(logging.INFO)
file_handler.setFormatter(fmt)
console_handler = logging.StreamHandler(sys.stderr)
console_handler.setLevel(logging.ERROR)
console_handler.setFormatter(fmt)
root.addHandler(file_handler)
root.addHandler(console_handler)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Column order for output CSV
# ---------------------------------------------------------------------------
OUTPUT_COLUMNS = [
    "title", "abstract", "doi", "year", "authors",
    "journal", "keywords", "source_db", "entry_key",
]


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _strip_braces(value: str) -> str:
    """Strip surrounding curly braces that BibTeX uses for case protection."""
    value = value.strip()
    while value.startswith("{") and value.endswith("}"):
        value = value[1:-1].strip()
    return value


def _normalize_ws(value: str) -> str:
    """Collapse internal newlines/tabs/multiple spaces into a single space."""
    return " ".join(value.split())


def _get_field(entry: dict, *candidates: str) -> str:
    """Case-insensitive field lookup; returns first match or empty string."""
    lower_entry = {k.lower(): v for k, v in entry.items()}
    for key in candidates:
        val = lower_entry.get(key.lower(), "")
        if val:
            return _normalize_ws(_strip_braces(str(val)))
    return ""


def _infer_source_db(entry: dict) -> str:
    """Infer database from BibTeX entry key prefix or source field."""
    key = entry.get("ID", "")
    if key.startswith("WOS:"):
        return "WOS"
    # IEEE Xplore exports use a purely numeric entry key (their internal document ID)
    if key.isdigit():
        return "IEEE"
    lower_entry = {k.lower(): v for k, v in entry.items()}
    source = lower_entry.get("source", "").lower()
    if "scopus" in source:
        return "Scopus"
    # Fallback: entries with an 'author' field that aren't Scopus are assumed WOS
    if "author" in lower_entry:
        return "WOS"
    return "BibTeX"


# ---------------------------------------------------------------------------
# Custom BibTeX parser (WOS + Scopus) — replaces bibtexparser
# ---------------------------------------------------------------------------

# Entry header: @article{ CITE_KEY, or @ARTICLE{CITE_KEY,
_ENTRY_HEAD_RE = re.compile(r'^@\w+\s*\{\s*(.+?)\s*,\s*$', re.IGNORECASE)
# Field start: FieldName = value...  (field names may contain hyphens)
_FIELD_RE = re.compile(r'^\s*([\w-]+)\s*=\s*(.*)')


def _join_value(parts: List[str]) -> str:
    """Join multi-line value parts, strip trailing comma."""
    val = " ".join(parts).strip()
    if val.endswith(","):
        val = val[:-1].strip()
    return val


def _parse_bibtex_fields(lines: List[str]) -> Dict[str, str]:
    """
    Parse field lines from a single BibTeX entry.

    Tracks brace depth to correctly handle multi-line values and nested braces.
    Field keys are normalised to lowercase.
    """
    fields: Dict[str, str] = {}
    current_field: Optional[str] = None
    value_parts: List[str] = []
    depth = 0

    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue

        field_match = _FIELD_RE.match(line)

        if field_match and depth <= 0:
            # Save the previous field (if any)
            if current_field is not None:
                fields[current_field] = _join_value(value_parts)
            current_field = field_match.group(1).lower()
            rest = field_match.group(2)
            value_parts = [rest]
            depth = rest.count("{") - rest.count("}")
            if depth <= 0:
                # Single-line value: save immediately
                fields[current_field] = _join_value(value_parts)
                current_field = None
                value_parts = []
                depth = 0
        elif current_field is not None:
            value_parts.append(stripped)
            depth += stripped.count("{") - stripped.count("}")
            if depth <= 0:
                fields[current_field] = _join_value(value_parts)
                current_field = None
                value_parts = []
                depth = 0
        # else: line before any field (e.g., closing '}' of entry) — ignore

    # Flush any field still open at end of entry lines
    if current_field is not None:
        fields[current_field] = _join_value(value_parts)

    return fields


def _build_bibtex_record(cite_key: str, lines: List[str]) -> Dict:
    """Build a CSV record dict from a parsed BibTeX entry."""
    fields = _parse_bibtex_fields(lines)
    # Synthesise a compatible entry dict for the shared helpers
    entry = {"ID": cite_key}
    entry.update(fields)
    source_db = _infer_source_db(entry)
    return {
        "title":     _get_field(entry, "title"),
        "abstract":  _get_field(entry, "abstract"),
        "doi":       _get_field(entry, "doi"),
        "year":      _get_field(entry, "year"),
        "authors":   _get_field(entry, "author"),
        "journal":   _get_field(entry, "journal"),
        "keywords":  _get_field(entry, "author_keywords", "keywords"),
        "source_db": source_db,
        "entry_key": cite_key,
    }


_ENTRY_SPLICE_RE = re.compile(r'(?<=\})\s*(?=@[A-Za-z])')


def _normalize_bib_lines(raw_lines: List[str], file_path: str) -> List[str]:
    """Split lines where closing brace of one entry runs into start of the next.

    Scopus can export ``}@ENTRYTYPE{key,`` without a newline between entries.
    Such lines are silently split here and a warning is printed.
    """
    result: List[str] = []
    splice_count = 0
    for raw in raw_lines:
        line = raw.rstrip("\n")
        parts = _ENTRY_SPLICE_RE.split(line)
        if len(parts) > 1:
            splice_count += 1
        result.extend(parts)
    if splice_count:
        msg = (
            f"WARN: {file_path}: {splice_count} entry/entries had no newline "
            "before '@' (missing blank line in export); auto-corrected."
        )
        print(msg)
        logger.warning(msg)
    return result


def parse_bibtex(file_path: str) -> List[Dict]:
    """Parse a .bib file and return a list of record dicts."""
    logger.info("Parsing BibTeX file: %s", file_path)
    with open(file_path, encoding="utf-8", errors="replace") as fh:
        raw_lines = fh.readlines()
    lines = _normalize_bib_lines(raw_lines, file_path)

    records = []
    cite_key: Optional[str] = None
    entry_lines: List[str] = []

    for raw in lines:
        line = raw.rstrip("\n")
        head_match = _ENTRY_HEAD_RE.match(line)
        if head_match:
            # New entry starts — flush the previous entry
            if cite_key is not None:
                record = _build_bibtex_record(cite_key, entry_lines)
                records.append(record)
            cite_key = head_match.group(1)
            entry_lines = []
        elif cite_key is not None:
            entry_lines.append(line)

    # Flush the last entry
    if cite_key is not None:
        record = _build_bibtex_record(cite_key, entry_lines)
        records.append(record)

    logger.info("Parsed %d entries from %s", len(records), file_path)
    return records


# ---------------------------------------------------------------------------
# MEDLINE/PubMed tagged format parser
# ---------------------------------------------------------------------------

# Regex matching a tag line: 2-4 uppercase alphanum chars, optional spaces,
# dash, space, then the value.  e.g. "PMID- 12345", "TI  - Some title"
_TAG_RE = re.compile(r'^([A-Z][A-Z0-9]{1,3})\s*-\s(.+)$')
# Continuation lines start with 6 spaces
_CONT_RE = re.compile(r'^      (.+)$')


def _parse_pubmed_entry(lines: List[str]) -> Optional[Dict]:
    """Parse a single PubMed entry from its lines into a record dict."""
    fields: Dict[str, List[str]] = {}
    current_tag: Optional[str] = None

    for line in lines:
        cont_match = _CONT_RE.match(line)
        tag_match = _TAG_RE.match(line)
        if cont_match and current_tag:
            # Continuation of the previous field
            fields[current_tag].append(cont_match.group(1))
        elif tag_match:
            current_tag = tag_match.group(1)
            value = tag_match.group(2).strip()
            if current_tag not in fields:
                fields[current_tag] = []
            fields[current_tag].append(value)
        else:
            current_tag = None  # blank/unrecognised line

    if not fields:
        return None

    # --- title: join continuation lines, strip trailing period ---
    title_parts = fields.get("TI", [])
    title = " ".join(title_parts).strip()
    if title.endswith("."):
        title = title[:-1]

    # --- abstract ---
    abstract = " ".join(fields.get("AB", [])).strip()

    # --- doi: LID lines that end with [doi] ---
    doi = ""
    for lid in fields.get("LID", []):
        if lid.endswith("[doi]"):
            doi = lid[: -len("[doi]")].strip()
            break

    # --- year: first 4-digit sequence in DP ---
    year = ""
    dp_values = fields.get("DP", [])
    if dp_values:
        year_match = re.search(r'\d{4}', dp_values[0])
        if year_match:
            year = year_match.group(0)

    # --- authors: all FAU lines ---
    authors = "; ".join(fields.get("FAU", []))

    # --- journal: JT (full journal title) ---
    journal = fields.get("JT", [""])[0].strip() if fields.get("JT") else ""

    # --- keywords: all OT lines ---
    keywords = "; ".join(fields.get("OT", []))

    # --- PMID ---
    pmid = fields.get("PMID", [""])[0].strip() if fields.get("PMID") else ""

    return {
        "title": title,
        "abstract": abstract,
        "doi": doi,
        "year": year,
        "authors": authors,
        "journal": journal,
        "keywords": keywords,
        "source_db": "PubMed",
        "entry_key": pmid,
    }


def parse_pubmed(file_path: str) -> List[Dict]:
    """Parse a PubMed MEDLINE tagged .txt file; returns list of record dicts."""
    logger.info("Parsing PubMed file: %s", file_path)

    with open(file_path, encoding="utf-8", errors="replace") as fh:
        lines = fh.readlines()

    # Split into per-entry line groups on blank lines
    records = []
    current_lines: List[str] = []
    for raw in lines:
        stripped = raw.rstrip("\n")
        if stripped == "":
            if current_lines:
                record = _parse_pubmed_entry(current_lines)
                if record:
                    records.append(record)
                current_lines = []
        else:
            current_lines.append(stripped)
    # Handle last entry (file may not end with blank line)
    if current_lines:
        record = _parse_pubmed_entry(current_lines)
        if record:
            records.append(record)

    logger.info("Parsed %d entries from %s", len(records), file_path)
    return records


# ---------------------------------------------------------------------------
# Format detection
# ---------------------------------------------------------------------------

def _is_pubmed(file_path: str) -> bool:
    """Return True if the file looks like MEDLINE/PubMed tagged format."""
    with open(file_path, encoding="utf-8", errors="replace") as fh:
        for line in fh:
            line = line.strip()
            if line:
                return line.startswith("PMID")
    return False


def parse_file(file_path: str) -> List[Dict]:
    """Auto-detect format and parse the file."""
    path = Path(file_path)
    if path.suffix.lower() == ".bib":
        return parse_bibtex(file_path)
    elif path.suffix.lower() == ".txt":
        if _is_pubmed(file_path):
            return parse_pubmed(file_path)
        else:
            logger.error("Unrecognised .txt format in %s (expected PubMed MEDLINE)", file_path)
            print(f"ERROR: {file_path}: unrecognised .txt format (expected PubMed MEDLINE).")
            return []
    else:
        logger.debug("Skipping %s: unsupported extension '%s'", file_path, path.suffix)
        return []


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description=(
            "Convert bibliographic files (.bib or PubMed .txt) to a CSV file "
            "suitable for screening with screen.py."
        )
    )
    parser.add_argument(
        "files",
        nargs="+",
        metavar="FILE",
        help="One or more .bib or .txt bibliographic files.",
    )
    parser.add_argument(
        "-o", "--output",
        default=None,
        metavar="OUTPUT_CSV",
        help="Output CSV path (default: <first_input_stem>.csv).",
    )
    args = parser.parse_args()

    # Determine output path
    if args.output:
        output_path = args.output
    else:
        stem = Path(args.files[0]).stem
        output_path = f"{stem}.csv"
    output_path = get_unique_filename(output_path)

    # Parse all input files and merge
    all_records: List[Dict] = []
    for file_path in args.files:
        if not Path(file_path).exists():
            print(f"ERROR: File not found: {file_path}")
            logger.error("File not found: %s", file_path)
            continue
        records = parse_file(file_path)
        if not records and Path(file_path).suffix.lower() not in (".bib", ".txt"):
            continue  # silently skip unrecognised extensions
        all_records.extend(records)
        print(f"  {file_path}: {len(records)} entries")

    if not all_records:
        print("No records parsed. Nothing to write.")
        sys.exit(1)

    # Build DataFrame, ensure all columns present
    df = pd.DataFrame(all_records, columns=OUTPUT_COLUMNS)
    df = df.fillna("")

    # Warn about missing titles/abstracts
    empty_title = (df["title"].str.strip() == "").sum()
    empty_abstract = (df["abstract"].str.strip() == "").sum()
    if empty_title:
        print(f"WARN: {empty_title} entries have no title.")
    if empty_abstract:
        print(f"WARN: {empty_abstract} entries have no abstract.")

    df.to_csv(output_path, index=False, quoting=csv.QUOTE_ALL, encoding="utf-8-sig")
    print(f"\nWrote {len(df)} records to {output_path}")


if __name__ == "__main__":
    main()
