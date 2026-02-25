"""
Generate large synthetic bibliographic fixture files for performance testing.

Creates:
  tests/bibs/wos_large.bib    — 1000 WOS-format BibTeX entries
  tests/bibs/pubmed_large.txt — 1000 PubMed MEDLINE entries

Entries are derived from the small real files (wos_m.bib, mcda_pubmed.txt)
by repeating them in round-robin order with unique cite keys / PMIDs.

Usage:
    python tests/bibs/generate_large_fixtures.py
"""

import re
import sys
from pathlib import Path

BIBS_DIR = Path(__file__).parent
TARGET = 1000


# ---------------------------------------------------------------------------
# BibTeX large fixture
# ---------------------------------------------------------------------------

def _split_bibtex_entries(text: str) -> list[str]:
    """Split a .bib file into individual raw entry strings."""
    # Split at each @article / @ARTICLE / @CONFERENCE etc. header
    parts = re.split(r'(?=^@\w)', text, flags=re.MULTILINE)
    entries = [p.strip() for p in parts if p.strip().startswith("@")]
    return entries


def _replace_bibtex_key(entry: str, new_key: str) -> str:
    """Replace the cite key in a BibTeX entry string."""
    return re.sub(
        r'^(@\w+\s*\{)\s*[^,]+\s*,',
        lambda m: m.group(1) + new_key + ",",
        entry,
        count=1,
        flags=re.MULTILINE,
    )


def generate_wos_large(source_path: Path, output_path: Path, n: int = TARGET) -> None:
    text = source_path.read_text(encoding="utf-8", errors="replace")
    entries = _split_bibtex_entries(text)
    if not entries:
        print(f"ERROR: no entries found in {source_path}", file=sys.stderr)
        sys.exit(1)

    lines = []
    for i in range(n):
        template = entries[i % len(entries)]
        new_key = f"WOS:LARGE{i:05d}"
        entry = _replace_bibtex_key(template, new_key)
        lines.append(entry)
        lines.append("")  # blank line between entries

    output_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {n} entries to {output_path}")


# ---------------------------------------------------------------------------
# PubMed large fixture
# ---------------------------------------------------------------------------

def _split_pubmed_entries(text: str) -> list[str]:
    """Split a PubMed MEDLINE file into individual raw entry strings."""
    entries = []
    current: list[str] = []
    for line in text.splitlines():
        if line == "" and current:
            entries.append("\n".join(current))
            current = []
        else:
            current.append(line)
    if current:
        entries.append("\n".join(current))
    return [e for e in entries if e.strip()]


def _replace_pubmed_pmid(entry: str, new_pmid: str) -> str:
    """Replace the PMID line in a PubMed entry string."""
    lines = entry.splitlines()
    result = []
    replaced = False
    for line in lines:
        if not replaced and re.match(r'^PMID-', line):
            result.append(f"PMID- {new_pmid}")
            replaced = True
        else:
            result.append(line)
    if not replaced:
        result.insert(0, f"PMID- {new_pmid}")
    return "\n".join(result)


def generate_pubmed_large(source_path: Path, output_path: Path, n: int = TARGET) -> None:
    text = source_path.read_text(encoding="utf-8", errors="replace")
    entries = _split_pubmed_entries(text)
    if not entries:
        print(f"ERROR: no entries found in {source_path}", file=sys.stderr)
        sys.exit(1)

    blocks = []
    for i in range(n):
        template = entries[i % len(entries)]
        new_pmid = f"{90000001 + i}"
        entry = _replace_pubmed_pmid(template, new_pmid)
        blocks.append(entry)

    output_path.write_text("\n\n".join(blocks) + "\n", encoding="utf-8")
    print(f"Wrote {n} entries to {output_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    generate_wos_large(
        source_path=BIBS_DIR / "wos_m.bib",
        output_path=BIBS_DIR / "wos_large.bib",
    )
    generate_pubmed_large(
        source_path=BIBS_DIR / "mcda_pubmed.txt",
        output_path=BIBS_DIR / "pubmed_large.txt",
    )
