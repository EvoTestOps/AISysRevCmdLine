"""
Tests for bib2csv.py — bibliographic file parser.

Covers:
  - All real bib files in tests/bibs/ (entry counts, columns, source_db)
  - Fault-injected files (specific fault assertions per entry)
  - Unit tests for internal helper functions
  - Format auto-detection via parse_file()
"""

import pytest
from pathlib import Path

from bib2csv import (
    OUTPUT_COLUMNS,
    _infer_source_db,
    _is_pubmed,
    _normalize_ws,
    _strip_braces,
    parse_file,
)

BIBS_DIR = Path(__file__).parent / "bibs"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _by_key(records: list, key: str) -> dict | None:
    """Return the first record whose entry_key matches key, or None."""
    return next((r for r in records if r["entry_key"] == key), None)


def _records_with_key(records: list, key: str) -> list:
    """Return all records whose entry_key matches key."""
    return [r for r in records if r["entry_key"] == key]


# ---------------------------------------------------------------------------
# Real files: entry counts
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("filename,expected_count", [
    ("wos.bib",        236),
    ("wos_m.bib",       29),
    ("mcda_wos.bib",   210),
    ("scopus.bib",     263),
    ("scopus_m.bib",    24),
    ("mcda_scopus.bib",236),
    ("pubmed.txt",     335),
    ("mcda_pubmed.txt", 10),
])
def test_real_file_entry_count(filename, expected_count):
    records = parse_file(str(BIBS_DIR / filename))
    assert len(records) == expected_count, (
        f"{filename}: expected {expected_count} entries, got {len(records)}"
    )


# ---------------------------------------------------------------------------
# Real files: output structure
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("filename", [
    "wos.bib", "wos_m.bib", "mcda_wos.bib",
    "scopus.bib", "scopus_m.bib", "mcda_scopus.bib",
    "pubmed.txt", "mcda_pubmed.txt",
])
def test_real_file_has_correct_columns(filename):
    records = parse_file(str(BIBS_DIR / filename))
    assert records, f"{filename}: no records returned"
    for col in OUTPUT_COLUMNS:
        assert col in records[0], f"{filename}: missing column '{col}'"


@pytest.mark.parametrize("filename", [
    "wos.bib", "wos_m.bib", "mcda_wos.bib",
    "scopus.bib", "scopus_m.bib", "mcda_scopus.bib",
    "pubmed.txt", "mcda_pubmed.txt",
])
def test_real_file_no_none_values(filename):
    records = parse_file(str(BIBS_DIR / filename))
    for i, rec in enumerate(records):
        for col in OUTPUT_COLUMNS:
            assert rec[col] is not None, (
                f"{filename} entry {i}: column '{col}' is None"
            )


@pytest.mark.parametrize("filename,expected_source", [
    ("wos.bib",        "WOS"),
    ("wos_m.bib",      "WOS"),
    ("mcda_wos.bib",   "WOS"),
    ("scopus.bib",     "Scopus"),
    ("scopus_m.bib",   "Scopus"),
    ("mcda_scopus.bib","Scopus"),
    ("pubmed.txt",     "PubMed"),
    ("mcda_pubmed.txt","PubMed"),
])
def test_real_file_source_db(filename, expected_source):
    records = parse_file(str(BIBS_DIR / filename))
    sources = {r["source_db"] for r in records}
    assert sources == {expected_source}, (
        f"{filename}: expected source_db={expected_source!r}, got {sources}"
    )


@pytest.mark.parametrize("filename,min_fraction", [
    ("wos.bib",         0.95),
    ("scopus.bib",      0.95),
    ("pubmed.txt",      0.95),
    ("mcda_pubmed.txt", 0.90),
])
def test_real_file_mostly_has_title_and_abstract(filename, min_fraction):
    records = parse_file(str(BIBS_DIR / filename))
    n = len(records)
    with_title    = sum(1 for r in records if r["title"].strip())
    with_abstract = sum(1 for r in records if r["abstract"].strip())
    assert with_title / n >= min_fraction, (
        f"{filename}: only {with_title}/{n} entries have a title"
    )
    assert with_abstract / n >= min_fraction, (
        f"{filename}: only {with_abstract}/{n} entries have an abstract"
    )


# ---------------------------------------------------------------------------
# wos_faults.bib — per-fault assertions
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def wos_fault_records():
    return parse_file(str(BIBS_DIR / "wos_faults.bib"))


def test_wos_faults_entry_count(wos_fault_records):
    assert len(wos_fault_records) == 11


def test_wos_fault_baseline(wos_fault_records):
    rec = _by_key(wos_fault_records, "WOS:FAULT_BASELINE")
    assert rec is not None
    assert rec["title"].strip() != ""
    assert rec["abstract"].strip() != ""
    assert rec["source_db"] == "WOS"


def test_wos_fault_f1_missing_abstract(wos_fault_records):
    """F1: Abstract field absent → empty string."""
    rec = _by_key(wos_fault_records, "WOS:FAULT_F1")
    assert rec is not None
    assert rec["abstract"] == ""


def test_wos_fault_f2_empty_abstract(wos_fault_records):
    """F2: Abstract = {} → empty string."""
    rec = _by_key(wos_fault_records, "WOS:FAULT_F2")
    assert rec is not None
    assert rec["abstract"] == ""


def test_wos_fault_f3_missing_title(wos_fault_records):
    """F3: Title field absent → empty string."""
    rec = _by_key(wos_fault_records, "WOS:FAULT_F3")
    assert rec is not None
    assert rec["title"] == ""


def test_wos_fault_f4_empty_title(wos_fault_records):
    """F4: Title = {} → empty string."""
    rec = _by_key(wos_fault_records, "WOS:FAULT_F4")
    assert rec is not None
    assert rec["title"] == ""


def test_wos_fault_f5_html_entities_preserved(wos_fault_records):
    """F5: HTML entities in abstract are passed through unchanged."""
    rec = _by_key(wos_fault_records, "WOS:FAULT_F5")
    assert rec is not None
    assert "&amp;" in rec["abstract"]


def test_wos_fault_f6_latex_preserved(wos_fault_records):
    """F6: LaTeX markup in abstract is passed through unchanged."""
    rec = _by_key(wos_fault_records, "WOS:FAULT_F6")
    assert rec is not None
    # At least one of the LaTeX markers must be present
    assert r"\textit" in rec["abstract"] or "$" in rec["abstract"]


def test_wos_fault_f7_doi_as_url(wos_fault_records):
    """F7: DOI stored as full https:// URL is preserved as-is."""
    rec = _by_key(wos_fault_records, "WOS:FAULT_F7")
    assert rec is not None
    assert rec["doi"].startswith("https://")


def test_wos_fault_f8_short_abstract(wos_fault_records):
    """F8: Very short abstract is not dropped."""
    rec = _by_key(wos_fault_records, "WOS:FAULT_F8")
    assert rec is not None
    assert rec["abstract"].strip() != ""
    assert len(rec["abstract"].split()) <= 5


def test_wos_fault_f10_duplicate_key(wos_fault_records):
    """F10: Two entries sharing the same cite key are both parsed."""
    dupes = _records_with_key(wos_fault_records, "WOS:FAULT_F10a")
    assert len(dupes) == 2, (
        f"Expected 2 entries with key WOS:FAULT_F10a, got {len(dupes)}"
    )
    titles = {r["title"] for r in dupes}
    assert len(titles) == 2, "Duplicate-key entries should have distinct titles"


# ---------------------------------------------------------------------------
# scopus_faults.bib — per-fault assertions
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def scopus_fault_records():
    return parse_file(str(BIBS_DIR / "scopus_faults.bib"))


def test_scopus_faults_entry_count(scopus_fault_records):
    assert len(scopus_fault_records) == 9


def test_scopus_fault_baseline(scopus_fault_records):
    rec = _by_key(scopus_fault_records, "ScopusFault_Baseline")
    assert rec is not None
    assert rec["title"].strip() != ""
    assert rec["abstract"].strip() != ""
    assert rec["source_db"] == "Scopus"


def test_scopus_fault_f1_missing_abstract(scopus_fault_records):
    rec = _by_key(scopus_fault_records, "ScopusFault_F1")
    assert rec is not None
    assert rec["abstract"] == ""


def test_scopus_fault_f2_empty_abstract(scopus_fault_records):
    rec = _by_key(scopus_fault_records, "ScopusFault_F2")
    assert rec is not None
    assert rec["abstract"] == ""


def test_scopus_fault_f3_missing_title(scopus_fault_records):
    rec = _by_key(scopus_fault_records, "ScopusFault_F3")
    assert rec is not None
    assert rec["title"] == ""


def test_scopus_fault_f4_empty_title(scopus_fault_records):
    rec = _by_key(scopus_fault_records, "ScopusFault_F4")
    assert rec is not None
    assert rec["title"] == ""


def test_scopus_fault_f5_html_entities_preserved(scopus_fault_records):
    rec = _by_key(scopus_fault_records, "ScopusFault_F5")
    assert rec is not None
    assert "&amp;" in rec["title"] or "&amp;" in rec["abstract"]


def test_scopus_fault_f7_doi_as_url(scopus_fault_records):
    rec = _by_key(scopus_fault_records, "ScopusFault_F7")
    assert rec is not None
    assert rec["doi"].startswith("https://")


def test_scopus_fault_f9_csv_special_chars(scopus_fault_records):
    """F9: Abstract with commas, quotes, and semicolons is preserved intact."""
    rec = _by_key(scopus_fault_records, "ScopusFault_F9")
    assert rec is not None
    # Must contain CSV-special characters
    assert "," in rec["abstract"]
    assert '"' in rec["abstract"]
    assert ";" in rec["abstract"]


def test_scopus_fault_f15_two_digit_year(scopus_fault_records):
    """F15: Two-digit year is passed through as-is."""
    rec = _by_key(scopus_fault_records, "ScopusFault_F15")
    assert rec is not None
    assert rec["year"] == "22"


# ---------------------------------------------------------------------------
# pubmed_faults.txt — per-fault assertions
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def pubmed_fault_records():
    return parse_file(str(BIBS_DIR / "pubmed_faults.txt"))


def test_pubmed_faults_entry_count(pubmed_fault_records):
    assert len(pubmed_fault_records) == 10


def test_pubmed_fault_baseline(pubmed_fault_records):
    rec = _by_key(pubmed_fault_records, "99000001")
    assert rec is not None
    assert rec["title"].strip() != ""
    assert rec["abstract"].strip() != ""
    assert rec["source_db"] == "PubMed"
    assert rec["doi"] == "10.1371/journal.pone.0268950"
    assert rec["year"] == "2022"


def test_pubmed_fault_f1_missing_abstract(pubmed_fault_records):
    """F1: No AB tag → empty abstract."""
    rec = _by_key(pubmed_fault_records, "99000002")
    assert rec is not None
    assert rec["abstract"] == ""
    assert rec["title"].strip() != ""


def test_pubmed_fault_f3_missing_title(pubmed_fault_records):
    """F3: No TI tag → empty title."""
    rec = _by_key(pubmed_fault_records, "99000003")
    assert rec is not None
    assert rec["title"] == ""
    assert rec["abstract"].strip() != ""


def test_pubmed_fault_f5_html_entities_preserved(pubmed_fault_records):
    """F5: HTML entities in AB are passed through unchanged."""
    rec = _by_key(pubmed_fault_records, "99000004")
    assert rec is not None
    assert "&amp;" in rec["abstract"]


def test_pubmed_fault_f7_doi_as_url(pubmed_fault_records):
    """F7: LID contains https:// URL as DOI value."""
    rec = _by_key(pubmed_fault_records, "99000005")
    assert rec is not None
    assert rec["doi"].startswith("https://")


def test_pubmed_fault_f12_wrong_indent_truncates_abstract(pubmed_fault_records):
    """F12: 4-space continuation lines are not recognised → abstract is first line only."""
    rec = _by_key(pubmed_fault_records, "99000006")
    assert rec is not None
    # The continuation text (4-space lines) should NOT appear in the abstract
    assert "four spaces" not in rec["abstract"]
    # Only the first AB line should be captured
    assert rec["abstract"].startswith("Benefit-risk assessment")


def test_pubmed_fault_f13_lid_without_doi_qualifier(pubmed_fault_records):
    """F13: LID line present but no [doi] qualifier → empty DOI."""
    rec = _by_key(pubmed_fault_records, "99000007")
    assert rec is not None
    assert rec["doi"] == ""


def test_pubmed_fault_f14_missing_pmid(pubmed_fault_records):
    """F14: No PMID tag → empty entry_key; rest of entry still parsed."""
    no_pmid = [r for r in pubmed_fault_records if r["entry_key"] == ""]
    assert len(no_pmid) == 1
    assert no_pmid[0]["title"].strip() != ""


def test_pubmed_fault_f15a_unusual_year_extractable(pubmed_fault_records):
    """F15a: DP = 'Spring 2014' — year extracted from 4-digit sequence."""
    rec = _by_key(pubmed_fault_records, "99000009")
    assert rec is not None
    assert rec["year"] == "2014"


def test_pubmed_fault_f15b_unusual_year_not_extractable(pubmed_fault_records):
    """F15b: DP = '22' — no 4-digit year, year is empty string."""
    rec = _by_key(pubmed_fault_records, "99000010")
    assert rec is not None
    assert rec["year"] == ""


# ---------------------------------------------------------------------------
# Unit tests: _strip_braces
# ---------------------------------------------------------------------------

def test_strip_braces_single():
    assert _strip_braces("{value}") == "value"


def test_strip_braces_double():
    assert _strip_braces("{{value}}") == "value"


def test_strip_braces_no_braces():
    assert _strip_braces("no braces") == "no braces"


def test_strip_braces_nested_inner_preserved():
    # Only outermost braces are stripped; inner braces remain
    assert _strip_braces("{nested {braces}}") == "nested {braces}"


def test_strip_braces_empty():
    assert _strip_braces("{}") == ""


def test_strip_braces_whitespace_around():
    assert _strip_braces("  {value}  ") == "value"


# ---------------------------------------------------------------------------
# Unit tests: _normalize_ws
# ---------------------------------------------------------------------------

def test_normalize_ws_multiple_spaces():
    assert _normalize_ws("a  b   c") == "a b c"


def test_normalize_ws_newline():
    assert _normalize_ws("a\nb") == "a b"


def test_normalize_ws_tab():
    assert _normalize_ws("a\tb") == "a b"


def test_normalize_ws_mixed():
    assert _normalize_ws("  a \n b\t c  ") == "a b c"


def test_normalize_ws_already_clean():
    assert _normalize_ws("clean string") == "clean string"


# ---------------------------------------------------------------------------
# Unit tests: _infer_source_db
# ---------------------------------------------------------------------------

def test_infer_source_db_wos_key():
    assert _infer_source_db({"ID": "WOS:000123456789"}) == "WOS"


def test_infer_source_db_scopus_source_field():
    assert _infer_source_db({"ID": "Smith2022", "source": "Scopus"}) == "Scopus"


def test_infer_source_db_scopus_case_insensitive():
    assert _infer_source_db({"ID": "Smith2022", "source": "SCOPUS"}) == "Scopus"


def test_infer_source_db_wos_author_field_fallback():
    # Entries with an 'author' field but no WOS: prefix and no Scopus source
    assert _infer_source_db({"ID": "SomeKey", "author": "Smith, J"}) == "WOS"


def test_infer_source_db_unknown_falls_back_to_bibtex():
    assert _infer_source_db({"ID": "Unknown2022"}) == "BibTeX"


# ---------------------------------------------------------------------------
# Unit tests: _is_pubmed (format detection)
# ---------------------------------------------------------------------------

def test_is_pubmed_true_for_pubmed_file():
    assert _is_pubmed(str(BIBS_DIR / "pubmed.txt")) is True


def test_is_pubmed_true_for_mcda_pubmed():
    assert _is_pubmed(str(BIBS_DIR / "mcda_pubmed.txt")) is True


def test_is_pubmed_true_for_pubmed_faults():
    assert _is_pubmed(str(BIBS_DIR / "pubmed_faults.txt")) is True


# ---------------------------------------------------------------------------
# parse_file: format auto-detection
# ---------------------------------------------------------------------------

def test_parse_file_bib_extension_returns_records():
    records = parse_file(str(BIBS_DIR / "wos_m.bib"))
    assert len(records) > 0
    assert all(r["source_db"] == "WOS" for r in records)


def test_parse_file_txt_extension_pubmed_returns_records():
    records = parse_file(str(BIBS_DIR / "mcda_pubmed.txt"))
    assert len(records) > 0
    assert all(r["source_db"] == "PubMed" for r in records)


def test_parse_file_unsupported_extension_returns_empty(tmp_path):
    bad_file = tmp_path / "data.ris"
    bad_file.write_text("TY  - JOUR\n")
    records = parse_file(str(bad_file))
    assert records == []


def test_parse_file_txt_non_pubmed_returns_empty(tmp_path):
    not_pubmed = tmp_path / "other.txt"
    not_pubmed.write_text("This is not a PubMed file.\n")
    records = parse_file(str(not_pubmed))
    assert records == []


# ---------------------------------------------------------------------------
# Performance tests: large generated fixtures (run with: pytest -m slow)
# ---------------------------------------------------------------------------

@pytest.mark.slow
@pytest.mark.parametrize("filename,expected_count,expected_source", [
    ("wos_large.bib",     1000, "WOS"),
    ("pubmed_large.txt",  1000, "PubMed"),
])
def test_large_file(filename, expected_count, expected_source):
    """Parse a 1000-entry file; verifies count and source_db."""
    path = BIBS_DIR / filename
    if not path.exists():
        pytest.skip(
            f"{filename} not found — run "
            f"python tests/bibs/generate_large_fixtures.py first"
        )
    records = parse_file(str(path))
    assert len(records) == expected_count, (
        f"{filename}: expected {expected_count} entries, got {len(records)}"
    )
    assert all(r["source_db"] == expected_source for r in records), (
        f"{filename}: not all entries have source_db={expected_source!r}"
    )
