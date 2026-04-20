#!/usr/bin/env python3
"""Scopus API utility — key validation, quota checks, and search."""

import argparse
import csv
import json
import sys
from pathlib import Path

import re

import requests

BASE_URL = "https://api.elsevier.com"
KEY_FILE = Path("~/scopus.key")
DATA_DIR = Path("data")
PAGE_SIZE = 25  # Scopus default; max is 200 but 25 is safe


def load_api_key(key_file: Path = KEY_FILE) -> str:
    key_file = key_file.expanduser()
    if not key_file.exists():
        sys.exit(f"API key file not found: {key_file}")
    key = key_file.read_text().strip()
    if not key:
        sys.exit(f"API key file is empty: {key_file}")
    return key


def load_conf(conf_file: Path) -> str:
    """Read conf file, strip comment lines (starting with #), return query string."""
    lines = conf_file.read_text(encoding="utf-8").splitlines()
    query = " ".join(
        line.strip() for line in lines
        if line.strip() and not line.strip().startswith("#")
    )
    if not query:
        sys.exit(f"No query found in {conf_file} (all lines are comments or empty).")
    return query


def make_headers(api_key: str) -> dict:
    return {
        "X-ELS-APIKey": api_key,
        "Accept": "application/json",
    }


def get_unique_output_path(base_path: Path) -> Path:
    if not base_path.exists():
        return base_path
    counter = 1
    while True:
        candidate = base_path.with_stem(f"{base_path.stem}_{counter:02d}")
        if not candidate.exists():
            return candidate
        counter += 1


# ---------------------------------------------------------------------------
# -i  info
# ---------------------------------------------------------------------------

def check_key_and_quota(api_key: str) -> None:
    """Run a minimal search query and report key status and rate-limit headers."""
    url = f"{BASE_URL}/content/search/scopus"
    params = {"query": "title(test)", "count": 1}

    print(f"Endpoint : {url}")
    print(f"API key  : {api_key[:8]}...{api_key[-4:]}")
    print()

    try:
        resp = requests.get(url, headers=make_headers(api_key), params=params, timeout=15)
    except requests.RequestException as exc:
        sys.exit(f"Request failed: {exc}")

    print(f"HTTP status : {resp.status_code} {resp.reason}")
    els_status = resp.headers.get("X-ELS-Status", "—")
    print(f"ELS status  : {els_status}")

    quota_headers = {
        "X-RateLimit-Limit": "Weekly quota (total requests)",
        "X-RateLimit-Remaining": "Weekly quota remaining",
        "X-RateLimit-Reset": "Quota reset (epoch seconds)",
        "X-ELS-ReqID": "Request ID",
        "X-ELS-APIKey-Remaining": "Key requests remaining",
    }
    print()
    print("Quota / rate-limit info:")
    found_any = False
    for header, label in quota_headers.items():
        val = resp.headers.get(header)
        if val is not None:
            print(f"  {label:35s} : {val}")
            found_any = True
    if not found_any:
        print("  (no quota headers returned)")

    print()
    print("All response headers:")
    for k, v in resp.headers.items():
        print(f"  {k}: {v}")

    if resp.status_code == 200:
        data = resp.json()
        total = data.get("search-results", {}).get("opensearch:totalResults", "?")
        print()
        print(f"Search results total (sanity check): {total}")
    elif resp.status_code == 401:
        print()
        print("Authentication failed — check your API key.")
        try:
            print(json.dumps(resp.json(), indent=2))
        except Exception:
            print(resp.text[:500])
        sys.exit(1)
    else:
        print()
        try:
            print(json.dumps(resp.json(), indent=2))
        except Exception:
            print(resp.text[:500])
        sys.exit(1)


# ---------------------------------------------------------------------------
# -s  search
# ---------------------------------------------------------------------------

# Fields to request from the API and how to map them to output CSV columns.
# The Scopus Search API query syntax is identical to the web advanced search.
FIELD_MAP = [
    ("dc:title",              "title"),
    ("dc:description",        "abstract"),
    ("prism:doi",             "doi"),
    ("prism:coverDate",       "year"),          # YYYY-MM-DD; we keep only year
    ("dc:creator",            "authors"),
    ("prism:publicationName", "journal"),
    ("authkeywords",          "keywords"),
    ("eid",                   "entry_key"),
    ("prism:volume",          "volume"),
    ("article-number",        "article_number"),
    ("citedby-count",         "cited_by"),
    ("prism:aggregationType", "type"),
    ("subtypeDescription",    "subtype"),
]
API_FIELDS = ",".join(f for f, _ in FIELD_MAP)
CSV_COLUMNS = [col for _, col in FIELD_MAP] + ["source_db", "bibtex"]

_AGGREGATION_TO_BIBTEX = {
    "Journal": "article",
    "Conference Proceeding": "inproceedings",
    "Book": "book",
    "Book Series": "incollection",
    "Trade Journal": "article",
}

_TITLE_STOP_WORDS = {
    "a", "an", "the", "of", "in", "on", "at", "to", "for", "and", "or",
    "but", "with", "by", "from", "into", "that", "this", "is", "are",
    "was", "be", "as", "its", "it", "via", "over", "under",
}


def _cite_key(row: dict) -> str:
    # Last name: dc:creator returns "Lastname, F." or "Lastname F." — take up to first comma or space
    author = row.get("authors", "").strip()
    lastname = re.split(r"[,\s]", author)[0] if author else "unknown"
    lastname = re.sub(r"[^a-zA-Z]", "", lastname).lower() or "unknown"

    year = row.get("year", "").strip() or "0000"

    title_words = re.findall(r"[a-zA-Z]+", row.get("title", "").lower())
    sig_word = next((w for w in title_words if w not in _TITLE_STOP_WORDS), "untitled")

    return f"{lastname}{year}{sig_word}"


def _make_bibtex(row: dict) -> str:
    bib_type = _AGGREGATION_TO_BIBTEX.get(row.get("type", ""), "misc")
    cite_key = _cite_key(row)
    raw_doi = row.get("doi", "").removeprefix("https://doi.org/")

    field_label = "booktitle" if bib_type == "inproceedings" else "journal"
    pairs = [
        ("title",          row.get("title", "")),
        ("author",         row.get("authors", "")),
        (field_label,      row.get("journal", "")),
        ("year",           row.get("year", "")),
        ("volume",         row.get("volume", "")),
        ("article-number", row.get("article_number", "")),
        ("doi",            raw_doi),
        ("abstract",       row.get("abstract", "")),
        ("keywords",       row.get("keywords", "")),
    ]
    body = ", ".join(f"{k} = {{{v}}}" for k, v in pairs if v)
    return f"@{bib_type}{{{cite_key}, {body}}}"


def _parse_entry(entry: dict) -> dict:
    row: dict = {}
    for api_field, col in FIELD_MAP:
        val = entry.get(api_field, "")
        if col == "year" and val:
            val = val[:4]  # YYYY-MM-DD → YYYY
        if col == "doi" and val:
            val = f"https://doi.org/{val}"
        row[col] = val or ""
    row["source_db"] = "Scopus"
    row["bibtex"] = _make_bibtex(row)
    return row


def _check_query_syntax(query: str) -> None:
    """Catch obvious syntax errors before hitting the API."""
    depth = 0
    in_string = False
    for ch in query:
        if ch == '"':
            in_string = not in_string
        if not in_string:
            if ch == '(':
                depth += 1
            elif ch == ')':
                depth -= 1
            if depth < 0:
                break
    if depth != 0:
        extra = "opening" if depth > 0 else "closing"
        count = abs(depth)
        print(f"\nQuery syntax warning: {count} unmatched {extra} parenthesis(es).")
        print("Please compare your query with the Scopus Advanced Search web interface:")
        print("  https://www.scopus.com/search/form.uri#advanced")
        print(f"\nFull query:\n  {query}\n")
        sys.exit(1)


def _fetch_page(api_key: str, query: str, start: int) -> dict:
    url = f"{BASE_URL}/content/search/scopus"
    params = {
        "query": query,
        "count": PAGE_SIZE,
        "start": start,
        "field": API_FIELDS,
        "view": "COMPLETE",  # required to get abstracts (dc:description)
    }
    resp = requests.get(url, headers=make_headers(api_key), params=params, timeout=30)
    if resp.status_code != 200:
        try:
            detail = resp.json()
            status_code = detail.get("service-error", {}).get("status", {}).get("statusCode", "")
            if status_code == "INVALID_INPUT":
                print(f"\nScopus rejected the query (INVALID_INPUT).")
                print("Please verify your query matches what you entered in the Scopus Advanced Search web interface:")
                print("  https://www.scopus.com/search/form.uri#advanced")
                print(f"\nFull query:\n  {query}\n")
                print("Common mistakes: unmatched parentheses, unquoted multi-word phrases (e.g. use \"Large Language Model\" not Large Language Model).")
                sys.exit(1)
        except Exception:
            detail = resp.text[:300]
        sys.exit(f"API error {resp.status_code}: {detail}")
    return resp.json()


def search(api_key: str, conf_file: Path, n: int | None) -> None:
    query = load_conf(conf_file)
    print(f"Query : {query[:120]}{'...' if len(query) > 120 else ''}")
    _check_query_syntax(query)

    # First page to find total results
    data = _fetch_page(api_key, query, start=0)
    results = data.get("search-results", {})
    total_available = int(results.get("opensearch:totalResults", 0))
    total_fetch = total_available if n is None else min(n, total_available)
    print(f"Total results in Scopus : {total_available}")
    print(f"Fetching                : {total_fetch}")

    if total_fetch == 0:
        print("No results — nothing to save.")
        return

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    out_path = get_unique_output_path(DATA_DIR / f"scopus_{conf_file.stem}.csv")

    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        writer.writeheader()

        fetched = 0
        entries = results.get("entry", [])
        for entry in entries:
            if fetched >= total_fetch:
                break
            writer.writerow(_parse_entry(entry))
            fetched += 1

        # Paginate through remaining pages
        while fetched < total_fetch:
            page_data = _fetch_page(api_key, query, start=fetched)
            entries = page_data.get("search-results", {}).get("entry", [])
            if not entries:
                break
            for entry in entries:
                if fetched >= total_fetch:
                    break
                writer.writerow(_parse_entry(entry))
                fetched += 1
            print(f"  fetched {fetched}/{total_fetch} ...", end="\r")

    print(f"\nSaved {fetched} records to {out_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Scopus API utility.")
    parser.add_argument(
        "--key-file",
        type=Path,
        default=KEY_FILE,
        help=f"Path to API key file (default: {KEY_FILE})",
    )
    parser.add_argument(
        "-i", "--info",
        action="store_true",
        help="Validate API key and report quota/rate-limit info.",
    )
    parser.add_argument(
        "-s", "--search",
        metavar="CONF",
        type=Path,
        help="Search using query from CONF file and save results to data/.",
    )
    parser.add_argument(
        "-n",
        metavar="N",
        type=int,
        default=None,
        help="Max number of results to fetch (default: all).",
    )
    args = parser.parse_args()

    if not args.info and args.search is None:
        parser.print_help()
        sys.exit(0)

    api_key = load_api_key(args.key_file)

    if args.info:
        check_key_and_quota(api_key)

    if args.search is not None:
        if not args.search.exists():
            sys.exit(f"Conf file not found: {args.search}")
        search(api_key, args.search, args.n)


if __name__ == "__main__":
    main()
