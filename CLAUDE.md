# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

AISysRevCmdLine is a command-line tool for automated systematic literature review (SLR) screening using LLMs via the OpenRouter API. It screens academic papers (title + abstract) against inclusion/exclusion criteria and returns structured decisions. Command-line equivalent of the web-based [AISysRev](https://github.com/EvoTestOps/AISysRev).

## Commands

```bash
# Run tests
pytest

# Download the SESR-Eval benchmark dataset from Zenodo
python download_data.py [-o data/] [--no-extract] [--no-verify]

# Convert bibliographic files (.bib / PubMed .txt) to CSV
python bib2csv.py <file1> [file2 ...] [-o output.csv]

# Run the main screening tool
python screen.py <csv_file> -n <count|all> -c criteria.conf -m models.conf

# Run per-criterion boolean screening
python screen_boolean.py <csv_file> -n <count|all> -c criteria_screen_boolean.yml -m models.conf

# Run classification (probability-based)
python classify.py <csv_file> -n all -p 0.5

# Run single-choice classification
python classify_single.py <csv_file> -n 10

# Generate embeddings
python embed.py <csv_file> -n all -p 0.7

# Plot embeddings (UMAP + Plotly)
python plot.py <csv_file> -c
```

Python environment: conda, or venv. Python version: 3.14.

## Architecture

**Entry points** — each is a standalone CLI script using argparse:
- `download_data.py` — Downloads the SESR-Eval benchmark dataset from Zenodo. Extracts only `data/3-processed-data/`, writes `criteria.conf` and `instructions.txt` into each study subfolder from `secondary_study_data.csv`, copies the authoritative primary study CSV to `primary_correct.csv`, and removes the original `primary_study_data*.csv` files.
- `bib2csv.py` — Bibliographic format converter. Converts WOS/Scopus `.bib` files and PubMed MEDLINE `.txt` exports to a CSV ready for screening. Multiple files can be merged in one call. Output columns: `title, abstract, doi, year, authors, journal, keywords, source_db, entry_key`.
- `screen.py` — Main screening pipeline. Sends papers to LLMs with criteria, collects structured JSON responses, flattens them into CSV columns.
- `screen_boolean.py` — Per-criterion screening pipeline. Sends each criterion as a separate LLM call, combines per-criterion probabilities using fuzzy boolean logic (AND=MIN, OR=MAX, NOT=1−p) over a criteria tree defined in YAML.
- `classify.py` — Multi-class probability classification.
- `classify_single.py` — Single best-fit class assignment using dynamic Pydantic models.
- `embed.py` — Generate vector embeddings via OpenRouter `/embeddings` endpoint.
- `plot.py` — UMAP dimensionality reduction + Plotly interactive HTML visualization.

*bib2csv.py (bibliographic import):*
1. Auto-detect format per file: `.bib` → `parse_bibtex()` (uses `bibtexparser`); `.txt` starting with `PMID` → `parse_pubmed()` (custom MEDLINE line parser)
2. BibTeX: field lookup is case-insensitive; WOS double-braces `{{value}}` and Scopus single-braces `{value}` both handled; `source_db` inferred from cite-key prefix (`WOS:`) or `source` field
3. PubMed: tag regex `^([A-Z][A-Z0-9]{1,3})\s*-\s(.+)$`; continuation lines require exactly 6 leading spaces; multi-valued tags (FAU, OT, LID) collected; DOI extracted from `LID` lines ending with `[doi]`
4. Merge all records into one DataFrame → save CSV

**Shared utilities:**
- `helpers.py` — CSV validation (auto-detects delimiter, normalizes headers), API key loading (`~/openrouter.key`), model config loading, unique filename generation.
- `async_api.py` — Async API infrastructure with two parallel stacks:
  1. **pydantic_ai Agent stack** (httpx-based, for structured outputs):
     - `screen.py`, `screen_boolean.py`: `process_all_models_agent` → `create_agent` + `process_batch_agent` → `_call_agent` → httpx client
     - `classify.py`, `classify_single.py`: `create_agent` → `process_batch_agent` → `_call_agent` → httpx client
  2. **aiohttp stack** (for `/embeddings` endpoint):
     - `embed.py`: `process_batch_aiohttp` → `retry_aiohttp_call` + `make_openrouter_headers`

**Core data flows:**

*screen.py (screening task):*
1. `validate_csv()` → `generate_prompts()` using criteria.conf + prompts/prompt_screen.txt
2. `process_all_models_agent()` — async concurrent API calls with semaphore-limited concurrency per model
3. Parse structured JSON → `flatten_nested_json()` into flat CSV columns
4. `add_average_probability()` → save enriched CSV

*classify.py (multi-class probability classification):*
1. `validate_csv()` → load criteria_classify.yml → generate classification prompts
2. `create_agent()` → `process_batch_agent()` — async calls for probability distribution across classes
3. Parse structured JSON with class probabilities → flatten into CSV columns
4. Save enriched CSV with probability columns for each class

*classify_single.py (single best-fit classification):*
1. `validate_csv()` → load criteria_classify.yml → dynamically create Pydantic models for each taxonomy
2. `create_agent()` → `process_batch_agent()` — async calls for single-choice classification
3. Parse structured JSON with single class assignment → flatten into CSV columns
4. Save enriched CSV with selected class labels

*screen_boolean.py (per-criterion boolean screening):*
1. `validate_csv()` → load `criteria_screen_boolean.yml` (boolean criteria tree) → `generate_prompts()` using `prompts/prompt_screen_boolean.txt` (one prompt per paper per criterion)
2. `process_all_models_agent()` — one async call per paper per criterion
3. `fuzzy_eval()` — apply fuzzy boolean logic across criteria tree per paper
4. Compute `(inclusion_prob, exclusion_prob, overall_prob, binary_decision)` → save enriched CSV

*embed.py (vector embeddings):*
1. `validate_csv()` → load embed.conf → generate embedding texts
2. `process_batch_aiohttp()` with `retry_aiohttp_call()` — direct aiohttp POST to `/embeddings` endpoint
3. Extract vector embeddings → add as columns to CSV
4. Save enriched CSV with embedding dimensions as columns

*plot.py (visualization):*
1. Read CSV with embeddings → extract embedding columns
2. Apply UMAP dimensionality reduction to 2D/3D
3. Create interactive Plotly scatter plot (colored by classification/screening results)
4. Save as standalone HTML file

**Structured output enforcement:** Pydantic models (`StructuredResponse`, `Decision`, `Criterion`) in entry scripts force LLMs to return valid JSON via `response_format: json_schema`. Criterion names are regex-normalized to `IC1`, `IC2`, `EC1`, `EC2` format.

**Retry/error handling:**
- pydantic_ai stack uses `AsyncTenacityTransport` (httpx with tenacity) for automatic retry on 429, 502-504 with exponential backoff + jitter.
- aiohttp stack uses manual retry logic in `retry_aiohttp_call()` with similar backoff strategy.
- Both stacks abort on permanent errors (400, 401, 403, 404, 405, 422) and deduplicate error messages across batch.

## Configuration Files

User-customizable `.conf` and `.yml` files (gitignored — copy from `.example` files):
- `models.conf` — One OpenRouter model ID per line
- `criteria.conf` — Inclusion/exclusion criteria text
- `criteria_classify.yml` — YAML classification taxonomies
- `criteria_screen_boolean.yml` — Boolean criteria tree for `screen_boolean.py` (tracked in git as a working example)
- `embed.conf` — Text prefix for embeddings

Prompt templates (tracked in git, not meant to be edited by users):
- `prompts/prompt_screen.txt` — Template for `screen.py` with `{0}`-`{3}` placeholders
- `prompts/prompt_classify.txt` — Template for `classify.py`
- `prompts/prompt_classify_single.txt` — Template for `classify_single.py`
- `prompts/prompt_generate_classes.txt` — Template for `generate_classes.py`
- `prompts/prompt_screen_boolean.txt` — Template for `screen_boolean.py`

## API Integration

Uses OpenRouter API (`https://openrouter.ai/api/v1/`) with `temperature: 0`, `top_p: 0.1`, and `response_format: json_schema` for deterministic structured output. API key read from `~/openrouter.key`.

## Testing

Tests are in `tests/` with test fixtures in `tests/test_csv_files/` (CSV edge cases) and `tests/bibs/` (bibliographic files).

- `tests/test_csv_reader.py` — CSV validation edge cases (delimiter detection, missing/duplicate columns, header normalization).
- `tests/test_bib2csv.py` — Full test suite for `bib2csv.py`: entry counts for all 8 real bib files, `source_db` correctness, no-`None` values, and per-fault assertions for all 15 fault categories injected in `tests/bibs/wos_faults.bib`, `tests/bibs/scopus_faults.bib`, and `tests/bibs/pubmed_faults.txt`.

Integration testing of screening/classification scripts is done by running commands against OpenRouter with default settings (typically first 10 papers only, keeping costs low).
