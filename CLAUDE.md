# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

AISysRevCmdLine is a command-line tool for automated systematic literature review (SLR) screening using LLMs via the OpenRouter API. It screens academic papers (title + abstract) against inclusion/exclusion criteria and returns structured decisions. Command-line equivalent of the web-based [AISysRev](https://github.com/EvoTestOps/AISysRev).

## Commands

```bash
# Run tests
pytest

# Run the main screening tool
uv run screen.py <csv_file> -n <count|all> -c criteria.conf -m models.conf

# Run classification (probability-based)
uv run classify.py <csv_file> -n all -p 0.5

# Run single-choice classification
uv run classify_single.py <csv_file> -n 10

# Generate embeddings
uv run embed.py <csv_file> -n all -p 0.7

# Plot embeddings (UMAP + Plotly)
uv run plot.py <csv_file> -c
```

Package management uses `uv`. Python version: 3.14.

## Architecture

**Entry points** — each is a standalone CLI script using argparse:
- `screen.py` — Main screening pipeline. Sends papers to LLMs with criteria, collects structured JSON responses, flattens them into CSV columns.
- `classify.py` — Multi-class probability classification.
- `classify_single.py` — Single best-fit class assignment using dynamic Pydantic models.
- `embed.py` — Generate vector embeddings via OpenRouter `/embeddings` endpoint.
- `plot.py` — UMAP dimensionality reduction + Plotly interactive HTML visualization.

**Shared utilities** in `helpers.py`: CSV validation (auto-detects delimiter, normalizes headers), API key loading (`~/openrouter.key`), model config loading, unique filename generation.

**Core data flow in screen.py:**
1. `validate_csv()` → `generate_prompts()` using criteria.conf + prompt.conf
2. `process_all_models()` — async concurrent API calls with semaphore-limited concurrency per model
3. Parse structured JSON → `flatten_nested_json()` into flat CSV columns
4. `add_average_probability()` → save enriched CSV

**Structured output enforcement:** Pydantic models (`StructuredResponse`, `Decision`, `Criterion`) force LLMs to return valid JSON. Criterion names are regex-normalized to `IC1`, `IC2`, `EC1`, `EC2` format.

**Retry/error handling:** `AsyncTenacityTransport` with exponential backoff for rate limits (429) and server errors (502-504).

## Configuration Files

User-customizable `.conf` and `.yml` files (gitignored — copy from `.example` files):
- `models.conf` — One OpenRouter model ID per line
- `criteria.conf` — Inclusion/exclusion criteria text
- `criteria_classify.yml` — YAML classification taxonomies
- `prompt.conf` / `prompt_classify.conf` / `prompt_classify_single.conf` — Prompt templates with `{0}`-`{3}` placeholders
- `criteria_embed.conf` — Text prefix for embeddings

## API Integration

Uses OpenRouter API (`https://openrouter.ai/api/v1/`) with `temperature: 0`, `top_p: 0.1`, and `response_format: json_schema` for deterministic structured output. API key read from `~/openrouter.key`.

## Testing

Tests are in `tests/` with test CSV fixtures in `tests/test_csv_files/`. Current test coverage focuses on CSV validation edge cases (delimiter detection, missing/duplicate columns, header normalization).
