# Systematic Review Screening with LLM APIs

**Automated screening of academic articles using Large Language Models (LLMs) via OpenRouter API.**

This script processes a CSV file of academic articles (with `title` and `abstract` columns), queries multiple LLM models for structured relevance decisions, and outputs an enriched CSV with the results.
It is meant to be a command line equivalent of the AISysRev web tool.

---
## Features
- **Structured LLM Responses:** Uses Pydantic models to enforce structured JSON output from LLMs.
- **Concurrent API Calls:** Efficiently processes multiple articles and models in parallel.
- **Model selection:** Select models you want to run from OpenRouter.
- **Inclusion / Exclusion Criteria:** Customize inclusion/exclusion criteria and instructions.
- **Error Handling:** Semi-robust retry logic and error reporting.
- **Progress Tracking:** Real-time progress bars with `tqdm`.

## Known Issues
Not all models return valid responses, e.g., OpenAI ones and older Llama models.
