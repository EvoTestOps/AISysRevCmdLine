# Systematic Review Screening with LLMs via OpenRouter

**Automated screening of academic articles using Large Language Models (LLMs) via OpenRouter API.**

This script processes a CSV file of academic articles (with `title` and `abstract` columns), queries multiple LLM models for structured relevance decisions, and outputs an enriched CSV with the results.
It is meant to be a command line equivalent of the [AISysRev](https://github.com/EvoTestOps/AISysRev) web tool.

---
## Features
- **Bibliographic import:** `bib2csv.py` converts Web of Science / Scopus `.bib` exports and PubMed MEDLINE `.txt` exports to a CSV ready for screening. Multiple files can be merged in one call.
- **Structured LLM Responses:** Uses Pydantic AI to enforce structured JSON output from LLMs.
- **Concurrent API Calls:** Efficiently processes multiple articles and models in parallel
- **Model selection:** Copy the example file and adjust models you want to run from OpenRouter your settings:
  ```bash
  cp models.conf.example models.conf
- **Inclusion / Exclusion Criteria:** Customize inclusion/exclusion criteria and instructions by copying the example file and adjusting it:
  ```bash
  cp criteria.conf.example criteria.conf
- **Boolean screening:** `screen_boolean.py` screens papers per-criterion — each criterion is sent as a separate LLM call. Per-criterion probabilities are combined using fuzzy boolean logic (AND=MIN, OR=MAX, NOT=1−p) over a criteria tree defined in YAML. A final binary include/exclude decision is derived from the overall probability (threshold 0.5).
- **Error Handling:** Semi-robust retry logic and error reporting.
- **Progress Tracking:** Real-time progress bars with `tqdm`.



## Importing bibliographic files

If your papers are in a database export rather than a CSV, convert them first:

```bash
# Single file (WOS, Scopus, or PubMed)
python bib2csv.py export.bib -o papers.csv
python bib2csv.py pubmed_export.txt -o papers.csv
# Merge multiple files from different databases
python bib2csv.py wos.bib scopus.bib pubmed.txt -o papers.csv
# Wildcards also work
python bib2csv.py bibs/* -o papers.csv
```

Supported formats:
- **Web of Science** — `.bib` export (`@article{ WOS:... }`)
- **Scopus** — `.bib` export (`@ARTICLE{...}`)
- **PubMed** — MEDLINE tagged `.txt` export (lines starting with `PMID-`)

Output CSV always contains: `title, abstract, doi, year, authors, journal, keywords, source_db, entry_key`

---

After customizing as shown above, you can run it
```bash
uv run screen.py <csv_file_with_columns_named_title_and_abstract> -n all
```
You can also customize input files. 
```bash 
uv run screen.py <csv_file_with_columns_named_title_and_abstract> -n <number of papers> -c <criteria_file> -m <models_file>
```
Then you see output like this:
<img width="2142" height="226" alt="{8BA04E17-0F83-4531-AB75-69C66F3F4E29}" src="https://github.com/user-attachments/assets/fad9c8a5-e24b-4716-989c-a74173b8f3eb" />

After that an enriched CSV file is produced with LLM responses. 

## Known Issues
Not all models return valid responses, e.g., older Llama models. Default setting of screen.py only runs 10 first rows, which is handy if you want to collects statistics on how good the models performs in terms of giving valid JSON responses and not blowing up your OpenRouter credits.  If you choose your models carefully you may see good success rate in getting valid output, as in image below
<img width="848" height="130" alt="{33D49869-9DBB-46B4-9A79-B1819A2612F7}" src="https://github.com/user-attachments/assets/1e252878-f9f2-4fae-89d1-6288eca12bd2" />

Ensure your CSV is properly escaped. Google sheet to CSV export may produce CSV lines that contain line breaks without being properly escaped. If that is the case run find "\n" replace " " in Google sheet with regular experssions enabled before exporting to CSV.
