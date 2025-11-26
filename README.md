# Systematic Review Screening with LLMs via OpenRouter

**Automated screening of academic articles using Large Language Models (LLMs) via OpenRouter API.**

This script processes a CSV file of academic articles (with `title` and `abstract` columns), queries multiple LLM models for structured relevance decisions, and outputs an enriched CSV with the results.
It is meant to be a command line equivalent of the [AISysRev](https://github.com/EvoTestOps/AISysRev) web tool.

---
## Features
- **Structured LLM Responses:** Uses Pydantic models to enforce structured JSON output from LLMs.
- **Concurrent API Calls:** Efficiently processes multiple articles and models in parallel
- **Model selection:** Copy the example file and adjust models you want to run from OpenRouter your settings:
  ```bash
  cp models.md.example models.md
- **Inclusion / Exclusion Criteria:** Customize inclusion/exclusion criteria and instructions by copying the example file and adjusting it:
  ```bash
  cp criteria.md.example criteria.md
- **Error Handling:** Semi-robust retry logic and error reporting.
- **Progress Tracking:** Real-time progress bars with `tqdm`.



After customizing as shown above, you can run it
```bash
python screen.py <csv_file_with_columns_named_title_and_abstract> all
```
Then you see output like this:
<img width="2142" height="226" alt="{8BA04E17-0F83-4531-AB75-69C66F3F4E29}" src="https://github.com/user-attachments/assets/fad9c8a5-e24b-4716-989c-a74173b8f3eb" />

After that an enriched CSV file is produced with LLM responses. 

## Known Issues
Not all models return valid responses, e.g., older Llama models. Default setting of screen.py only runs 10 first rows, which is handy if you want to collects statistics on how good the models performs in terms of giving valid JSON responses and not blowing up your OpenRouter credits.  If you choose your models carefully you may see good success rate in getting valid output, as in image below
<img width="848" height="130" alt="{33D49869-9DBB-46B4-9A79-B1819A2612F7}" src="https://github.com/user-attachments/assets/1e252878-f9f2-4fae-89d1-6288eca12bd2" />

Ensure your CSV is properly escaped. Google sheet to CSV export may produce CSV lines that contain line breaks without being properly escaped. If that is the case run find "\n" replace " " in Google sheet with regular experssions enabled before exporting to CSV.
