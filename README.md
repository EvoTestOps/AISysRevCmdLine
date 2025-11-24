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
python screen.py <csv_file_with_columns_named_title_and_abstract>
```
Then you see output like this:
<img width="2142" height="226" alt="{8BA04E17-0F83-4531-AB75-69C66F3F4E29}" src="https://github.com/user-attachments/assets/fad9c8a5-e24b-4716-989c-a74173b8f3eb" />

After that an enriched CSV file is produced with LLM responses. 

## Known Issues
Not all models return valid responses, e.g., OpenAI ones and older Llama models.

There is also screen_test.py that does the same thing as screen.py but collects statistics if you want to know how the models are performing in terms of giving correct responses 

```bash
python screen_test.py <csv_file_with_columns_named_title_and_abstract> 100
```
Then you see stats like below. The default is 10 papers but in the above a 100 was specified. 

<img width="908" height="157" alt="{228C07BA-4807-4D22-A57D-ACDEF60C7459}" src="https://github.com/user-attachments/assets/00e60954-2728-4962-9a0b-d9b19f94c092" />
