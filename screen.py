# Screens paper based on title and abstract. Lots of code and prompts retrieved from SESR-eval paper replication package
# https://arxiv.org/abs/2507.19027
# TODO might want to consider removing binary and Likert decision as they cost money and at least Mika is not using them for anything.
# Probability decision is enough and can always be convertedy to binary or Likert later if needed. Well likert might be a bit tricky to convert from probability.
import argparse
import asyncio
import csv
import logging
import sys
import pandas as pd
import json
import os
import re
from typing import Dict, List, Optional, Tuple
from pydantic import BaseModel, field_validator, Field as PydanticField
from enum import Enum
from pydantic_ai.output import ToolOutput
from pathlib import Path

from async_api import process_all_models_agent

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


class LikertDecision(str, Enum):
    stronglyDisagree = "1"
    disagree = "2"
    somewhatDisagree = "3"
    neitherAgreeOrDisagree = "4"
    somewhatAgree = "5"
    agree = "6"
    stronglyAgree = "7"


class Decision(BaseModel, extra="forbid"):
    binary_decision: bool = PydanticField(
        description="Whether the criterion or relevance is clearly met (true) or not (false)."
    )
    probability_decision: float = PydanticField(
        description="The likelihood, that the criterion applies or the primary study is relevant."
    )
    likert_decision: LikertDecision = PydanticField(
        description="Likert scale decision."
    )
    reason: str = PydanticField(description="Reason for the decision.")

class Criterion(BaseModel, extra="forbid"):
    name: str = PydanticField(
        description="Criterion ID. E.g. IC1, EC1. Do NOT include description text."
    )
    decision: Decision = PydanticField(description="Decision for the criterion.")

    @field_validator('name', mode='before')
    @classmethod
    def normalize_id(cls, v: str) -> str:
        if not isinstance(v, str):
            return str(v)

        # 1. Extract the basic parts: (Letter I or E) and (Digits)
        # This ignores any 'C' in the middle and any trailing text
        match = re.search(r'([IE])(?:C)?(\d+)', v, re.IGNORECASE)

        if match:
            prefix = match.group(1).upper() # 'I' or 'E'
            number = match.group(2)         # '1', '2', etc.
            # 2. FORCE the format to 'IC' or 'EC'
            # To use the short format (I1), change this to: return f"{prefix}{number}"
            return f"{prefix}C{number}"

        return v.strip().replace(" ", "_")

class BinaryDecision(str, Enum):
    include = "Include"
    exclude = "Exclude"


class StructuredResponse(BaseModel, extra="forbid"):
    overall_decision: Decision
    inclusion_criteria: list[Criterion]
    exclusion_criteria: list[Criterion]


# --- Helper Functions ---
def detect_delimiter(file_path: str) -> str:
    with open(file_path, "r") as f:
        first_line = f.readline()
    if "," in first_line:
        return ","
    elif "\t" in first_line:
        return "\t"
    elif ";" in first_line:
        return ";"
    else:
        return ","  # default


system_prompt = "You are an expert research assistant."


def validate_csv(file_path: str, n_rows: Optional[int] = None) -> pd.DataFrame:
    delimiter = detect_delimiter(file_path)
    required_columns = {"title", "abstract"}
    header_row_index = -1
    with open(file_path, "r") as f:
        reader = csv.reader(f, delimiter=delimiter)
        for i, row in enumerate(reader):
            if i >= 20:
                break
            headers = [h.strip().lower() for h in row]
            if required_columns.issubset(headers):
                header_row_index = i
                break
    if header_row_index == -1:
        logger.error(
            "Error: Required columns (title, abstract) not found in the first 20 rows."
        )
        sys.exit(
            "Error: Required columns (title, abstract) not found in the first 20 rows."
        )
    with open(file_path, "r") as f:
        reader = csv.reader(f, delimiter=delimiter)
        for _ in range(header_row_index):
            next(reader)
        headers = [h.strip().lower() for h in next(reader)]
        column_counts = {}
        for col in headers:
            column_counts[col] = column_counts.get(col, 0) + 1
            if column_counts[col] > 1 and col in required_columns:
                logger.error(f"Error: Duplicate column detected: {col}")
                sys.exit(f"Error: Duplicate column detected: {col}")
        empty_title = 0
        empty_abstract = 0
        for i, row in enumerate(reader, 1):
            if not row[headers.index("title")].strip():
                empty_title += 1
            if not row[headers.index("abstract")].strip():
                empty_abstract += 1
        if empty_title > 0:
            logger.warning(f"WARN: {empty_title} titles are empty.")
            print(f"WARN: {empty_title} titles are empty.")
        if empty_abstract > 0:
            logger.warning(f"WARN: {empty_abstract} abstracts are empty.")
            print(f"WARN: {empty_abstract} abstracts are empty.")
    df = pd.read_csv(
        file_path, delimiter=delimiter, header=header_row_index, nrows=n_rows
    )
    df.columns = df.columns.str.strip().str.lower()
    return df


def load_api_key(key_path: str) -> str:
    try:
        with open(os.path.expanduser(key_path), "r") as file:
            api_key = file.read().strip()
        if not api_key:
            logger.error("Error: OpenRouter API key file is empty.")
            sys.exit("Error: OpenRouter API key file is empty.")
        return api_key
    except FileNotFoundError:
        logger.error(f"Error: OpenRouter API key file not found at {key_path}")
        sys.exit(f"Error: OpenRouter API key file not found at {key_path}")


def load_models(models_file: str) -> List[str]:
    with open(models_file, "r") as file:
        models = [
            line.strip().strip('"')
            for line in file
            if line.strip() and not line.strip().startswith("#")
        ]
    return models


def generate_unique_model_keys(models: List[str]) -> List[str]:
    seen = {}
    unique_keys = []
    for model in models:
        if model not in seen:
            seen[model] = 1
            unique_keys.append(model)
        else:
            seen[model] += 1
            unique_keys.append(f"{model}_{seen[model]}")
    return unique_keys


def generate_prompts(
    df: pd.DataFrame, criteria: str, additional_instructions: str
) -> List[str]:
    with open("prompts/prompt.txt", "r") as file:
        prompt_template = file.read()
    prompts = []
    for _, row in df.iterrows():
        prompt = prompt_template.format(
            row["title"], row["abstract"], criteria, additional_instructions
        )
        prompts.append(prompt)
    return prompts


def generate_output_filename(input_csv: str, criteria_path: str, models_path: str) -> str:
    # 1. Identify the input folder
    input_path = Path(input_csv)
    input_dir = input_path.parent

    # 2. Extract stems for naming
    csv_stem = input_path.stem
    c_stem = Path(criteria_path).stem
    m_stem = Path(models_path).stem

    # 3. Combine into base format
    base_name = f"{csv_stem}_{c_stem}_{m_stem}"

    # 4. Check for existing files in that specific directory
    output_path = input_dir / f"{base_name}.csv"

    counter = 1
    while output_path.exists():
        output_path = input_dir / f"{base_name}_{counter:02d}.csv"
        counter += 1

    return str(output_path)


def save_enriched_csv(df: pd.DataFrame, output_file: str) -> None:
    df.to_csv(output_file, index=False)
    logger.info(f"\nEnriched data saved to {output_file}")
    print(f"\nEnriched data saved to {output_file}")


def flatten_nested_json(
    nested_dict: Dict, parent_key: str = "", sep: str = "_"
) -> Dict:
    flattened = {}
    for key, value in nested_dict.items():
        new_key = f"{parent_key}{sep}{key}" if parent_key else key

        if isinstance(value, dict):
            # Recursively flatten dictionaries
            flattened.update(flatten_nested_json(value, new_key, sep))

        elif isinstance(value, list):
            # Special handling for criteria lists to avoid raw JSON strings
            for item in value:
                if isinstance(item, dict) and "name" in item:
                    # Use the criterion name (e.g., IC1) as part of the key
                    criterion_id = item["name"]
                    criterion_data = item.get("decision", {})
                    # Flatten the decision object under the criterion ID
                    crit_flattened = flatten_nested_json(
                        criterion_data, f"{new_key}{sep}{criterion_id}", sep
                    )
                    flattened.update(crit_flattened)
                else:
                    # Fallback for other lists
                    flattened[new_key] = json.dumps(value)
        else:
            flattened[new_key] = value
    return flattened


def run_nested_async_processing(
    df: pd.DataFrame, prompts: List[str], models: List[str], api_key: str
) -> Tuple[pd.DataFrame, Dict[str, Tuple[int, int]], List[str]]:
    df = df.copy().reset_index(drop=True)
    model_keys = generate_unique_model_keys(models)
    model_results = asyncio.run(
        process_all_models_agent(
            prompts,
            models,
            api_key,
            system_prompt=system_prompt,
            output_type=ToolOutput(StructuredResponse, name="structured_response"),
            max_concurrent_per_model=20,
        )
    )
    stats = {}
    for model_idx, model in enumerate(models):
        unique_key = model_keys[model_idx]
        logger.info(f"\nMerging results for model: {unique_key}")
        print(f"\nMerging results for model: {unique_key}")
        results = model_results[model_idx]
        successes = 0
        failures = 0
        for i, result in enumerate(results):
            if result:
                try:
                    parsed = result.model_dump()
                    flattened = flatten_nested_json(parsed)
                    for col, value in flattened.items():
                        col_name = f"{unique_key}_{col}"
                        if col_name not in df.columns:
                            df[col_name] = None
                        df.at[i, col_name] = value
                    successes += 1
                except json.JSONDecodeError as e:
                    logger.error(
                        f"Failed to parse JSON for row {i} (model {unique_key}): {e}, first 20 characters of response: ~ {result}"
                    )
                    print(
                        f"\nFailed to parse JSON for row {i} (model {unique_key}): {e}, first 20 characters of response: ~ {result}"
                    )
                    col_name = f"{unique_key}_error"
                    if col_name not in df.columns:
                        df[col_name] = None
                    df.at[i, col_name] = f"Failed to parse JSON: {e}"
                    failures += 1
                except Exception as e:
                    logger.error(
                        f"Failed to validate response for row {i} (model {unique_key}): {e}"
                    )
                    print(
                        f"\nFailed to validate response for row {i} (model {unique_key}): {e}"
                    )
                    col_name = f"{unique_key}_error"
                    if col_name not in df.columns:
                        df[col_name] = None
                    df.at[i, col_name] = f"Failed to validate response: {e}"
                    failures += 1
            else:
                col_name = f"{unique_key}_error"
                if col_name not in df.columns:
                    df[col_name] = None
                df.at[i, col_name] = "No response from API"
                failures += 1
        stats[unique_key] = (successes, failures)
    return df, stats, model_keys


def add_average_probability(df: pd.DataFrame, model_keys: List[str]) -> pd.DataFrame:
    df["average_probability"] = None
    df["min_probability"] = None
    df["max_probability"] = None

    for i, row in df.iterrows():
        probabilities = []
        for key in model_keys:
            col_name = f"{key}_overall_decision_probability_decision"
            if col_name in df.columns and pd.notna(row[col_name]):
                probabilities.append(row[col_name])

        if probabilities:
            df.at[i, "average_probability"] = round(
                sum(probabilities) / len(probabilities), 4
            )
            df.at[i, "min_probability"] = round(min(probabilities), 4)
            df.at[i, "max_probability"] = round(max(probabilities), 4)

    # Move the new columns to the front
    cols = ["average_probability", "min_probability", "max_probability"] + [
        col
        for col in df.columns
        if col not in ["average_probability", "min_probability", "max_probability"]
    ]
    df = df[cols]
    return df


# --- Main ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Screen papers based on title and abstract using LLMs.")

    parser.add_argument("csv_file", help="Path to the input CSV file.")
    parser.add_argument("-n", "--n_rows", default="10",
                        help="Number of rows to process (default: 10). Use 'all' for the entire file.")
    parser.add_argument("-c", "--criteria", default="criteria.conf",
                        help="Path to the criteria config file (default: criteria.conf)")
    parser.add_argument("-m", "--models", default="models.conf",
                        help="Path to the models list file (default: models.conf)")
    args = parser.parse_args()

    # Logic for n_rows: Convert to int unless it's "all"
    n_rows = None if args.n_rows.lower() == "all" else int(args.n_rows)

    # Load configuration files
    api_key = load_api_key("~/openrouter.key")
    models = load_models(args.models)

    # Load Criteria
    try:
        with open(args.criteria, "r") as file:
            criteria = "".join(
                line for line in file if not line.strip().startswith("#")
            ).strip()
    except FileNotFoundError:
        sys.exit(f"Error: Criteria file '{args.criteria}' not found.")

    # Load additional instructions if file exists
    additional_instructions = ""
    logger.info("Validating CSV file")
    print(f"Config: Rows={args.n_rows}, Criteria={args.criteria}, Models={args.models}")

    df = validate_csv(args.csv_file, n_rows=n_rows)

    logger.info(f"In total {len(df)} articles.")
    print(f"In total {len(df)} articles.")
    print(f"Criteria:\n ------------------\n {criteria[:300]}... \n ------------------")
    print("Generating prompts:")
    logger.info("Generating prompts")
    prompts = generate_prompts(df, criteria, additional_instructions)
    enriched_df, stats, model_keys = run_nested_async_processing(
        df, prompts, models, api_key
    )
    enriched_df = add_average_probability(enriched_df, model_keys)
    output_file = generate_output_filename(args.csv_file, args.criteria, args.models)
    save_enriched_csv(enriched_df, output_file)
    print("\nModel statistics:")
    for model_key, (success, failure) in stats.items():
        print(f"{model_key}: {success} successes, {failure} failures")
