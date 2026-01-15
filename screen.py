# Screens paper based on title and abstract. Lots of code and prompts retrieved from SESR-eval paper replication package
# https://arxiv.org/abs/2507.19027
# TODO might want to consider removing binary and Likert decision as they cost money and at least Mika is not using them for anything.
# Probability decision is enough and can always be convertedy to binary or Likert later if needed. Well likert might be a bit tricky to convert from probability.

import asyncio
import csv
import sys
import pandas as pd
import json
import os
from typing import Dict, List, Optional, Tuple
from pydantic import BaseModel, Field as PydanticField
from enum import Enum
from pydantic_ai.output import ToolOutput
from tqdm.asyncio import tqdm_asyncio


# --- Pydantic Models and Enums ---
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
        description="Criterion ID. E.g. IC1, IC2, IC3 etc.. for inclusion criteria or EC1, EC2, EC3 etc.. for exclusion criteria"
    )
    decision: Decision = PydanticField(description="Decision for the criterion.")


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
                sys.exit(f"Error: Duplicate column detected: {col}")
        empty_title = 0
        empty_abstract = 0
        for i, row in enumerate(reader, 1):
            if not row[headers.index("title")].strip():
                empty_title += 1
            if not row[headers.index("abstract")].strip():
                empty_abstract += 1
        if empty_title > 0:
            print(f"WARN: {empty_title} titles are empty.")
        if empty_abstract > 0:
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
            sys.exit("Error: OpenRouter API key file is empty.")
        return api_key
    except FileNotFoundError:
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
    with open("prompt.md", "r") as file:
        prompt_template = file.read()
    prompts = []
    for _, row in df.iterrows():
        prompt = prompt_template.format(
            row["title"], row["abstract"], criteria, additional_instructions
        )
        prompts.append(prompt)
    return prompts


def generate_output_filename(input_filename: str, model_keys: List[str]) -> str:
    base, ext = os.path.splitext(input_filename)
    models_str = "_".join(m.replace("/", "-") for m in model_keys)
    output_filename = f"{base}_LLMs_{models_str}{ext}"
    counter = 1
    while os.path.exists(output_filename):
        output_filename = f"{base}_LLMs_{models_str}_{counter:02d}{ext}"
        counter += 1
    return output_filename


def save_enriched_csv(df: pd.DataFrame, output_file: str) -> None:
    df.to_csv(output_file, index=False)
    print(f"\nEnriched data saved to {output_file}")


def flatten_nested_json(
    nested_dict: Dict, parent_key: str = "", sep: str = "_"
) -> Dict:
    flattened = {}
    for key, value in nested_dict.items():
        new_key = f"{parent_key}{sep}{key}" if parent_key else key
        if isinstance(value, dict):
            flattened.update(flatten_nested_json(value, new_key, sep))
        elif isinstance(value, list):
            flattened[new_key] = json.dumps(value)
        else:
            flattened[new_key] = value
    return flattened


# --- Async API Functions ---
async def call_openrouter_async(
    prompt: str,
    model_name: str,
    api_key: str,
    semaphore: asyncio.Semaphore,
    max_retries: int = 3,
) -> Optional[StructuredResponse]:
    async with semaphore:
        for attempt in range(max_retries):
            try:
                from pydantic_ai import Agent
                from pydantic_ai.models.openrouter import (
                    OpenRouterModel,
                    OpenRouterModelSettings,
                )
                from pydantic_ai.providers.openrouter import OpenRouterProvider

                settings = OpenRouterModelSettings(
                    openrouter_provider={
                        "require_parameters": True,
                        "data_collection": "deny",
                    },
                    extra_headers={
                        "X-Title": "AISysRev",
                        "HTTP-Referer": "https://github.com/EvoTestOps/AISysRev",
                    },
                    # By default we fix temperature and top_p
                    temperature=0,
                    top_p=0.1,
                )
                model = OpenRouterModel(
                    model_name,
                    provider=OpenRouterProvider(api_key=api_key),
                    settings=settings,
                )
                agent = Agent(
                    model,
                    system_prompt="You are an expert research assistant.",
                    retries=max_retries,
                    output_type=ToolOutput(
                        StructuredResponse, name="structured_response"
                    ),
                )
                result = await agent.run(prompt)
                return result.output
            except Exception as e:
                print(f"LLM call failed for model {model_name}: {e}")
                await asyncio.sleep(2 ** (attempt + 2))
        return None


async def process_prompts_for_model(
    prompts: List[str],
    model: str,
    api_key: str,
    max_concurrent_per_model: int = 20,
) -> List[Optional[StructuredResponse]]:
    semaphore = asyncio.Semaphore(max_concurrent_per_model)
    tasks = [
        call_openrouter_async(prompt, model, api_key, semaphore) for prompt in prompts
    ]
    return await tqdm_asyncio.gather(*tasks, desc=f"Processing {model}")


async def process_all_models(
    prompts: List[str],
    models: List[str],
    api_key: str,
    max_concurrent_per_model: int = 20,
) -> Tuple[List[List[Optional[StructuredResponse]]], List[str]]:
    print(
        f"Processing {len(models)} models with {max_concurrent_per_model} concurrent prompts per model..."
    )
    model_keys = generate_unique_model_keys(models)
    model_tasks = [
        process_prompts_for_model(prompts, model, api_key, max_concurrent_per_model)
        for model in models
    ]
    return await tqdm_asyncio.gather(*model_tasks, desc="Processing models"), model_keys


def run_nested_async_processing(
    df: pd.DataFrame, prompts: List[str], models: List[str], api_key: str
) -> Tuple[pd.DataFrame, Dict[str, Tuple[int, int]], List[str]]:
    df = df.copy().reset_index(drop=True)
    model_results, model_keys = asyncio.run(
        process_all_models(prompts, models, api_key, max_concurrent_per_model=20)
    )
    stats = {}
    for model_idx, model in enumerate(models):
        unique_key = model_keys[model_idx]
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
                    print(
                        f"\nFailed to parse JSON for row {i} (model {unique_key}): {e}, first 20 characters of response: ~ {result}"
                    )
                    col_name = f"{unique_key}_error"
                    if col_name not in df.columns:
                        df[col_name] = None
                    df.at[i, col_name] = f"Failed to parse JSON: {e}"
                    failures += 1
                except Exception as e:
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
    if len(sys.argv) not in (2, 3):
        sys.exit("Usage: python screen_test.py <csv_file> [n_rows]")
    csv_file = sys.argv[1]
    n_rows_arg = sys.argv[2] if len(sys.argv) == 3 else "10"
    # If "all" is passed, set n_rows to None (or a very large number)
    n_rows = None if n_rows_arg.lower() == "all" else int(n_rows_arg)
    api_key = load_api_key("~/openrouter.key")
    models = load_models("models.md")
    with open("criteria.md", "r") as file:
        criteria = "".join(
            line for line in file if not line.strip().startswith("#")
        ).strip()
    with open("json_instruction_prompt.txt", "r") as file:
        additional_instructions = file.read()
    additional_instructions = ""
    print("Validating CSV file:")
    df = validate_csv(csv_file, n_rows=n_rows)
    print(f"In total {len(df)} articles.")
    print(f"Criteria:\n ------------------\n {criteria[:300]}... \n ------------------")
    print("Generating prompts:")
    prompts = generate_prompts(df, criteria, additional_instructions)
    enriched_df, stats, model_keys = run_nested_async_processing(
        df, prompts, models, api_key
    )
    enriched_df = add_average_probability(enriched_df, model_keys)
    output_file = generate_output_filename(csv_file, model_keys)
    save_enriched_csv(enriched_df, output_file)
    print("\nModel statistics:")
    for model_key, (success, failure) in stats.items():
        print(f"{model_key}: {success} successes, {failure} failures")
