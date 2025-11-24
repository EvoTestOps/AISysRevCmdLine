import asyncio
import aiohttp
import csv
import sys
import pandas as pd
import json
import os
import random
from typing import Dict, List, Optional, Tuple
from pydantic import BaseModel, Field as PydanticField
from enum import Enum
from tqdm.asyncio import tqdm_asyncio
from tqdm import tqdm

# --- Pydantic Models and Enums ---
class LikertDecision(Enum):
    stronglyDisagree = "1"
    disagree = "2"
    somewhatDisagree = "3"
    neitherAgreeOrDisagree = "4"
    somewhatAgree = "5"
    agree = "6"
    stronglyAgree = "7"

class Decision(BaseModel, extra="forbid"):
    binary_decision: bool = PydanticField(description="Whether the criterion or relevance is clearly met (true) or not (false).")
    probability_decision: float = PydanticField(description="The likelihood, that the criterion applies or the primary study is relevant.")
    likert_decision: LikertDecision = PydanticField(description="Likert scale decision.")
    reason: str = PydanticField(description="Reason for the decision.")

class Criterion(BaseModel, extra="forbid"):
    name: str = PydanticField(description="Criterion ID. E.g. IC1, IC2, IC3 etc.. for inclusion criteria or EC1, EC2, EC3 etc.. for exclusion criteria")
    decision: Decision = PydanticField(description="Decision for the criterion.")

class BinaryDecision(Enum):
    include = "Include"
    exclude = "Exclude"

class StructuredResponse(BaseModel, extra="forbid"):
    overall_decision: Decision
    inclusion_criteria: list[Criterion]
    exclusion_criteria: list[Criterion]

# --- Helper Functions ---
def detect_delimiter(file_path: str) -> str:
    with open(file_path, 'r') as f:
        first_line = f.readline()
    return ';' if ';' in first_line else ','

def validate_csv(file_path: str, n_rows: int = None) -> pd.DataFrame:
    delimiter = detect_delimiter(file_path)
    required_columns = {'title', 'abstract'}
    found_columns = set()
    column_counts = {}
    with open(file_path, 'r') as f:
        reader = csv.reader(f, delimiter=delimiter)
        headers = [h.strip().lower() for h in next(reader)]
        for col in headers:
            column_counts[col] = column_counts.get(col, 0) + 1
            if col in required_columns:
                found_columns.add(col)
        if not required_columns.issubset(found_columns):
            missing = required_columns - found_columns
            sys.exit(f"Error: Missing required columns: {missing}")
        for col, count in column_counts.items():
            if col in required_columns and count > 1:
                sys.exit(f"Error: Duplicate column detected: {col}")
        empty_title = 0
        empty_abstract = 0
        f.seek(0)
        next(reader)
        for i, row in enumerate(reader, 1):
            if not row[headers.index('title')].strip():
                empty_title += 1
            if not row[headers.index('abstract')].strip():
                empty_abstract += 1
        if empty_title > 0:
            print(f"WARN: {empty_title} titles are empty.")
        if empty_abstract > 0:
            print(f"WARN: {empty_abstract} abstracts are empty.")
    df = pd.read_csv(file_path, delimiter=delimiter, nrows=n_rows)
    df.columns = df.columns.str.strip().str.lower()
    return df

def load_api_key(key_path: str) -> str:
    try:
        with open(os.path.expanduser(key_path), 'r') as file:
            api_key = file.read().strip()
        if not api_key:
            sys.exit("Error: OpenRouter API key file is empty.")
        return api_key
    except FileNotFoundError:
        sys.exit(f"Error: OpenRouter API key file not found at {key_path}")

def load_models(models_file: str) -> List[str]:
    with open(models_file, 'r') as file:
        models = [
            line.strip().strip('"')
            for line in file
            if line.strip() and not line.strip().startswith('#')
        ]
    return models

def generate_prompts(df: pd.DataFrame, criteria: str, additional_instructions: str) -> List[str]:
    with open('prompt.md', 'r') as file:
        prompt_template = file.read()
    prompts = []
    for _, row in df.iterrows():
        prompt = prompt_template.format(row['title'], row['abstract'], criteria, additional_instructions)
        prompts.append(prompt)
    return prompts

def generate_output_filename(input_filename: str, models: List[str]) -> str:
    base, ext = os.path.splitext(input_filename)
    models_str = "_".join(m.replace("/", "-") for m in models)
    output_filename = f"{base}_LLMs_{models_str}{ext}"
    counter = 1
    while os.path.exists(output_filename):
        output_filename = f"{base}_LLMs_{models_str}_{counter:02d}{ext}"
        counter += 1
    return output_filename

def save_enriched_csv(df: pd.DataFrame, output_file: str) -> None:
    df.to_csv(output_file, index=False)
    print(f"\nEnriched data saved to {output_file}")

def flatten_nested_json(nested_dict: Dict, parent_key: str = '', sep: str = '_') -> Dict:
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
    model: str,
    api_key: str,
    session: aiohttp.ClientSession,
    semaphore: asyncio.Semaphore,
    max_retries: int = 3,
) -> Optional[Dict]:
    url = "https://openrouter.ai/api/v1/chat/completions"
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": "structured_response",
                "strict": True,
                "schema": StructuredResponse.model_json_schema(),
            },
        },
    }
    async with semaphore:
        for attempt in range(max_retries):
            try:
                async with session.post(url, headers=headers, json=payload, timeout=aiohttp.ClientTimeout(total=120)) as response:
                    if response.status == 429:
                        retry_after = int(response.headers.get("Retry-After", 5))
                        jitter = random.uniform(0, 5)
                        retry_after = retry_after + jitter + attempt * retry_after
                        print(f"\nRate limited for model {model}. Retrying after {retry_after} seconds...")
                        await asyncio.sleep(retry_after)
                        continue
                    response.raise_for_status()
                    return  await response.json()
            except Exception as e:
                print(f"\nAttempt {attempt + 1} failed for model {model}: {e}")
                await asyncio.sleep(2 ** (attempt + 2))
        return None

async def process_prompts_for_model(
    prompts: List[str],
    model: str,
    api_key: str,
    max_concurrent_per_model: int = 20,
) -> Tuple[List[Optional[Dict]], int, int]:
    semaphore = asyncio.Semaphore(max_concurrent_per_model)
    async with aiohttp.ClientSession() as session:
        tasks = [
            call_openrouter_async(prompt, model, api_key, session, semaphore)
            for prompt in prompts
        ]
        return await tqdm_asyncio.gather(*tasks, desc=f"Processing {model}")


async def process_all_models(
    prompts: List[str],
    models: List[str],
    api_key: str,
    max_concurrent_per_model: int = 20,
) -> Tuple[List[List[Optional[Dict]]], Dict[str, Tuple[int, int]]]:
    print(f"Processing {len(models)} models with {max_concurrent_per_model} concurrent prompts per model...")
    model_tasks = [
        process_prompts_for_model(prompts, model, api_key, max_concurrent_per_model)
        for model in models
    ]
    return  await tqdm_asyncio.gather(*model_tasks, desc="Processing models")
 
def run_nested_async_processing(df: pd.DataFrame, prompts: List[str], models: List[str], api_key: str) -> Tuple[pd.DataFrame, Dict[str, Tuple[int, int]]]:
    df = df.copy().reset_index(drop=True)
    model_results = asyncio.run(process_all_models(prompts, models, api_key, max_concurrent_per_model=20))
    stats = {}  # Initialize the stats dictionary here
    for model_idx, model in enumerate(models):
        print(f"\nMerging results for model: {model}")
        results = model_results[model_idx]
        successes = 0
        failures = 0
        for i, result in enumerate(results):
            if result:
                try:
                    # Parse the JSON content
                    parsed = json.loads(result['choices'][0]['message']['content'])
                    # Validate with Pydantic does not work for some reason check this. 
                    # structured_response = StructuredResponse(**parsed)
                    # Flatten the validated response
                    flattened = flatten_nested_json(parsed)
                    for col, value in flattened.items():
                        col_name = f"{model}_{col}"
                        if col_name not in df.columns:
                            df[col_name] = None
                        df.at[i, col_name] = value
                    successes += 1
                except json.JSONDecodeError as e:
                    print(f"\nFailed to parse JSON for row {i} (model {model}): {e}")
                    col_name = f"{model}_error"
                    if col_name not in df.columns:
                        df[col_name] = None
                    df.at[i, col_name] = f"Failed to parse JSON: {e}"
                    failures += 1
                except Exception as e:  # Catches Pydantic validation errors and other exceptions
                    print(f"\nFailed to validate response for row {i} (model {model}): {e}")
                    col_name = f"{model}_error"
                    if col_name not in df.columns:
                        df[col_name] = None
                    df.at[i, col_name] = f"Failed to validate response: {e}"
                    failures += 1
            else:
                col_name = f"{model}_error"
                if col_name not in df.columns:
                    df[col_name] = None
                df.at[i, col_name] = "No response from API"
                failures += 1
        stats[model] = (successes, failures)
    return df, stats


def add_average_probability(df: pd.DataFrame, models: List[str]) -> pd.DataFrame:
    df["average_probability"] = None
    for i, row in df.iterrows():
        probabilities = []
        for model in models:
            col_name = f"{model}_overall_decision_probability_decision"
            if col_name in df.columns and pd.notna(row[col_name]):
                probabilities.append(row[col_name])
        if probabilities:
            df.at[i, "average_probability"] = round(sum(probabilities) / len(probabilities), 4)
    cols = ['average_probability'] + [col for col in df.columns if col != 'average_probability']
    df = df[cols]
    return df

# --- Main ---
if __name__ == "__main__":
    if len(sys.argv) not in (2, 3):
        sys.exit("Usage: python screen_test.py <csv_file> [n_rows]")
    csv_file = sys.argv[1]
    n_rows = int(sys.argv[2]) if len(sys.argv) == 3 else 10
    api_key = load_api_key("~/openrouter.key")
    models = load_models("models.md")
    output_file = generate_output_filename(csv_file, models)
    with open('criteria.md', 'r') as file:
        criteria = ''.join(
            line for line in file
            if not line.strip().startswith('#')
        ).strip()
    with open('json_instruction_prompt.txt', 'r') as file:
        additional_instructions = file.read()
    print("Validating CSV file:")
    df = validate_csv(csv_file, n_rows=n_rows)
    print(f"In total {len(df)} articles.")
    print(f"Criteria:\n ------------------\n {criteria[:300]}... \n ------------------")
    print("Generating prompts:")
    prompts = generate_prompts(df, criteria, additional_instructions)
    enriched_df, stats = run_nested_async_processing(df, prompts, models, api_key)
    enriched_df = add_average_probability(enriched_df, models)
    save_enriched_csv(enriched_df, output_file)
    print("\nModel statistics:")
    for model, (success, failure) in stats.items():
        print(f"{model}: {success} successes, {failure} failures")
