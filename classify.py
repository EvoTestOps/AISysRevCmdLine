# Classifies paper using OpenRouter API based on classification criteria specified in criteria_classify.yml.
# Saves the classification probabilities to new CSV files.
# Classifications are given as probablities for each class specified in criteria_classify.yml.
# This means that per paper we may generate up to 100 prompots, e.g. 10 classifications with 10 classes each.

import sys
import pandas as pd
import argparse
import os
import yaml
import aiohttp
import asyncio
import random
from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Set, Tuple
from tqdm.asyncio import tqdm_asyncio

from helpers import validate_csv, load_models, load_api_key


class StructuredResponse(BaseModel, extra="forbid"):
    probability_decision: float = Field(
        description="A probability that the study belongs to the specified class"
    )


def assign_chosen_class_per_paper(
    paper_results_dict: Dict[str, Optional[float]], classification_names: Set[str]
) -> Dict[str, str]:
    """
    Analyzes the classification probabilities for a single paper and assigns a 'chosen' class
    for each classification type (e.g., 'classification1_chosen').

    The logic is:
    1. Group all class probabilities by their classification name.
    2. For each classification:
       - If at least one class has a probability >= 0.5, choose the class with the highest probability.
       - Otherwise, assign the class 'other'.

    Args:
        paper_results_dict: A dictionary of {probability_column_name: probability_value}.
        classification_names: A set of base classification names (e.g., {'classification1', 'classification2'}).

    Returns:
        A dictionary of {chosen_column_name: chosen_class_name} to be added to the paper's data.
    """
    chosen_classes = {}

    # Group probabilities by base classification name
    grouped_probs: Dict[str, List[Tuple[str, float]]] = {
        name: [] for name in classification_names
    }

    for col_name, probability in paper_results_dict.items():
        # Find the base classification name (e.g., from 'Classification_Type_Class_Name')
        for class_name in classification_names:
            if col_name.startswith(class_name):
                # Only consider non-None probabilities for decision making
                if probability is not None:
                    # Extract the specific class name from the column name
                    specific_class_name = col_name[len(class_name) + 1 :].replace(
                        "_", " "
                    )
                    grouped_probs[class_name].append((specific_class_name, probability))
                break

    # Apply the assignment logic for each classification type
    for class_name, probs_list in grouped_probs.items():
        chosen_col_name = f"{class_name}_chosen"

        # Filter for probabilities >= 0.5
        high_confidence_probs = [
            (cname, prob) for cname, prob in probs_list if prob >= 0.5
        ]

        if high_confidence_probs:
            # If any class has >= 0.5, pick the one with the highest probability
            # The key for max is the probability value (item[1])
            highest_class, _ = max(high_confidence_probs, key=lambda item: item[1])
            chosen_classes[chosen_col_name] = highest_class
        else:
            # Otherwise, auto-assign 'other'
            chosen_classes[chosen_col_name] = "other"

    return chosen_classes


def load_classification_criteria(yml_file: str) -> Dict:
    with open(yml_file, "r") as file:
        criteria = yaml.safe_load(file)
    return criteria


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
    if model.startswith("openai/"):
        schema = StructuredResponse.model_json_schema()
    else:
        schema = StructuredResponse.model_json_schema()
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "provider": {"order": ["google-vertex", "fireworks", "mistral"]},
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": "structured_response",
                "strict": True,
                "schema": schema,
            },
        },
    }
    async with semaphore:
        for attempt in range(max_retries):
            try:
                async with session.post(
                    url,
                    headers=headers,
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=120),
                ) as response:
                    if response.status == 429:
                        retry_after = int(response.headers.get("Retry-After", 5))
                        jitter = random.uniform(0, 5)
                        retry_after = retry_after + jitter + attempt * retry_after
                        print(
                            f"\nRate limited for model {model}. Retrying after {retry_after} seconds..."
                        )
                        await asyncio.sleep(retry_after)
                        continue
                    response.raise_for_status()
                    return await response.json()
            except Exception as e:
                print(f"\nAttempt {attempt + 1} failed for model {model}: {e}")
                await asyncio.sleep(2 ** (attempt + 2))
        return None


async def process_prompts_for_model(
    prompts: List[str],
    model: str,
    api_key: str,
    max_concurrent_per_model: int = 20,
) -> List[Optional[Dict]]:
    semaphore = asyncio.Semaphore(max_concurrent_per_model)
    async with aiohttp.ClientSession() as session:
        tasks = [
            call_openrouter_async(prompt, model, api_key, session, semaphore)
            for prompt in prompts
        ]
        return await tqdm_asyncio.gather(*tasks, desc=f"Processing {model}")


def generate_prompts(df: pd.DataFrame, criteria: Dict) -> List[str]:
    with open("prompt_classify.conf", "r") as file:
        prompt_template = file.read()

    prompts = []
    sr_context = criteria["Systematic review context"]

    for _, row in df.iterrows():
        title = row["title"]
        abstract = row["abstract"]

        for classification in criteria["Classifications"]:
            classification_name = classification["name"]
            for class_name in classification["classes"]:
                prompt = prompt_template.format(
                    title, abstract, sr_context, classification_name, class_name
                )
                prompts.append(prompt)

    return prompts


def get_unique_filename(base_path: str) -> str:
    """
    Generate a unique filename by appending a number if the file already exists.
    """
    base_dir, base_name = os.path.split(base_path)
    name, ext = os.path.splitext(base_name)

    counter = 1
    new_path = base_path
    while os.path.exists(new_path):
        new_path = os.path.join(base_dir, f"{name}_{counter}{ext}")
        counter += 1

    return new_path


def parse_results(
    prompts: List[str], results: List[Optional[Dict]], df: pd.DataFrame, criteria: Dict
) -> pd.DataFrame:
    """
    Parses the structured JSON results from the LLMs and adds the probability
    decision for each classification as new columns in the DataFrame, placing
    them *before* the original columns, and adds the newly assigned chosen class.

    Args:
        prompts: The list of prompts generated for the LLMs.
        results: The list of JSON responses from the LLMs.
        df: The original DataFrame of papers.
        criteria: The loaded classification criteria (to get class names).

    Returns:
        The updated DataFrame with new classification columns first.
    """
    if len(prompts) != len(results):
        print("Warning: Mismatch between number of prompts and results.")

    classification_data = []

    n_papers = len(df)

    # Get a set of base classification names for use in assign_chosen_class_per_paper
    classification_names = {
        c["name"].replace(" ", "_").replace("-", "_")
        for c in criteria["Classifications"]
    }

    classes_per_paper = sum(len(c["classes"]) for c in criteria["Classifications"])

    expected_prompt_count = n_papers * classes_per_paper
    if len(prompts) != expected_prompt_count:
        print(
            f"Error: Expected {expected_prompt_count} prompts, got {len(prompts)}. Results parsing may be inaccurate."
        )
        return df

    # Iterate through the results, one block per paper
    for i in range(n_papers):
        paper_results = results[i * classes_per_paper : (i + 1) * classes_per_paper]
        paper_results_dict = {}

        res_idx = 0
        for classification in criteria["Classifications"]:
            classification_name = classification["name"]
            for class_name in classification["classes"]:
                # Column name for the probability score
                col_name = f"{classification_name}_{class_name}".replace(
                    " ", "_"
                ).replace("-", "_")

                result = paper_results[res_idx]
                probability = None

                if result and "choices" in result and result["choices"]:
                    try:
                        content_str = result["choices"][0]["message"]["content"]
                        structured_response = StructuredResponse.model_validate_json(
                            content_str
                        )
                        probability = structured_response.probability_decision

                        if not (0.0 <= probability <= 1.0):
                            print(
                                f"Warning: Probability {probability} out of range for paper index {i}, class {col_name}. Setting to NaN."
                            )
                            probability = None

                    except Exception:
                        probability = None

                paper_results_dict[col_name] = probability
                res_idx += 1

        # --- NEW FEATURE: Assign the chosen class ---
        chosen_classes_dict = assign_chosen_class_per_paper(
            paper_results_dict, classification_names
        )

        # Merge probability and chosen class results
        final_paper_results = {**chosen_classes_dict, **paper_results_dict}
        classification_data.append(final_paper_results)
        # --- END NEW FEATURE ---

    # Convert the list of dictionaries to a DataFrame
    classification_df = pd.DataFrame(classification_data)

    # Concatenate the new classification columns (*including the chosen class columns*)
    # *before* the original DataFrame.
    # We put 'classification_df' first in the list passed to pd.concat.
    df = pd.concat(
        [classification_df.reset_index(drop=True), df.reset_index(drop=True)], axis=1
    )

    return df


def main():
    parser = argparse.ArgumentParser(description="Process papers from a CSV file.")
    parser.add_argument("csv_file", type=str, help="Path to the CSV file.")
    parser.add_argument(
        "-n",
        type=str,
        default="10",
        help="Number of papers to process (integer) or 'all' for all papers.",
    )
    parser.add_argument(
        "-p",
        type=float,
        default=None,
        help="Probability threshold for processed papers (0-1).",
    )
    args = parser.parse_args()

    api_key = load_api_key("~/openrouter.key")
    models = load_models("models_classify.conf")

    if args.n.lower() == "all":
        n_rows = None
    else:
        try:
            n_rows = int(args.n)
        except ValueError:
            sys.exit("Error: -n must be an integer or 'all'.")

    if args.p is not None and not (0 <= args.p <= 1):
        sys.exit("Error: -p must be between 0 and 1.")

    # Load the original CSV
    original_df = validate_csv(args.csv_file, n_rows=n_rows, require_avg_prob=args.p)
    n_papers = len(original_df)
    print(f"Processing {n_papers} papers.")

    criteria = load_classification_criteria("criteria_classify.yml")
    prompts = generate_prompts(original_df, criteria)
    print(f"Generated {len(prompts)} prompts.")
    # Calculate Total Characters
    total_characters = sum(len(prompt) for prompt in prompts)

    # Estimate Tokens (Total Characters / 4)
    estimated_tokens = total_characters // 4
    estimated_m_tokens = estimated_tokens / 1_000_000
    print(
        f"Estimated token count (for pricing input): **{estimated_m_tokens:.6f} M tokens**"
    )
    # Get the directory of the input file
    input_dir = os.path.dirname(os.path.abspath(args.csv_file))
    original_filename = os.path.splitext(os.path.basename(args.csv_file))[0]

    # Process prompts with each model
    for model in models:
        print(f"\nProcessing with model: {model}")
        results = asyncio.run(process_prompts_for_model(prompts, model, api_key))

        # Create a fresh copy of the original DataFrame for each model
        df = original_df.copy()
        df = parse_results(prompts, results, df, criteria)

        # Generate output filename and path
        model_name_clean = model.replace("/", "_")
        output_filename = (
            f"{original_filename}_LLM_classification_{model_name_clean}.csv"
        )
        output_path = os.path.join(input_dir, output_filename)
        output_path = get_unique_filename(output_path)

        df.to_csv(output_path, index=False)
        print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
