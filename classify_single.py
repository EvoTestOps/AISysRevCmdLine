import sys
import pandas as pd
import argparse
import os
import yaml
import aiohttp
import asyncio
import random
from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Tuple
from tqdm.asyncio import tqdm_asyncio

from helpers import validate_csv, load_models, load_api_key

# --- 1. Structured Response Model Creator ---


def create_structured_response_model(classification_classes: List[str]) -> type(
    BaseModel
):
    """Dynamically create a Pydantic model for the LLM's single-class selection."""
    class_names_str = ", ".join([f"'{c}'" for c in classification_classes])

    class SingleClassResponse(BaseModel, extra="forbid"):
        chosen_class: str = Field(
            description=f"The single class that best matches the paper's abstract and title. Must be one of: {class_names_str}, or 'other' if none match."
        )

    return SingleClassResponse


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
    structured_schema: Dict = None,
) -> Optional[Dict]:
    url = "https://openrouter.ai/api/v1/chat/completions"
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}

    schema = structured_schema if structured_schema is not None else {}

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
    tasks: List[Tuple[str, Dict]],
    model: str,
    api_key: str,
    max_concurrent_per_model: int = 20,
) -> List[Optional[Dict]]:
    semaphore = asyncio.Semaphore(max_concurrent_per_model)
    async with aiohttp.ClientSession() as session:
        api_call_tasks = [
            call_openrouter_async(
                prompt, model, api_key, session, semaphore, structured_schema=schema
            )
            for prompt, schema in tasks
        ]
        return await tqdm_asyncio.gather(*api_call_tasks, desc=f"Processing {model}")


# --- 2. Final, Simplified generate_prompts (Optimized for your template) ---
def generate_prompts(df: pd.DataFrame, criteria: Dict) -> List[Tuple[str, Dict]]:
    """
    Generates one prompt per paper per classification type, passing the full
    list of classes to the {4} placeholder as per the user's template structure.
    """
    # NOTE: Ensure 'prompt_classify_single.conf' is in the current directory
    with open("prompt_classify_single.conf", "r") as file:
        prompt_template = file.read()

    tasks: List[Tuple[str, Dict]] = []
    sr_context = criteria["Systematic review context"]

    for _, row in df.iterrows():
        title = row["title"]
        abstract = row["abstract"]

        for classification in criteria["Classifications"]:
            classification_name = classification["name"]

            # All classes + 'other' are options
            class_options = classification["classes"] + ["other"]

            # The list of classes is formatted as a string for the prompt
            class_list_string = ", ".join([f"'{c}'" for c in classification["classes"]])

            # Create the dynamic Pydantic model and its schema
            ResponseModel = create_structured_response_model(class_options)
            schema = ResponseModel.model_json_schema()

            # The arguments correspond to the placeholders in your prompt template:
            # {0}: title, {1}: abstract, {2}: sr_context, {3}: classification_name
            # {4}: class_list_string (the list of options)
            prompt = prompt_template.format(
                title,
                abstract,
                sr_context,
                classification_name,
                class_list_string,  # This will fill the "### Class: {4}" line
            )
            tasks.append((prompt, schema))

    return tasks


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


# --- 3. parse_results (Unchanged, as it was correct for the single-choice logic) ---
def parse_results(
    tasks: List[Tuple[str, Dict]],
    results: List[Optional[Dict]],
    df: pd.DataFrame,
    criteria: Dict,
) -> pd.DataFrame:
    if len(tasks) != len(results):
        print(
            "Warning: Mismatch between number of tasks and results. Skipping remaining papers."
        )
        results = results[: len(tasks)]

    classification_data = []

    n_papers = len(df)

    classification_objects = criteria["Classifications"]
    n_classifications_per_paper = len(classification_objects)

    expected_result_count = n_papers * n_classifications_per_paper
    if len(results) != expected_result_count:
        print(
            f"Error: Expected {expected_result_count} results, got {len(results)}. Results parsing may be inaccurate."
        )

    for i in range(n_papers):
        start_index = i * n_classifications_per_paper
        end_index = (i + 1) * n_classifications_per_paper
        paper_results = results[start_index:end_index]

        if len(paper_results) < n_classifications_per_paper:
            print(
                f"Warning: Only {len(paper_results)} results found for paper {i}. Filling remaining with 'N/A'."
            )

        paper_results_dict = {}

        for res_idx, classification in enumerate(classification_objects):
            classification_raw_name = classification["name"]
            classification_name = classification_raw_name.replace(" ", "_").replace(
                "-", "_"
            )
            chosen_col_name = f"{classification_name}_chosen"

            result = paper_results[res_idx] if res_idx < len(paper_results) else None
            chosen_class = "API_FAILED_OR_INVALID"

            if result and "choices" in result and result["choices"]:
                try:
                    content_str = result["choices"][0]["message"]["content"]

                    class_options = classification["classes"] + ["other"]
                    ResponseModel = create_structured_response_model(class_options)

                    structured_response = ResponseModel.model_validate_json(content_str)
                    chosen_class = structured_response.chosen_class

                    if chosen_class not in class_options:
                        print(
                            f"Warning: LLM returned invalid class '{chosen_class}' for paper index {i}, classification {classification_name}. Setting to 'other'."
                        )
                        chosen_class = "other"

                except Exception:
                    chosen_class = "other"

            paper_results_dict[chosen_col_name] = chosen_class

        classification_data.append(paper_results_dict)

    classification_df = pd.DataFrame(classification_data)
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

    original_df = validate_csv(args.csv_file, n_rows=n_rows, require_avg_prob=args.p)
    n_papers = len(original_df)
    print(f"Processing {n_papers} papers.")

    criteria = load_classification_criteria("criteria_classify.yml")

    tasks = generate_prompts(original_df, criteria)
    print(f"Generated {len(tasks)} prompts.")

    total_characters = sum(len(prompt) for prompt, _ in tasks)
    estimated_tokens = total_characters // 4
    estimated_m_tokens = estimated_tokens / 1_000_000
    print(
        f"Estimated token count (for pricing input): **{estimated_m_tokens:.6f} M tokens**"
    )

    input_dir = os.path.dirname(os.path.abspath(args.csv_file))
    original_filename = os.path.splitext(os.path.basename(args.csv_file))[0]

    for model in models:
        print(f"\nProcessing with model: {model}")

        results = asyncio.run(process_prompts_for_model(tasks, model, api_key))

        df = original_df.copy()
        df = parse_results(tasks, results, df, criteria)

        model_name_clean = model.replace("/", "_")
        output_filename = f"{original_filename}_LLM_single_choice_classification_{model_name_clean}.csv"
        output_path = os.path.join(input_dir, output_filename)
        output_path = get_unique_filename(output_path)

        df.to_csv(output_path, index=False)
        print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
