import sys
import pandas as pd
import argparse
import os
import yaml
import asyncio
from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Tuple
from pydantic_ai.output import ToolOutput

from helpers import validate_csv, load_models, load_api_key, get_unique_filename
from async_api import create_agent, process_batch_agent

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


# --- 2. Prompt generation ---
def generate_prompts(df: pd.DataFrame, criteria: Dict) -> List[Tuple[str, List[str]]]:
    """
    Generates one prompt per paper per classification type.
    Returns list of (prompt, class_options) tuples to group by classification type.
    """
    with open("prompts/prompt_classify_single.txt", "r") as file:
        prompt_template = file.read()

    tasks: List[Tuple[str, List[str]]] = []
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

            prompt = prompt_template.format(
                title,
                abstract,
                sr_context,
                classification_name,
                class_list_string,
            )
            tasks.append((prompt, class_options))

    return tasks


# --- 3. parse_results ---
def parse_results(
    tasks: List[Tuple[str, List[str]]],
    results: List[Optional[BaseModel]],
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

            if result is not None:
                try:
                    chosen_class = result.chosen_class
                    class_options = classification["classes"] + ["other"]

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


async def process_tasks_for_model(
    tasks: List[Tuple[str, List[str]]],
    model_name: str,
    api_key: str,
) -> List[Optional[BaseModel]]:
    """
    Process tasks grouped by classification type.
    Creates one Agent per unique set of class options.
    """
    # Group tasks by class_options (as tuple for hashing)
    groups: Dict[tuple, List[Tuple[int, str]]] = {}
    for idx, (prompt, class_options) in enumerate(tasks):
        key = tuple(class_options)
        if key not in groups:
            groups[key] = []
        groups[key].append((idx, prompt))

    # Process each group with its own Agent
    all_results: List[Optional[BaseModel]] = [None] * len(tasks)

    for class_options_tuple, indexed_prompts in groups.items():
        class_options = list(class_options_tuple)
        ResponseModel = create_structured_response_model(class_options)
        agent = create_agent(
            model_name,
            api_key,
            output_type=ToolOutput(ResponseModel, name="structured_response"),
        )
        prompts = [p for _, p in indexed_prompts]
        indices = [i for i, _ in indexed_prompts]

        results = await process_batch_agent(prompts, agent, model_name)

        for orig_idx, result in zip(indices, results):
            all_results[orig_idx] = result

    return all_results


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

        results = asyncio.run(process_tasks_for_model(tasks, model, api_key))

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
