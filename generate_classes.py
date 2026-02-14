# Tries to generate classification classes using OpenRouter API and saves them to Markdown files.
# Based on trial runs results are not very good. It is suggesteded that your manayually create classification classes instead.
# Specify classes in criteria_classify.yml and use classify_papers.py to classify papers based on those classes.abs
# See criteria_classify.yml.example for an example of how to specify classes.

import sys
import argparse
import json
from openai.lib._pydantic import to_strict_json_schema
import requests
from pathlib import Path
from pydantic import BaseModel, Field
from typing import List
from helpers import validate_csv, load_models, load_api_key


class ClassRule(BaseModel):
    value: str = Field(description="Rule text, max 10 words")


class ClassificationClass(BaseModel):
    title: str = Field(description="Class title, max 10 words")
    criteria: List[ClassRule] = Field(description="List of 2-5 rules for the class")


class ClassificationGenerationResponse(BaseModel):
    classes: List[ClassificationClass] = Field(
        description="List of 3-10 mutually exclusive classes, each with a title and 2-5 criteria rules."
    )


def call_openrouter(prompt: str, model: str, api_key: str):
    """Call OpenRouter API with the given prompt and model."""
    url = "https://openrouter.ai/api/v1/chat/completions"
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    if model.startswith("openai/"):
        schema = to_strict_json_schema(ClassificationGenerationResponse)
    else:
        schema = ClassificationGenerationResponse.model_json_schema()

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
    response = requests.post(url, json=payload, headers=headers)
    response.raise_for_status()
    content = response.json()
    structured_response = content["choices"][0]["message"]["content"]
    # Parse the JSON string into a Python dict
    return json.loads(structured_response)


def save_response_to_md(
    response: dict, csv_path: Path, model: str, n_papers: int
) -> None:
    """Save only the classification results to a Markdown file, avoiding overwrites."""
    sanitized_model = "".join(
        c if c.isalnum() or c in ["-", "_"] else "_" for c in model
    )
    base_output_filename = csv_path.with_stem(
        f"{csv_path.stem}_n_{n_papers}_LLM_class_gen_{sanitized_model}"
    ).with_suffix(".md")

    # Check if file exists and find a non-conflicting name
    output_filename = base_output_filename
    counter = 1
    while output_filename.exists():
        output_filename = csv_path.with_stem(
            f"{csv_path.stem}_n_{n_papers}_LLM_class_gen_{sanitized_model}({counter})"
        ).with_suffix(".md")
        counter += 1

    with open(output_filename, "w") as md_file:
        md_file.write("# Classes\n\n")
        for class_data in response.get("classes", []):
            md_file.write(f"## {class_data['title']}\n\n")
            md_file.write("- **Criteria:**\n")
            for rule in class_data.get("criteria", []):
                md_file.write(f"  - {rule['value']}\n")
            md_file.write("\n")
    print(f"Response saved to: {output_filename}")


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
    models = load_models("models_generate_classes.conf")

    # Parse -n
    if args.n.lower() == "all":
        n_rows = None
    else:
        try:
            n_rows = int(args.n)
        except ValueError:
            sys.exit("Error: -n must be an integer or 'all'.")

    # Validate -p
    if args.p is not None and not (0 <= args.p <= 1):
        sys.exit("Error: -p must be between 0 and 1.")

    # Load and filter data
    df = validate_csv(args.csv_file, n_rows=n_rows, require_avg_prob=args.p)
    n_papers = len(df)
    print(f"Processing {n_papers} papers.")
    # Continue with your processing logic here

    with open("prompt_generate_classes.conf", "r") as file:
        prompt_template = file.read()

    studies = []
    for _, row in df.iterrows():
        studies.append(f"Title: {row['title']}\nAbstract: {row['abstract']}")
    studies_str = "\n\n".join(studies)

    prompt = prompt_template.format(studies_str)

    # csv_path = Path(args.csv_file)
    # base_filename = csv_path.stem  # e.g., "data" for "data.csv"

    # Count tokens
    token_count = len(prompt) // 4  # Rough estimate: 1 token ~ 4 characters
    print(f"Estimated Prompt size: {token_count} tokens")
    for model in models:
        print(f"Calling model: {model}")
        response = call_openrouter(prompt, model, api_key)
        save_response_to_md(response, Path(args.csv_file), model, n_papers)


if __name__ == "__main__":
    main()
