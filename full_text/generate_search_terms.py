# Generates search terms and synonyms for each inclusion/exclusion criterion in a
# criteria YAML file by querying OpenRouter LLMs.
#
# Usage (from project root or full_text/):
#   python full_text/generate_search_terms.py <yml_file> [-m models.conf] [-o output.yml]
#
# Output: a new YAML file identical in structure to the input, with each leaf
# criterion augmented by a `search_terms` block keyed by model ID.

import argparse
import asyncio
import logging
import sys
import yaml
from pathlib import Path
from typing import List, Optional
from pydantic import BaseModel, Field
from pydantic_ai.output import ToolOutput

# Project root is one level up from this script's directory
sys.path.insert(0, str(Path(__file__).parent.parent))

from async_api import process_all_models_agent
from helpers import load_api_key, load_models, get_unique_filename

# --- Logging ---
logging.getLogger().handlers.clear()
root = logging.getLogger()
root.setLevel(logging.INFO)
fmt = logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
file_handler = logging.FileHandler("app.log", mode="a", encoding="utf-8")
file_handler.setLevel(logging.INFO)
file_handler.setFormatter(fmt)
console_handler = logging.StreamHandler(sys.stderr)
console_handler.setLevel(logging.ERROR)
console_handler.setFormatter(fmt)
root.addHandler(file_handler)
root.addHandler(console_handler)
logger = logging.getLogger(__name__)

system_prompt = "You are an expert in systematic literature reviews."

_SCRIPT_DIR = Path(__file__).parent
_PROMPT_PATH = _SCRIPT_DIR.parent / "prompts" / "prompt_search_terms.txt"
_DEFAULT_MODELS = str(_SCRIPT_DIR / "models.conf")


# --- Pydantic model ---

class SearchTermsResponse(BaseModel, extra="forbid"):
    search_terms: List[str] = Field(
        description="List of search terms and synonyms for the criterion. "
                    "Each entry is a concise word or short phrase an author might use."
    )


# --- YAML loading & traversal ---

def load_criteria(yml_path: str) -> dict:
    try:
        with open(yml_path, "r") as f:
            return yaml.safe_load(f)
    except FileNotFoundError:
        sys.exit(f"Error: Criteria file '{yml_path}' not found.")


def extract_leaf_criteria(node: dict) -> list[dict]:
    """Recursively collect leaf criterion nodes (those with 'id') in DFS order."""
    if "id" in node:
        return [node]
    result = []
    for child in node.get("criteria", []):
        result.extend(extract_leaf_criteria(child))
    return result


# --- Prompt generation ---

def generate_prompts(title: str, abstract: str, leaf_criteria: list[dict], prompt_template: str) -> list[str]:
    """One prompt per criterion."""
    return [prompt_template.format(title, abstract, c["description"]) for c in leaf_criteria]


# --- Result injection ---

def build_results_map(models: list[str], leaf_criteria: list[dict],
                      all_model_results: list[list[Optional[SearchTermsResponse]]]) -> dict:
    """Build {criterion_id: {model_name: [terms]}} from process_all_models_agent output."""
    results_by_id: dict = {c["id"]: {} for c in leaf_criteria}
    for model_idx, model_name in enumerate(models):
        model_results = all_model_results[model_idx]
        for crit_idx, crit in enumerate(leaf_criteria):
            response = model_results[crit_idx]
            if response is not None:
                results_by_id[crit["id"]][model_name] = response.search_terms
            else:
                logger.warning("No result for model=%s criterion=%s", model_name, crit["id"])
    return results_by_id


def inject_search_terms(leaf_criteria: list[dict], results_by_id: dict) -> None:
    """Add search_terms in-place to each leaf criterion dict."""
    for crit in leaf_criteria:
        crit["search_terms"] = results_by_id.get(crit["id"], {})


# --- Main ---

async def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate search terms for each criterion in a criteria YAML via OpenRouter LLMs."
    )
    parser.add_argument("yml_file", help="Path to the criteria YAML file")
    parser.add_argument("-m", "--models", default=_DEFAULT_MODELS,
                        help=f"Models config file (default: {_DEFAULT_MODELS})")
    parser.add_argument("-o", "--output", default=None,
                        help="Output YAML path (default: <input>_terms.yml)")
    args = parser.parse_args()

    # Resolve output path
    input_path = Path(args.yml_file)
    if args.output:
        output_path = args.output
    else:
        default_out = input_path.parent / f"{input_path.stem}_terms.yml"
        output_path = get_unique_filename(str(default_out))

    # Load inputs
    api_key = load_api_key("~/openrouter.key")
    models = load_models(args.models)
    criteria_yml = load_criteria(args.yml_file)

    title = criteria_yml.get("title", "")
    abstract = criteria_yml.get("abstract", "")
    if not title:
        sys.exit("Error: 'title' field missing from YAML.")
    if not abstract:
        sys.exit("Error: 'abstract' field missing from YAML.")

    # Extract all leaf criteria from inclusion and exclusion branches
    leaf_criteria: list[dict] = []
    for section in ("inclusion", "exclusion"):
        if section in criteria_yml:
            leaf_criteria.extend(extract_leaf_criteria(criteria_yml[section]))

    if not leaf_criteria:
        sys.exit("Error: No leaf criteria found in YAML.")

    print(f"Found {len(leaf_criteria)} criteria: {[c['id'] for c in leaf_criteria]}")
    print(f"Querying {len(models)} model(s): {models}")

    # Load prompt template
    try:
        prompt_template = _PROMPT_PATH.read_text(encoding="utf-8")
        lines = prompt_template.splitlines()
        content_lines = [l for l in lines if not l.startswith("#")]
        prompt_template = "\n".join(content_lines).strip()
    except FileNotFoundError:
        sys.exit(f"Error: Prompt template not found at {_PROMPT_PATH}")

    prompts = generate_prompts(title, abstract, leaf_criteria, prompt_template)

    # Query LLMs
    all_model_results = await process_all_models_agent(
        prompts=prompts,
        models=models,
        api_key=api_key,
        system_prompt=system_prompt,
        output_type=ToolOutput(SearchTermsResponse, name="search_terms_response"),
    )

    # Inject results into YAML tree
    results_by_id = build_results_map(models, leaf_criteria, all_model_results)
    inject_search_terms(leaf_criteria, results_by_id)

    # Write output YAML
    with open(output_path, "w", encoding="utf-8") as f:
        yaml.dump(criteria_yml, f, default_flow_style=False, allow_unicode=True, sort_keys=False)

    print(f"Saved: {output_path}")


if __name__ == "__main__":
    asyncio.run(main())
