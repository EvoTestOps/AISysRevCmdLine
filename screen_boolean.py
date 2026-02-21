# Screens papers using per-criterion LLM calls and fuzzy boolean logic.
# Unlike screen.py (which sends all criteria in one prompt), this script:
#   1. Reads criteria_screen_boolean.yml — a tree of inclusion/exclusion criteria with AND/OR operators
#   2. Sends each criterion to the LLM separately as a neutral question (no IC/EC label)
#   3. Combines per-criterion probabilities using Fuzzy Logic:
#        AND = MIN(A, B)    OR = MAX(A, B)    NOT = 1 - A  (applied to exclusion branch)
#   4. Derives binary_decision from overall_probability >= 0.5

import argparse
import asyncio
import logging
import sys
import yaml
import pandas as pd
from typing import Any, Dict, List, Optional, Tuple
from pydantic import BaseModel, Field
from pydantic_ai.output import ToolOutput
from pathlib import Path

from async_api import process_all_models_agent
from helpers import validate_csv, load_api_key, load_models, get_unique_filename

# --- Logging ---
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

system_prompt = "You are an expert research assistant."


# --- Pydantic model ---
# Simple: one response per criterion. No IC/EC labeling — LLM only sees the description.
class CriterionResponse(BaseModel, extra="forbid"):
    probability_decision: float = Field(
        description="The likelihood, that the criterion applies or the primary study is relevant. "
                    "A float between 0.000 and 1.000: closer to 1.000 means extremely likely (very strong match), "
                    "closer to 0.000 means extremely unlikely (very weak or no match). "
                    "Use intermediate values, not just 0.000 or 1.000."
    )
    reason: str = Field(description="Reasoning for the probability estimate.")


# --- Criteria loading ---

def load_criteria(yml_path: str) -> dict:
    """Load criteria YAML (inclusion/exclusion tree with boolean operators)."""
    try:
        with open(yml_path, "r") as f:
            return yaml.safe_load(f)
    except FileNotFoundError:
        sys.exit(f"Error: Criteria file '{yml_path}' not found.")


def extract_leaf_criteria(node: dict) -> list[dict]:
    """Recursively collect leaf criteria nodes (those with 'id' + 'description') in DFS order."""
    if "id" in node:
        return [node]
    result = []
    for child in node.get("criteria", []):
        result.extend(extract_leaf_criteria(child))
    return result


# --- Prompt generation ---

def generate_prompts(df: pd.DataFrame, leaf_criteria: list[dict], prompt_template: str) -> list[str]:
    """Generate N_papers × N_criteria prompts in paper-major order.
    Each prompt asks neutrally whether one criterion applies — no IC/EC label."""
    prompts = []
    for _, row in df.iterrows():
        for crit in leaf_criteria:
            prompts.append(
                prompt_template.format(row["title"], row["abstract"], crit["description"])
            )
    return prompts


# --- Fuzzy logic ---

def fuzzy_eval(node: dict, criterion_probs: dict[str, Optional[float]]) -> Optional[float]:
    """Recursively evaluate the criteria tree using fuzzy logic.
    Leaf: return criterion probability (negated if negate=true).
    Group: AND = MIN, OR = MAX of child probabilities (None children skipped).
    """
    if "id" in node:
        prob = criterion_probs.get(node["id"])
        if prob is None:
            return None
        return round(1.0 - prob, 4) if node.get("negate", False) else round(prob, 4)
    # group node
    child_probs = [
        p for child in node.get("criteria", [])
        if (p := fuzzy_eval(child, criterion_probs)) is not None
    ]
    if not child_probs:
        return None
    op = node.get("operator", "AND").upper()
    return round(min(child_probs), 4) if op == "AND" else round(max(child_probs), 4)


def compute_overall(
    criteria_yml: dict,
    criterion_probs: dict[str, Optional[float]],
) -> Tuple[Optional[float], Optional[float], Optional[float], Optional[bool]]:
    """Return (inclusion_prob, exclusion_prob, overall_prob, binary_decision)."""
    incl = fuzzy_eval(criteria_yml["inclusion"], criterion_probs) if "inclusion" in criteria_yml else None
    excl = fuzzy_eval(criteria_yml["exclusion"], criterion_probs) if "exclusion" in criteria_yml else None

    if incl is not None and excl is not None:
        overall = round(min(incl, 1.0 - excl), 4)   # fuzzy AND(inclusion, NOT exclusion)
    elif incl is not None:
        overall = incl
    elif excl is not None:
        overall = round(1.0 - excl, 4)
    else:
        overall = None

    binary = (overall >= 0.5) if overall is not None else None
    return incl, excl, overall, binary


# --- Result processing ---

def generate_unique_model_keys(models: List[str]) -> List[str]:
    seen: Dict[str, int] = {}
    unique_keys = []
    for model in models:
        if model not in seen:
            seen[model] = 1
            unique_keys.append(model)
        else:
            seen[model] += 1
            unique_keys.append(f"{model}_{seen[model]}")
    return unique_keys


class ColResolver:
    """Resolves column names to avoid overwriting pre-existing columns.
    If the intended name already exists in the input DataFrame, appends _2, _3, ...
    until a free name is found. Caches all resolutions so every row uses the same name."""

    def __init__(self, existing_cols: set[str]):
        self._pre_run = existing_cols
        self._map: dict[str, str] = {}

    def resolve(self, col: str) -> str:
        if col in self._map:
            return self._map[col]
        allocated = set(self._map.values())
        candidate = col
        counter = 2
        while candidate in self._pre_run or candidate in allocated:
            candidate = f"{col}_{counter}"
            counter += 1
        self._map[col] = candidate
        return candidate


def process_results_boolean(
    model_results: List[List[Optional[Any]]],
    df: pd.DataFrame,
    models: List[str],
    model_keys: List[str],
    leaf_criteria: list[dict],
    criteria_yml: dict,
    resolver: ColResolver,
) -> Tuple[pd.DataFrame, Dict[str, Tuple[int, int]], List[str]]:
    """Parse per-criterion LLM results, apply fuzzy logic, write columns to df.
    Returns (df, stats, overall_prob_cols) — overall_prob_cols are the resolved
    _overall_probability column names, passed on to add_overall_probability_stats."""
    n_papers = len(df)
    n_criteria = len(leaf_criteria)
    stats = {}
    overall_prob_cols: List[str] = []

    for model_idx, model in enumerate(models):
        unique_key = model_keys[model_idx]
        logger.info(f"Merging results for model: {unique_key}")
        print(f"\nMerging results for model: {unique_key}")
        results = model_results[model_idx]  # flat list of n_papers * n_criteria items

        # Resolve all output column names for this model upfront — consistent across all papers
        prob_cols  = {c["id"]: resolver.resolve(f"{unique_key}_{c['id']}_probability") for c in leaf_criteria}
        reason_cols = {c["id"]: resolver.resolve(f"{unique_key}_{c['id']}_reason")      for c in leaf_criteria}
        incl_col   = resolver.resolve(f"{unique_key}_inclusion_probability")
        excl_col   = resolver.resolve(f"{unique_key}_exclusion_probability")
        overall_col = resolver.resolve(f"{unique_key}_overall_probability")
        binary_col  = resolver.resolve(f"{unique_key}_binary_decision")
        overall_prob_cols.append(overall_col)

        successes = 0
        failures = 0

        for paper_i in range(n_papers):
            paper_results = results[paper_i * n_criteria : (paper_i + 1) * n_criteria]
            criterion_probs: dict[str, Optional[float]] = {}

            for crit_j, crit in enumerate(leaf_criteria):
                crit_id = crit["id"]
                result = paper_results[crit_j]

                if result is not None:
                    try:
                        prob = result.probability_decision
                        reason = result.reason
                        if not (0.0 <= prob <= 1.0):
                            logger.warning(
                                f"Probability {prob} out of range for paper {paper_i}, {crit_id}. Setting to None."
                            )
                            prob = None
                        criterion_probs[crit_id] = prob
                        _set_col(df, paper_i, prob_cols[crit_id], prob)
                        _set_col(df, paper_i, reason_cols[crit_id], reason)
                        if prob is not None:
                            successes += 1
                        else:
                            failures += 1
                    except Exception as e:
                        logger.error(f"Failed to parse result for paper {paper_i}, {crit_id}: {e}")
                        criterion_probs[crit_id] = None
                        _set_col(df, paper_i, resolver.resolve(f"{unique_key}_{crit_id}_error"), str(e))
                        failures += 1
                else:
                    criterion_probs[crit_id] = None
                    _set_col(df, paper_i, resolver.resolve(f"{unique_key}_{crit_id}_error"), "No response from API")
                    failures += 1

            # Apply fuzzy boolean logic across this paper's criteria
            incl, excl, overall, binary = compute_overall(criteria_yml, criterion_probs)
            _set_col(df, paper_i, incl_col, incl)
            _set_col(df, paper_i, excl_col, excl)
            _set_col(df, paper_i, overall_col, overall)
            _set_col(df, paper_i, binary_col, binary)

        stats[unique_key] = (successes, failures)

    return df, stats, overall_prob_cols


def _set_col(df: pd.DataFrame, row_i: int, col: str, value: Any) -> None:
    """Write a value to df at row_i for col, creating the column if needed."""
    if col not in df.columns:
        df[col] = None
    df.at[row_i, col] = value


# --- Cross-model aggregation ---

def add_overall_probability_stats(
    df: pd.DataFrame, overall_prob_cols: List[str], resolver: ColResolver
) -> pd.DataFrame:
    """Compute average/min/max overall_probability across models and prepend as columns."""
    avg_col = resolver.resolve("average_overall_probability")
    min_col = resolver.resolve("min_overall_probability")
    max_col = resolver.resolve("max_overall_probability")
    agg_cols = [avg_col, min_col, max_col]
    for col in agg_cols:
        df[col] = None

    for i, row in df.iterrows():
        probs = [row[col] for col in overall_prob_cols if col in df.columns and pd.notna(row[col])]
        if probs:
            df.at[i, avg_col] = round(sum(probs) / len(probs), 4)
            df.at[i, min_col] = round(min(probs), 4)
            df.at[i, max_col] = round(max(probs), 4)

    front = agg_cols + [c for c in df.columns if c not in agg_cols]
    return df[front]


# --- Output ---

def generate_output_filename(input_csv: str, criteria_path: str, models_path: str) -> str:
    input_path = Path(input_csv)
    base = f"{input_path.stem}_{Path(criteria_path).stem}_{Path(models_path).stem}"
    output = input_path.parent / f"{base}.csv"
    counter = 1
    while output.exists():
        output = input_path.parent / f"{base}_{counter:02d}.csv"
        counter += 1
    return str(output)


def save_enriched_csv(df: pd.DataFrame, output_file: str) -> None:
    df.to_csv(output_file, index=False)
    logger.info(f"Enriched data saved to {output_file}")
    print(f"\nEnriched data saved to {output_file}")


# --- Main ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Screen papers per-criterion with fuzzy boolean logic."
    )
    parser.add_argument("csv_file", help="Path to the input CSV file.")
    parser.add_argument(
        "-n", "--n_rows", default="10",
        help="Number of rows to process (default: 10). Use 'all' for the entire file.",
    )
    parser.add_argument(
        "-c", "--criteria", default="criteria_screen_boolean.yml",
        help="Path to the boolean criteria YAML file (default: criteria_screen_boolean.yml)",
    )
    parser.add_argument(
        "-m", "--models", default="models.conf",
        help="Path to the models list file (default: models.conf)",
    )
    args = parser.parse_args()

    n_rows = None if args.n_rows.lower() == "all" else int(args.n_rows)

    api_key = load_api_key("~/openrouter.key")
    models = load_models(args.models)
    criteria_yml = load_criteria(args.criteria)

    try:
        with open("prompts/prompt_boolean.txt", "r") as f:
            prompt_template = f.read()
    except FileNotFoundError:
        sys.exit("Error: 'prompts/prompt_boolean.txt' not found.")

    leaf_inclusion = extract_leaf_criteria(criteria_yml.get("inclusion", {}))
    leaf_exclusion = extract_leaf_criteria(criteria_yml.get("exclusion", {}))
    all_leaf_criteria = leaf_inclusion + leaf_exclusion

    print(f"Config: Rows={args.n_rows}, Criteria={args.criteria}, Models={args.models}")
    print(f"Criteria: {len(leaf_inclusion)} inclusion, {len(leaf_exclusion)} exclusion "
          f"({len(all_leaf_criteria)} total leaf criteria)")

    df = validate_csv(args.csv_file, n_rows=n_rows)
    print(f"In total {len(df)} articles.")
    logger.info(f"In total {len(df)} articles.")

    prompts = generate_prompts(df, all_leaf_criteria, prompt_template)
    print(f"Generated {len(prompts)} prompts ({len(df)} papers * {len(all_leaf_criteria)} criteria).")
    logger.info("Generating prompts")

    df = df.copy().reset_index(drop=True)
    model_keys = generate_unique_model_keys(models)
    resolver = ColResolver(set(df.columns))

    model_results = asyncio.run(
        process_all_models_agent(
            prompts,
            models,
            api_key,
            system_prompt=system_prompt,
            output_type=ToolOutput(CriterionResponse, name="criterion_response"),
            max_concurrent_per_model=20,
        )
    )

    df, stats, overall_prob_cols = process_results_boolean(
        model_results, df, models, model_keys, all_leaf_criteria, criteria_yml, resolver
    )
    df = add_overall_probability_stats(df, overall_prob_cols, resolver)
    output_file = generate_output_filename(args.csv_file, args.criteria, args.models)
    save_enriched_csv(df, output_file)

    print("\nModel statistics (criterion-level calls):")
    for model_key, (success, failure) in stats.items():
        print(f"  {model_key}: {success} successes, {failure} failures")
