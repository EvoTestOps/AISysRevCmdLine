# generate_embeddings.py

import sys
import pandas as pd
import argparse
import os
import random
import aiohttp
import asyncio
from typing import List, Optional
from tqdm.asyncio import tqdm_asyncio

# Assuming 'helpers' is in the same directory or accessible
import helpers

# --- I/O and Network Functions (Kept from original) ---


async def call_openrouter_embedding_async(
    text_input: str,
    model: str,
    api_key: str,
    session: aiohttp.ClientSession,
    semaphore: asyncio.Semaphore,
    max_retries: int = 3,
) -> Optional[List[float]]:
    """Calls OpenRouter's embedding endpoint and returns the vector."""
    url = "https://openrouter.ai/api/v1/embeddings"
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}

    payload = {"model": model, "input": text_input, "encoding_format": "float"}

    async with semaphore:
        for attempt in range(max_retries):
            try:
                # Use a standard timeout for API calls
                async with session.post(
                    url,
                    headers=headers,
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=60),
                ) as response:
                    if response.status == 429:
                        retry_after = int(response.headers.get("Retry-After", 5))
                        jitter = random.uniform(0, 5)
                        retry_after = retry_after + jitter + attempt * retry_after
                        print(
                            f"\nRate limited for embedding model {model}. Retrying after {retry_after:.2f} seconds..."
                        )
                        await asyncio.sleep(retry_after)
                        continue

                    response.raise_for_status()

                    result = await response.json()

                    if result and "data" in result and len(result["data"]) > 0:
                        return result["data"][0]["embedding"]
                    return None

            except Exception as e:
                # Exponential backoff for other errors
                print(
                    f"\nAttempt {attempt + 1} failed for embedding model {model}: {e}"
                )
                await asyncio.sleep(2 ** (attempt + 2))
        return None


async def process_texts_for_embedding(
    texts: List[str],
    model: str,
    api_key: str,
    max_concurrent_per_model: int = 20,
) -> List[Optional[List[float]]]:
    """Processes texts for embedding creation using concurrent async calls."""
    semaphore = asyncio.Semaphore(max_concurrent_per_model)
    # Set a higher limit for aiohttp.ClientSession to prevent 'Too many open files' error
    conn = aiohttp.TCPConnector(limit=max_concurrent_per_model * 2)
    async with aiohttp.ClientSession(connector=conn) as session:
        tasks = [
            call_openrouter_embedding_async(text, model, api_key, session, semaphore)
            for text in texts
        ]
        return await tqdm_asyncio.gather(*tasks, desc=f"Embedding with {model}")


def generate_prompts(df: pd.DataFrame) -> List[str]:
    """
    Generates a list of strings (texts) to be embedded using the 'criteria_embed.md' prefix.
    """
    inputs = []

    prompt_prefix = ""
    try:
        # NOTE: This file is assumed to exist in the same run directory
        with open("criteria_embed.md", "r") as file:
            # Strip lines and ignore lines starting with '#' (Markdown headers)
            content_lines = [line.rstrip() for line in file if not line.startswith("#")]
            prompt_prefix = "\n".join(content_lines)

    except FileNotFoundError:
        print(
            "Warning: 'criteria_embed.md' not found. Using an empty prefix for embeddings."
        )

    for _, row in df.iterrows():
        title = row.get("title", "")
        abstract = row.get("abstract", "")

        # Structure the input for the LLM embedding
        combined_text = f"{prompt_prefix}\n\nTITLE: {title}\nABSTRACT: {abstract}"
        inputs.append(combined_text)

    return inputs


# --- Main Logic for Embedding Generation ---


def main():
    parser = argparse.ArgumentParser(
        description="Generate LLM embeddings for papers from a CSV file using OpenRouter."
    )
    parser.add_argument("csv_file", type=str, help="Path to the input CSV file.")
    parser.add_argument(
        "-n",
        type=str,
        default="all",
        help="Number of papers to process (integer) or 'all' for all papers.",
    )
    parser.add_argument(
        "-p",
        type=float,
        default=None,
        help="Probability threshold (0-1) for filtering input papers before embedding.",
    )

    args = parser.parse_args()

    # --- Input Validation ---
    if args.p is not None and not (0 <= args.p <= 1):
        sys.exit("Error: -p (pre-processing threshold) must be between 0 and 1.")

    # Load API Key and set the specific embedding model
    api_key = helpers.load_api_key("~/openrouter.key")
    # Assuming load_models returns a list, and we need the first one
    embedding_model = helpers.load_models("models_embed.md")[0]

    # Determine number of rows
    n_rows = int(args.n) if args.n.lower() != "all" else None

    PROBABILITY_COLUMN = "average_probability"

    # Load the original CSV with pre-filtering
    original_df = helpers.validate_csv(
        args.csv_file,
        n_rows=n_rows,
        require_avg_prob=args.p,
    )

    # Check for probability column if filtering is used
    if (args.p is not None) and (PROBABILITY_COLUMN not in original_df.columns):
        sys.exit(
            f"Error: DataFrame must contain a '{PROBABILITY_COLUMN}' column when using -p."
        )

    n_papers = len(original_df)
    if n_papers == 0:
        sys.exit("No papers remaining after filters. Exiting.")

    print(f"Processing {n_papers} papers after applying -p filter (if any).")

    # Generate the texts to be embedded
    embedding_inputs = generate_prompts(original_df)
    print(f"Generated {len(embedding_inputs)} inputs for embedding.")

    # Process prompts with the embedding model
    print(f"\nProcessing with embedding model: {embedding_model}")
    embeddings = asyncio.run(
        process_texts_for_embedding(embedding_inputs, embedding_model, api_key)
    )

    # Create a fresh copy of the original DataFrame to store results
    df = original_df.copy()

    # Store the resulting vectors as a new column
    # Clean up model name for safe column and file naming
    embedding_col_name = f"{embedding_model.replace('/', '_')}_embedding"
    df[embedding_col_name] = embeddings

    # Drop rows where embedding failed (vector is None)
    initial_rows = len(df)
    df.dropna(subset=[embedding_col_name], inplace=True)
    if len(df) < initial_rows:
        print(f"Dropped {initial_rows - len(df)} rows where embedding failed.")

    if len(df) == 0:
        sys.exit("No embeddings successfully generated. Exiting.")

    # Generate output filename and path
    input_dir = os.path.dirname(os.path.abspath(args.csv_file))
    original_filename = os.path.splitext(os.path.basename(args.csv_file))[0]
    model_name_clean = embedding_model.replace("/", "_")
    output_filename_prefix = f"{original_filename}_LLM_embeddings_{model_name_clean}"
    output_path = os.path.join(input_dir, f"{output_filename_prefix}.csv")
    output_path = helpers.get_unique_filename(output_path)

    # Save the new DataFrame with embeddings
    df.to_csv(output_path, index=False)
    print(f"\n✅ Results saved to {output_path}")


if __name__ == "__main__":
    main()
