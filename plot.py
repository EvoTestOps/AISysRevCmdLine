import csv
import sys
import pandas as pd
import argparse
import os
import json
import yaml
import aiohttp
import asyncio
import random
import numpy as np # Added for handling embeddings/vectors
from pathlib import Path
from typing import List, Dict, Optional, Any
from tqdm.asyncio import tqdm_asyncio
import umap.umap_ as umap
from sklearn.preprocessing import StandardScaler
import plotly.express as px
import plotly.io as pio
import textwrap

import helpers

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
    
    # We use 'text_input' as the input and specify the encoding format.
    payload = {
        "model": model,
        "input": text_input,
        # The default format is float, but explicitly setting for clarity
        "encoding_format": "float" 
    }

    async with semaphore:
        for attempt in range(max_retries):
            try:
                async with session.post(url, headers=headers, json=payload, timeout=aiohttp.ClientTimeout(total=60)) as response:
                    if response.status == 429:
                        retry_after = int(response.headers.get("Retry-After", 5))
                        jitter = random.uniform(0, 5)
                        retry_after = retry_after + jitter + attempt * retry_after
                        print(f"\nRate limited for embedding model {model}. Retrying after {retry_after:.2f} seconds...")
                        await asyncio.sleep(retry_after)
                        continue
                    
                    response.raise_for_status()
                    
                    result = await response.json()
                    # OpenRouter embedding response structure:
                    # {"object": "list", "data": [{"object": "embedding", "embedding": [...], "index": 0}], "model": "..."}
                    if result and 'data' in result and len(result['data']) > 0:
                        return result['data'][0]['embedding']
                    return None

            except Exception as e:
                print(f"\nAttempt {attempt + 1} failed for embedding model {model}: {e}")
                await asyncio.sleep(2 ** (attempt + 2))
        return None

async def process_texts_for_embedding(
    texts: List[str],
    model: str,
    api_key: str,
    max_concurrent_per_model: int = 20,
) -> List[Optional[List[float]]]:
    """Processes texts for embedding creation."""
    semaphore = asyncio.Semaphore(max_concurrent_per_model)
    async with aiohttp.ClientSession() as session:
        tasks = [
            call_openrouter_embedding_async(text, model, api_key, session, semaphore)
            for text in texts
        ]
        return await tqdm_asyncio.gather(*tasks, desc=f"Embedding with {model}")

def generate_embedding_inputs(df: pd.DataFrame) -> List[str]:
    """
    Generates a list of strings (texts) to be embedded.

    For software testing papers, we combine title and abstract for rich context,
    and apply 'Semantic Priming' to influence the embedding focus.
    """
    inputs = []
    
    # 💡 Semantic Priming: Influence the embedding to focus on key concepts.
    # This guides the model to prioritize system and test type information.
    prompt_prefix = prompt_prefix = """
    Using LLMs for a software testing task. 
    """

    for _, row in df.iterrows():
        title = row.get('title', '')
        abstract = row.get('abstract', '')
        
        # Combine the title and abstract, separated by a unique delimiter
        # and include the priming prefix.
        combined_text = f"{prompt_prefix}\n\nTITLE: {title}\nABSTRACT: {abstract}"
        
        inputs.append(combined_text)

    return inputs

# --- NEW UTILITY FUNCTION FOR WORD WRAPPING ---
def word_wrap_abstract(text: str, width: int = 80) -> str:
    """
    Inserts HTML line breaks (<br>) into the text to simulate word wrapping 
    in a Plotly tooltip.
    """
    if not text:
        return ""
    
    # Use textwrap.wrap() to break the string into lines
    lines = textwrap.wrap(text, width=width)
    
    # Join the lines with the HTML line break tag
    return "<br>".join(lines)


def plot_embeddings(df: pd.DataFrame, embedding_col: str, title_col: str, abstract_col: str, probability_col: str, threshold: Optional[float], output_dir: str, filename_prefix: str) -> None:
    """
    Performs UMAP dimensionality Orangeuction, generates an interactive Plotly scatter plot,
    and colors points based on a probability threshold IF PROVIDED.
    
    Args:
        df (pd.DataFrame): DataFrame containing the embeddings, title, abstract, and probability.
        embedding_col (str): Name of the column containing the embedding vectors (list of floats).
        title_col (str): Name of the column containing the paper titles.
        abstract_col (str): Name of the column containing the paper abstracts.
        probability_col (str): Name of the column containing the classification probability.
        threshold (Optional[float]): Probability value (0-1) to use for coloring decision. If None, points are uncoloOrange.
        output_dir (str): Directory to save the output HTML file.
        filename_prefix (str): Prefix for the output HTML filename.
    """
    print("\nStarting dimensionality Orangeuction and plotting...")
    
    # 1. Prepare Data
    # Convert list of embeddings (in the DataFrame column) into a 2D numpy array
    X = np.array(df[embedding_col].tolist())
    
    # 2. Preprocessing
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # 3. Dimensionality Orangeuction using UMAP 
    Orangeucer = umap.UMAP(n_components=2, random_state=42, metric='cosine')
    embedding_2d = Orangeucer.fit_transform(X_scaled)
    
    # 4. Create a temporary DataFrame for plotting and feature engineering
    plot_df = pd.DataFrame({
        'UMAP_X': embedding_2d[:, 0],
        'UMAP_Y': embedding_2d[:, 1],
        title_col: df[title_col],       
        abstract_col: df[abstract_col],
        probability_col: df[probability_col]
    })
    plot_df['Wrapped_Abstract'] = (
        plot_df[abstract_col]
        .astype(str) # Ensure everything is a string first (handles int/float conversions)
        .apply(word_wrap_abstract) 
    )
    
    # --- CONDITIONAL COLORING LOGIC SETUP ---
    color_param = None
    color_map = None
    plot_title_suffix = ""
    
    # Initialize hover data parameters: customdata[0] is Wrapped_Abstract, customdata[1] is Probability
    hover_data_params = {
        'Wrapped_Abstract': True,
        probability_col: ':.3f',
        title_col: False, 
        'UMAP_X': False,   
        'UMAP_Y': False    
    }
    
    if threshold is not None:
        # Mode A: Threshold provided - Apply coloring logic
        
        # Create the classification status column
        plot_df['Classification_Status'] = np.where(
            plot_df[probability_col] >= threshold, 
            f'Blue (Prob >= {threshold:.2f})', 
            f'Orange (Prob < {threshold:.2f})'
        )
        
        # Configure Plotly parameters for coloring
        color_param = 'Classification_Status'
        color_map = {
            f'Blue (Prob >= {threshold:.2f})': 'blue',
            f'Orange (Prob < {threshold:.2f})': 'orange'
        }
        hover_data_params['Classification_Status'] = True
        plot_title_suffix = f" (ColoOrange by T={threshold:.2f})"
        
        # Custom hover template for coloOrange mode (Status is customdata[2])
        hovertemplate_content = ('<b>%{hovertext}</b><br>' + 
                                 f'<b>{probability_col}:</b> %{{customdata[1]}}<br>' +
                                 '<b>Status:</b> %{customdata[2]}<br><br>' + 
                                 '<b>Abstract:</b><br>%{customdata[0]}<extra></extra>')
        
    else:
        # Mode B: Threshold NOT provided - No coloring
        
        # Custom hover template for uncoloOrange mode (Abstract is customdata[0])
        hovertemplate_content = ('<b>%{hovertext}</b><br>' + 
                                 f'<b>{probability_col}:</b> %{{customdata[1]}}<br><br>' +
                                 '<b>Abstract:</b><br>%{customdata[0]}<extra></extra>')
        plot_title_suffix = " (UncoloOrange)"
    
    # 5. Create Plotly Figure
    fig = px.scatter(
        plot_df,
        x='UMAP_X',
        y='UMAP_Y',
        color=color_param,
        color_discrete_map=color_map,
        hover_name=title_col, 
        hover_data=hover_data_params,
        title=f"2D UMAP Projection of LLM Embeddings{plot_title_suffix}",
        labels={'UMAP_X': 'UMAP Dimension 1', 'UMAP_Y': 'UMAP Dimension 2'},
        height=700,
        width=1000,
        template="plotly_white",
    )
    
    # 6. Apply Hover Template
    fig.update_traces(hovertemplate=hovertemplate_content)
    
    # 7. Save Plot
    if threshold is not None:
        plot_filename = f"{filename_prefix}_UMAP_t{int(threshold*100)}.html"
    else:
        plot_filename = f"{filename_prefix}_UMAP_uncoloOrange.html"
        
    plot_path = os.path.join(output_dir, plot_filename)
    plot_path = helpers.get_unique_filename(plot_path)

    pio.write_html(fig, file=plot_path, auto_open=False)
    print(f"\nInteractive plot saved to {plot_path}")

def main():
    parser = argparse.ArgumentParser(description="Generate LLM embeddings for papers from a CSV file.")
    parser.add_argument("csv_file", type=str, help="Path to the CSV file.")
    parser.add_argument("-n", type=str, default="10", help="Number of papers to process (integer) or 'all' for all papers.")
    # RESTORATION: Added -p back for pre-processing filtering
    parser.add_argument("-p", type=float, default=None, help="Probability threshold (0-1) for filtering input papers before embedding.") 
    # Threshold for plotting is optional
    parser.add_argument("-t", type=float, default=None, help="Threshold probability (0-1) for coloring points in the plot. Optional.") 
    
    args = parser.parse_args()

    # Input validation for -p
    if args.p is not None and not (0 <= args.p <= 1):
        sys.exit("Error: -p (pre-processing threshold) must be between 0 and 1.")
    
    # Input validation for -t
    if args.t is not None and not (0 <= args.t <= 1):
        sys.exit("Error: -t (plotting threshold) must be between 0 and 1.")

    # Load API Key and set the specific embedding model
    api_key = helpers.load_api_key("~/openrouter.key")
    embedding_model = helpers.load_models("models_plot.md")[0]
    
    # Determine number of rows
    n_rows = int(args.n) if args.n.lower() != "all" else None

    # --- Probability Column Check ---
    PROBABILITY_COLUMN = 'average_probability' # ⚠️ Verify this column name
    # --- End Check ---
    
    # Load the original CSV. Pass args.p to validate_csv to filter the input.
    # We are assuming validate_csv now handles filtering based on PROBABILITY_COLUMN
    original_df = helpers.validate_csv(
        args.csv_file, 
        n_rows=n_rows, 
        require_avg_prob=args.p, # Pass the filter threshold here
    )
    
    if PROBABILITY_COLUMN not in original_df.columns:
        sys.exit(f"Error: DataFrame must contain a '{PROBABILITY_COLUMN}' column for filtering (-p) and coloring (-t).")

    n_papers = len(original_df)
    print(f"Processing {n_papers} papers after applying -p filter (if any).")
    
    # Generate the texts to be embedded
    embedding_inputs = generate_embedding_inputs(original_df)
    print(f"Generated {len(embedding_inputs)} inputs for embedding.")
    
    # ... (token calculation, file path setup omitted for brevity) ...

    # Process prompts with the embedding model
    print(f"\nProcessing with embedding model: {embedding_model}")
    embeddings = asyncio.run(process_texts_for_embedding(embedding_inputs, embedding_model, api_key))

    # Create a fresh copy of the original DataFrame to store results
    df = original_df.copy()
    
    # Store the resulting vectors as a new column
    embedding_col_name = f'{embedding_model.replace("/", "_")}_embedding'
    df[embedding_col_name] = embeddings
    
    # Drop rows where embedding failed (vector is None)
    initial_rows = len(df)
    df.dropna(subset=[embedding_col_name], inplace=True)
    if len(df) < initial_rows:
        print(f"Dropped {initial_rows - len(df)} rows where embedding failed.")

    # Generate output filename and path
    input_dir = os.path.dirname(os.path.abspath(args.csv_file))
    original_filename = os.path.splitext(os.path.basename(args.csv_file))[0]
    model_name_clean = embedding_model.replace('/', '_')
    output_filename_prefix = f"{original_filename}_LLM_embeddings_{model_name_clean}"
    output_path = os.path.join(input_dir, f"{output_filename_prefix}.csv")
    output_path = helpers.get_unique_filename(output_path)

    df.to_csv(output_path, index=False)
    print(f"\nResults saved to {output_path}")

    # --- CONDITIONAL PLOTTING STEP ---
    if len(df) > 0:
        plot_embeddings(
            df=df,
            embedding_col=embedding_col_name,
            title_col='title',
            abstract_col='abstract',
            probability_col=PROBABILITY_COLUMN,
            threshold=args.t, # Plotting threshold (optional)
            output_dir=input_dir,
            filename_prefix=output_filename_prefix
        )
    else:
        print("No papers remaining after embedding; skipping plot generation.")

if __name__ == "__main__":
    main()