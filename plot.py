# plot_embeddings.py (Final, Corrected Version with Color/Marker Cycling)

import pandas as pd
import argparse
import os
import sys
import numpy as np
import umap.umap_ as umap
from sklearn.preprocessing import StandardScaler
import plotly.express as px
import plotly.io as pio
import textwrap
from typing import Optional, List, Dict, Any

# Assuming helpers.py exists and contains get_unique_filename
import helpers 

# --- NEW CONSTANTS FOR COLOR/MARKER CYCLING ---
# Max 5 colors as requested
COLOR_CYCLE = ['#1F77B4', '#D62728', '#2CA02C', '#000000', '#FF7F0E']
# Common Plotly marker symbols for cycling
MARKER_CYCLE = ['circle', 'cross', 'diamond', 'square', 'star'] 


# --- Utility Functions ---

def word_wrap_abstract(text: str, width: int = 80) -> str:
    """
    Inserts HTML line breaks (<br>) into the text to simulate word wrapping 
    in a Plotly tooltip.
    """
    if not text:
        return ""
    
    lines = textwrap.wrap(text, width=width)
    return "<br>".join(lines)


def select_classification_column(df: pd.DataFrame) -> Optional[str]:
    """
    Identifies columns ending with '_chosen' and prompts the user to select one.
    Returns the selected column name or None if no valid selection is made or user quits.
    """
    chosen_cols = [col for col in df.columns if col.endswith('_chosen')]
    
    if not chosen_cols:
        print("\n❌ No classification columns ending with '_chosen' were found in the CSV.")
        return None

    print("\n--- Available Classification Columns for Plot Coloring (Select a number) ---")
    for i, col in enumerate(chosen_cols):
        unique_count = df[col].nunique(dropna=True)
        print(f"[{i + 1}] {col} (Categories: {unique_count})")
    
    while True:
        try:
            choice = input("Enter number to select column (or 'q' to skip plotting): ").strip()
            if choice.lower() == 'q':
                return None
            
            index = int(choice) - 1
            
            if 0 <= index < len(chosen_cols):
                selected_col = chosen_cols[index]
                return selected_col
            else:
                print(f"Invalid selection. Please enter a number between 1 and {len(chosen_cols)}.")
        
        except ValueError:
            print("Invalid input. Please enter a number or 'q'.")


# --- Main Plotting Logic ---

def plot_embeddings(df: pd.DataFrame, embedding_col: str, title_col: str, abstract_col: str, probability_col: str, threshold: Optional[float], classification_col: Optional[str], output_dir: str, filename_prefix: str) -> None:
    """
    Performs UMAP dimensionality reduction and generates an interactive Plotly scatter plot.
    Applies custom color/marker cycling if classification_col is provided.
    """
    print(f"\nUsing embedding column: {embedding_col}")
    print("Starting dimensionality reduction and plotting...")
    
    # 1. Prepare Data and UMAP
    try:
        # Assuming the list of floats was saved as a string, use eval to convert back
        X = np.array(df[embedding_col].apply(lambda x: eval(str(x))).tolist())
    except Exception as e:
        print(f"Error: Could not convert embedding column '{embedding_col}' to a NumPy array. Data might be malformed: {e}")
        return
        
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # UMAP Reduction
    reducer = umap.UMAP(n_components=2, random_state=42, metric='cosine')
    embedding_2d = reducer.fit_transform(X_scaled)
    
    # 2. Create base DataFrame for plotting
    plot_df_data = {
        'UMAP_X': embedding_2d[:, 0],
        'UMAP_Y': embedding_2d[:, 1],
        title_col: df[title_col],       
        abstract_col: df[abstract_col],
    }
    
    has_probability = probability_col in df.columns
    if has_probability:
        plot_df_data[probability_col] = df[probability_col]
    
    plot_df = pd.DataFrame(plot_df_data)

    if classification_col is not None and classification_col in df.columns:
        plot_df[classification_col] = df[classification_col].astype(str).fillna("N/A")
    else:
        classification_col = None 
    
    plot_df['Wrapped_Abstract'] = (
        plot_df[abstract_col]
        .astype(str)
        .apply(word_wrap_abstract) 
    )
    
    # --- CONDITIONAL COLORING/MARKER LOGIC SETUP ---
    color_param = None
    color_map = None
    symbol_param = None
    symbol_map = None # Initialize symbol map
    plot_title_suffix = " (Uncolored)"
    
    hover_data_params = {
        'Wrapped_Abstract': True,
        title_col: False, 
        'UMAP_X': False,   
        'UMAP_Y': False    
    }
    
    if threshold is not None and has_probability: 
        # Binary threshold coloring
        plot_df['Classification_Status'] = np.where(
            plot_df[probability_col] >= threshold, 
            f'Blue (Prob >= {threshold:.2f})', 
            f'Orange (Prob < {threshold:.2f})'
        )
        color_param = 'Classification_Status'
        color_map = {f'Blue (Prob >= {threshold:.2f})': 'blue', f'Orange (Prob < {threshold:.2f})': 'orange'}
        hover_data_params['Classification_Status'] = True
        hover_data_params[probability_col] = ':.3f' 
        plot_title_suffix = f" (Colored by T={threshold:.2f})"
        
    elif classification_col is not None: 
        # --- APPLY CUSTOM COLOR/MARKER CYCLING ---
        categories = sorted(plot_df[classification_col].unique())
        
        category_color_map = {}
        category_marker_map = {}
        
        for i, cat in enumerate(categories):
            # Color cycles (0 to 4)
            color_index = i % len(COLOR_CYCLE)
            # Marker increments every 5 colors (0 to 4, then repeats)
            marker_index = i // len(COLOR_CYCLE) % len(MARKER_CYCLE)
            
            # Key for maps is the original class name
            category_color_map[cat] = COLOR_CYCLE[color_index]
            category_marker_map[cat] = MARKER_CYCLE[marker_index]

        # Use the original classification column name for both color and symbol.
        color_param = classification_col 
        symbol_param = classification_col 
        
        # Pass the custom mappings to Plotly.
        color_map = category_color_map 
        symbol_map = category_marker_map 
        
        hover_data_params[classification_col] = True
        if has_probability:
             hover_data_params[probability_col] = ':.3f' 
        plot_title_suffix = f" (Colored/Markered by Column: {classification_col})"
        
    else:
        if has_probability:
            hover_data_params[probability_col] = ':.3f'
        plot_title_suffix = " (Uncolored)"
    
    # 3. Create Plotly Figure (symbol and symbol_map parameters added)
    fig = px.scatter(
        plot_df,
        x='UMAP_X',
        y='UMAP_Y',
        color=color_param,
        color_discrete_map=color_map, 
        symbol=symbol_param, 
        symbol_map=symbol_map, # Uses the custom marker mapping
        hover_name=title_col, 
        hover_data=hover_data_params,
        title=f"2D UMAP Projection of LLM Embeddings{plot_title_suffix}",
        labels={'UMAP_X': 'UMAP Dimension 1', 'UMAP_Y': 'UMAP Dimension 2'},
        height=700,
        width=1000,
        template="plotly_white",
    )
    
    # 4. Apply Hover Template
    hovertemplate_content = '<b>%{hovertext}</b><br>'
    
    custom_data_list = [plot_df['Wrapped_Abstract']]
    
    # Build Custom Data and Hover Template content sequentially
    if has_probability:
        hovertemplate_content += f'<b>{probability_col}:</b> %{{customdata[1]}}<br>'
        custom_data_list.append(plot_df[probability_col].apply(lambda x: f'{x:.3f}'))
    
    if threshold is not None and has_probability: 
        # customdata[2] is Classification_Status
        custom_data_list.append(plot_df['Classification_Status'])
        hovertemplate_content += '<b>Status:</b> %{{customdata[2]}}<br>'
        
    elif classification_col is not None:
        # customdata[2] is classification_col
        custom_data_list.append(plot_df[classification_col])
        hovertemplate_content += f'<b>{classification_col}:</b> %{{customdata[2]}}<br>'

    # Final Hover template structure
    hovertemplate_content += '<br><b>Abstract:</b><br>%{customdata[0]}<extra></extra>'
    
    fig.update_traces(
        customdata=np.stack(custom_data_list, axis=-1),
        hovertemplate=hovertemplate_content
    )

    # 5. Save Plot
    if classification_col is not None:
        safe_col_name = classification_col.replace('_chosen', '').replace('_', '-')
        plot_filename = f"{filename_prefix}_UMAP_c{safe_col_name}.html"
    elif threshold is not None and has_probability:
        plot_filename = f"{filename_prefix}_UMAP_t{int(threshold*100)}.html"
    else:
        plot_filename = f"{filename_prefix}_UMAP_uncolored.html"
        
    plot_path = os.path.join(output_dir, plot_filename)
    plot_path = helpers.get_unique_filename(plot_path)

    pio.write_html(fig, file=plot_path, auto_open=False)
    print(f"\n✅ Interactive plot saved to {plot_path}")

# --- User Interaction and CLI Logic (In main function) ---

def main():
    parser = argparse.ArgumentParser(description="Generate an interactive Plotly scatter plot from an embedding CSV file.")
    parser.add_argument("csv_file", type=str, help="Path to the CSV file containing the LLM embeddings.")
    parser.add_argument("-t", type=float, default=None, help="Probability threshold (0-1) for coloring points in the plot (Blue/Orange). Cannot be used with -c.") 
    parser.add_argument("-c", action='store_true', help="Use a classification column ending in '_chosen' for categorical coloring. Prompts user for selection. Cannot be used with -t.")
    
    args = parser.parse_args()
    
    # --- Input Validation ---
    if args.t is not None and not (0 <= args.t <= 1):
        sys.exit("Error: -t (plotting threshold) must be between 0 and 1.")

    if args.t is not None and args.c:
        sys.exit("Error: Cannot use both -t (probability threshold coloring) and -c (classification column selection). Choose one.")

    # Load the CSV file
    try:
        df = pd.read_csv(args.csv_file)
    except FileNotFoundError:
        sys.exit(f"Error: CSV file not found at {args.csv_file}")
    
    if len(df.columns) == 0:
        sys.exit("Error: CSV file is empty or malformed (no columns found).")

    # --- 1. Identify Embedding Column (Always the last one) ---
    embedding_col_name = df.columns[-1]
    
    # --- 2. Validate Required Columns ---
    required_cols = ['title', 'abstract']
    missing_cols = [col for col in required_cols if col not in df.columns]
    
    if len(missing_cols) > 0:
        sys.exit(f"Error: CSV is missing required columns: {', '.join(missing_cols)}")
        
    PROBABILITY_COLUMN = 'average_probability'
    if args.t is not None and PROBABILITY_COLUMN not in df.columns:
        sys.exit(f"Error: DataFrame must contain a '{PROBABILITY_COLUMN}' column when using -t.")

    # --- 3. Handle User Interaction for Classification Column ---
    classification_col_name = None
    if args.c:
        classification_col_name = select_classification_column(df)
        if classification_col_name is None and args.t is None:
             sys.exit("No classification column selected and no threshold specified. Exiting without plotting.")
    # ----------------------------------------------------------------

    if len(df) == 0:
          print("Input DataFrame is empty. Skipping plot generation.")
          return

    # Derive output directory and filename prefix
    input_dir = os.path.dirname(os.path.abspath(args.csv_file))
    output_filename_prefix = os.path.splitext(os.path.basename(args.csv_file))[0]
    
    # Plot the embeddings
    plot_embeddings(
        df=df,
        embedding_col=embedding_col_name,
        title_col='title',
        abstract_col='abstract',
        probability_col=PROBABILITY_COLUMN,
        threshold=args.t, 
        classification_col=classification_col_name, 
        output_dir=input_dir,
        filename_prefix=output_filename_prefix
    )

if __name__ == "__main__":
    main()