#Functions to validate CSV files for required columns and data quality.
#Paper count can be limited with paper count and probability. This probability comes screen.py as output.abs
#Thus it allows you to select only the papers that are mostly likely to be included based in title abstact screening.
#Paper count is applied first, then probability filtering if given
#Paper count is 10 by default. Specifying all gives all papers.

import csv
import sys
import pandas as pd
from typing import Set, Optional, List
import os
import requests
import time


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

def detect_delimiter(file_path: str) -> str:
    with open(file_path, 'r') as f:
        first_line = f.readline()
    if ',' in first_line:
        return ','
    elif '\t' in first_line:
        return '\t'
    elif ';' in first_line:
        return ';'
    else:
        return ','  # default


def validate_doi_csv(file_path: str, n_rows: int = None) -> pd.DataFrame:
    """
    Validates a CSV for the presence of a 'doi' column and issues warnings 
    if empty DOI values are found.
    """
    delimiter = detect_delimiter(file_path)
    required_columns = {'doi'}
    header_row_index = -1
    
    # 1. Search for the header row (scanning first 20 rows)
    max_matches = -1
    best_missing_columns: Set[str] = required_columns.copy()

    with open(file_path, 'r') as f:
        reader = csv.reader(f, delimiter=delimiter)
        for i, row in enumerate(reader):
            if i >= 20:
                break
            
            headers = [h.strip().lower() for h in row]
            present_columns = required_columns.intersection(set(headers))
            num_matches = len(present_columns)
            current_missing_columns = required_columns - present_columns

            if not current_missing_columns:
                header_row_index = i
                break
            
            if num_matches > max_matches:
                max_matches = num_matches
                best_missing_columns = current_missing_columns

        if header_row_index == -1:
            sys.exit(f"Error: Could not find a valid header row containing 'doi'.")

    # 2. Duplicate Check
    with open(file_path, 'r') as f:
        reader = csv.reader(f, delimiter=delimiter)
        for _ in range(header_row_index):
            next(reader)
        headers = [h.strip().lower() for h in next(reader)]
        if headers.count('doi') > 1:
            sys.exit(f"Error: Duplicate 'doi' column detected.")

    # 3. Load DataFrame
    df = pd.read_csv(file_path, delimiter=delimiter, header=header_row_index, nrows=n_rows)
    df.columns = df.columns.str.strip().str.lower()

    # 4. Warning for empty DOIs
    # Check for NaN, null, or whitespace-only strings
    empty_doi_mask = df['doi'].isna() | (df['doi'].astype(str).str.strip() == '')
    empty_doi_count = empty_doi_mask.sum()
    
    if empty_doi_count > 0:
        print(f"WARN: {empty_doi_count} DOI entries are empty.")

    return df

def validate_csv(file_path: str, n_rows: int = None, require_avg_prob: float = None) -> pd.DataFrame:
    delimiter = detect_delimiter(file_path)
    required_columns = {'title', 'abstract'}
    if require_avg_prob is not None:
        required_columns.add('average_probability')

    header_row_index = -1
    
    # Initialize this to None. It will store the set of missing columns 
    # from the first row checked (i=0).
    # Memory for the best candidate
    max_matches = -1
    best_missing_columns: Set[str] = required_columns.copy()

    with open(file_path, 'r') as f:
        reader = csv.reader(f, delimiter=delimiter)
        
        for i, row in enumerate(reader):
            if i >= 20:
                break
            headers = [h.strip().lower() for h in row]
            present_columns = required_columns.intersection(set(headers))
            num_matches = len(present_columns)
            current_missing_columns = required_columns - present_columns
            # 1. Check for a perfect match (the goal)
            if not current_missing_columns:
                header_row_index = i
                break  # Found the perfect header, exit immediately
            # 2. Check if this row is the best candidate so far and update memory
            if num_matches > max_matches:
                max_matches = num_matches
                best_missing_columns = current_missing_columns

        if header_row_index == -1:
            # If no perfect header was found after checking 20 rows
            # If max_matches is still -1, it means the file had no valid headers.
            if max_matches == -1:
                 sys.exit(f"Error: Could not find a valid header row in the first 20 rows. Missing all required column(s): **{', '.join(sorted(required_columns))}**.")
            
            # Report the missing columns from the best matching row found (max_matches > -1)
            sys.exit(f"Error: Could not find a valid header row in the first 20 rows. Missing column(s): {', '.join(sorted(best_missing_columns))}.")

    with open(file_path, 'r') as f:
        reader = csv.reader(f, delimiter=delimiter)
        for _ in range(header_row_index):
            next(reader)
        headers = [h.strip().lower() for h in next(reader)]
        column_counts = {}
        for col in headers:
            column_counts[col] = column_counts.get(col, 0) + 1
            if column_counts[col] > 1 and col in required_columns:
                sys.exit(f"Error: Duplicate column detected: {col}")

    df = pd.read_csv(file_path, delimiter=delimiter, header=header_row_index, nrows=n_rows)
    df.columns = df.columns.str.strip().str.lower()

    # Check for empty strings and NaN in title and abstract
    empty_title = (df['title'].isna() | (df['title'].str.strip() == '')).sum()
    empty_abstract = (df['abstract'].isna() | (df['abstract'].str.strip() == '')).sum()
    if empty_title > 0:
        print(f"WARN: {empty_title} titles are empty.")
    if empty_abstract > 0:
        print(f"WARN: {empty_abstract} abstracts are empty.")

    if require_avg_prob is not None:
        # Convert invalid values to NaN
        df['average_probability'] = pd.to_numeric(df['average_probability'], errors='coerce')
        valid_probs = df['average_probability'].dropna()
        if len(valid_probs) > 0:
            print(f"\nAverage Probability Statistics (before filtering):")
            print(f"- Total papers with probability: {len(valid_probs)}")
            print(f"- Average: {valid_probs.mean():.3f}")
            print(f"- Minimum: {valid_probs.min():.3f}")
            print(f"- Maximum: {valid_probs.max():.3f}")
            print(f"- Median: {valid_probs.median():.3f}")
            print(f"- Papers with probability >= {require_avg_prob}: {len(valid_probs[valid_probs >= require_avg_prob])}")
        invalid_avg_prob = df['average_probability'].isna().sum()
        if invalid_avg_prob > 0:
            print(f"WARN: {invalid_avg_prob} average_probability values are not between 0 and 1.")
        # Filter out rows with NaN in average_probability
        #df = df.dropna(subset=['average_probability'])
        # Now filter by require_avg_prob
        df = df[df['average_probability'] >= require_avg_prob]

    return df




def download_pdfs_by_doi(doi_list: list, output_folder: str = "pdf"):
    """
    Downloads PDFs from a list of DOIs and saves them to a folder.
    Replaces '/' with '_' in filenames to ensure they are valid.
    """
    # Create the folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        print(f"Created folder: {output_folder}")

    # Headers to mimic a browser (some repositories block basic python scripts)
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
    }

    for doi in doi_list:
        if not doi or pd.isna(doi):
            continue

        # Clean filename: replace '/' with '_' to avoid path errors
        clean_filename = doi.replace('/', '_') + ".pdf"
        file_path = os.path.join(output_folder, clean_filename)

        # Skip if file already exists
        if os.path.exists(file_path):
            print(f"Skipping: {doi} (Already exists)")
            continue

        # Construct the download URL (using sci-hub or a specific resolver)
        # Note: DOI links usually go to landing pages; 
        # actual PDF extraction often requires a specific API or Sci-Hub URL.
        download_url = f"https://doi.org/{doi}" 

        try:
            print(f"Downloading: {doi}...", end="\r")
            response = requests.get(download_url, headers=headers, timeout=30)
            
            if response.status_code == 200 and 'application/pdf' in response.headers.get('Content-Type', ''):
                with open(file_path, 'wb') as f:
                    f.write(response.content)
                print(f"Saved: {clean_filename}         ")
            else:
                print(f"Failed: {doi} (Not a direct PDF link or access denied)")
                
        except Exception as e:
            print(f"Error downloading {doi}: {e}")
        
        # Polite delay to avoid rate-limiting
        time.sleep(1)