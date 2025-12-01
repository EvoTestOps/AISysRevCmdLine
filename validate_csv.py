#Functions to validate CSV files for required columns and data quality.
#Paper count can be limited with paper count and probability. This probability comes screen.py as output.abs
#Thus it allows you to select only the papers that are mostly likely to be included based in title abstact screening.
#Paper count is applied first, then probability filtering if given
#Paper count is 10 by default. Specifying all gives all papers.

import csv
import sys
import pandas as pd
from typing import Set, Optional



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