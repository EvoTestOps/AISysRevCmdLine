import pandas as pd
import os

# Create a directory for test files
os.makedirs("test_csv_files", exist_ok=True)

# normal.csv
pd.DataFrame({
    "title": ["Paper 1", "Paper 2", "Paper 3", ""],
    "abstract": ["This is the abstract for paper 1.", "", "This is the abstract for paper 3.", "This is an abstract with no title."]
}).to_csv("test_csv_files/normal.csv", index=False)

# missing_title.csv
pd.DataFrame({
    "abstract": ["This is the abstract for paper 1.", "This is the abstract for paper 2."]
}).to_csv("test_csv_files/missing_title.csv", index=False)

# duplicate_title.csv
pd.DataFrame({
    "title": ["Paper 1", "Paper 2"],
    "Title": ["Paper 1", "Paper 2"],
    "abstract": ["This is the abstract for paper 1.", "This is the abstract for paper 2."]
}).to_csv("test_csv_files/duplicate_title.csv", index=False)

# missing_abstract.csv
pd.DataFrame({
    "title": ["Paper 1", "Paper 2"],
    "other": ["This is not an abstract.", "This is not an abstract."]
}).to_csv("test_csv_files/missing_abstract.csv", index=False)

# duplicate_abstract.csv
pd.DataFrame({
    "title": ["Paper 1", "Paper 2"],
    "abstract": ["This is the abstract for paper 1.", "This is the abstract for paper 2."],
    "Abstract": ["This is the abstract for paper 1.", "This is the abstract for paper 2."]
}).to_csv("test_csv_files/duplicate_abstract.csv", index=False)

# empty_file.csv
pd.DataFrame(columns=["title", "abstract"]).to_csv("test_csv_files/empty_file.csv", index=False)

# comma_delimited.csv
pd.DataFrame({
    "title": ["Paper 1", "Paper 2"],
    "abstract": ["This is the abstract for paper 1.", "This is the abstract for paper 2."]
}).to_csv("test_csv_files/comma_delimited.csv", index=False)

# semicolon_delimited.csv
pd.DataFrame({
    "title": ["Paper 1", "Paper 2"],
    "abstract": ["This is the abstract for paper 1.", "This is the abstract for paper 2."]
}).to_csv("test_csv_files/semicolon_delimited.csv", index=False, sep=';')

# whitespace_headers.csv
pd.DataFrame({
    "  title  ": ["Paper 1", "Paper 2"],
    "  abstract  ": ["This is the abstract for paper 1.", "This is the abstract for paper 2."]
}).to_csv("test_csv_files/whitespace_headers.csv", index=False)

# mixed_delimiters.csv (edge case: first row comma, second row semicolon)
with open("test_csv_files/mixed_delimiters.csv", "w") as f:
    f.write("title,abstract\n")
    f.write("Paper 1;This is the abstract for paper 1.\n")
