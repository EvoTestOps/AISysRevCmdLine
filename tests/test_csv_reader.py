import pytest
import pandas as pd
import os
from screen import validate_csv

TEST_DIR = "tests/test_csv_files"

def test_validate_csv_normal():
    """Test that normal.csv passes validation and issues warnings for empty fields."""
    df = validate_csv(f"{TEST_DIR}/normal.csv")
    assert isinstance(df, pd.DataFrame)
    # If you want to check warnings, you can capture stdout in pytest
    # (see below for an example using capsys)

def test_validate_csv_missing_title():
    """Test that missing_title.csv raises an error for missing 'title' column."""
    with pytest.raises(SystemExit) as e:
        validate_csv(f"{TEST_DIR}/missing_title.csv")
    assert "Error: Missing required columns: {'title'}" in str(e.value)

def test_validate_csv_missing_abstract():
    """Test that missing_abstract.csv raises an error for missing 'abstract' column."""
    with pytest.raises(SystemExit) as e:
        validate_csv(f"{TEST_DIR}/missing_abstract.csv")
    assert "Error: Missing required columns: {'abstract'}" in str(e.value)

def test_validate_csv_duplicate_title():
    """Test that duplicate_title.csv raises an error for duplicate 'title' column."""
    with pytest.raises(SystemExit) as e:
        validate_csv(f"{TEST_DIR}/duplicate_title.csv")
    assert "Error: Duplicate column detected: title" in str(e.value)

def test_validate_csv_duplicate_abstract():
    """Test that duplicate_abstract.csv raises an error for duplicate 'abstract' column."""
    with pytest.raises(SystemExit) as e:
        validate_csv(f"{TEST_DIR}/duplicate_abstract.csv")
    assert "Error: Duplicate column detected: abstract" in str(e.value)

def test_validate_csv_empty_file():
    """Test that empty_file.csv passes validation (only headers)."""
    df = validate_csv(f"{TEST_DIR}/empty_file.csv")
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 0

def test_validate_csv_comma_delimited():
    """Test that comma_delimited.csv passes validation."""
    df = validate_csv(f"{TEST_DIR}/comma_delimited.csv")
    assert isinstance(df, pd.DataFrame)

def test_validate_csv_semicolon_delimited():
    """Test that semicolon_delimited.csv passes validation."""
    df = validate_csv(f"{TEST_DIR}/semicolon_delimited.csv")
    assert isinstance(df, pd.DataFrame)

def test_validate_csv_whitespace_headers():
    """Test that whitespace_headers.csv passes validation."""
    df = validate_csv(f"{TEST_DIR}/whitespace_headers.csv")
    assert isinstance(df, pd.DataFrame)

# Optional: If you want to check warnings (e.g., empty titles/abstracts)
def test_validate_csv_warnings(capsys):
    """Test that warnings are printed for empty titles/abstracts in normal.csv."""
    validate_csv(f"{TEST_DIR}/normal.csv")
    captured = capsys.readouterr()
    assert "Warning: 1 titles are empty." in captured.out
    assert "Warning: 1 abstracts are empty." in captured.out


# import pytest
# import subprocess
# import os

# TEST_DIR = "tests/test_csv_files"

# def run_script(csv_file):
#     return subprocess.run(
#         ["python", "csv_reader.py", f"{TEST_DIR}/{csv_file}"],
#         capture_output=True, text=True
#     )

# def test_normal():
#     result = run_script("normal.csv")
#     assert result.returncode == 0
#     assert "Warning: 1 titles are empty." in result.stdout
#     assert "Warning: 1 abstracts are empty." in result.stdout

# def test_missing_title():
#     result = run_script("missing_title.csv")
#     assert result.returncode != 0
#     assert "Error: Missing required columns: {'title'}" in result.stderr

# def test_missing_abstract():
#     result = run_script("missing_abstract.csv")
#     assert result.returncode != 0
#     assert "Error: Missing required columns: {'abstract'}" in result.stderr

# def test_duplicate_title():
#     result = run_script("duplicate_title.csv")
#     assert result.returncode != 0
#     assert "Error: Duplicate column detected: title" in result.stderr

# def test_duplicate_abstract():
#     result = run_script("duplicate_abstract.csv")
#     assert result.returncode != 0
#     assert "Error: Duplicate column detected: abstract" in result.stderr

# def test_empty_file():
#     result = run_script("empty_file.csv")
#     assert result.returncode == 0

# def test_comma_delimited():
#     result = run_script("comma_delimited.csv")
#     assert result.returncode == 0

# def test_semicolon_delimited():
#     result = run_script("semicolon_delimited.csv")
#     assert result.returncode == 0

# def test_whitespace_headers():
#     result = run_script("whitespace_headers.csv")
#     assert result.returncode == 0
