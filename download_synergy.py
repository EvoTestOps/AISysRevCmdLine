"""
Prepare SYNERGY dataset files for use with AISysRevCmdLine screening tools.

Fetches eligibility criteria from the datasets.toml index on GitHub and
downloads paper CSVs (title/abstract/label_included) from OpenAlex via the
synergy-dataset package, installed in an isolated venv to avoid polluting the
project environment.

Each dataset is stored in its own subfolder under data/synergy/, mirroring
the data/sesr/ layout created by download_sesr.py:

    data/synergy/
        Wolters_2018/
            primary_correct.csv   ← papers (title, abstract, label_included)
            criteria.conf         ← eligibility criteria from datasets.toml
        Hall_2012/
            ...

Usage:
    python download_synergy.py                  # download all active datasets (default)
    python download_synergy.py Wolters_2018     # download one dataset
    python download_synergy.py --list
    python download_synergy.py --data-dir data/synergy
"""

import argparse
import subprocess
import sys
import tempfile
import tomllib
import urllib.request
from pathlib import Path

DATASETS_TOML_URL = (
    "https://raw.githubusercontent.com/asreview/synergy-dataset/master/datasets.toml"
)
TOML_CACHE = Path.home() / ".cache" / "synergy_dataset" / "datasets.toml"
DEFAULT_DATA_DIR = Path("data/synergy")
VENV_PATH = Path.home() / ".synergy_dataset_env"


def fetch_toml(local_path: str | None, refresh: bool) -> bytes:
    if local_path:
        return Path(local_path).expanduser().read_bytes()
    if TOML_CACHE.exists() and not refresh:
        return TOML_CACHE.read_bytes()
    print("Fetching datasets.toml from GitHub...", flush=True)
    with urllib.request.urlopen(DATASETS_TOML_URL, timeout=30) as resp:
        data = resp.read()
    TOML_CACHE.parent.mkdir(parents=True, exist_ok=True)
    TOML_CACHE.write_bytes(data)
    return data


def load_datasets(local_path: str | None, refresh: bool) -> list[dict]:
    return tomllib.loads(fetch_toml(local_path, refresh).decode())["datasets"]


def ensure_venv(verbose: bool = True) -> Path:
    """Return path to venv Python, creating and populating the venv if needed."""
    venv_python = VENV_PATH / "bin" / "python"
    if not venv_python.exists():
        if verbose:
            print(f"Creating isolated venv at {VENV_PATH} ...", flush=True)
        subprocess.run(
            [sys.executable, "-m", "venv", str(VENV_PATH)],
            check=True,
            stdout=subprocess.DEVNULL,
        )
        pip = VENV_PATH / "bin" / "pip"
        if verbose:
            print("Installing synergy-dataset and pandas into venv...", flush=True)
        subprocess.run(
            [str(pip), "install", "--quiet", "synergy-dataset", "pandas"],
            check=True,
            stdout=subprocess.DEVNULL,
        )
    return venv_python


def download_dataset(key: str, data_dir: Path, verbose: bool = True) -> Path:
    """Download papers CSV into data_dir/{key}/primary_correct.csv and return that path."""
    venv_python = ensure_venv(verbose=verbose)
    # synergy_dataset refuses to write into a non-empty folder, so use a temp dir
    with tempfile.TemporaryDirectory() as tmpdir:
        subprocess.run(
            [
                str(venv_python), "-m", "synergy_dataset", "get",
                "--ignore-legal",
                "-d", key,
                "-o", tmpdir,
            ],
            check=True,
            stdout=subprocess.DEVNULL,
        )
        dataset_dir = data_dir / key
        dataset_dir.mkdir(parents=True, exist_ok=True)
        dest = dataset_dir / "primary_correct.csv"
        (Path(tmpdir) / f"{key}.csv").rename(dest)
    return dest


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Prepare SYNERGY dataset files for AISysRevCmdLine screening.\n"
            "Downloads paper CSVs (isolated venv) and writes criteria.conf files."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "dataset",
        nargs="?",
        default="all",
        metavar="DATASET",
        help="Dataset key (e.g. Wolters_2018) or 'all' (default).",
    )
    parser.add_argument(
        "--list", "-l",
        action="store_true",
        help="List available datasets and whether they have eligibility criteria.",
    )
    parser.add_argument(
        "--toml",
        metavar="PATH",
        help="Local path to datasets.toml (default: fetch from GitHub and cache).",
    )
    parser.add_argument(
        "--refresh",
        action="store_true",
        help="Re-fetch datasets.toml from GitHub even if a local cache exists.",
    )
    parser.add_argument(
        "--data-dir",
        metavar="DIR",
        default=str(DEFAULT_DATA_DIR),
        help=f"Root directory for synergy datasets (default: {DEFAULT_DATA_DIR}).",
    )
    parser.add_argument(
        "--include-inactive",
        action="store_true",
        help="Also process datasets marked active=false in datasets.toml.",
    )
    args = parser.parse_args()

    datasets = load_datasets(args.toml, args.refresh)
    data_dir = Path(args.data_dir).expanduser()

    if args.list:
        print(f"{'Key':<30} {'Active':<8} {'Has criteria'}")
        print("-" * 55)
        for ds in datasets:
            key = ds.get("key", "?")
            active = ds.get("active", True)
            has_criteria = bool(
                ds.get("publication", {}).get("eligibility_criteria", "").strip()
            )
            print(
                f"{key:<30} {'yes' if active else 'no':<8} {'yes' if has_criteria else 'no'}"
            )
        return

    # Resolve target datasets
    if args.dataset == "all":
        targets = datasets
    else:
        targets = [ds for ds in datasets if ds.get("key") == args.dataset]
        if not targets:
            print(
                f"Error: dataset '{args.dataset}' not found in datasets.toml.",
                file=sys.stderr,
            )
            sys.exit(1)

    if not args.include_inactive:
        inactive = [ds["key"] for ds in targets if not ds.get("active", True)]
        if inactive:
            print(f"Skipping {len(inactive)} inactive datasets (use --include-inactive to include them).")
        targets = [ds for ds in targets if ds.get("active", True)]

    single = len(targets) == 1

    for ds in targets:
        key = ds.get("key", "")
        dataset_dir = data_dir / key

        # Ensure paper CSV exists in data/synergy/{key}/primary_correct.csv
        csv_path = dataset_dir / "primary_correct.csv"
        if csv_path.exists():
            status = "already downloaded"
        else:
            if single:
                print(f"  Downloading {key} papers from OpenAlex...", flush=True)
            try:
                csv_path = download_dataset(key, data_dir, verbose=single)
                status = "downloaded"
            except subprocess.CalledProcessError as e:
                print(f"  {key}: ERROR – {e}", file=sys.stderr)
                continue

        # Extract and write eligibility criteria into the same subfolder
        eligibility = ds.get("publication", {}).get("eligibility_criteria", "").strip()
        criteria_path = dataset_dir / "criteria.conf"
        if eligibility:
            criteria_path.write_text(eligibility, encoding="utf-8")
            criteria_status = "criteria.conf written"
        else:
            criteria_status = "WARNING: no eligibility_criteria in datasets.toml"

        if single:
            print(f"  Papers:   {csv_path}  ({status})")
            print(f"  Criteria: {criteria_path}") if eligibility else print(f"  {criteria_status}")
            if eligibility:
                print(f"  Run:")
                print(f"    python screen.py {csv_path} -n 10 -c {criteria_path} -m models.conf")
        else:
            flag = "!" if not eligibility else " "
            print(f"  {flag} {key:<35} {status}, {criteria_status}")


if __name__ == "__main__":
    main()
