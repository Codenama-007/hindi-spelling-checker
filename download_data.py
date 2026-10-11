"""
Step 1 - download a Hindi text dataset from Kaggle.

Credentials (any ONE of these):
    A) Environment variables:  KAGGLE_USERNAME + KAGGLE_KEY   (or KAGGLE_API_TOKEN)
    B) A file  kaggle_token.txt  next to this script, containing either
           username:key            (old-style key)
       or  your_token_string       (single token)
    C) ~/.kaggle/kaggle.json    (the classic method)

Usage:
    python download_data.py --search "hindi news"          # find candidate datasets
    python download_data.py --dataset owner/dataset-slug   # download + unzip
"""
import argparse
import os
import sys

import config

TOKEN_FILE = config.ROOT / "kaggle_token.txt"


def load_credentials():
    """Set Kaggle env vars from kaggle_token.txt. Must run BEFORE importing kaggle."""
    already_set = os.environ.get("KAGGLE_API_TOKEN") or (
        os.environ.get("KAGGLE_USERNAME") and os.environ.get("KAGGLE_KEY")
    )
    if already_set:
        print("Using Kaggle credentials from environment variables.")
        return
    if TOKEN_FILE.exists():
        value = TOKEN_FILE.read_text(encoding="utf8").strip()
        if not value:
            sys.exit(f"{TOKEN_FILE.name} is empty.")
        if ":" in value:
            os.environ["KAGGLE_USERNAME"], os.environ["KAGGLE_KEY"] = value.split(":", 1)
        else:
            os.environ["KAGGLE_API_TOKEN"] = value
        print(f"Using Kaggle credentials from {TOKEN_FILE.name}.")
    else:
        print("No env vars or kaggle_token.txt found - falling back to ~/.kaggle/kaggle.json")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--search", help="search Kaggle datasets by keyword")
    ap.add_argument("--dataset", help="owner/slug of the dataset to download")
    args = ap.parse_args()

    if not (args.search or args.dataset):
        ap.print_help()
        return

    load_credentials()

    try:
        # Newer kaggle package versions (1.7+) print a cosmetic "Authentication
        # required" notice on import (it checks for OAuth login first, finds
        # none, and warns) - safe to ignore as long as a token is set, since
        # token-based calls still work. Do NOT call api.authenticate() again
        # here: a second call re-triggers that same warning and, in practice,
        # can leave the api object unable to return results.
        import kaggle
        api = kaggle.api
    except Exception as e:
        sys.exit(f"Could not load the Kaggle API: {e}\n"
                 "Check your token (Settings -> API -> Create New Token), or run:\n"
                 "  pip install -U kaggle")

    if args.search:
        results = list(api.dataset_list(search=args.search))[:20]
        if not results:
            sys.exit("No datasets returned. If you saw an 'Authentication required' "
                     "notice above and got no results, your token may be invalid - "
                     "regenerate one at https://www.kaggle.com/settings/api")
        for d in results:
            print(f"{d.ref:60s} {getattr(d, 'total_bytes', '')!s:>12}  {d.title}")
    else:
        config.KAGGLE_DIR.mkdir(parents=True, exist_ok=True)
        api.dataset_download_files(args.dataset, path=str(config.KAGGLE_DIR), unzip=True)
        print("Downloaded to", config.KAGGLE_DIR)


if __name__ == "__main__":
    main() 