# Hindi Spelling Checker

Pipeline: **Kaggle data -> scraper -> NLP preprocessing -> vocabulary/model -> model check -> Streamlit demo**

## Run order

```
pip install -r requirements.txt

python test_model.py                       # 0. sanity-check the model (offline, no data needed)

python download_data.py --search "hindi"   # 1. find a Kaggle dataset  (see Kaggle setup below)
python download_data.py --dataset owner/slug
python scraper.py wiki --requests 100 --delay 3   # 2. scrape Hindi Wikipedia (optional, run either/both)
python build_vocab.py --min-count 3        # 3. NLP preprocessing -> data/processed/vocab.json
python test_model.py --vocab               # 4. verify the REAL model
streamlit run app.py                       # 5. demo
```

Steps 1 and 2 can run in either order, or you can use just one of them. `build_vocab.py`
reads everything under `data/raw/`, so run it after at least one of them has produced data.
If you run `build_vocab.py` with no data at all, it falls back to a tiny seed corpus for
demo purposes only - good enough to prove the pipeline works, too small for real accuracy.

## Kaggle setup

Credentials are checked in this order:

1. Environment variables `KAGGLE_USERNAME` + `KAGGLE_KEY` (or `KAGGLE_API_TOKEN` for the
   newer single-token accounts).
2. A `kaggle_token.txt` file next to the scripts, containing either `username:key` or a
   single token string. This file is already in `.gitignore` - never commit it.
3. The classic `~/.kaggle/kaggle.json`.

Get a token from kaggle.com -> your profile -> Settings -> API -> Create New Token.

## Scraping notes

`scraper.py wiki` backs off automatically on HTTP 429 (rate limited), honoring Wikipedia's
`Retry-After` header, and gives up cleanly after 6 failed retries. Ctrl+C stops it safely -
everything scraped so far is already saved to `data/raw/scraped_corpus.txt`, and reruns skip
duplicates. Set your email in the `UA` string at the top of `scraper.py` per Wikimedia's API
etiquette.

`scraper.py web --seed URL` is a generic crawler for any site you have permission to scrape -
it obeys robots.txt, rate-limits itself, and stays on the seed's domain. Check each site's
terms of use first.

## Files

| File | Role (Urdu-repo equivalent) |
|---|---|
| `preprocessing.py` | normalize, tokenize, stopwords, stemming, phonetic key (`preprocessing.py`) |
| `spellchecker.py` | the model: candidates + ranking (`model.py` / `predict.py`) |
| `download_data.py` | Kaggle dataset search/download (step 1) |
| `scraper.py` | Wikipedia / generic web scraper (step 2) |
| `build_vocab.py` | corpus -> vocab.json (`dataset_processing.py`, `prepare_training_data.py`) |
| `test_model.py` | correctness + accuracy check (`data_test.py`, `evaluate.py`) |
| `app.py` | Streamlit frontend (`app.py`) |

Tuning knobs (`config.py` or sidebar in the app): `MIN_WORD_COUNT` (higher = ignore typos
that exist in the corpus), `EDIT_PENALTY`, `CONTEXT_WEIGHT`.

Known limits: detects non-word errors only (not real-word errors like `का`/`की`); quality
depends on corpus size.
