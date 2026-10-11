"""Central paths and tunable settings for the Hindi spelling checker."""
from pathlib import Path

ROOT = Path(__file__).parent
RAW_DIR = ROOT / "data" / "raw"
KAGGLE_DIR = RAW_DIR / "kaggle"
SCRAPED_FILE = RAW_DIR / "scraped_corpus.txt"
SEED_FILE = ROOT / "data" / "seed_corpus.txt"      # tiny corpus, used for tests / demo fallback
PROCESSED_DIR = ROOT / "data" / "processed"
VOCAB_FILE = PROCESSED_DIR / "vocab.json"

# Words seen fewer times than this are treated as noise (typos inside the corpus).
MIN_WORD_COUNT = 3
MIN_BIGRAM_COUNT = 2

# Ranking: score = log(freq) - EDIT_PENALTY * cost + CONTEXT_WEIGHT * context
EDIT_PENALTY = 4.0
CONTEXT_WEIGHT = 1.0
