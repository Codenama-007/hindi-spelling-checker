"""
Step 3 - turn Kaggle + scraped text into the spell checker's vocabulary.

    python build_vocab.py                    # uses data/raw/** (falls back to the seed corpus)
    python build_vocab.py --text-cols headline,content --min-count 5

Reads .csv/.tsv/.txt/.json/.jsonl under data/raw/kaggle plus data/raw/scraped_corpus.txt.
Text columns in CSVs are auto-detected (columns that are mostly Devanagari).
Writes data/processed/vocab.json  (word counts + bigram counts).
"""
import argparse
import json
from collections import Counter, defaultdict

import pandas as pd

import config
from preprocessing import devanagari_ratio, has_devanagari, normalize_text, phonetic_key, split_sentences, tokenize_words


def iter_text_files():
    if config.KAGGLE_DIR.exists():
        for p in sorted(config.KAGGLE_DIR.rglob("*")):
            if p.suffix.lower() in {".csv", ".tsv", ".txt", ".json", ".jsonl"}:
                yield p
    if config.SCRAPED_FILE.exists():
        yield config.SCRAPED_FILE


def _read_frame(path):
    if path.suffix.lower() == ".json":
        return pd.read_json(path)
    if path.suffix.lower() == ".jsonl":
        return pd.read_json(path, lines=True)
    sep = "\t" if path.suffix.lower() == ".tsv" else ","
    return pd.read_csv(path, sep=sep, on_bad_lines="skip", low_memory=False)


def iter_documents(path, text_cols):
    """Yield raw text documents from a file."""
    if path.suffix.lower() == ".txt":
        with open(path, encoding="utf8", errors="ignore") as f:
            yield from f
        return
    df = _read_frame(path)
    cols = text_cols or [c for c in df.columns if df[c].dtype == object and
                         df[c].dropna().astype(str).head(200).map(devanagari_ratio).mean() > 0.3]
    print(f"  {path.name}: {len(df):,} rows, text columns -> {cols or 'NONE FOUND'}")
    for c in cols:
        yield from df[c].dropna().astype(str)


def count_lines(documents, counts=None, bigrams=None):
    """Tokenise documents sentence by sentence; accumulate word and bigram counts."""
    counts = Counter() if counts is None else counts
    bigrams = Counter() if bigrams is None else bigrams
    for doc in documents:
        if not has_devanagari(doc):
            continue
        for sent in split_sentences(normalize_text(doc)):
            toks = tokenize_words(sent)
            counts.update(toks)
            bigrams.update(zip(toks, toks[1:]))
    return counts, bigrams


def filter_minority_spellings(words: dict, min_ratio: float, min_cluster_count: int):
    """A large raw corpus contains real typos too (e.g. 'सिक्षा' for 'शिक्षा'), and
    if they occur often enough they pass the --min-count filter and get treated as
    valid words, which defeats the spell checker. Within each phonetic cluster
    (words that normalize to the same phonetic_key - e.g. श/ष/स, ी/ि confusions),
    keep only the dominant spelling(s); drop minority spellings that are rare
    relative to the cluster's most common member. Clusters whose top member is
    below `min_cluster_count` are left alone - too rare to confidently call one
    spelling "correct" over another."""
    clusters = defaultdict(list)
    for w, c in words.items():
        clusters[phonetic_key(w)].append((w, c))
    keep, dropped = {}, []
    for members in clusters.values():
        top_count = max(c for _, c in members)
        if len(members) == 1 or top_count < min_cluster_count:
            keep.update(members)
            continue
        for w, c in members:
            if c >= top_count * min_ratio:
                keep[w] = c
            else:
                dropped.append((w, c))
    if dropped:
        dropped.sort(key=lambda x: -x[1])
        sample = ", ".join(f"{w}({c})" for w, c in dropped[:10])
        print(f"  Dropped {len(dropped)} minority spellings within phonetic clusters "
              f"(e.g. {sample})")
    return keep

# Specific, well-known Hindi typos that don't share a phonetic_key with their correct
# spelling (so the cluster filter above can't catch them) but are common enough, even
# in semi-formal text, to clear --min-count on a large corpus. Hand-curated, so there's
# no risk of this wrongly merging two distinct real words the way a blanket phonetic
# rule could (e.g. द/ध generally: दान "donation" vs धान "paddy" are both real, different
# words - a blanket merge would risk deleting whichever is rarer in the corpus).
# Add more pairs here as you spot them during testing.
COMMON_MISSPELLINGS = {
    "विध्यार्थी": "विद्यार्थी",
    "विध्या": "विद्या",
    "विध्यालय": "विद्यालय",
}


def remove_common_misspellings(words: dict):
    removed = []
    for typo, correct in COMMON_MISSPELLINGS.items():
        if typo in words and correct in words and words[correct] > words[typo]:
            removed.append((typo, words.pop(typo)))
    if removed:
        print(f"  Removed {len(removed)} words from the curated common-misspellings "
              f"list: {', '.join(f'{w}({c})' for w, c in removed)}")
    return words
def filter_missing_trailing_nasal(words: dict, min_ratio: float, min_cluster_count: int):
    """Catches a very common Hindi typo pattern the phonetic-cluster filter can't:
    dropping the word-final nasal mark entirely (किताबें -> किताबे, लड़कियां -> लड़किया).
    These don't share a phonetic_key since one word is simply the other with ं or ँ
    appended, not a substitution. For every such pair where both spellings are in the
    vocabulary, keep only the dominant one - same relative-frequency rule as the
    phonetic-cluster filter above."""
    pairs = [(w, w + mark) for w in words for mark in ("ं", "ँ") if w + mark in words]
    to_drop = set()
    for bare, nasal in pairs:
        c_bare, c_nasal = words[bare], words[nasal]
        if max(c_bare, c_nasal) < min_cluster_count:
            continue
        if c_bare < c_nasal * min_ratio:
            to_drop.add(bare)
        elif c_nasal < c_bare * min_ratio:
            to_drop.add(nasal)
    for w in to_drop:
        words.pop(w, None)
    if to_drop:
        print(f"  Dropped {len(to_drop)} words missing/extra a trailing ं/ँ relative "
              f"to a much more common counterpart spelling")
    return words

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--text-cols", help="comma-separated CSV columns to use")
    ap.add_argument("--min-count", type=int, default=config.MIN_WORD_COUNT)
    ap.add_argument("--min-bigram", type=int, default=config.MIN_BIGRAM_COUNT)
    ap.add_argument("--max-bigrams", type=int, default=500_000)
    ap.add_argument("--min-ratio", type=float, default=0.03,
                    help="within a phonetic cluster, drop spellings below this fraction "
                         "of the cluster's top count")
    ap.add_argument("--min-cluster-count", type=int, default=50,
                    help="only apply cluster filtering when the cluster's top word "
                         "occurs at least this many times")
    ap.add_argument("--no-cluster-filter", action="store_true",
                    help="skip the phonetic-cluster minority filter")
    args = ap.parse_args()
    cols = args.text_cols.split(",") if args.text_cols else None

    counts, bigrams = Counter(), Counter()
    files = list(iter_text_files())
    if not files:
        print("No raw data found - using the tiny seed corpus (demo only).")
        files, args.min_count, args.min_bigram = [config.SEED_FILE], 1, 1
    for p in files:
        print("Reading", p)
        count_lines(iter_documents(p, cols), counts, bigrams)

    words = {w: c for w, c in counts.items() if c >= args.min_count}
    before = len(words)
    if not args.no_cluster_filter:
        words = filter_minority_spellings(words, args.min_ratio, args.min_cluster_count)
        words = filter_missing_trailing_nasal(words, args.min_ratio, args.min_cluster_count)
    words = remove_common_misspellings(words)
    print(f"  Vocabulary: {before:,} -> {len(words):,} words after filtering")
    keep_bi = {f"{a} {b}": c for (a, b), c in bigrams.most_common(args.max_bigrams)
               if c >= args.min_bigram and a in words and b in words}
    config.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    config.VOCAB_FILE.write_text(json.dumps({"words": words, "bigrams": keep_bi},
                                            ensure_ascii=False), encoding="utf8")
    print(f"\nTokens: {sum(counts.values()):,} | raw types: {len(counts):,} | "
          f"kept words: {len(words):,} | bigrams: {len(keep_bi):,}\nSaved -> {config.VOCAB_FILE}")


if __name__ == "__main__":
    main()