"""
Is the spell checker working correctly?  Run:

    python test_model.py                 # self-contained: builds a model from the seed corpus
    python test_model.py --vocab         # test the real model in data/processed/vocab.json
    python test_model.py --vocab --top3 0.85 --samples 2000

Checks
  1. Preprocessing sanity (normalize / tokenize / phonetic key)
  2. Known words are never "corrected"
  3. Hand-written typo cases (matra, sibilant, nasal, missing / extra letter)
  4. Sentence-level detection and correction
  5. Synthetic evaluation: corrupt real vocabulary words, measure top-1 / top-3 accuracy
  6. Speed
Exit code is 1 if any check fails, so it can also run in CI.
"""
import argparse
import random
import sys
import time

import config
from preprocessing import normalize_text, phonetic_key, tokenize_words
from spellchecker import ALPHABET, HindiSpellChecker

PASS, FAIL = "PASS", "FAIL"
results = []


def check(name, ok, detail=""):
    results.append(ok)
    print(f"  [{PASS if ok else FAIL}] {name}" + (f"  ({detail})" if detail else ""))


# ------------------------------------------------------------------ 1. preprocessing
def test_preprocessing():
    print("\n1. Preprocessing")
    check("zero-width chars removed", normalize_text("भा\u200dरत") == "भारत")
    check("whitespace collapsed", normalize_text("  भारत   देश ") == "भारत देश")
    check("tokenizer ignores Latin/digits/punctuation",
          tokenize_words("मैं India से 5 दिन में आया।") == ["मैं", "से", "दिन", "में", "आया"])
    check("phonetic key: ी/ि", phonetic_key("परीवार") == phonetic_key("परिवार"))
    check("phonetic key: श/ष/स", phonetic_key("सिक्षा") == phonetic_key("शिक्षा"))
    check("phonetic key: न्द/ंद", phonetic_key("हिन्दी") == phonetic_key("हिंदी"))


# ------------------------------------------------------------------ 2/3. words
KNOWN = ["भारत", "शिक्षा", "विद्यार्थी", "परिवार", "पुस्तकालय"]
TYPOS = {  # typo -> expected correction (all words exist in the seed corpus)
    "सिक्षा": "शिक्षा",         # sibilant confusion
    "परीवार": "परिवार",         # long/short ई
    "विध्यार्थी": "विद्यार्थी",   # द -> ध
    "मोसम": "मौसम",            # ौ -> ो
    "किताबे": "किताबें",         # missing ें - a deletion typo; on a large real corpus
    "पसद": "पसंद",            # missing ं    this needs --min-count tuning, not just the
    "पुस्तकलय": "पुस्तकालय",     # missing ा    phonetic-cluster filter, since deleted
    "मेहनथ": "मेहनत",          # wrong letter  characters don't share a phonetic key.
    "बहूत": "बहुत",            # long/short ऊ
    # Note: 'भरत' (a real Hindi name/word) was deliberately left out - it isn't
    # actually a typo of 'भारत', so it isn't a fair test case.
}


def test_words(sc):
    print("\n2. Known words stay untouched")
    for w in KNOWN:
        if sc.is_known(w):
            check(f"'{w}' unchanged", sc.correct(w) == w)
        else:
            print(f"  [SKIP] '{w}' not in this vocabulary")

    print("\n3. Hand-written typo cases")
    hits = 0
    for typo, gold in TYPOS.items():
        if not sc.is_known(gold):
            print(f"  [SKIP] '{gold}' not in this vocabulary")
            continue
        top = [c for c, _ in sc.suggest(typo, n=3)]
        ok = bool(top) and top[0] == gold
        hits += ok
        check(f"{typo} -> {gold}", ok, f"got {top[:3]}")
    return hits


# ------------------------------------------------------------------ 4. sentences
def test_sentences(sc):
    print("\n4. Sentence level")
    bad = "मुझे हिंदी भाषा बहूत पसद है।"
    text, res = sc.check_text(bad)
    flagged = [r["word"] for r in res if not r["known"]]
    check("flags exactly the misspelled words", set(flagged) == {"बहूत", "पसद"}, f"flagged {flagged}")
    check("corrects sentence", sc.correct_text(bad) == "मुझे हिंदी भाषा बहुत पसंद है।",
          sc.correct_text(bad))
    good = "मुझे हिंदी भाषा बहुत पसंद है।"
    check("clean sentence unchanged", sc.correct_text(good) == good)
    check("English / digits ignored", sc.correct_text("Hello 123 भारत") == "Hello 123 भारत")
    check("empty input safe", sc.correct_text("") == "")


# ------------------------------------------------------------------ 5. synthetic evaluation
SWAPS = [("ि", "ी"), ("ी", "ि"), ("ु", "ू"), ("ू", "ु"), ("श", "ष"), ("ष", "श"), ("स", "श"),
         ("ं", "ँ"), ("ँ", "ं"), ("ौ", "ो"), ("ै", "े"), ("द", "ध"), ("त", "थ"), ("ड", "ढ")]


def corrupt(word, rng):
    """Apply one realistic error; return None if no error could be applied."""
    kind = rng.choice(["swap", "delete", "insert", "transpose"])
    if kind == "swap":
        opts = [(i, b) for i, ch in enumerate(word) for a, b in SWAPS if ch == a]
        if not opts:
            return None
        i, b = rng.choice(opts)
        return word[:i] + b + word[i + 1:]
    if kind == "delete":
        i = rng.randrange(1, len(word))
        return word[:i] + word[i + 1:]
    if kind == "insert":
        i = rng.randrange(1, len(word))
        return word[:i] + rng.choice("ाेीकरनतमस") + word[i:]
    i = rng.randrange(0, len(word) - 1)
    return word[:i] + word[i + 1] + word[i] + word[i + 2:]


def test_synthetic(sc, samples, top3_min, top1_min, seed=7):
    print(f"\n5. Synthetic evaluation ({samples} corrupted words)")
    rng = random.Random(seed)
    pool = sorted((w for w, c in sc.freq.items() if len(w) >= 4 and c >= 2), key=lambda w: -sc.freq[w])
    pool = pool[:5000] or sorted(w for w in sc.freq if len(w) >= 4)
    t1 = t3 = n = 0
    misses = []
    start = time.time()
    while n < samples and pool:
        gold = rng.choice(pool)
        typo = corrupt(gold, rng)
        if not typo or typo == gold or sc.is_known(typo):
            continue
        top = [c for c, _ in sc.suggest(typo, n=3)]
        n += 1
        t1 += bool(top) and top[0] == gold
        t3 += gold in top
        if gold not in top and len(misses) < 8:
            misses.append((typo, gold, top))
        if time.time() - start > 120:
            break
    elapsed = time.time() - start
    a1, a3 = t1 / max(n, 1), t3 / max(n, 1)
    print(f"     evaluated {n} | top-1 = {a1:.1%} | top-3 = {a3:.1%} | {1000 * elapsed / max(n, 1):.1f} ms/word")
    for typo, gold, top in misses:
        print(f"     miss: {typo} (wanted {gold}) got {top}")
    check(f"top-1 accuracy >= {top1_min:.0%}", a1 >= top1_min, f"{a1:.1%}")
    check(f"top-3 accuracy >= {top3_min:.0%}", a3 >= top3_min, f"{a3:.1%}")
    return a1, a3, 1000 * elapsed / max(n, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vocab", action="store_true", help="use data/processed/vocab.json instead of the seed corpus")
    ap.add_argument("--samples", type=int, default=500)
    ap.add_argument("--top1", type=float, default=0.60)
    ap.add_argument("--top3", type=float, default=0.80)
    ap.add_argument("--min-freq", type=int, default=1)
    args = ap.parse_args()

    if args.vocab:
        print(f"Loading {config.VOCAB_FILE}")
        sc = HindiSpellChecker.load(min_freq=args.min_freq)
    else:
        print(f"Building model from seed corpus {config.SEED_FILE}")
        lines = config.SEED_FILE.read_text(encoding="utf8").splitlines()
        sc = HindiSpellChecker.from_sentences(lines)
    print("Model:", sc.stats())

    test_preprocessing()
    test_words(sc)
    test_sentences(sc)
    _, _, ms = test_synthetic(sc, args.samples, args.top3, args.top1)
    check("speed under 250 ms/word", ms < 250, f"{ms:.1f} ms")

    passed = sum(results)
    print(f"\n{'=' * 50}\n{passed}/{len(results)} checks passed")
    if passed != len(results):
        print("MODEL CHECK FAILED")
        sys.exit(1)
    print("MODEL CHECK PASSED")


if __name__ == "__main__":
    main()
