"""
Hindi spelling checker.

Approach (Norvig-style noisy-channel, adapted for Devanagari):
  * Vocabulary + word frequencies come from the Kaggle / scraped corpus.
  * A word not in the vocabulary is an error.
  * Candidates come from three sources, each with an edit "cost":
        0.0  word is already valid
        0.5  same phonetic key      (ी/ि, ू/ु, श/ष/स, ण/न, ँ/ं, nukta, न्द/ंद ...)
        1.0  one edit away          (delete / transpose / replace / insert)
        1.5  one edit + phonetic slip
        2.5  two edits (only if nothing closer exists, and only for short words)
  * Ranking:  log(freq) - EDIT_PENALTY*cost + CONTEXT_WEIGHT * bigram_context
"""
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

import config
from preprocessing import normalize_text, phonetic_key, tokenize_with_spans, tokenize_words

# Devanagari letters + signs used when generating edits.
ALPHABET = "".join(
    chr(c) for r in (range(0x0901, 0x0904), range(0x0905, 0x093A), range(0x093C, 0x094E))
    for c in r
)

def _is_nasalization_insertion(original: str, candidate: str) -> bool:
    """True if candidate = original with one ं or ँ inserted somewhere. Dropping the
    word-final nasal mark (पसंद -> पसद, किताबें -> किताबे) is probably the single most
    common shortcut in casual Hindi typing, so this edit type gets a lower cost than
    a generic insertion/deletion."""
    if len(candidate) != len(original) + 1:
        return False
    for i, ch in enumerate(candidate):
        if ch in ("ं", "ँ") and candidate[:i] + candidate[i + 1:] == original:
            return True
    return False


NASALIZATION_COST = 0.08  # lower than the phonetic tier (0.5): a dropped ं/ँ is a very
                          # reliable signal on its own, so this should win even against
                          # an unrelated word that's up to ~40x more frequent
                          
def edits1(word: str) -> set:
    splits = [(word[:i], word[i:]) for i in range(len(word) + 1)]
    deletes = [a + b[1:] for a, b in splits if b]
    transposes = [a + b[1] + b[0] + b[2:] for a, b in splits if len(b) > 1]
    replaces = [a + c + b[1:] for a, b in splits if b for c in ALPHABET]
    inserts = [a + c + b for a, b in splits for c in ALPHABET]
    return set(deletes + transposes + replaces + inserts)


class HindiSpellChecker:
    def __init__(self, word_freq, bigrams=None, min_freq=1,
                 edit_penalty=config.EDIT_PENALTY, context_weight=config.CONTEXT_WEIGHT):
        self.freq = {w: c for w, c in word_freq.items() if c >= min_freq}
        self.bigrams = bigrams or {}
        self.edit_penalty = edit_penalty
        self.context_weight = context_weight
        self.key_index = defaultdict(list)
        for w in self.freq:
            self.key_index[phonetic_key(w)].append(w)

    # ------------------------------------------------------------ construction
    @classmethod
    def load(cls, path=config.VOCAB_FILE, min_freq=1, **kw):
        data = json.loads(Path(path).read_text(encoding="utf8"))
        bigrams = {tuple(k.split(" ", 1)): v for k, v in data.get("bigrams", {}).items()}
        return cls(data["words"], bigrams, min_freq=min_freq, **kw)

    @classmethod
    def from_sentences(cls, sentences, min_freq=1, **kw):
        words, bigrams = Counter(), Counter()
        for s in sentences:
            toks = tokenize_words(normalize_text(s))
            words.update(toks)
            bigrams.update(zip(toks, toks[1:]))
        return cls(words, dict(bigrams), min_freq=min_freq, **kw)

    # ------------------------------------------------------------ core API
    def is_known(self, word: str) -> bool:
        return word in self.freq

    def candidates(self, word: str) -> dict:
        """Return {candidate: edit_cost}."""
        if word in self.freq:
            return {word: 0.0}
        cands = {}
        key = phonetic_key(word)
        for c in self.key_index.get(key, ()):
            cands[c] = 0.5
        for e in edits1(word):
            if e in self.freq:
                cost = NASALIZATION_COST if _is_nasalization_insertion(word, e) else 1.0
                if e not in cands or cost < cands[e]:
                    cands[e] = cost
        for e in edits1(key):
            for c in self.key_index.get(e, ()):
                cands.setdefault(c, 1.5)
        if cands or len(word) > 9:
            return cands
        for e1 in edits1(key):
            for e2 in edits1(e1):
                for c in self.key_index.get(e2, ()):
                    cands.setdefault(c, 2.5)
        return cands

    def _context(self, cand, prev, nxt) -> float:
        score = 0.0
        if prev:
            score += math.log1p(self.bigrams.get((prev, cand), 0))
        if nxt:
            score += math.log1p(self.bigrams.get((cand, nxt), 0))
        return score

    def suggest(self, word: str, prev=None, nxt=None, n: int = 5):
        """Ranked [(candidate, score)]; empty list if there is nothing plausible."""
        cands = self.candidates(word)
        scored = [
            (c, math.log(self.freq[c]) - self.edit_penalty * cost
             + self.context_weight * self._context(c, prev, nxt))
            for c, cost in cands.items()
        ]
        scored.sort(key=lambda x: (-x[1], x[0]))
        return scored[:n]

    def correct(self, word: str, prev=None, nxt=None) -> str:
        if self.is_known(word):
            return word
        s = self.suggest(word, prev, nxt, n=1)
        return s[0][0] if s else word

    # ------------------------------------------------------------ text level
    def check_text(self, text: str, n: int = 5):
        """Return one dict per Devanagari word: word, start, end, known, suggestions."""
        text = normalize_text(text)
        spans = tokenize_with_spans(text)
        out = []
        for i, (w, s, e) in enumerate(spans):
            known = self.is_known(w)
            prev = spans[i - 1][0] if i > 0 else None
            nxt = spans[i + 1][0] if i + 1 < len(spans) else None
            sugg = [] if known else self.suggest(w, prev, nxt, n)
            out.append({"word": w, "start": s, "end": e, "known": known,
                        "suggestions": [c for c, _ in sugg]})
        return text, out

    def correct_text(self, text: str) -> str:
        text, results = self.check_text(text, n=1)
        for r in reversed(results):
            if not r["known"] and r["suggestions"]:
                text = text[: r["start"]] + r["suggestions"][0] + text[r["end"]:]
        return text

    def stats(self) -> dict:
        return {"vocabulary_size": len(self.freq),
                "total_tokens": sum(self.freq.values()),
                "bigrams": len(self.bigrams)}
