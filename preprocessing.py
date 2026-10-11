"""
Hindi NLP preprocessing (counterpart of the Urdu repo's preprocessing.py).

Pure-Python, no external NLP model needed:
    1. Normalization   - Unicode NFC, strip zero-width chars, unify punctuation/whitespace
    2. Tokenization    - Devanagari word tokens (keeps character offsets)
    3. Stopword removal
    4. Light stemming  - rule-based suffix stripping
    5. Phonetic key    - collapses common Hindi spelling confusions (used by the spell checker)
"""
import re
import unicodedata

# ---------------------------------------------------------------- 1. NORMALIZATION
_ZERO_WIDTH = dict.fromkeys(map(ord, "\u200b\u200c\u200d\u2060\ufeff"), None)
_PUNCT_MAP = str.maketrans({"॥": "।", "\u00a0": " ", "“": '"', "”": '"', "‘": "'", "’": "'"})


def normalize_text(text) -> str:
    """NFC-normalize (this also splits nukta letters like क़ into क + ़), drop
    zero-width characters, unify punctuation and collapse whitespace."""
    if text is None or (isinstance(text, float) and text != text):  # None / NaN
        return ""
    text = unicodedata.normalize("NFC", str(text))
    text = text.translate(_ZERO_WIDTH).translate(_PUNCT_MAP)
    return re.sub(r"\s+", " ", text).strip()


# ---------------------------------------------------------------- 2. TOKENIZATION
# Devanagari letters/signs/matras, excluding danda (U+0964/0965) and Devanagari digits.
WORD_RE = re.compile(r"[\u0900-\u0963\u0971-\u097F]+")
_LETTER_RE = re.compile(r"[\u0904-\u0939\u0958-\u0961]")
_SENT_SPLIT_RE = re.compile(r"[।!?\n]+|\.(?:\s|$)")


def has_devanagari(text: str) -> bool:
    return bool(_LETTER_RE.search(text or ""))


def devanagari_ratio(text: str) -> float:
    letters = [c for c in text if c.isalpha()]
    if not letters:
        return 0.0
    return sum(1 for c in letters if "\u0900" <= c <= "\u097f") / len(letters)


def tokenize_words(text: str):
    """Return Devanagari word tokens (no punctuation, no digits, no Latin)."""
    return [m.group() for m in WORD_RE.finditer(text or "") if _LETTER_RE.search(m.group())]


def tokenize_with_spans(text: str):
    """Return [(word, start, end)] - used by the spell checker to rewrite text in place."""
    return [(m.group(), m.start(), m.end()) for m in WORD_RE.finditer(text or "")
            if _LETTER_RE.search(m.group())]


def split_sentences(text: str):
    return [s.strip() for s in _SENT_SPLIT_RE.split(text or "") if s and s.strip()]


def tokenize_text(text: str):
    """Tokens including punctuation, like the Urdu pipeline (e.g. '।')."""
    return re.findall(r"[\u0900-\u0963\u0971-\u097F]+|[।?!,.;:\-\"'()]", text or "")


# ---------------------------------------------------------------- 3. STOPWORDS
HINDI_STOPWORDS = set("""
अपना अपनी अपने अभी आदि आप इन इनका इन्हीं इन्हें इन्हों इस इसका इसकी इसके इसमें इसी इसे उन उनका उनकी उनके
उनको उन्हीं उन्हें उन्हों उस उसके उसी उसे एक एवं ऐसे और कई कर करता करते करना करने करें कहते कहा का काफ़ी
कि कितना किन्हें किन्हों किया किर किस किसी किसे की कुछ कुल के को कोई कौन कौनसा गया घर जब जहाँ जा जितना जिन
जिन्हें जिन्हों जिस जिसे जैसा जैसे जो तक तब तरह तो था थी थे दिया दुसरा
दूसरे दो द्वारा न नहीं ना नीचे ने पर पहले पूरा पे फिर बनी बहुत बाद बाला बिलकुल भी भीतर मगर मे में
यदि यह यहाँ यही या ये रखें रहा रहे लिए लिये लेकिन व वर्ग वह वहाँ वहीं वाले वे वो सकता
सकते सबसे सभी साथ सारा से सो संग ही हुआ हुई हुए है हैं हो होता होती होते होना होने हूँ हम हमें हमारा
हमारी हमारे मैं मुझे मेरा मेरी मेरे तुम तुम्हारा आपका आपकी
""".split())


def remove_stopwords(tokens):
    return [t for t in tokens if t not in HINDI_STOPWORDS]


# ---------------------------------------------------------------- 4. STEMMING
_SUFFIXES = sorted([
    "ाइयाँ", "ाइयों", "ियाँ", "ियों", "ाओं", "ाएँ", "ाएं", "ाना", "ाता", "ाती", "ाते", "ाया", "ाये",
    "ेगा", "ेगी", "ेंगे", "ेंगी", "ूँगा", "ूँगी", "ीयों", "ीय",
    "ों", "ें", "ीं", "ाँ", "ता", "ती", "ते", "ना", "ने", "ा", "े", "ी", "ो",
], key=len, reverse=True)


def stem_word(word: str, min_len: int = 2) -> str:
    """Light rule-based stemmer: strip the longest matching inflectional suffix."""
    for suf in _SUFFIXES:
        if word.endswith(suf) and len(word) - len(suf) >= min_len:
            return word[: -len(suf)]
    return word


def stem_tokens(tokens):
    return [stem_word(t) for t in tokens]


# ---------------------------------------------------------------- 5. PHONETIC KEY
# Collapses spellings Hindi writers commonly confuse, so "परीवार" and "परिवार",
# "सिक्षा" and "शिक्षा", "हिन्दी" and "हिंदी" all land on the same key.
_KEY_MAP = str.maketrans({
    "ी": "ि", "ू": "ु", "ँ": "ं", "ष": "स", "श": "स", "ण": "न", "ॉ": "ा", "ऑ": "आ", "ॅ": "े",
    "ङ": "न", "ञ": "न", "़": None, "ऩ": "न", "ऱ": "र", "ऴ": "ळ",
    "ौ": "ो", "ै": "े",  # vowel-length/diphthong confusions (मौसम/मोसम, है/हे)
})
_NASAL_CONJUNCT = re.compile(r"[नमणङञ]्(?=[\u0915-\u0939])")


def phonetic_key(word: str) -> str:
    word = _NASAL_CONJUNCT.sub("ं", word)
    return word.translate(_KEY_MAP)


# ---------------------------------------------------------------- PIPELINE
def preprocess_text(text) -> str:
    """Normalize -> tokenize -> remove stopwords -> stem. Returns a string."""
    tokens = tokenize_words(normalize_text(text))
    return " ".join(stem_tokens(remove_stopwords(tokens)))


def analyze_text(text) -> dict:
    """Every stage of the pipeline (for Streamlit / demo / report)."""
    normalized = normalize_text(text)
    tokens = tokenize_text(normalized)
    words = tokenize_words(normalized)
    no_stop = remove_stopwords(words)
    return {
        "original": text,
        "normalized": normalized,
        "sentences": split_sentences(normalized),
        "tokens": tokens,
        "stopword_removed": no_stop,
        "stemmed": stem_tokens(no_stop),
    }


if __name__ == "__main__":
    sample = "मैं भारत से हूँ और मुझे हिंदी भाषा बहुत पसंद है।"
    for k, v in analyze_text(sample).items():
        print(f"{k:>17}: {v}")
