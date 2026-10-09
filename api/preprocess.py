"""Text cleaning shared by train.py and the API (same steps in both places)."""
import re
from functools import lru_cache

import nltk
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS


def _load_stopwords():
    try:
        from nltk.corpus import stopwords
        try:
            return set(stopwords.words("english"))
        except LookupError:
            nltk.download("stopwords", quiet=True)
            return set(stopwords.words("english"))
    except Exception:
        # No internet / NLTK data missing: use scikit-learn's built-in list.
        return set(ENGLISH_STOP_WORDS)


_stemmer = PorterStemmer()
STOP_WORDS = _load_stopwords()


@lru_cache(maxsize=50000)
def _stem(word):
    return _stemmer.stem(word)


def clean_text(text):
    text = str(text)
    text = re.sub(r"<[^>]+>", " ", text)                  # remove HTML tags
    text = re.sub(r"http\S+|www\.\S+", " url ", text)     # replace links
    text = re.sub(r"[^a-zA-Z]", " ", text).lower()        # letters only, lowercase
    return " ".join(_stem(w) for w in text.split() if w not in STOP_WORDS)
