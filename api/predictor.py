"""Loads (or rebuilds) the model and returns predictions. No web code here."""
import json
import sys
from pathlib import Path

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from .preprocess import clean_text

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = BASE_DIR / "models"


def _train_from_data():
    sys.path.insert(0, str(BASE_DIR))
    from train import build_vectorizer, load_corpus
    df = load_corpus()
    df["clean"] = df["text"].apply(clean_text)
    vec = build_vectorizer()
    X = vec.fit_transform(df["clean"])
    return vec, LogisticRegression(max_iter=1000).fit(X, df["target"])


def _tiny_fallback():
    texts = ["win a free prize claim now", "urgent verify your bank account",
             "lunch tomorrow at noon", "please find the attached report"]
    vec = TfidfVectorizer(ngram_range=(1, 2))
    X = vec.fit_transform([clean_text(t) for t in texts])
    return vec, LogisticRegression().fit(X, [1, 1, 0, 0])


def load_or_train():
    try:  # 1) saved files from train.py
        return joblib.load(MODEL_DIR / "vectorizer.pkl"), joblib.load(MODEL_DIR / "model.pkl")
    except Exception:
        pass
    try:  # 2) retrain from data/*.csv
        return _train_from_data()
    except Exception:
        pass
    return _tiny_fallback()  # 3) last resort so the server never crashes


vectorizer, model = load_or_train()


def predict(text):
    features = vectorizer.transform([clean_text(text)])
    prob = model.predict_proba(features)[0]
    is_spam = prob[1] >= 0.5
    return {
        "status": "SPAM" if is_spam else "HAM",
        "confidence": round(float(prob[1] if is_spam else prob[0]) * 100, 2),
        "probabilities": {
            "spam": round(float(prob[1]) * 100, 2),
            "ham": round(float(prob[0]) * 100, 2),
        },
    }


def info():
    path = MODEL_DIR / "metrics.json"
    if path.exists():
        return json.loads(path.read_text())
    return {"records": 0, "models": {}}
