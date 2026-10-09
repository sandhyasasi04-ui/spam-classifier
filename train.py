"""Train the spam models, print the comparison, save the production model."""
import json
import sys
from pathlib import Path

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, f1_score, precision_score,
                             recall_score)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC

BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR))
from api.preprocess import clean_text  # noqa: E402

DATA_DIR = BASE_DIR / "data"
MODEL_DIR = BASE_DIR / "models"
OUT_DIR = BASE_DIR / "outputs"

LABEL_COLS = ["label", "v1", "category", "class", "target"]
TEXT_COLS = ["text", "message", "v2", "email", "body", "content"]


def load_corpus():
    """Merge every CSV in data/ into one corpus with columns: label, text."""
    frames = []
    for path in sorted(DATA_DIR.glob("*.csv")):
        df = pd.read_csv(path, encoding="latin-1")
        cols = {c.lower().strip(): c for c in df.columns}
        label_col = next((cols[c] for c in LABEL_COLS if c in cols), None)
        text_col = next((cols[c] for c in TEXT_COLS if c in cols), None)
        if label_col is None or text_col is None:
            print(f"[skip] {path.name}: could not find label/text columns")
            continue
        part = df[[label_col, text_col]].copy()
        part.columns = ["label", "text"]
        frames.append(part)
        print(f"[load] {path.name}: {len(part)} rows")
    if not frames:
        raise SystemExit("No usable CSV found in data/. Put spam.csv there.")
    df = pd.concat(frames, ignore_index=True).dropna().drop_duplicates()
    df["target"] = df["label"].astype(str).str.lower().str.strip().isin(["spam", "1"]).astype(int)
    return df


def build_vectorizer():
    return TfidfVectorizer(max_features=5000, ngram_range=(1, 2))


def main():
    df = load_corpus()
    df["clean"] = df["text"].apply(clean_text)
    df = df[df["clean"].str.len() > 0]
    print(f"\nMaster corpus: {len(df)} records "
          f"({int(df.target.sum())} spam, {int((1 - df.target).sum())} ham)")

    vectorizer = build_vectorizer()
    X = vectorizer.fit_transform(df["clean"])
    y = df["target"]
    print(f"TF-IDF feature matrix: shape {X.shape}\n")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42)

    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Naive Bayes": MultinomialNB(),
        "SVM": LinearSVC(),
    }
    metrics = {}
    for name, clf in models.items():
        clf.fit(X_train, y_train)
        pred = clf.predict(X_test)
        metrics[name] = {
            "accuracy": round(accuracy_score(y_test, pred) * 100, 2),
            "precision": round(precision_score(y_test, pred, zero_division=0) * 100, 2),
            "recall": round(recall_score(y_test, pred) * 100, 2),
            "f1": round(f1_score(y_test, pred) * 100, 2),
        }
        print(f"=== {name}: accuracy = {metrics[name]['accuracy']}% ===")
        print(classification_report(y_test, pred, target_names=["ham", "spam"]))

    MODEL_DIR.mkdir(exist_ok=True)
    joblib.dump(vectorizer, MODEL_DIR / "vectorizer.pkl")
    joblib.dump(models["Logistic Regression"], MODEL_DIR / "model.pkl")
    info = {
        "records": int(len(df)),
        "spam": int(df.target.sum()),
        "ham": int((1 - df.target).sum()),
        "features": int(X.shape[1]),
        "models": metrics,
    }
    (MODEL_DIR / "metrics.json").write_text(json.dumps(info, indent=2))
    print("Saved models/model.pkl, models/vectorizer.pkl, models/metrics.json")

    make_charts(models["Logistic Regression"], vectorizer, y_test,
                models["Logistic Regression"].predict(X_test))


def make_charts(clf, vectorizer, y_test, pred):
    """Confusion matrix + top spam words (skipped if matplotlib is missing)."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed - skipping charts "
              "(pip install matplotlib seaborn to enable)")
        return
    OUT_DIR.mkdir(exist_ok=True)

    cm = confusion_matrix(y_test, pred)
    fig, ax = plt.subplots(figsize=(4.5, 4))
    try:
        import seaborn as sns
        sns.heatmap(cm, annot=True, fmt="d", cmap="Purples", ax=ax,
                    xticklabels=["ham", "spam"], yticklabels=["ham", "spam"])
    except ImportError:
        ax.imshow(cm, cmap="Purples")
        for (i, j), v in __import__("numpy").ndenumerate(cm):
            ax.text(j, i, v, ha="center", va="center")
        ax.set_xticks([0, 1], ["ham", "spam"])
        ax.set_yticks([0, 1], ["ham", "spam"])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Confusion matrix (Logistic Regression)")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "confusion_matrix.png", dpi=150)
    plt.close(fig)

    names = vectorizer.get_feature_names_out()
    top = clf.coef_[0].argsort()[-15:]
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.barh([names[i] for i in top], clf.coef_[0][top], color="#6a4fc4")
    ax.set_title("Top 15 spam-weighted words")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "top_spam_words.png", dpi=150)
    plt.close(fig)
    print("Saved outputs/confusion_matrix.png and outputs/top_spam_words.png")


if __name__ == "__main__":
    main()
