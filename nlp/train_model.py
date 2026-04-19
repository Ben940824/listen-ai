"""
train_model.py
--------------
Reads data/labeled_posts.csv (produced by label_posts.py) and:

  1. Evaluates the baseline lexicon-based algorithm (from app.py) on the labeled data.
  2. Trains a TF-IDF + Logistic Regression classifier on 80% of the data.
  3. Evaluates the new model on the remaining 20%.
  4. Saves the trained vectorizer and classifier to nlp/model/ for use by app.py.
  5. Prints a side-by-side comparison report.

Usage:
    python train_model.py [--data PATH] [--model-dir DIR] [--test-size FLOAT]
"""

import argparse
import time
import csv
import re
import sys
from pathlib import Path

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split

_HERE = Path(__file__).parent
DEFAULT_DATA = _HERE.parent / "data" / "labeled_posts.csv"
DEFAULT_MODEL_DIR = _HERE / "model"

# ---------------------------------------------------------------------------
# Lexicon-based baseline (replicated from app.py so this script is standalone)
# ---------------------------------------------------------------------------
POSITIVE_WORDS = {
    "good", "great", "excellent", "love", "awesome", "happy", "amazing",
    "nice", "best", "positive", "fast", "smooth", "reliable",
}
POSITIVE_WORDS_ZH = {
    "好", "很好", "優秀", "喜歡", "讚", "開心", "高興", "棒",
    "最佳", "正面", "快速", "順暢", "可靠", "滿意", "推薦",
}
NEGATIVE_WORDS = {
    "bad", "terrible", "awful", "hate", "worst", "slow", "bug", "bugs",
    "issue", "issues", "angry", "broken", "negative", "expensive",
}
NEGATIVE_WORDS_ZH = {
    "差", "糟糕", "很糟", "討厭", "最差", "慢", "錯誤", "問題",
    "生氣", "壞掉", "負面", "昂貴", "失望", "卡頓",
}
NEGATION_WORDS = {"not", "never", "no", "hardly", "不", "沒", "無", "未", "別", "不是"}

POSITIVE_ALL = POSITIVE_WORDS | POSITIVE_WORDS_ZH
NEGATIVE_ALL = NEGATIVE_WORDS | NEGATIVE_WORDS_ZH

CJK_LEXICON_TERMS = sorted(
    POSITIVE_WORDS_ZH | NEGATIVE_WORDS_ZH | {w for w in NEGATION_WORDS if re.search(r"[\u4e00-\u9fff]", w)},
    key=len, reverse=True,
)


def _tokenize_cjk_segment(segment: str) -> list:
    tokens = []
    idx = 0
    while idx < len(segment):
        match = ""
        for term in CJK_LEXICON_TERMS:
            if segment.startswith(term, idx):
                match = term
                break
        if match:
            tokens.append(match)
            idx += len(match)
        else:
            tokens.append(segment[idx])
            idx += 1
    return tokens


def _tokenize(text: str) -> list:
    raw_tokens = re.findall(r"[a-zA-Z']+|[\u4e00-\u9fff]+", text.lower())
    tokens = []
    for raw in raw_tokens:
        if re.fullmatch(r"[\u4e00-\u9fff]+", raw):
            tokens.extend(_tokenize_cjk_segment(raw))
        else:
            tokens.append(raw)
    return tokens


def lexicon_predict(text: str) -> str:
    """Replicate the baseline classify_text logic from app.py."""
    tokens = _tokenize(text)
    score = 0
    prev = ["", ""]
    for token in tokens:
        negated = any(p in NEGATION_WORDS for p in prev)
        if token in POSITIVE_ALL:
            score += -1 if negated else 1
        elif token in NEGATIVE_ALL:
            score += 1 if negated else -1
        prev = [prev[-1], token]
    if score > 0:
        return "positive"
    if score < 0:
        return "negative"
    return "neutral"


# ---------------------------------------------------------------------------
# Training & evaluation
# ---------------------------------------------------------------------------

def evaluate_lexicon(texts: list, labels: list) -> dict:
    """Run lexicon predictions and return metrics."""
    start = time.perf_counter()
    preds = [lexicon_predict(t) for t in texts]
    elapsed = time.perf_counter() - start

    acc = accuracy_score(labels, preds)
    report = classification_report(labels, preds, labels=["positive", "neutral", "negative"],
                                   output_dict=True, zero_division=0)
    return {
        "accuracy": acc,
        "macro_f1": report["macro avg"]["f1-score"],
        "inference_time_s": elapsed,
        "report": report,
    }


def train_tfidf_lr(
    train_texts: list,
    train_labels: list,
    test_texts: list,
    test_labels: list,
    model_dir: Path,
) -> dict:
    """
    Train TF-IDF + Logistic Regression, save artifacts, return metrics on test set.
    """
    # Vectorise
    vectorizer = TfidfVectorizer(
        analyzer="char_wb",     # character n-grams work well for mixed CJK/Latin text
        ngram_range=(2, 4),
        max_features=30_000,
        sublinear_tf=True,
    )
    X_train = vectorizer.fit_transform(train_texts)
    X_test = vectorizer.transform(test_texts)

    # Train
    clf = LogisticRegression(
        C=1.0,
        max_iter=1000,
        solver="lbfgs",
        multi_class="multinomial",
        random_state=42,
    )
    clf.fit(X_train, train_labels)

    # Evaluate
    start = time.perf_counter()
    preds = clf.predict(X_test)
    elapsed = time.perf_counter() - start

    acc = accuracy_score(test_labels, preds)
    report = classification_report(test_labels, preds,
                                   labels=["positive", "neutral", "negative"],
                                   output_dict=True, zero_division=0)

    # Persist model artifacts
    model_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(vectorizer, model_dir / "tfidf_vectorizer.pkl")
    joblib.dump(clf,        model_dir / "lr_classifier.pkl")
    print(f"\nModel artifacts saved to: {model_dir}")

    return {
        "accuracy": acc,
        "macro_f1": report["macro avg"]["f1-score"],
        "inference_time_s": elapsed,
        "report": report,
    }


def print_comparison(
    lexicon_metrics: dict,
    ml_metrics: dict,
    n_test: int,
    n_train: int,
) -> None:
    """Print a formatted side-by-side comparison table."""
    w = 28
    sep = "-" * (w * 3 + 4)

    print("\n" + "=" * (w * 3 + 4))
    print("  SENTIMENT ANALYSIS ALGORITHM COMPARISON")
    print("=" * (w * 3 + 4))
    print(f"  Train / Test split  : {n_train} / {n_test} samples")
    print(sep)
    print(f"{'Metric':<{w}} {'Lexicon-based':>{w}} {'TF-IDF + LR':>{w}}")
    print(sep)

    def fmt(v, is_time=False):
        return f"{v*1000:.2f} ms" if is_time else f"{v:.4f}"

    metrics = [
        ("Accuracy",        "accuracy",        False),
        ("Macro F1-score",  "macro_f1",        False),
        ("Inference time",  "inference_time_s", True),
    ]
    for label, key, is_time in metrics:
        lv = lexicon_metrics[key]
        mv = ml_metrics[key]
        print(f"{label:<{w}} {fmt(lv, is_time):>{w}} {fmt(mv, is_time):>{w}}")

    print(sep)
    print("\nPer-class F1 (new model):")
    r = ml_metrics["report"]
    for cls in ["positive", "neutral", "negative"]:
        f1 = r[cls]["f1-score"]
        prec = r[cls]["precision"]
        rec = r[cls]["recall"]
        print(f"  {cls:<10}  precision={prec:.3f}  recall={rec:.3f}  f1={f1:.3f}")

    print("\nConclusion:")
    delta_acc = ml_metrics["accuracy"] - lexicon_metrics["accuracy"]
    if delta_acc > 0:
        print(f"  TF-IDF + LR improves accuracy by {delta_acc*100:.2f}pp over the lexicon baseline.")
    else:
        print(f"  TF-IDF + LR accuracy is {abs(delta_acc)*100:.2f}pp lower than the lexicon baseline.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Train TF-IDF+LR and compare with lexicon baseline.")
    parser.add_argument("--data",       type=Path,  default=DEFAULT_DATA)
    parser.add_argument("--model-dir",  type=Path,  default=DEFAULT_MODEL_DIR)
    parser.add_argument("--test-size",  type=float, default=0.2,
                        help="Fraction of data used for testing (default: 0.2)")
    args = parser.parse_args()

    if not args.data.exists():
        print(f"[error] Labeled dataset not found at: {args.data}")
        print("        Run label_posts.py first to generate it.")
        sys.exit(1)

    df = pd.read_csv(args.data)
    required_cols = {"content", "label"}
    if not required_cols.issubset(df.columns):
        print(f"[error] CSV must contain columns: {required_cols}")
        sys.exit(1)

    df = df.dropna(subset=["content", "label"])
    df = df[df["label"].isin(["positive", "neutral", "negative"])]

    texts  = df["content"].tolist()
    labels = df["label"].tolist()

    print(f"Dataset: {len(df)} samples")
    print(f"  positive : {labels.count('positive')}")
    print(f"  neutral  : {labels.count('neutral')}")
    print(f"  negative : {labels.count('negative')}")

    train_texts, test_texts, train_labels, test_labels = train_test_split(
        texts, labels, test_size=args.test_size, random_state=42, stratify=labels
    )

    print(f"\nEvaluating lexicon baseline on {len(test_texts)} test samples...")
    lexicon_m = evaluate_lexicon(test_texts, test_labels)

    print(f"Training TF-IDF + LR on {len(train_texts)} samples...")
    ml_m = train_tfidf_lr(train_texts, train_labels, test_texts, test_labels, args.model_dir)

    print_comparison(lexicon_m, ml_m, len(test_texts), len(train_texts))


if __name__ == "__main__":
    main()
