#!/usr/bin/env python3
"""
Step 2 – Train TF-IDF + Logistic Regression and compare against the original
lexicon-based classifier.

Expects nlp/data/labeled_posts.csv to already exist (produced by label_data.py).
Splits data 80/20 (stratified), then:
  1. Evaluates the original lexicon classifier on the test split.
  2. Trains a TF-IDF + LogisticRegression pipeline on the train split.
  3. Evaluates the new model on the test split.
  4. Prints a side-by-side comparison table.
  5. Saves the trained pipeline to nlp/model/sentiment_model.pkl.

Usage (from repository root):
    python nlp/scripts/train_model.py
"""

from __future__ import annotations

import pickle
import sys
import time
from collections import Counter
from pathlib import Path

from dotenv import load_dotenv

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
load_dotenv(_REPO_ROOT / ".env", override=False)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
NLP_DIR = Path(__file__).resolve().parent.parent          # nlp/
LABELED_CSV = NLP_DIR / "data" / "labeled_posts.csv"
MODEL_PATH = NLP_DIR / "model" / "sentiment_model.pkl"

# ---------------------------------------------------------------------------
# Standalone lexicon implementation (mirrors app.py without requiring fastapi)
# ---------------------------------------------------------------------------
import re  # noqa: E402

_POSITIVE_WORDS = {
    "good", "great", "excellent", "love", "awesome", "happy", "amazing",
    "nice", "best", "positive", "fast", "smooth", "reliable",
    "好", "很好", "優秀", "喜歡", "讚", "開心", "高興", "棒",
    "最佳", "正面", "快速", "順暢", "可靠", "滿意", "推薦",
}
_NEGATIVE_WORDS = {
    "bad", "terrible", "awful", "hate", "worst", "slow", "bug", "bugs",
    "issue", "issues", "angry", "broken", "negative", "expensive",
    "差", "糟糕", "很糟", "討厭", "最差", "慢", "錯誤", "問題",
    "生氣", "壞掉", "負面", "昂貴", "失望", "卡頓",
}
_NEGATION = {"not", "never", "no", "hardly", "不", "沒", "無", "未", "別", "不是"}
_CJK_TERMS = sorted(
    {w for w in _POSITIVE_WORDS | _NEGATIVE_WORDS | _NEGATION if re.search(r"[\u4e00-\u9fff]", w)},
    key=len, reverse=True,
)


def _tokenize(text: str) -> list[str]:
    def _cjk_seg(seg: str) -> list[str]:
        tokens, idx = [], 0
        while idx < len(seg):
            m = next((t for t in _CJK_TERMS if seg.startswith(t, idx)), "")
            tokens.append(m if m else seg[idx])
            idx += len(m) if m else 1
        return tokens

    out: list[str] = []
    for raw in re.findall(r"[a-zA-Z']+|[\u4e00-\u9fff]+", text.lower()):
        out.extend(_cjk_seg(raw) if re.fullmatch(r"[\u4e00-\u9fff]+", raw) else [raw])
    return out


def _lexicon_classify(text: str) -> str:
    """Standalone reimplementation of the original lexicon-based classifier."""
    tokens, score, prev = _tokenize(text), 0, ["", ""]
    for tok in tokens:
        neg = any(p in _NEGATION for p in prev)
        if tok in _POSITIVE_WORDS:
            score += -1 if neg else 1
        elif tok in _NEGATIVE_WORDS:
            score += 1 if neg else -1
        prev = [prev[-1], tok]
    return "positive" if score > 0 else "negative" if score < 0 else "neutral"


def load_labeled_data() -> tuple[list[str], list[str]]:
    """Return (texts, labels) lists from the labeled CSV."""
    import csv  # noqa: PLC0415
    with open(LABELED_CSV, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    texts = [r["content"] for r in rows]
    labels = [r["label"] for r in rows]
    return texts, labels


# ---------------------------------------------------------------------------
# Baseline evaluation
# ---------------------------------------------------------------------------

def evaluate_lexicon(texts: list[str], labels: list[str]) -> tuple[float, float, float]:
    """Evaluate the original rule-based lexicon classifier."""
    from sklearn.metrics import accuracy_score, classification_report, f1_score  # noqa: PLC0415

    print("\n" + "=" * 60)
    print("Baseline: Lexicon-based Classifier")
    print("=" * 60)

    predictions: list[str] = []
    start = time.perf_counter()
    for text in texts:
        predictions.append(_lexicon_classify(text))
    elapsed = time.perf_counter() - start

    acc = accuracy_score(labels, predictions)
    f1 = f1_score(labels, predictions, average="macro", zero_division=0)
    ms_per_sample = elapsed / len(texts) * 1000

    print(f"Accuracy:          {acc:.4f}")
    print(f"F1 (macro):        {f1:.4f}")
    print(f"Inference time:    {elapsed:.3f}s total  |  {ms_per_sample:.3f} ms/sample")
    print()
    print(classification_report(labels, predictions, zero_division=0))

    return acc, f1, ms_per_sample


# ---------------------------------------------------------------------------
# TF-IDF + Logistic Regression
# ---------------------------------------------------------------------------

def build_pipeline():
    """Construct the sklearn Pipeline."""
    from sklearn.feature_extraction.text import TfidfVectorizer  # noqa: PLC0415
    from sklearn.linear_model import LogisticRegression  # noqa: PLC0415
    from sklearn.pipeline import Pipeline  # noqa: PLC0415

    # Character n-gram TF-IDF is effective for Chinese text because it does
    # not rely on word boundary detection and captures common character
    # combinations (e.g. bigrams "喜歡", "問題", "超好") directly.
    return Pipeline([
        ("tfidf", TfidfVectorizer(
            analyzer="char_wb",   # character n-grams within word boundaries
            ngram_range=(2, 4),   # bigrams through 4-grams
            max_features=50_000,
            sublinear_tf=True,    # apply log(1 + tf) scaling
        )),
        ("clf", LogisticRegression(
            max_iter=1000,
            C=1.0,
            solver="lbfgs",
            multi_class="multinomial",
        )),
    ])


def evaluate_tfidf(
    pipeline, X_test: list[str], y_test: list[str]
) -> tuple[float, float, float]:
    """Evaluate the trained TF-IDF pipeline on the test split."""
    from sklearn.metrics import accuracy_score, classification_report, f1_score  # noqa: PLC0415

    print("\n" + "=" * 60)
    print("New Model: TF-IDF + Logistic Regression")
    print("=" * 60)

    start = time.perf_counter()
    predictions = pipeline.predict(X_test)
    elapsed = time.perf_counter() - start

    acc = accuracy_score(y_test, predictions)
    f1 = f1_score(y_test, predictions, average="macro", zero_division=0)
    ms_per_sample = elapsed / len(X_test) * 1000

    print(f"Accuracy:          {acc:.4f}")
    print(f"F1 (macro):        {f1:.4f}")
    print(f"Inference time:    {elapsed:.3f}s total  |  {ms_per_sample:.3f} ms/sample")
    print()
    print(classification_report(y_test, predictions, zero_division=0))

    return acc, f1, ms_per_sample


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    if not LABELED_CSV.exists():
        print(
            f"ERROR: {LABELED_CSV} not found.\n"
            "Run `python nlp/scripts/label_data.py` first to generate labeled data.",
            file=sys.stderr,
        )
        sys.exit(1)

    from sklearn.model_selection import train_test_split  # noqa: PLC0415

    texts, labels = load_labeled_data()
    print(f"Loaded {len(texts)} labeled samples")
    print(f"Label distribution: {dict(Counter(labels))}")

    # Use stratified split only when every class has at least 2 members.
    # With heavily skewed data (e.g. only 1 negative sample) stratify will
    # raise ValueError, so we fall back to a plain random split.
    min_class_count = min(Counter(labels).values())
    use_stratify = labels if min_class_count >= 2 else None
    if use_stratify is None:
        print("Warning: some classes have < 2 samples; skipping stratified split.")

    X_train, X_test, y_train, y_test = train_test_split(
        texts, labels,
        test_size=0.2,
        random_state=42,
        stratify=use_stratify,
    )
    print(f"Train: {len(X_train)}  |  Test: {len(X_test)}")

    # ------------------------------------------------------------------
    # 1. Baseline
    # ------------------------------------------------------------------
    lex_acc, lex_f1, lex_ms = evaluate_lexicon(X_test, y_test)

    # ------------------------------------------------------------------
    # 2. Train new model
    # ------------------------------------------------------------------
    print("\nTraining TF-IDF + Logistic Regression …")
    pipeline = build_pipeline()
    t0 = time.perf_counter()
    pipeline.fit(X_train, y_train)
    train_sec = time.perf_counter() - t0
    print(f"Training complete in {train_sec:.2f}s")

    # ------------------------------------------------------------------
    # 3. Evaluate new model
    # ------------------------------------------------------------------
    tfidf_acc, tfidf_f1, tfidf_ms = evaluate_tfidf(pipeline, X_test, y_test)

    # ------------------------------------------------------------------
    # 4. Save model
    # ------------------------------------------------------------------
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(pipeline, f)
    print(f"\nModel saved → {MODEL_PATH}")

    # ------------------------------------------------------------------
    # 5. Comparison table (copy-paste friendly for the report)
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("Comparison Summary")
    print("=" * 70)
    header = f"{'Method':<42} {'Accuracy':>9} {'F1 macro':>10} {'ms/sample':>11}"
    print(header)
    print("-" * 70)
    print(f"{'Lexicon-based (baseline)':<42} {lex_acc:>9.4f} {lex_f1:>10.4f} {lex_ms:>11.3f}")
    print(f"{'TF-IDF + Logistic Regression':<42} {tfidf_acc:>9.4f} {tfidf_f1:>10.4f} {tfidf_ms:>11.3f}")
    print()


if __name__ == "__main__":
    main()
