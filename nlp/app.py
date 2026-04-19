"""
app.py
------
FastAPI NLP service for ListenAI.

Sentiment classification strategy (chosen at startup):
  1. TF-IDF + Logistic Regression  — loaded from model/ directory if present.
  2. Lexicon-based fallback         — used when no trained model is found.

The model directory is controlled by the MODEL_DIR environment variable
(default: ./model relative to this file).
"""

import os
import re
from collections import Counter
from pathlib import Path

from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI(title="listen-ai-nlp")

# ---------------------------------------------------------------------------
# Lexicon data (used as fallback when no trained model is present)
# ---------------------------------------------------------------------------
POSITIVE_WORDS = {
    "good", "great", "excellent", "love", "awesome", "happy", "amazing",
    "nice", "best", "positive", "fast", "smooth", "reliable",
}
POSITIVE_WORDS_ZH_TW = {
    "好", "很好", "優秀", "喜歡", "讚", "開心", "高興", "棒",
    "最佳", "正面", "快速", "順暢", "可靠", "滿意", "推薦",
}
NEGATIVE_WORDS = {
    "bad", "terrible", "awful", "hate", "worst", "slow", "bug", "bugs",
    "issue", "issues", "angry", "broken", "negative", "expensive",
}
NEGATIVE_WORDS_ZH_TW = {
    "差", "糟糕", "很糟", "討厭", "最差", "慢", "錯誤", "問題",
    "生氣", "壞掉", "負面", "昂貴", "失望", "卡頓",
}
NEGATION_WORDS = {"not", "never", "no", "hardly", "不", "沒", "無", "未", "別", "不是"}

POSITIVE_WORDS_ALL = POSITIVE_WORDS | POSITIVE_WORDS_ZH_TW
NEGATIVE_WORDS_ALL = NEGATIVE_WORDS | NEGATIVE_WORDS_ZH_TW

CJK_LEXICON_TERMS = sorted(
    POSITIVE_WORDS_ZH_TW | NEGATIVE_WORDS_ZH_TW
    | {w for w in NEGATION_WORDS if re.search(r"[\u4e00-\u9fff]", w)},
    key=len,
    reverse=True,
)


def _tokenize_cjk_segment(segment: str) -> list[str]:
    tokens: list[str] = []
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


def tokenize(text: str) -> list[str]:
    raw_tokens = re.findall(r"[a-zA-Z']+|[\u4e00-\u9fff]+", text.lower())
    tokens: list[str] = []
    for raw in raw_tokens:
        if re.fullmatch(r"[\u4e00-\u9fff]+", raw):
            tokens.extend(_tokenize_cjk_segment(raw))
        else:
            tokens.append(raw)
    return tokens


def classify_text_lexicon(text: str) -> tuple[str, int]:
    """Original lexicon-based classifier (kept for fallback and backwards compatibility)."""
    tokens = tokenize(text)
    score = 0
    previous_tokens = ["", ""]
    for token in tokens:
        is_negated = any(prev in NEGATION_WORDS for prev in previous_tokens)
        if token in POSITIVE_WORDS_ALL:
            score += -1 if is_negated else 1
        elif token in NEGATIVE_WORDS_ALL:
            score += 1 if is_negated else -1
        previous_tokens = [previous_tokens[-1], token]
    if score > 0:
        return "positive", score
    if score < 0:
        return "negative", score
    return "neutral", score


# ---------------------------------------------------------------------------
# TF-IDF + Logistic Regression model loader
# ---------------------------------------------------------------------------
_MODEL_DIR = Path(os.getenv("MODEL_DIR", str(Path(__file__).parent / "model")))

# Single sklearn Pipeline saved by train_model.py (TF-IDF + LogisticRegression).
_pipeline = None
_using_ml_model = False


def _load_ml_model() -> None:
    """
    Attempt to load the trained sklearn Pipeline from model/sentiment_model.pkl.
    Sets _using_ml_model = True on success; falls back to lexicon on any error.
    """
    global _pipeline, _using_ml_model

    model_path = Path(_MODEL_DIR) / "sentiment_model.pkl"

    if not model_path.exists():
        print(
            f"[nlp] No trained model found at {model_path}. "
            "Using lexicon-based fallback."
        )
        return

    try:
        import pickle  # stdlib; always available

        with open(model_path, "rb") as f:
            _pipeline = pickle.load(f)
        _using_ml_model = True
        print(f"[nlp] TF-IDF + LR model loaded from {model_path}.")
    except Exception as exc:
        print(f"[nlp] Failed to load model ({exc}). Using lexicon fallback.")


_load_ml_model()


# ---------------------------------------------------------------------------
# Unified classification interface
# ---------------------------------------------------------------------------

def classify_text(text: str) -> tuple[str, int]:
    """
    Classify a single text using the best available algorithm.
    Returns (label, score) where score is 0 for the ML path (no numeric score).
    """
    if _using_ml_model:
        label: str = _pipeline.predict([text])[0]
        return label, 0

    return classify_text_lexicon(text)


# ---------------------------------------------------------------------------
# API models
# ---------------------------------------------------------------------------

class SentimentRequest(BaseModel):
    texts: list[str]


class SentimentItem(BaseModel):
    text: str
    label: str
    score: int


class SentimentResponse(BaseModel):
    sentiment_percentage: dict[str, float]
    classifications: list[SentimentItem]


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health")
def health() -> dict[str, str]:
    algorithm = "tfidf-lr" if _using_ml_model else "lexicon"
    return {
        "status": "ok",
        "service": "nlp",
        "port": os.getenv("NLP_PORT", "8001"),
        "algorithm": algorithm,
    }


@app.post("/sentiment", response_model=SentimentResponse)
def sentiment(req: SentimentRequest) -> SentimentResponse:
    results: list[SentimentItem] = []
    counts = Counter({"positive": 0, "neutral": 0, "negative": 0})

    for text in req.texts:
        label, score = classify_text(text)
        counts[label] += 1
        results.append(SentimentItem(text=text, label=label, score=score))

    total = max(1, len(req.texts))
    sentiment_percentage = {
        "positive": round((counts["positive"] / total) * 100, 2),
        "neutral":  round((counts["neutral"]  / total) * 100, 2),
        "negative": round((counts["negative"] / total) * 100, 2),
    }

    return SentimentResponse(
        sentiment_percentage=sentiment_percentage,
        classifications=results,
    )
