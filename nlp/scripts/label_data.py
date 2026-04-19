#!/usr/bin/env python3
"""
Step 1 – Automatic data labeling via OpenAI API.

Reads posts.csv from the project data directory, draws a stratified random
sample of SAMPLE_SIZE posts, calls gpt-4o-mini to classify each post as
"positive", "neutral", or "negative", and writes the result to
nlp/data/labeled_posts.csv.

Usage:
    export OPENAI_API_KEY="sk-..."
    python nlp/scripts/label_data.py

Optional arguments:
    --sample   Number of posts to label (default: 300)
    --csv      Path to posts.csv
    --out      Path to output labeled CSV
    --seed     Random seed (default: 42)
"""

from __future__ import annotations

import argparse
import csv
import os
import random
import sys
import time
from collections import Counter
from pathlib import Path

from dotenv import load_dotenv

# Load .env from the repository root (listen-ai/) so OPENAI_API_KEY is
# available without requiring the user to export it manually each session.
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
load_dotenv(_REPO_ROOT / ".env", override=False)

# ---------------------------------------------------------------------------
# Paths (relative to repository root)
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parent.parent.parent   # → listen-ai/
POSTS_CSV_DEFAULT = REPO_ROOT / "data" / "posts.csv"
OUTPUT_CSV_DEFAULT = Path(__file__).resolve().parent.parent / "data" / "labeled_posts.csv"

# ---------------------------------------------------------------------------
# Prompt template
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = (
    "You are a sentiment analysis assistant. "
    "Classify the following Chinese social media post as positive, neutral, or negative. "
    "Reply with exactly one word: positive, neutral, or negative."
)

USER_TEMPLATE = "Post: {content}"

VALID_LABELS = {"positive", "neutral", "negative"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Label posts.csv via OpenAI API")
    parser.add_argument("--sample", type=int, default=300, help="Number of posts to label")
    parser.add_argument("--csv", default=str(POSTS_CSV_DEFAULT), help="Path to posts.csv")
    parser.add_argument("--out", default=str(OUTPUT_CSV_DEFAULT), help="Output labeled CSV path")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    return parser.parse_args()


def label_with_openai(client, content: str) -> str:
    """Send one post to gpt-4o-mini and return a normalized label."""
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": USER_TEMPLATE.format(content=content)},
        ],
        max_tokens=5,
        temperature=0,
    )
    raw = response.choices[0].message.content.strip().lower()
    # Guard against unexpected model output
    for label in VALID_LABELS:
        if label in raw:
            return label
    return "neutral"


def main() -> None:
    args = parse_args()

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("ERROR: OPENAI_API_KEY environment variable is not set.", file=sys.stderr)
        sys.exit(1)

    # Lazy import so the script fails fast if key is missing
    from openai import OpenAI  # noqa: PLC0415
    client = OpenAI(api_key=api_key)

    # -----------------------------------------------------------------------
    # Load posts
    # -----------------------------------------------------------------------
    posts_path = Path(args.csv)
    if not posts_path.exists():
        print(f"ERROR: {posts_path} does not exist.", file=sys.stderr)
        sys.exit(1)

    with open(posts_path, encoding="utf-8") as f:
        all_posts = list(csv.DictReader(f))

    print(f"Total posts in CSV: {len(all_posts)}")

    # -----------------------------------------------------------------------
    # Random sample
    # -----------------------------------------------------------------------
    random.seed(args.seed)
    sample_size = min(args.sample, len(all_posts))
    sampled = random.sample(all_posts, sample_size)
    print(f"Sampling {sample_size} posts for labeling (seed={args.seed})...")

    # -----------------------------------------------------------------------
    # Label each post
    # -----------------------------------------------------------------------
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    results: list[dict[str, str]] = []
    errors = 0

    for i, post in enumerate(sampled, start=1):
        content = post["content"]
        try:
            label = label_with_openai(client, content)
        except Exception as exc:  # noqa: BLE001
            print(f"  [{i}/{sample_size}] API error: {exc} – defaulting to 'neutral'")
            label = "neutral"
            errors += 1

        results.append({"content": content, "label": label})
        print(f"  [{i:>3}/{sample_size}] {label:<8} | {content[:60]}")

        # Avoid exceeding rate limits: pause every 20 requests
        if i % 20 == 0:
            time.sleep(1)

    # -----------------------------------------------------------------------
    # Write output CSV
    # -----------------------------------------------------------------------
    with open(out_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["content", "label"])
        writer.writeheader()
        writer.writerows(results)

    print(f"\nLabeled data written to: {out_path}")
    dist = Counter(r["label"] for r in results)
    print(f"Label distribution: {dict(dist)}")
    if errors:
        print(f"API errors (defaulted to neutral): {errors}")


if __name__ == "__main__":
    main()
