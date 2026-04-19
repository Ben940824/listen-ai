"""
label_posts.py
--------------
Uses OpenAI gpt-4o-mini to assign a sentiment label (positive / neutral / negative)
to every post in data/posts.csv, then writes the result to data/labeled_posts.csv.

Usage:
    export OPENAI_API_KEY="sk-..."
    python label_posts.py [--input PATH] [--output PATH] [--model MODEL]
"""

import argparse
import csv
import os
import sys
import time
from pathlib import Path

from openai import OpenAI, RateLimitError, APIError

_HERE = Path(__file__).parent
DEFAULT_INPUT = _HERE.parent / "data" / "posts.csv"
DEFAULT_OUTPUT = _HERE.parent / "data" / "labeled_posts.csv"

SYSTEM_PROMPT = """You are a sentiment classification expert.
Classify the given social media post into exactly one of three labels:
  - positive
  - neutral
  - negative

Rules:
1. Reply with ONLY one of the three label words above, nothing else.
2. Consider irony and context carefully.
3. If the post is mixed or factual without emotional tone, use "neutral".
"""

USER_PROMPT_TEMPLATE = 'Post:\n"""\n{content}\n"""'


def classify_with_llm(client: OpenAI, content: str, model: str, retries: int = 3) -> str:
    """Send a single post to the LLM and return the predicted label."""
    for attempt in range(retries):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": USER_PROMPT_TEMPLATE.format(content=content)},
                ],
                max_tokens=5,
                temperature=0.0,
            )
            label = response.choices[0].message.content.strip().lower()

            if label not in ("positive", "neutral", "negative"):
                print(f"  [warn] unexpected label '{label}', defaulting to 'neutral'")
                label = "neutral"

            return label

        except RateLimitError:
            wait = 2 ** (attempt + 1)
            print(f"  [rate-limit] waiting {wait}s before retry {attempt + 1}/{retries}")
            time.sleep(wait)

        except APIError as exc:
            print(f"  [api-error] {exc}, retrying {attempt + 1}/{retries}")
            time.sleep(2)

    print("  [error] all retries exhausted, defaulting to 'neutral'")
    return "neutral"


def label_posts(
    input_path: Path,
    output_path: Path,
    model: str = "gpt-4o-mini",
    delay: float = 0.3,
) -> None:
    """
    Read posts from input_path, label each one, and write to output_path.
    Supports resume: skips rows already present in output_path.
    """
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("[error] OPENAI_API_KEY environment variable is not set.")
        sys.exit(1)

    client = OpenAI(api_key=api_key)

    # Load already-labeled rows so interrupted runs can be resumed
    already_labeled: set[str] = set()
    if output_path.exists():
        with open(output_path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                already_labeled.add(row["content"])
        print(f"[resume] {len(already_labeled)} posts already labeled, skipping them.")

    with open(input_path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    total = len(rows)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    write_mode = "a" if already_labeled else "w"
    with open(output_path, write_mode, newline="", encoding="utf-8") as out_f:
        writer = csv.DictWriter(out_f, fieldnames=["name", "date", "content", "label"])
        if write_mode == "w":
            writer.writeheader()

        for idx, row in enumerate(rows, start=1):
            content = row["content"]

            if content in already_labeled:
                continue

            print(f"[{idx}/{total}] Labeling: {content[:60]}...")
            label = classify_with_llm(client, content, model)
            print(f"         -> {label}")

            writer.writerow({
                "name": row["name"],
                "date": row["date"],
                "content": content,
                "label": label,
            })
            out_f.flush()

            time.sleep(delay)

    print(f"\nDone. Labeled CSV saved to: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Label posts.csv with LLM sentiment.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--model", type=str, default="gpt-4o-mini")
    parser.add_argument("--delay", type=float, default=0.3,
                        help="Seconds to sleep between API calls (default: 0.3)")
    args = parser.parse_args()

    label_posts(args.input, args.output, args.model, args.delay)
