import json
import os
import re

from pydantic import ValidationError

from src.llm import call_llm
from src.ingest.schema import (
    ExtractionResult,
    EXTRACTION_SYSTEM_PROMPT,
    build_extraction_prompt,
)

RAW_PAPERS_PATH = "data/raw_papers.json"
OUTPUT_PATH = "data/extracted.json"
MAX_RETRIES = 3


def strip_json_fences(text: str) -> str:
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    return text.strip()


def extract_one(paper: dict) -> ExtractionResult | None:
    prompt = build_extraction_prompt(
        title=paper["title"],
        abstract=paper["abstract"],
        authors=paper["authors"],
    )

    for attempt in range(1, MAX_RETRIES + 1):
        raw = call_llm(
            prompt=prompt,
            system=EXTRACTION_SYSTEM_PROMPT,
        )
        cleaned = strip_json_fences(raw)

        try:
            data = json.loads(cleaned)
            return ExtractionResult.model_validate(data)
        except (json.JSONDecodeError, ValidationError) as e:
            print(f"  attempt {attempt}/{MAX_RETRIES} failed: {e}")

            if attempt == MAX_RETRIES:
                print(f"  giving up on {paper['arxiv_id']}")
                return None

    return None


def load_cache() -> dict:
    if os.path.exists(OUTPUT_PATH):
        with open(OUTPUT_PATH) as f:
            return json.load(f)
    return {}


def save_cache(cache: dict):
    with open(OUTPUT_PATH, "w") as f:
        json.dump(cache, f, indent=2)


if __name__ == "__main__":
    with open(RAW_PAPERS_PATH) as f:
        papers = json.load(f)

    cache = load_cache()
    print(f"Loaded {len(papers)} papers, {len(cache)} already extracted")

    for paper in papers:
        arxiv_id = paper["arxiv_id"]

        if arxiv_id in cache:
            print(f"skip (cached): {paper['title'][:60]}")
            continue

        print(f"extracting: {paper['title'][:60]}")
        result = extract_one(paper)

        if result is not None:
            cache[arxiv_id] = {
                "title": paper["title"],
                "entities": [e.model_dump() for e in result.entities],
                "relations": [r.model_dump() for r in result.relations],
            }

            save_cache(cache)

            print(
                f"  -> {len(result.entities)} entities, "
                f"{len(result.relations)} relations"
            )

    print(f"Done. {len(cache)}/{len(papers)} papers extracted successfully.")
