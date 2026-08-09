import json
import re
from pydantic import BaseModel, Field
from typing import Literal

from src.llm import call_llm

Route = Literal["graph", "vector", "both"]

ROUTER_SYSTEM_PROMPT = """You are a query router for a research-paper question answering system. \
You decide whether a question should be answered using a knowledge graph (good for \
relationships, connections, multi-hop reasoning, comparisons across entities) or a \
vector similarity search (good for single facts, definitions, "what does paper X say" \
questions). You respond with ONLY valid JSON, no markdown fences, no explanation."""

ROUTER_PROMPT_TEMPLATE = """Classify this question into one route:

- "graph": requires connecting multiple entities, multi-hop reasoning, comparisons, \
or aggregations across relationships (e.g. "which papers extend a method proposed \
by X", "what tasks have papers by author Y addressed", "which authors have worked \
on both A and B")
- "vector": a single-fact or definitional question answerable from one passage \
(e.g. "what does paper X propose", "what is method Y")
- "both": the question is ambiguous, or genuinely needs both relationship context \
and passage-level detail

Examples:
Q: "What does the DSRAG paper propose?"
A: {{"route": "vector", "confidence": 0.95}}

Q: "Which papers propose methods that extend BERT?"
A: {{"route": "graph", "confidence": 0.9}}

Q: "What are the main approaches to reducing hallucination in RAG?"
A: {{"route": "both", "confidence": 0.6}}

Now classify:
Q: "{question}"

Respond with ONLY: {{"route": "graph" | "vector" | "both", "confidence": 0.0-1.0}}"""


class RouteDecision(BaseModel):
    route: Route
    confidence: float = Field(ge=0.0, le=1.0)


def strip_json_fences(text: str) -> str:
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    return text.strip()


def route_question(question: str, confidence_threshold: float = 0.7) -> RouteDecision:
    prompt = ROUTER_PROMPT_TEMPLATE.format(question=question)
    raw = call_llm(prompt=prompt, system=ROUTER_SYSTEM_PROMPT)
    cleaned = strip_json_fences(raw)

    try:
        data = json.loads(cleaned)
        decision = RouteDecision.model_validate(data)
    except Exception:
        return RouteDecision(route="both", confidence=0.0)

    if decision.confidence < confidence_threshold and decision.route != "both":
        return RouteDecision(route="both", confidence=decision.confidence)

    return decision


if __name__ == "__main__":
    test_questions = [
        "What does the DSRAG paper propose?",
        "Which papers propose methods that extend RAG?",
        "What is knowledge graph based retrieval?",
    ]

    for q in test_questions:
        decision = route_question(q)
        print(f"Q: {q}")
        print(f"  -> route={decision.route} confidence={decision.confidence}\n")
