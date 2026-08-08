"""
Defines the strict schema Claude/Groq must return during entity/relationship
extraction, and the prompt that instructs it to do so.

Keeping the schema in one place means Phase 1 (extraction), Phase 2
(graph writes), and later validation all import the same source of truth
instead of drifting out of sync.
"""

from pydantic import BaseModel, Field
from typing import Literal

EntityType = Literal["Paper", "Author", "Institution", "Method", "Dataset", "Task"]

RelationType = Literal[
    "AUTHORED_BY",
    "AFFILIATED_WITH",
    "CITES",
    "PROPOSES",
    "USES_METHOD",
    "EVALUATED_ON",
    "ADDRESSES",
    "EXTENDS",
    "APPLIED_TO",
]


class ExtractedEntity(BaseModel):
    entity_type: EntityType
    canonical_name: str = Field(description="Normalized name, e.g. 'BERT' not 'the BERT model'")


class ExtractedRelation(BaseModel):
    source_entity: str
    source_type: EntityType
    relation_type: RelationType
    target_entity: str
    target_type: EntityType
    confidence: float = Field(ge=0.0, le=1.0)


class ExtractionResult(BaseModel):
    entities: list[ExtractedEntity]
    relations: list[ExtractedRelation]


EXTRACTION_SYSTEM_PROMPT = """You are an information extraction system. You extract entities \
and relationships from academic paper abstracts into a strict JSON schema. \
You never invent facts not stated or clearly implied in the text. \
You respond with ONLY valid JSON, no markdown fences, no preamble, no explanation."""


def build_extraction_prompt(title: str, abstract: str, authors: list[str]) -> str:
    author_list = ", ".join(authors)
    return f"""Extract entities and relationships from this paper into this exact JSON schema:

{{
  "entities": [
    {{"entity_type": "Paper" | "Author" | "Institution" | "Method" | "Dataset" | "Task", "canonical_name": "string"}}
  ],
  "relations": [
    {{"source_entity": "string", "source_type": "...", "relation_type": "AUTHORED_BY" | "AFFILIATED_WITH" | "CITES" | "PROPOSES" | "USES_METHOD" | "EVALUATED_ON" | "ADDRESSES" | "EXTENDS" | "APPLIED_TO", "target_entity": "string", "target_type": "...", "confidence": 0.0-1.0}}
  ]
}}

Rules:
- Always include the paper itself as a Paper entity, using its title as canonical_name.
- Always include every listed author as an Author entity, with an AUTHORED_BY relation from the Paper to each Author.
- Only extract Method, Dataset, and Task entities that are explicitly named in the abstract (e.g. "BERT", "HotpotQA", "question answering"). Do not invent generic ones.
- Do not extract Institution or CITES relations from this abstract alone — those require full-text, which we don't have yet. Leave them out.
- confidence reflects how explicitly the abstract states the relationship (1.0 = stated directly, 0.6 = reasonably implied).

Paper title: {title}
Authors: {author_list}
Abstract: {abstract}

Respond with ONLY the JSON object."""
