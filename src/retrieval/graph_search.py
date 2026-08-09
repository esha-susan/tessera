import json
import os
import re
from pydantic import BaseModel, Field
from typing import Literal
from neo4j import GraphDatabase
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

from src.llm import call_llm

load_dotenv()

NEO4J_URI = os.getenv("NEO4J_URI")
NEO4J_USER = os.getenv("NEO4J_USER")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")

ENTITY_LABELS = {"Paper", "Author", "Institution", "Method", "Dataset", "Task"}
QueryType = Literal["neighbors", "extends_chain", "shared_task"]
EntityType = Literal["Paper", "Author", "Institution", "Method", "Dataset", "Task"]

_model = SentenceTransformer("all-MiniLM-L6-v2")
_driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))

GRAPH_PLAN_SYSTEM_PROMPT = """You extract entities from a question about research papers and \
pick which graph query template applies. You respond with ONLY valid JSON, no markdown fences."""

GRAPH_PLAN_PROMPT_TEMPLATE = """Given this question, extract the entity being asked about and \
choose ONE query type:

- "neighbors": general "what is X connected to" questions
- "extends_chain": "what does X extend / build on" questions
- "shared_task": "who else works on the same problem as X" questions. Extract the \
TASK or TOPIC itself as the entity, not a paper or author mentioned only as context.

Entity types: Paper, Author, Institution, Method, Dataset, Task

Examples:
Q: "What does TagRAG extend?"
A: {{"query_type": "extends_chain", "entity_name": "TagRAG", "entity_type": "Method"}}

Q: "Who else works on statute retrieval besides the LegalMALR authors?"
A: {{"query_type": "shared_task", "entity_name": "statute retrieval", "entity_type": "Task"}}

Q: "What is RAG connected to?"
A: {{"query_type": "neighbors", "entity_name": "RAG", "entity_type": "Method"}}

Now classify:
Question: "{question}"

Respond with ONLY: {{"query_type": "neighbors" | "extends_chain" | "shared_task", "entity_name": "string", "entity_type": "Paper" | "Author" | "Institution" | "Method" | "Dataset" | "Task"}}"""

class GraphQueryPlan(BaseModel):
    query_type: QueryType
    entity_name: str
    entity_type: EntityType


def strip_json_fences(text: str) -> str:
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    return text.strip()


def plan_graph_query(question: str) -> GraphQueryPlan | None:
    prompt = GRAPH_PLAN_PROMPT_TEMPLATE.format(question=question)
    raw = call_llm(prompt=prompt, system=GRAPH_PLAN_SYSTEM_PROMPT)
    cleaned = strip_json_fences(raw)

    try:
        data = json.loads(cleaned)
        return GraphQueryPlan.model_validate(data)
    except Exception:
        return None


def resolve_entity(name: str, entity_type: str, threshold: float = 0.6) -> str | None:
    with _driver.session() as session:
        result = session.run(
            f"MATCH (n:{entity_type}) RETURN n.canonical_name AS name"
        )
        candidates = [r["name"] for r in result]

    if not candidates:
        return None

    name_lower = name.lower()
    substring_matches = [
        c for c in candidates
        if name_lower in c.lower() or c.lower() in name_lower
    ]
    if len(substring_matches) == 1:
        return substring_matches[0]
    if len(substring_matches) > 1:
        candidates = substring_matches

    query_emb = _model.encode(name, normalize_embeddings=True)
    candidate_embs = _model.encode(candidates, normalize_embeddings=True)
    similarities = candidate_embs @ query_emb

    best_idx = int(similarities.argmax())
    best_score = float(similarities[best_idx])

    if best_score < threshold:
        return None

    return candidates[best_idx]

def run_neighbors(entity_name: str, entity_type: str) -> list[dict]:
    with _driver.session() as session:
        result = session.run(
            f"""
            MATCH (n:{entity_type} {{canonical_name: $name}})-[r]-(m)
            RETURN type(r) AS relation, labels(m)[0] AS neighbor_type, m.canonical_name AS neighbor
            LIMIT 25
            """,
            name=entity_name,
        )
        return [dict(r) for r in result]


def run_extends_chain(entity_name: str) -> list[dict]:
    with _driver.session() as session:
        result = session.run(
            """
            MATCH (m:Method {canonical_name: $name})-[:EXTENDS*1..3]->(base:Method)
            RETURN base.canonical_name AS extends
            """,
            name=entity_name,
        )
        return [dict(r) for r in result]


def run_shared_task(entity_name: str) -> list[dict]:
    with _driver.session() as session:
        result = session.run(
            """
            MATCH (a1:Author)<-[:AUTHORED_BY]-(p1:Paper)-[:ADDRESSES]->(t:Task {canonical_name: $name})<-[:ADDRESSES]-(p2:Paper)-[:AUTHORED_BY]->(a2:Author)
            WHERE a1 <> a2 AND p1 <> p2
            RETURN DISTINCT p1.canonical_name AS paper1, p2.canonical_name AS paper2
            LIMIT 15
            """,
            name=entity_name,
        )
        return [dict(r) for r in result]


def graph_search(question: str) -> dict:
    plan = plan_graph_query(question)
    if plan is None:
        return {"resolved_entity": None, "results": [], "note": "could not plan query"}

    resolved_name = resolve_entity(plan.entity_name, plan.entity_type)
    if resolved_name is None:
        return {"resolved_entity": None, "results": [], "note": f"no match for '{plan.entity_name}'"}

    if plan.query_type == "neighbors":
        results = run_neighbors(resolved_name, plan.entity_type)
    elif plan.query_type == "extends_chain":
        results = run_extends_chain(resolved_name)
    elif plan.query_type == "shared_task":
        results = run_shared_task(resolved_name)
    else:
        results = []

    return {
        "resolved_entity": resolved_name,
        "query_type": plan.query_type,
        "results": results,
    }


if __name__ == "__main__":
    test_questions = [
        "What does TagRAG extend?",
        "Who else works on statute retrieval besides the LegalMALR authors?",
        "What is RAG connected to?",
    ]

    for q in test_questions:
        print(f"Q: {q}")
        result = graph_search(q)
        print(json.dumps(result, indent=2))
        print()
