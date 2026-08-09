import re
from src.llm import call_llm
from src.retrieval.router import route_question
from src.retrieval.vector_search import vector_search
from src.retrieval.graph_search import graph_search

ANSWER_SYSTEM_PROMPT = """You answer questions about research papers using only the provided context. \
Every claim must be followed by a citation to the context id it came from, like [C_id] or [G1]. \
Never state a fact without a citation. If the context doesn't contain enough information, say so."""


def graph_results_to_statements(graph_result: dict) -> list[str]:
    entity = graph_result.get("resolved_entity")
    query_type = graph_result.get("query_type")
    results = graph_result.get("results", [])
    statements = []

    if not entity or not results:
        return statements

    if query_type == "neighbors":
        for r in results:
            statements.append(f"{entity} is connected to {r['neighbor']} ({r['neighbor_type']}) via {r['relation']}")
    elif query_type == "extends_chain":
        for r in results:
            statements.append(f"{entity} extends {r['extends']}")
    elif query_type == "shared_task":
        for r in results:
            statements.append(f"'{r['paper1']}' and '{r['paper2']}' both address the task '{entity}'")

    return statements


def build_context(vector_results: list[dict], graph_result: dict | None) -> list[dict]:
    context = []

    for r in vector_results:
        context.append({
            "id": f"C_{r['chunk_id']}",
            "source": "vector",
            "text": r["chunk_text"],
        })

    if graph_result:
        statements = graph_results_to_statements(graph_result)
        for i, stmt in enumerate(statements):
            context.append({
                "id": f"G{i + 1}",
                "source": "graph",
                "text": stmt,
            })

    return context


def format_context(context: list[dict]) -> str:
    lines = []
    for c in context:
        label = "PASSAGE" if c["source"] == "vector" else "GRAPH FACT"
        lines.append(f"[{c['id']}] ({label}) {c['text']}")
    return "\n\n".join(lines)


def extract_citations(answer: str) -> set[str]:
    return set(re.findall(r"\[([A-Za-z0-9_]+)\]", answer))


def generate_answer(question: str, context: list[dict], max_retries: int = 2) -> str:
    valid_ids = {c["id"] for c in context}
    context_text = format_context(context)

    prompt = f"""Context:
{context_text}

Question: {question}

Answer using only the context above. Cite every claim with the bracketed id it came from, e.g. [C_2509.10467v1] or [G1]."""

    for attempt in range(max_retries + 1):
        answer = call_llm(prompt=prompt, system=ANSWER_SYSTEM_PROMPT)
        used_citations = extract_citations(answer)
        invalid = used_citations - valid_ids

        if not invalid:
            return answer

        prompt += f"\n\nYour previous answer used invalid citation(s): {invalid}. Only cite ids that appear in the context above: {sorted(valid_ids)}."

    return "Could not generate a fully grounded answer within the retry limit."


def answer_question(question: str) -> dict:
    decision = route_question(question)

    vector_results = []
    graph_result = None

    if decision.route in ("vector", "both"):
        vector_results = vector_search(question, k=5)

    if decision.route in ("graph", "both"):
        graph_result = graph_search(question)

    context = build_context(vector_results, graph_result)

    if not context:
        return {
            "question": question,
            "route": decision.route,
            "answer": "No relevant information found.",
            "context": [],
        }

    answer = generate_answer(question, context)

    return {
        "question": question,
        "route": decision.route,
        "answer": answer,
        "context": context,
    }


if __name__ == "__main__":
    test_questions = [
        "What does TagRAG extend?",
        "What does the DSRAG paper propose?",
        "Who else works on statute retrieval besides the LegalMALR authors?",
    ]

    for q in test_questions:
        result = answer_question(q)
        print(f"Q: {result['question']}")
        print(f"Route: {result['route']}")
        print(f"Answer: {result['answer']}")
        print()
