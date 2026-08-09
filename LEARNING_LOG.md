# Tessera — File Reference

What each file does, and the one key concept it taught me.

---

## docker-compose.yml
Starts Neo4j and Postgres as local containers with one command
(`docker compose up -d`). Defines ports, passwords, and persistent storage
for each.
**Concept:** Docker runs a full database as an isolated container instead of
installing it on the OS directly.

## .env
Holds secrets — LLM API key, Neo4j password, Postgres connection string.
Never committed to git.
**Concept:** secrets stay out of source code and out of git history, since a
leaked key gets exploited fast even if deleted later.

## requirements.txt
List of Python packages the project depends on (`pip install -r requirements.txt`
installs all of them).

## src/llm.py
Single function, `call_llm(prompt, system)`, that sends a prompt to whichever
LLM provider is set in `.env` (Groq right now) and returns plain text.
**Concept:** adapter pattern — every other file calls this one function and
never talks to the Groq/Gemini/Claude SDK directly, so switching providers
means changing one `.env` value, not the codebase.

## src/ingest/fetch_papers.py
Queries arXiv's public API for papers matching a search term, pulls out
title, abstract, authors, and arXiv id for each, saves them to
`data/raw_papers.json`.
**Concept:** arXiv's API returns XML (Atom format), not JSON — parsed with
Python's built-in `xml.etree.ElementTree`.

## data/raw_papers.json
Raw output of `fetch_papers.py` — the input to Phase 1's extraction step.

## src/ingest/schema.py
Defines the strict extraction schema (pydantic models for entities and
relations, restricted to the fixed ontology) and the prompt that instructs
the LLM to return only that JSON shape.
**Concept:** `pydantic.Literal` types restrict a field to an exact set of
allowed values — this is what actually enforces "only 6 entity types, only
9 relation types" in code, not just as prompt instructions.

## src/ingest/extract.py
Loops over `data/raw_papers.json`, sends each paper's title/abstract to the
LLM via `call_llm()`, validates the JSON response against the schema,
retries up to 3 times on failure, and saves results incrementally to
`data/extracted.json`.
**Concept:** caching by document id, saved after every item, means a crash
mid-run doesn't cost re-extracting (and re-paying for) everything already
done — only the run resumes from where it stopped. Also: pydantic's
`Literal` validation caught a real near-miss on the first run (Llama
returned `"ADDRESS"` instead of `"ADDRESSES"`), which triggered the retry
loop and succeeded on the 2nd attempt — schema validation working exactly
as intended, not just in theory.

## data/extracted.json
Validated entities and relations per paper, output of `extract.py`. Input
to Phase 2 (writing into Neo4j).

---

*(new entries added here as each phase is built)*

---

## Ontology

**Entities:** Paper, Author, Institution, Method, Dataset, Task

**Relationships:** AUTHORED_BY, AFFILIATED_WITH, CITES, PROPOSES, USES_METHOD,
EVALUATED_ON, ADDRESSES, EXTENDS, APPLIED_TO

## src/ingest/resolve_entities.py
Merges duplicate entities extracted with different surface names but
referring to the same thing (e.g. "RAG" and "retrieval-augmented
generation"). Two-pass approach: exact-match normalization first, then
embedding similarity (via a local sentence-transformers model) for the
rest. Outputs canonical entities with alias lists, and rewrites all
relations to point at canonical names, to `data/resolved.json`.
**Concept:** embedding similarity should only merge entities where "close
in meaning" genuinely implies "the same thing" — Method/Task/Dataset
names qualify, but Paper and Author don't, since two different papers or
people can sound similar without being the same. Applying fuzzy matching
there actually merged distinct papers on the first run; fixed by
restricting embedding-based merging to a specific set of entity types
and using exact-match only for identity types.
**Concept:** union-find (disjoint set) — a data structure for clustering
items that are pairwise similar into groups, handling transitive matches
(A~B, B~C means all three belong together) without comparing everything
to everything repeatedly.
**Concept:** cosine similarity across many pairs at once via a single
matrix multiply (`embeddings @ embeddings.T`) instead of looping.

## src/ingest/write_graph.py
Writes `data/resolved.json` into Neo4j — one MERGE per entity (keyed on
canonical_name) and one MERGE per relation. Creates a uniqueness
constraint per entity label first, so MERGE is genuinely idempotent:
re-running this script after adding more papers only adds new
nodes/edges, never duplicates existing ones.
**Concept:** Cypher can't parameterize relationship types the way it can
values — the type has to be part of the query string itself. Since our
relation types come from a fixed, pydantic-validated set (not raw model
text), we defensively assert each value is in that known set before
building the query, so this never becomes an actual injection risk.
**Verified with real Cypher queries in the Neo4j browser** — found a
genuine 3-hop connection between two different papers (different author
teams) both addressing "statute retrieval," reachable only by walking
Author → Paper → Task ← Paper ← Author. This is the core thing plain
vector search over abstracts would likely miss.

## src/ingest/build_vector_index.py
Embeds each paper's title+abstract (one chunk per paper, using the same
sentence-transformers model as entity resolution) and stores it in a
`paper_chunks` Postgres table alongside a pgvector HNSW index. Uses
`arxiv_id` as `chunk_id` — the shared key between this table and the
Neo4j graph, so later a vector search hit can be traced back into the
graph, and vice versa.
**Concept:** `ON CONFLICT DO UPDATE` (Postgres) is the same idempotency
idea as Neo4j's `MERGE` — re-running ingestion updates existing rows
rather than duplicating them.
**Concept:** pgvector's `<=>` operator is cosine *distance*; subtracting
from 1 converts it to a more intuitive similarity score. Ordering by
`<=>` is what actually uses the HNSW index.

## src/ingest/test_search.py
Quick manual check: embeds a natural-language question, finds the
closest chunks by cosine similarity. Verified good results — a query
about multi-hop QA correctly surfaced the most topically relevant papers,
confirming the embedding + index pipeline works end to end.

## src/retrieval/router.py
Classifies an incoming question as "graph", "vector", or "both", using a
few-shot prompt + strict JSON schema (same validate/retry pattern as
extraction). If the model's confidence is below a threshold, forces
"both" rather than guessing — runs both retrieval paths and merges,
instead of risking a wrong single-path answer. Verified against 3 test
questions — routed correctly (single-paper fact -> vector, relationship
question -> graph).
**Concept:** few-shot prompting — showing the model 2-3 worked examples
of the exact input/output shape wanted, before the real question,
noticeably improves classification reliability over describing the task
in words alone.
**Concept:** fail toward safety, not toward silence — if the router call
itself errors out, default to "both" rather than crashing or returning
nothing, so a router failure never means the user gets no answer.

## src/retrieval/vector_search.py
Reusable version of the earlier test search — embeds a query, returns
the top-k most similar chunks with similarity scores. Used by the router
for the "vector" path.

## src/retrieval/graph_search.py
The graph path: one LLM call extracts the entity + picks a query type
from a fixed 3-template library (neighbors / extends_chain /
shared_task) — never freeform Cypher. The extracted entity name is then
resolved against real graph node names using substring match first,
falling back to embedding similarity for looser phrasing.
**Concept:** substring matching handles cases embedding similarity
misses — e.g. "LegalMALR" (short) vs. the full paper title containing it
scored below the embedding threshold, but is trivially caught by a
literal substring check. Cheap checks before expensive ones.
**Concept:** few-shot examples need to cover every question *shape* you
care about, not just be present in general — the shared_task template
kept failing (extracting the wrong entity, e.g. "LegalMALR" instead of
the actual task "statute retrieval") until an example matching that
exact shape was added to the prompt. Description of the rule in words
wasn't enough; the model needed to see the pattern once.
**Verified**: all 3 query types (neighbors, extends_chain, shared_task)
correctly resolve entities and return results matching the manual Cypher
queries run earlier in the Neo4j browser.

## src/generation/merge.py
Ties everything together: routes the question, fetches from vector
and/or graph paths, converts graph results into readable statements,
assembles labeled context (PASSAGE vs GRAPH FACT, each with a citation
id), generates an answer, and validates every citation the model used
actually exists in the retrieved context — regenerating with a
correction prompt if not. `answer_question()` is the single entry point
the whole system funnels through.
**Concept:** citation validation via regex extraction + set comparison —
after generation, pull every [id] out of the answer text and check each
one against the actual set of retrieved context ids. This is what turns
"ask the model to cite sources" (which it can still get wrong) into "cite
sources it's provably not allowed to invent."
**Verified**: all 3 test questions produced correctly-cited, accurate
answers on the first attempt, across both routes and a mixed
graph+vector scenario.
