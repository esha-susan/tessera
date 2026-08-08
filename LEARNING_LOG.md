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
