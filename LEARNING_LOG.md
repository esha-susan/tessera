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

---

*(new entries added here as each phase is built)*

---

## Ontology

**Entities:** Paper, Author, Institution, Method, Dataset, Task

**Relationships:** AUTHORED_BY, AFFILIATED_WITH, CITES, PROPOSES, USES_METHOD,
EVALUATED_ON, ADDRESSES, EXTENDS, APPLIED_TO
