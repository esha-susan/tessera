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

# Learning Log 01 — arXiv Paper Ingestion

**File:** `arxiv_loader.py`
**Purpose:** Fetch research papers from arXiv and save their basic metadata as JSON.

---

## 1. Where does this file fit in the project?

This is the **data ingestion stage** of the Knowledge Graph RAG system.

Before we can build a knowledge graph, we need research papers as our source data.

The overall flow starts approximately like this:

```text
arXiv
   ↓
arXiv API
   ↓
arxiv_loader.py
   ↓
raw_papers.json
   ↓
LLM-based information extraction
   ↓
Structured entities and relationships
   ↓
Knowledge Graph
```

So this file **does not build the knowledge graph**.

Its job is simply to collect relevant research papers and store their basic information in a clean format that later stages can process.

---

# 2. What does the file do?

The file performs five main tasks:

1. Defines a search query for relevant arXiv papers.
2. Sends the query to the arXiv API.
3. Receives the search results as XML.
4. Extracts useful information from each paper.
5. Saves the extracted information to `data/raw_papers.json`.

The information extracted for each paper is:

```text
arXiv ID
Title
Abstract
Authors
Publication date
```

---

# 3. Imports

```python
import json
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
```

Each module has a different purpose.

### `json`

Python's built-in JSON module.

It is used later to save our list of paper dictionaries into:

```text
data/raw_papers.json
```

---

### `urllib.parse`

Used to encode the search query so that it can safely be included in a URL.

For example, a query containing spaces and special characters needs to be converted into a URL-safe representation.

This is done using:

```python
urllib.parse.quote(query)
```

---

### `urllib.request`

Used to make an HTTP request to the arXiv API.

The important operation is:

```python
urllib.request.urlopen(url)
```

This allows our Python program to communicate with the arXiv server.

---

### `xml.etree.ElementTree`

The arXiv API returns its results as **XML**, rather than JSON.

`ElementTree` allows Python to parse that XML and navigate through elements such as:

```text
entry
title
summary
author
published
id
```

We import it as:

```python
import xml.etree.ElementTree as ET
```

so that we can use the shorter name `ET`.

---

# 4. arXiv API endpoint

```python
ARXIV_API = "http://export.arxiv.org/api/query"
```

An API is an interface that allows one program to communicate with another service.

Instead of manually opening arXiv in a browser and searching for papers, our Python program sends a request directly to the arXiv API.

Conceptually:

```text
Our Python program
        ↓
    HTTP request
        ↓
     arXiv API
        ↓
    XML response
        ↓
Our Python program
```

---

# 5. XML namespaces

```python
NS = {
    "atom": "http://www.w3.org/2005/Atom",
    "arxiv": "http://arxiv.org/schemas/atom",
}
```

The arXiv response uses XML namespaces.

A namespace helps distinguish XML elements that may have the same name but belong to different XML vocabularies.

The important namespace here is:

```text
http://www.w3.org/2005/Atom
```

which is given the shorter name:

```text
atom
```

This allows us to write:

```python
entry.find("atom:title", NS)
```

instead of dealing with the full namespace every time.

---

# 6. Search query

```python
SEARCH_QUERY = (
    'cat:cs.CL AND '
    '(abs:"retrieval augmented generation" '
    'OR abs:"dense retrieval" '
    'OR abs:"knowledge graph")'
)
```

This defines which papers we want to retrieve.

### `cat:cs.CL`

Restricts the search to the arXiv category:

```text
cs.CL
```

which is **Computation and Language**, a computer science category related to NLP and language technologies.

### `abs:`

Means that the search should consider the paper's abstract.

For example:

```text
abs:"knowledge graph"
```

looks for "knowledge graph" in the abstract.

### `OR`

Means that a paper can match any of the listed topics:

```text
retrieval augmented generation
OR
dense retrieval
OR
knowledge graph
```

### `AND`

The paper must also belong to the specified category.

Therefore the query roughly means:

> Find relevant papers in the Computation and Language category whose abstracts mention retrieval-augmented generation, dense retrieval, or knowledge graphs.

---

# 7. Maximum number of results

```python
MAX_RESULTS = 20
```

This limits the number of papers retrieved to 20.

It prevents the ingestion step from downloading an unnecessarily large number of papers.

---

# 8. `fetch_papers()` function

```python
def fetch_papers(query: str, max_results: int) -> list[dict]:
```

This function is responsible for the entire arXiv retrieval process.

### Inputs

```text
query
```

The arXiv search query.

```text
max_results
```

The maximum number of papers to retrieve.

### Type hints

```python
query: str
```

means `query` is expected to be a string.

```python
max_results: int
```

means `max_results` is expected to be an integer.

```python
-> list[dict]
```

means the function is expected to return a list of dictionaries.

For example:

```python
[
    {
        "title": "Paper 1",
        "authors": ["Alice", "Bob"]
    },
    {
        "title": "Paper 2",
        "authors": ["Charlie"]
    }
]
```

---

# 9. Constructing the API parameters

```python
params = (
    f"search_query={urllib.parse.quote(query)}"
    f"&start=0"
    f"&max_results={max_results}"
    f"&sortBy=relevance"
)
```

This constructs the parameters that will be attached to the API URL.

There are four important parameters.

### `search_query`

Contains the encoded search query.

```python
urllib.parse.quote(query)
```

converts the query into a URL-safe form.

### `start=0`

Starts retrieving results from the beginning of the result set.

### `max_results`

Specifies how many papers should be returned.

### `sortBy=relevance`

Requests that arXiv order the results according to relevance.

---

# 10. Constructing the final URL

```python
url = f"{ARXIV_API}?{params}"
```

This combines:

```text
API endpoint
+
query parameters
```

into one URL.

Conceptually:

```text
http://export.arxiv.org/api/query?
    search_query=...
    &start=0
    &max_results=20
    &sortBy=relevance
```

This is the URL that Python will request.

---

# 11. Sending the HTTP request

```python
with urllib.request.urlopen(url) as response:
    raw_xml = response.read()
```

This sends the request to arXiv.

The server returns an HTTP response containing XML.

```python
response
```

represents that response.

Then:

```python
response.read()
```

reads the actual response content.

The XML is stored in:

```python
raw_xml
```

The process is:

```text
Python
  ↓
HTTP request
  ↓
arXiv
  ↓
XML response
  ↓
raw_xml
```

---

# 12. Why use `with`?

```python
with urllib.request.urlopen(url) as response:
```

`with` is Python's context-manager syntax.

It ensures that the network resource is properly cleaned up after it is used.

Conceptually:

```text
Open connection
      ↓
Use connection
      ↓
Close connection
```

The same idea is used later when opening the JSON file.

---

# 13. Parsing the XML

```python
root = ET.fromstring(raw_xml)
```

At this point, `raw_xml` is raw XML data.

`ET.fromstring()` parses it into an XML tree.

Conceptually:

```text
Raw XML
   ↓
ElementTree parser
   ↓
Structured XML tree
```

Now Python can search for elements such as:

```text
entry
title
summary
author
published
id
```

---

# 14. Create an empty paper list

```python
papers = []
```

This list will eventually contain one dictionary for every retrieved paper.

Initially:

```python
papers = []
```

After processing papers:

```python
papers = [
    {
        "arxiv_id": "...",
        "title": "...",
        "abstract": "...",
        "authors": [...],
        "published": "..."
    },
    ...
]
```

---

# 15. Process every paper

```python
for entry in root.findall("atom:entry", NS):
```

The arXiv XML response contains multiple `<entry>` elements.

Each `<entry>` represents one paper.

So this loop means:

> For every paper returned by arXiv, extract its information.

Conceptually:

```text
XML response
    │
    ├── entry → Paper 1
    ├── entry → Paper 2
    ├── entry → Paper 3
    └── ...
```

The loop processes each one.

---

# 16. Extracting the arXiv ID

```python
arxiv_id = entry.find("atom:id", NS).text.strip().split("/abs/")[-1]
```

The API gives us an ID URL such as:

```text
http://arxiv.org/abs/2606.13916
```

We want only:

```text
2606.13916
```

The operations happen in sequence:

```python
entry.find("atom:id", NS)
```

Finds the `<id>` element.

```python
.text
```

gets its text.

```python
.strip()
```

removes unnecessary whitespace.

```python
.split("/abs/")
```

splits the URL around `/abs/`.

Finally:

```python
[-1]
```

takes the last part, which is the actual arXiv ID.

---

# 17. Extracting the title

```python
title = entry.find("atom:title", NS).text.strip().replace("\n", " ")
```

This finds the title.

The operations are:

```text
find title
   ↓
get text
   ↓
remove surrounding whitespace
   ↓
replace newlines with spaces
```

The newline replacement makes the stored title cleaner.

Instead of:

```text
Knowledge Graph
for Retrieval Augmented Generation
```

we get:

```text
Knowledge Graph for Retrieval Augmented Generation
```

---

# 18. Extracting the abstract

```python
abstract = entry.find("atom:summary", NS).text.strip().replace("\n", " ")
```

In the arXiv API, the `<summary>` field contains the paper's abstract.

So this extracts and cleans the abstract.

The abstract is especially important for the later stages because the LLM will eventually use the paper information to extract structured knowledge.

---

# 19. Extracting the publication date

```python
published = entry.find("atom:published", NS).text.strip()[:10]
```

The API may return a full timestamp such as:

```text
2026-06-11T12:34:56Z
```

The code only needs the date:

```text
2026-06-11
```

Python slicing:

```python
[:10]
```

takes the first ten characters.

---

# 20. Extracting authors

```python
authors = [
    author.find("atom:name", NS).text
    for author in entry.findall("atom:author", NS)
]
```

This is a list comprehension.

It is equivalent to:

```python
authors = []

for author in entry.findall("atom:author", NS):
    name = author.find("atom:name", NS).text
    authors.append(name)
```

If the paper has:

```text
Alice
Bob
Charlie
```

then:

```python
authors
```

becomes:

```python
["Alice", "Bob", "Charlie"]
```

---

# 21. Creating our own paper representation

```python
papers.append(
    {
        "arxiv_id": arxiv_id,
        "title": title,
        "abstract": abstract,
        "authors": authors,
        "published": published,
    }
)
```

This is an important transformation.

The arXiv API gives us XML.

We don't want to pass the entire XML structure through the rest of the project.

Instead, we convert each paper into a simple Python dictionary.

Each paper now looks like:

```python
{
    "arxiv_id": "2606.13916",
    "title": "...",
    "abstract": "...",
    "authors": ["Author 1", "Author 2"],
    "published": "2026-06-11"
}
```

This is essentially a **normalization step** from the API's XML representation into the representation used by our project.

---

# 22. Return the papers

```python
return papers
```

Once all entries have been processed, the function returns the complete list.

The result is:

```python
[
    paper1,
    paper2,
    paper3,
    ...
]
```

where every paper is represented by a dictionary.

---

# 23. The `__main__` block

The actual Python code should be:

```python
if __name__ == "__main__":
```

This means:

> Run the following code only when this Python file is executed directly.

For example:

```bash
python arxiv_loader.py
```

will execute the code inside the block.

But if another Python file does:

```python
from arxiv_loader import fetch_papers
```

the function can be imported without automatically executing the code that downloads papers.

This makes the file reusable as a Python module.

---

# 24. Calling the function

```python
papers = fetch_papers(SEARCH_QUERY, MAX_RESULTS)
```

This passes:

```text
SEARCH_QUERY
      ↓
fetch_papers()
      ↓
MAX_RESULTS
```

The function then:

```text
Build URL
   ↓
Call arXiv
   ↓
Receive XML
   ↓
Parse XML
   ↓
Process every paper
   ↓
Create dictionaries
   ↓
Return list
```

The returned list is stored in:

```python
papers
```

---

# 25. Saving the results

```python
with open("data/raw_papers.json", "w") as f:
    json.dump(papers, f, indent=2)
```

This writes the Python list to a JSON file.

The file is:

```text
data/raw_papers.json
```

The `"w"` means the file is opened in write mode.

If it doesn't exist, it will be created.

---

## `json.dump()`

```python
json.dump(papers, f, indent=2)
```

converts the Python object into JSON and writes it to the file.

For example:

```python
papers = [
    {
        "title": "Knowledge Graph RAG",
        "authors": ["Alice"]
    }
]
```

becomes:

```json
[
  {
    "title": "Knowledge Graph RAG",
    "authors": [
      "Alice"
    ]
  }
]
```

`indent=2` makes the JSON human-readable.

---

# 26. Displaying the first three papers

```python
for paper in papers[:3]:
    print(f"  - {paper['title']} ({paper['arxiv_id']})")
```

`papers[:3]` selects the first three papers.

The code then prints their titles and IDs.

This isn't part of the actual data-processing pipeline.

It is mainly a quick **sanity check** to confirm that papers were successfully retrieved.

---

# 27. Complete data flow

The most important mental model for this file is:

```text
                 SEARCH QUERY
                      │
                      ▼
              ┌──────────────┐
              │  arXiv API   │
              └──────┬───────┘
                     │
                     │ HTTP request
                     ▼
                XML response
                     │
                     ▼
             ElementTree parser
                     │
                     ▼
              <entry> elements
                     │
          ┌──────────┼──────────┐
          ▼          ▼          ▼
        Paper 1    Paper 2    Paper 3
          │          │          │
          ▼          ▼          ▼
       Extract    Extract    Extract
       metadata   metadata   metadata
          │          │          │
          └──────────┼──────────┘
                     ▼
            Python dictionaries
                     │
                     ▼
              papers list
                     │
                     ▼
           json.dump() / JSON
                     │
                     ▼
           data/raw_papers.json
```

---

# 28. Where this fits into Tessera

This file represents **data acquisition**.

It does not yet understand the research knowledge contained in the papers.

For example, suppose an abstract says:

> "We propose a retrieval augmented generation system using a knowledge graph and evaluate it on the HotpotQA dataset."

This file simply stores that as text:

```python
{
    "abstract": "We propose a retrieval augmented generation..."
}
```

A later component can use an LLM to turn that unstructured text into structured knowledge such as:

```text
Paper
 ├── uses → Retrieval Augmented Generation
 ├── uses → Knowledge Graph
 ├── evaluated_on → HotpotQA
 └── proposes → Some Method
```

That structured information is what eventually becomes useful for constructing the knowledge graph.

So the distinction is:

```text
THIS FILE

Collect raw information
        ↓
        JSON


LATER FILES

Understand information
        ↓
Extract entities/relationships
        ↓
Knowledge Graph
```

---

# 29. Important concepts learned from this file

| Concept            | Meaning                                                  |
| ------------------ | -------------------------------------------------------- |
| API                | Interface allowing programs to communicate               |
| HTTP request       | Request sent from our program to a server                |
| XML                | Structured data format returned by arXiv                 |
| XML parser         | Converts XML into a structure Python can navigate        |
| Namespace          | Identifies the vocabulary an XML element belongs to      |
| URL encoding       | Converts text into a URL-safe representation             |
| JSON               | Structured data format used to store/transmit data       |
| List comprehension | Compact way of constructing a list                       |
| Type hint          | Indicates expected types of function inputs/outputs      |
| Context manager    | Safely manages resources such as files/connections       |
| `__main__`         | Allows code to run only when a file is executed directly |

---

# 30. Interview explanation

If asked:

### "What does your arXiv ingestion module do?"

A good explanation is:

> "The ingestion module queries the arXiv API for research papers relevant to our RAG and knowledge-graph domain. Since the API returns Atom XML, I parse the response using Python's ElementTree, extract metadata such as the paper ID, title, abstract, authors, and publication date, normalize each paper into a Python dictionary, and persist the collection as JSON for downstream knowledge extraction."

---

# 31. Questions an interviewer could ask

### Q1. Why use an API instead of scraping arXiv?

Because the API provides a structured and programmatic way to retrieve paper metadata without parsing arbitrary webpage HTML.

### Q2. Why is the response XML?

The arXiv API exposes its results using the Atom XML format, so the application uses `ElementTree` to parse it.

### Q3. Why do you encode the query?

Because the query becomes part of a URL, and spaces/special characters need URL encoding.

### Q4. Why save the papers as JSON?

JSON is simple, human-readable, and easy for Python and downstream processing components to consume.

### Q5. Why don't you create the knowledge graph here?

This module is responsible only for **data acquisition**. Knowledge extraction is a separate stage so that ingestion and semantic processing remain separated.

### Q6. What information do you extract?

```text
arXiv ID
Title
Abstract
Authors
Publication date
```

### Q7. What is the output of `fetch_papers()`?

A:

```python
list[dict]
```

where each dictionary represents one paper.

---

# 32. The one-sentence takeaway

> **`arxiv_loader.py` converts relevant research papers from the arXiv API's XML format into a clean list of Python dictionaries and persists them as `raw_papers.json`, providing the raw input for the later knowledge-extraction and knowledge-graph stages.**

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
