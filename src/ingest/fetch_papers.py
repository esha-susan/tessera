import json
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET

ARXIV_API = "http://export.arxiv.org/api/query"

NS = {
    "atom": "http://www.w3.org/2005/Atom",
    "arxiv": "http://arxiv.org/schemas/atom",
}

SEARCH_QUERY = (
    'cat:cs.CL AND '
    '(abs:"retrieval augmented generation" '
    'OR abs:"dense retrieval" '
    'OR abs:"knowledge graph")'
)

MAX_RESULTS = 20


def fetch_papers(query: str, max_results: int) -> list[dict]:
    params = (
        f"search_query={urllib.parse.quote(query)}"
        f"&start=0"
        f"&max_results={max_results}"
        f"&sortBy=relevance"
    )

    url = f"{ARXIV_API}?{params}"

    with urllib.request.urlopen(url) as response:
        raw_xml = response.read()

    root = ET.fromstring(raw_xml)
    papers = []

    for entry in root.findall("atom:entry", NS):
        arxiv_id = entry.find("atom:id", NS).text.strip().split("/abs/")[-1]
        title = entry.find("atom:title", NS).text.strip().replace("\n", " ")
        abstract = entry.find("atom:summary", NS).text.strip().replace("\n", " ")
        published = entry.find("atom:published", NS).text.strip()[:10]

        authors = [
            author.find("atom:name", NS).text
            for author in entry.findall("atom:author", NS)
        ]

        papers.append(
            {
                "arxiv_id": arxiv_id,
                "title": title,
                "abstract": abstract,
                "authors": authors,
                "published": published,
            }
        )

    return papers


if __name__ == "__main__":
    print(f"Querying arXiv for: {SEARCH_QUERY}")

    papers = fetch_papers(SEARCH_QUERY, MAX_RESULTS)

    print(f"Fetched {len(papers)} papers")

    with open("data/raw_papers.json", "w") as f:
        json.dump(papers, f, indent=2)

    print("Saved to data/raw_papers.json")

    for paper in papers[:3]:
        print(f"  - {paper['title']} ({paper['arxiv_id']})")
