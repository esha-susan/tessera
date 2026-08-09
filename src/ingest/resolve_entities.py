import json
import re
from collections import defaultdict
from sentence_transformers import SentenceTransformer
import numpy as np

EXTRACTED_PATH = "data/extracted.json"
OUTPUT_PATH = "data/resolved.json"
SIMILARITY_THRESHOLD = 0.82
FUZZY_MATCH_TYPES = {"Method", "Task", "Dataset"}

_model = SentenceTransformer("all-MiniLM-L6-v2")


def normalize(name: str) -> str:
    name = name.lower().strip()
    name = re.sub(r"\s+", " ", name)
    name = re.sub(r"[^\w\s\-]", "", name)
    return name


class UnionFind:
    def __init__(self, items):
        self.parent = {item: item for item in items}

    def find(self, x):
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[ra] = rb


def resolve_entities_for_type(names: list[str], entity_type: str) -> dict[str, str]:
    unique_names = list(set(names))
    if len(unique_names) <= 1:
        return {n: unique_names[0] if unique_names else n for n in names}

    uf = UnionFind(unique_names)

    by_normalized = defaultdict(list)
    for n in unique_names:
        by_normalized[normalize(n)].append(n)
    for group in by_normalized.values():
        for n in group[1:]:
            uf.union(n, group[0])

    if entity_type in FUZZY_MATCH_TYPES:
        embeddings = _model.encode(unique_names, normalize_embeddings=True)
        sim_matrix = embeddings @ embeddings.T

        for i in range(len(unique_names)):
            for j in range(i + 1, len(unique_names)):
                if sim_matrix[i][j] >= SIMILARITY_THRESHOLD:
                    uf.union(unique_names[i], unique_names[j])

    clusters = defaultdict(list)
    for n in unique_names:
        clusters[uf.find(n)].append(n)

    canonical_for_root = {
        root: min(members, key=len) for root, members in clusters.items()
    }

    return {n: canonical_for_root[uf.find(n)] for n in unique_names}


if __name__ == "__main__":
    with open(EXTRACTED_PATH) as f:
        data = json.load(f)

    names_by_type = defaultdict(list)
    all_relations = []

    for arxiv_id, paper in data.items():
        for e in paper["entities"]:
            names_by_type[e["entity_type"]].append(e["canonical_name"])
        for r in paper["relations"]:
            all_relations.append(r)

    declared = {(t, n) for t, names in names_by_type.items() for n in names}
    missing = set()
    for r in all_relations:
        for entity, etype in [
            (r["source_entity"], r["source_type"]),
            (r["target_entity"], r["target_type"]),
        ]:
            if (etype, entity) not in declared:
                missing.add((etype, entity))

    if missing:
        print(f"\nNote: {len(missing)} entities were referenced in relations but never declared — adding them:")
        for etype, entity in sorted(missing):
            print(f"  {etype}: {entity}")
            names_by_type[etype].append(entity)

    print("Entities per type before resolution:")
    for etype, names in names_by_type.items():
        print(f"  {etype}: {len(names)} raw, {len(set(names))} unique strings")

    rename_map = {}
    for etype, names in names_by_type.items():
        resolved = resolve_entities_for_type(names, etype)
        for raw, canonical in resolved.items():
            rename_map[(etype, raw)] = canonical

    aliases = defaultdict(set)
    for (etype, raw), canonical in rename_map.items():
        aliases[(etype, canonical)].add(raw)

    print("\nEntities per type after resolution:")
    for etype in names_by_type:
        canonicals = {c for (t, c) in aliases if t == etype}
        print(f"  {etype}: {len(canonicals)} canonical entities")

    resolved_relations = []
    for r in all_relations:
        resolved_relations.append(
            {
                "source_entity": rename_map[(r["source_type"], r["source_entity"])],
                "source_type": r["source_type"],
                "relation_type": r["relation_type"],
                "target_entity": rename_map[(r["target_type"], r["target_entity"])],
                "target_type": r["target_type"],
                "confidence": r["confidence"],
            }
        )

    resolved_entities = [
        {"entity_type": etype, "canonical_name": canonical, "aliases": sorted(names)}
        for (etype, canonical), names in aliases.items()
    ]

    with open(OUTPUT_PATH, "w") as f:
        json.dump(
            {"entities": resolved_entities, "relations": resolved_relations},
            f,
            indent=2,
        )

    print(f"\nSaved {len(resolved_entities)} entities, {len(resolved_relations)} relations to {OUTPUT_PATH}")
