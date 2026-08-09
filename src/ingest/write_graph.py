import json
import os
from neo4j import GraphDatabase
from dotenv import load_dotenv

load_dotenv()

RESOLVED_PATH = "data/resolved.json"

NEO4J_URI = os.getenv("NEO4J_URI")
NEO4J_USER = os.getenv("NEO4J_USER")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")

ENTITY_LABELS = {"Paper", "Author", "Institution", "Method", "Dataset", "Task"}

RELATION_TYPES = {
    "AUTHORED_BY",
    "AFFILIATED_WITH",
    "CITES",
    "PROPOSES",
    "USES_METHOD",
    "EVALUATED_ON",
    "ADDRESSES",
    "EXTENDS",
    "APPLIED_TO",
}


def get_driver():
    return GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))


def create_constraints(driver):
    with driver.session() as session:
        for label in ENTITY_LABELS:
            session.run(
                f"CREATE CONSTRAINT IF NOT EXISTS FOR (n:{label}) REQUIRE n.canonical_name IS UNIQUE"
            )


def write_entities(driver, entities):
    with driver.session() as session:
        for e in entities:
            label = e["entity_type"]
            if label not in ENTITY_LABELS:
                raise ValueError(f"Unknown entity label: {label}")
            session.run(
                f"""
                MERGE (n:{label} {{canonical_name: $name}})
                SET n.aliases = $aliases
                """,
                name=e["canonical_name"],
                aliases=e["aliases"],
            )


def write_relations(driver, relations):
    with driver.session() as session:
        for r in relations:
            source_label = r["source_type"]
            target_label = r["target_type"]
            rel_type = r["relation_type"]

            if source_label not in ENTITY_LABELS or target_label not in ENTITY_LABELS:
                raise ValueError(f"Unknown entity label in relation: {r}")
            if rel_type not in RELATION_TYPES:
                raise ValueError(f"Unknown relation type: {rel_type}")

            session.run(
                f"""
                MATCH (a:{source_label} {{canonical_name: $source}})
                MATCH (b:{target_label} {{canonical_name: $target}})
                MERGE (a)-[rel:{rel_type}]->(b)
                SET rel.confidence = $confidence
                """,
                source=r["source_entity"],
                target=r["target_entity"],
                confidence=r["confidence"],
            )


if __name__ == "__main__":
    with open(RESOLVED_PATH) as f:
        data = json.load(f)

    driver = get_driver()

    print("Creating uniqueness constraints...")
    create_constraints(driver)

    print(f"Writing {len(data['entities'])} entities...")
    write_entities(driver, data["entities"])

    print(f"Writing {len(data['relations'])} relations...")
    write_relations(driver, data["relations"])

    driver.close()
    print("Done.")
