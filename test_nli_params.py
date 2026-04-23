import requests
from graph import GraphDatabase
from collections import defaultdict
from ingest import ingest_text
import time

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass


WIKIPEDIA_URLS = [
    ("https://en.wikipedia.org/wiki/CBRE_Group", "CBRE_Group"),
    ("https://en.wikipedia.org/wiki/Cushman_%26_Wakefield", "Cushman_Wakefield"),
    ("https://en.wikipedia.org/wiki/Colliers_International", "Colliers"),
    ("https://en.wikipedia.org/wiki/JLL_(company)", "JLL"),
]


def fetch_wikipedia_content(url: str) -> str:
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    }
    response = requests.get(url, headers=headers)
    response.raise_for_status()
    return response.text


def test_nli_params(entailment_threshold: float, contradiction_threshold: float):
    print(f"\n{'=' * 60}")
    print(
        f"Testing with entailment={entailment_threshold}, contradiction={contradiction_threshold}"
    )
    print(f"{'=' * 60}")

    db = GraphDatabase(
        entailment_threshold=entailment_threshold,
        contradiction_threshold=contradiction_threshold,
    )

    for url, name in WIKIPEDIA_URLS:
        print(f"  Fetching {name}...")
        content = fetch_wikipedia_content(url)
        print(f"    Fetched {len(content)} chars, ingesting...")
        try:
            ingest_text(db, content, name)
            print(
                f"    Done: {len(db.get_all_nodes())} nodes, {len(db.get_all_edges())} edges so far"
            )
        except Exception as e:
            print(f"    Error: {e}")
        time.sleep(1)

    total_nodes = len(db.get_all_nodes())
    total_edges = len(db.get_all_edges())

    edges_by_source_target = defaultdict(list)
    for edge in db.get_all_edges():
        key = (edge.source_node_id, edge.target_node_id)
        edges_by_source_target[key].append(edge)

    merged_count = 0
    for key, edges in edges_by_source_target.items():
        if len(edges) > 1:
            merged_count += len(edges) - 1

    multi_source_edges = sum(1 for e in db.get_all_edges() if len(e.sources) > 1)

    print(f"\n  Results:")
    print(f"    Nodes: {total_nodes}")
    print(f"    Edges: {total_edges}")
    print(f"    Edges merged: {merged_count}")
    print(f"    Multi-source edges: {multi_source_edges}")

    total_sources = sum(len(e.sources) for e in db.get_all_edges())
    print(f"    Total source references: {total_sources}")

    print("\n  Sample merged edges:")
    merged_edges = [e for e in db.get_all_edges() if len(e.sources) > 1][:5]
    for edge in merged_edges:
        src = db.get_node(edge.source_node_id)
        tgt = db.get_node(edge.target_node_id)
        print(
            f"    {src.name} -> {tgt.name}: '{edge.content[:50]}...' | sources: {edge.sources}"
        )

    return {
        "entailment_threshold": entailment_threshold,
        "contradiction_threshold": contradiction_threshold,
        "nodes": total_nodes,
        "edges": total_edges,
        "merged": merged_count,
        "multi_source": multi_source_edges,
        "total_sources": total_sources,
    }


def main():
    print("Starting NLI parameter testing with real Wikipedia data...")
    print("Scale: NLI entailment scores typically range -3 to +4")

    param_combinations = [
        (1.0, 0.0),
        (2.0, 0.0),
        (3.0, 0.0),
        (3.5, 0.0),
    ]

    results = []
    for entailment, contradiction in param_combinations:
        result = test_nli_params(entailment, contradiction)
        results.append(result)
        print("\n" + "-" * 40 + "\n")

    print("\n" + "=" * 60)
    print("SUMMARY OF RESULTS")
    print("=" * 60)
    print(
        f"{'Entailment':<12} {'Contradiction':<14} {'Nodes':<8} {'Edges':<8} {'Merged':<8} {'MultiSrc':<8} {'Srcs':<8}"
    )
    print("-" * 70)
    for r in results:
        print(
            f"{r['entailment_threshold']:<12.1f} {r['contradiction_threshold']:<14.1f} {r['nodes']:<8} {r['edges']:<8} {r['merged']:<8} {r['multi_source']:<8} {r['total_sources']:<8}"
        )

    best = max(results, key=lambda x: x["multi_source"])
    print(
        f"\nBest by multi-source edges: entailment={best['entailment_threshold']}, contradiction={best['contradiction_threshold']}"
    )


if __name__ == "__main__":
    main()
