import io
import base64
import networkx as nx
import matplotlib.pyplot as plt
from flask import Flask, render_template_string, send_file
from graph import GraphDatabase

app = Flask(__name__)

HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>Graph Visualization</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; background: #f5f5f5; }
        h1 { color: #333; }
        .info { background: white; padding: 15px; border-radius: 8px; margin-bottom: 20px; }
        img { max-width: 100%; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
        .controls { margin-bottom: 20px; }
        a { color: #0066cc; }
    </style>
</head>
<body>
    <h1>Knowledge Graph Visualization</h1>
    <div class="info">
        <strong>Stats:</strong> {{ nodes|length }} nodes, {{ edges|length }} edges
    </div>
    <div class="controls">
        <a href="/">Full Graph</a> | 
        <a href="/subgraph">Subgraph (CBRE Group)</a> | 
        <a href="/identity">Identity Facts Only</a>
    </div>
    <img src="/plot.png" alt="Graph Visualization">
</body>
</html>
"""

_graph_db = None


def set_graph_db(db: GraphDatabase):
    global _graph_db
    _graph_db = db


@app.route("/")
def index():
    if _graph_db is None:
        return "No graph loaded", 500

    nodes = [
        {"id": n.id, "name": n.name, "facts": n.identity_facts}
        for n in _graph_db.get_all_nodes()
    ]
    edges = []
    for e in _graph_db.get_all_edges():
        src = _graph_db.get_node(e.source_node_id)
        tgt = _graph_db.get_node(e.target_node_id)
        edges.append(
            {
                "from": src.name if src else "Unknown",
                "to": tgt.name if tgt else "Unknown",
                "content": e.content,
            }
        )

    return render_template_string(HTML_TEMPLATE, nodes=nodes, edges=edges)


@app.route("/subgraph")
def subgraph():
    if _graph_db is None:
        return "No graph loaded", 500

    cbre_node = next((n for n in _graph_db.get_all_nodes() if "CBRE" in n.name), None)
    if not cbre_node:
        return "CBRE node not found", 404

    nodes = [
        {"id": cbre_node.id, "name": cbre_node.name, "facts": cbre_node.identity_facts}
    ]
    edges = []

    for edge in cbre_node.get_edges():
        target = _graph_db.get_node(edge.target_node_id)
        nodes.append(
            {"id": target.id, "name": target.name, "facts": target.identity_facts}
        )
        edges.append(
            {"from": cbre_node.name, "to": target.name, "content": edge.content}
        )

    return render_template_string(HTML_TEMPLATE, nodes=nodes, edges=edges)


@app.route("/identity")
def identity():
    if _graph_db is None:
        return "No graph loaded", 500

    nodes = []
    for n in _graph_db.get_all_nodes():
        if n.identity_facts:
            nodes.append({"id": n.id, "name": n.name, "facts": n.identity_facts})

    return render_template_string(HTML_TEMPLATE, nodes=nodes, edges=[])


@app.route("/plot.png")
def plot():
    if _graph_db is None:
        return "No graph loaded", 500

    fig = generate_plot(_graph_db)

    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=150)
    buf.seek(0)

    return send_file(buf, mimetype="image/png")


def generate_plot(db: GraphDatabase, center_node_id: int = None, depth: int = None):
    G = nx.DiGraph()

    if center_node_id is not None and depth is not None:
        center_node = db.get_node(center_node_id)
        if center_node:
            G.add_node(center_node.id, label=center_node.name)
            for edge in center_node.get_edges():
                G.add_node(
                    edge.target_node_id, label=db.get_node(edge.target_node_id).name
                )
                G.add_edge(
                    edge.source_node_id, edge.target_node_id, label=edge.content[:40]
                )
    else:
        for node in db.get_all_nodes():
            G.add_node(node.id, label=node.name)

        for edge in db.get_all_edges():
            G.add_edge(
                edge.source_node_id, edge.target_node_id, label=edge.content[:40]
            )

    fig, ax = plt.subplots(1, 1, figsize=(20, 16))

    try:
        pos = nx.spring_layout(G, k=2, iterations=50, seed=42)
    except:
        pos = nx.circular_layout(G)

    node_labels = {node[0]: node[1]["label"] for node in G.nodes(data=True)}
    nx.draw_networkx_nodes(
        G, pos, ax=ax, node_color="lightblue", node_size=1500, alpha=0.8
    )
    nx.draw_networkx_labels(G, pos, labels=node_labels, ax=ax, font_size=8)

    edge_labels = {(u, v): d["label"] for u, v, d in G.edges(data=True)}
    nx.draw_networkx_edges(
        G, pos, ax=ax, edge_color="gray", arrows=True, arrowsize=15, alpha=0.6
    )
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, ax=ax, font_size=6)

    ax.axis("off")
    ax.set_title(
        f"Graph: {len(db.get_all_nodes())} nodes, {len(db.get_all_edges())} edges"
    )

    return fig


def run_visualization_server(db: GraphDatabase, port: int = 5000):
    set_graph_db(db)
    print(f"Starting visualization server on http://localhost:{port}")
    print("Press Ctrl+C to stop")
    app.run(host="0.0.0.0", port=port, debug=False)


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "--port":
        port = int(sys.argv[2]) if len(sys.argv) > 2 else 5000
    else:
        port = 5000

    from main import fetch_wikipedia_content
    from ingest import ingest_text

    print("Loading CBRE data...")
    content = fetch_wikipedia_content("https://en.wikipedia.org/wiki/CBRE_Group")

    db = GraphDatabase()
    ingest_text(db, content, "wikipedia_cbre")

    print(f"Database: {len(db.get_all_nodes())} nodes, {len(db.get_all_edges())} edges")

    run_visualization_server(db, port=port)
