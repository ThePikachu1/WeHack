import os
import json
import uuid
import mimetypes
from pathlib import Path
from datetime import datetime

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

from flask import Flask, request, stream_with_context, Response, jsonify

from graph import GraphDatabase
from orchestration import SessionManager, create_agent
from sources import TextSourceProvider, WebSourceProvider, FileSourceProvider


app = Flask(__name__)
DATA_DIR = Path("./data")
DATA_DIR.mkdir(exist_ok=True)

CONVERSATIONS_DIR = DATA_DIR / "conversations"
CONVERSATIONS_DIR.mkdir(exist_ok=True)

ATTACHMENTS_DIR = DATA_DIR / "attachments"
ATTACHMENTS_DIR.mkdir(exist_ok=True)


def get_conversation_dir(conv_id: int) -> Path:
    d = CONVERSATIONS_DIR / str(conv_id)
    d.mkdir(exist_ok=True)
    return d


def load_conversation_meta(conv_id: int) -> dict:
    path = get_conversation_dir(conv_id) / "meta.json"
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return {"name": f"Conversation {conv_id}", "created_at": datetime.now().isoformat()}


def save_conversation_meta(conv_id: int, meta: dict) -> None:
    path = get_conversation_dir(conv_id) / "meta.json"
    with open(path, "w") as f:
        json.dump(meta, f)


@app.route("/new_conversation", methods=["POST"])
def new_conversation():
    data = request.get_json(silent=True) or {}
    name = data.get("name", f"Conversation {datetime.now().strftime('%Y-%m-%d %H:%M')}")

    conv_id = (
        len(list(CONVERSATIONS_DIR.iterdir())) + 1 if CONVERSATIONS_DIR.exists() else 1
    )

    meta = {"name": name, "created_at": datetime.now().isoformat()}
    save_conversation_meta(conv_id, meta)

    return jsonify({"conv_id": conv_id})


@app.route("/list_conversations", methods=["GET"])
def list_conversations():
    result = []
    for conv_dir in CONVERSATIONS_DIR.iterdir():
        if conv_dir.is_dir():
            try:
                conv_id = int(conv_dir.name)
                meta = load_conversation_meta(conv_id)
                result.append(
                    {"conv_id": conv_id, "name": meta.get("name", "Untitled")}
                )
            except ValueError:
                continue
    result.sort(key=lambda x: x["conv_id"])
    return jsonify(result)


@app.route("/conversations/<int:conv_id>/upload_attachment", methods=["POST"])
def upload_attachment(conv_id: int):
    filename = request.headers.get("Content-Disposition", "")
    content_type = request.headers.get("Content-Type", "application/octet-stream")

    if filename:
        import re

        match = re.search(r'filename="([^"]+)"', filename)
        if match:
            filename = match.group(1)
    else:
        filename = "unnamed"

    attachment_id = str(uuid.uuid4())
    attachment_path = ATTACHMENTS_DIR / f"{attachment_id}_{conv_id}"

    with open(attachment_path, "wb") as f:
        f.write(request.data)

    size = len(request.data)

    meta_path = get_conversation_dir(conv_id) / "attachments.json"
    if meta_path.exists():
        with open(meta_path) as f:
            attachments = json.load(f)
    else:
        attachments = []

    attachments.append(
        {
            "attachment_id": attachment_id,
            "name": filename,
            "type": content_type,
            "size": size,
        }
    )

    with open(meta_path, "w") as f:
        json.dump(attachments, f)

    return jsonify({"attachment_id": attachment_id})


@app.route("/conversations/<int:conv_id>/attachments/<attachment_id>", methods=["GET"])
def get_attachment(conv_id: int, attachment_id: str):
    attachment_path = ATTACHMENTS_DIR / f"{attachment_id}_{conv_id}"
    if not attachment_path.exists():
        return jsonify({"error": "Not found"}), 404

    meta_path = get_conversation_dir(conv_id) / "attachments.json"
    if meta_path.exists():
        with open(meta_path) as f:
            attachments = json.load(f)
        att = next(
            (a for a in attachments if a["attachment_id"] == attachment_id), None
        )
        content_type = att["type"] if att else "application/octet-stream"
    else:
        content_type = "application/octet-stream"

    return Response(
        attachment_path.read_bytes(),
        mimetype=content_type,
        headers={"Content-Disposition": f"attachment; filename={attachment_id}"},
    )


@app.route("/conversations/<int:conv_id>/attachments", methods=["GET"])
def list_attachments(conv_id: int):
    meta_path = get_conversation_dir(conv_id) / "attachments.json"
    if meta_path.exists():
        with open(meta_path) as f:
            attachments = json.load(f)
    else:
        attachments = []

    result = []
    for a in attachments:
        result.append(
            {
                "type": a["type"],
                "name": a["name"],
                "size": a["size"],
                "attachment_id": a["attachment_id"],
            }
        )
    return jsonify(result)


@app.route("/conversations/<int:conv_id>/history", methods=["GET"])
def get_history(conv_id: int):
    history_path = get_conversation_dir(conv_id) / "history.json"
    if history_path.exists():
        with open(history_path) as f:
            history = json.load(f)
    else:
        history = []
    return jsonify(history)


@app.route("/conversations/<int:conv_id>/sources", methods=["GET"])
def get_sources(conv_id: int):
    history = []
    history_path = get_conversation_dir(conv_id) / "history.json"
    if history_path.exists():
        with open(history_path) as f:
            history = json.load(f)

    sources_seen = {}
    for msg in history:
        atts = msg.get("attachments", [])
        for att_id in atts:
            sources_seen[att_id] = True

    meta_path = get_conversation_dir(conv_id) / "attachments.json"
    if meta_path.exists():
        with open(meta_path) as f:
            attachments = json.load(f)
    else:
        attachments = []

    result = []
    for a in attachments:
        if a["attachment_id"] in sources_seen:
            result.append(
                {
                    "type": a["type"],
                    "name": a["name"],
                    "size": a["size"],
                    "attachment_id": a["attachment_id"],
                }
            )

    return jsonify(result)


@app.route("/conversations/<int:conv_id>/graph", methods=["GET"])
def get_graph(conv_id: int):
    db = GraphDatabase()

    db_path = get_conversation_dir(conv_id) / "graph"
    if db_path.exists():
        try:
            db = GraphDatabase.from_saved(str(db_path))
        except Exception:
            pass

    edges_data = []
    for edge in db.edges.values():
        src = db.nodes.get(edge.source_node_id)
        tgt = db.nodes.get(edge.target_node_id)
        edges_data.append(
            {
                "from": edge.source_node_id,
                "to": edge.target_node_id,
                "content": edge.content,
                "sources": edge.sources,
            }
        )

    nodes_data = []
    for node in db.nodes.values():
        nodes_data.append(
            {
                "node_id": node.id,
                "label": node.name,
                "facts": [
                    {"content": f["fact"], "sources": f["source"]}
                    for f in node.identity_facts
                ],
            }
        )

    return jsonify(
        {
            "edges": edges_data,
            "nodes": nodes_data,
        }
    )


@app.route("/conversations/<int:conv_id>/chat", methods=["POST"])
def chat(conv_id: int):
    data = request.get_json()
    message = data.get("message", "")
    attachments = data.get("attachments", [])

    history_path = get_conversation_dir(conv_id) / "history.json"
    if history_path.exists():
        with open(history_path) as f:
            history = json.load(f)
    else:
        history = []

    db_path = get_conversation_dir(conv_id) / "graph"
    if db_path.exists():
        try:
            db = GraphDatabase.from_saved(str(db_path))
        except Exception:
            db = GraphDatabase()
    else:
        db = GraphDatabase()

    for att_id in attachments:
        attachment_path = ATTACHMENTS_DIR / f"{att_id}_{conv_id}"
        if attachment_path.exists():
            content = attachment_path.read_text(errors="replace")
            source = TextSourceProvider(content, f"attachment:{att_id}")
            try:
                db.ingest(source)
            except Exception as e:
                print(f"Error ingesting {att_id}: {e}")

    db.save(str(db_path))

    history.append(
        {
            "role": "user",
            "message": message,
            "attachments": attachments,
        }
    )

    with open(history_path, "w") as f:
        json.dump(history, f)

    from orchestration import create_agent, chat as do_chat

    try:
        agent = create_agent(db)
    except Exception as e:
        agent = None

    def generate():
        if agent and message:
            try:
                history_texts = []
                for msg in history[:-1]:
                    if msg.get("role") == "user":
                        history_texts.append(msg.get("message", ""))

                response = do_chat(
                    agent,
                    message,
                    history=[{"role": "user", "content": m} for m in history_texts]
                    if history_texts
                    else None,
                )

                for i, char in enumerate(response):
                    yield json.dumps(
                        {
                            "type": "token",
                            "content": char,
                            "token_id": i,
                        }
                    )

                history.append(
                    {
                        "role": "assistant",
                        "message": response,
                        "attachments": [],
                    }
                )

                yield json.dumps({"type": "done"})
            except Exception as e:
                yield json.dumps({"type": "error", "content": str(e)})
        else:
            yield json.dumps({"type": "token", "content": ""})
            yield json.dumps({"type": "done"})

    with open(history_path, "w") as f:
        json.dump(history, f)

    return Response(
        stream_with_context(generate()),
        mimetype="application/json-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
