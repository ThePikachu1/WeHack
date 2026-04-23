import os
import sys

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

from typing import List, Optional

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, AIMessage
from langgraph.prebuilt import create_react_agent
from langchain_core.tools import tool

from graph import GraphDatabase
from sources import SourceProvider


def _get_llm(
    model_name: str = "gemini-flash-latest", temperature: float = 0
) -> ChatGoogleGenerativeAI:
    """Get the LLM instance."""
    api_key = os.environ.get("GEMINI_API_KEY")
    return ChatGoogleGenerativeAI(
        model=model_name,
        temperature=temperature,
        convert_system_message_to_human=True,
        google_api_key=api_key,
    )


def format_subgraph_for_context(subgraph) -> str:
    """Format a subgraph for use as context in prompts."""
    if not subgraph.node_ids:
        return "No relevant facts found in the knowledge graph."

    lines = []
    lines.append("=== Knowledge Graph Context ===")

    nodes = subgraph.get_nodes()
    edges = subgraph.get_edges()

    for node in nodes:
        if node.identity_facts:
            lines.append(f"\n{node.name}:")
            for fact in node.identity_facts:
                sources_str = f" [sources: {', '.join(fact['source'])}]"
                lines.append(f"  - {fact['fact']}{sources_str}")

    if edges:
        lines.append("\nRelationships:")
        for edge in edges:
            src = subgraph.parent_db.get_node(edge.source_node_id)
            tgt = subgraph.parent_db.get_node(edge.target_node_id)
            src_name = src.name if src else "?"
            tgt_name = tgt.name if tgt else "?"
            sources_str = f" [sources: {', '.join(edge.sources)}]"
            lines.append(f"  - {src_name} -> {tgt_name}: {edge.content}{sources_str}")

    lines.append("\n================================")
    return "\n".join(lines)


@tool
def search_knowledge_graph(query: str, k: int = 10) -> str:
    """Search the knowledge graph for relevant facts about a query.

    Args:
        query: The search query (can be about companies, people, relationships, etc.)
        k: Maximum number of results to return (default: 10)

    Returns:
        Formatted context from the knowledge graph with sources
    """
    results = _graph_db.search_edges(query, k=k)

    if not results:
        return "No relevant facts found in the knowledge graph."

    lines = []
    lines.append(f"=== Found {len(results)} relevant facts ===\n")

    for edge, similarity in results:
        src_node = _graph_db.get_node(edge.source_node_id)
        tgt_node = _graph_db.get_node(edge.target_node_id)
        src_name = src_node.name if src_node else "?"
        tgt_name = tgt_node.name if tgt_node else "?"

        sources_str = f" [Sources: {', '.join(edge.sources)}]"

        lines.append(f"Fact ({similarity:.2f}): {edge.content}")
        lines.append(f"  From: {src_name} -> To: {tgt_name}{sources_str}")
        lines.append("")

    subgraph = _graph_db.get_related_subgraph(query, k=k)
    context_lines = format_subgraph_for_context(subgraph)

    return "\n".join(lines) + "\n\n" + context_lines


@tool
def get_subgraph_around_node(node_name: str, depth: int = 1) -> str:
    """Get all related facts around a specific node in the knowledge graph.

    Args:
        node_name: The name of the node to explore
        depth: How many hops of relationships to follow (default: 1)

    Returns:
        Formatted subgraph with sources
    """
    node = _graph_db.get_node_by_name(node_name)

    if not node:
        return f"Node '{node_name}' not found in the knowledge graph."

    subgraph = _graph_db.get_related_subgraph(node_name, k=50)
    return format_subgraph_for_context(subgraph)


@tool
def get_source_content(source_id: str) -> str:
    """Retrieve the original content from a specific source for more context.

    Args:
        source_id: The ID of the source (e.g., 'CBRE Group', 'JLL', etc.)

    Returns:
        The original content from that source
    """
    source = _graph_db.get_source(source_id)

    if not source:
        available_sources = list(_graph_db.sources.keys())
        return f"Source '{source_id}' not found. Available sources: {available_sources}"

    try:
        content = source.get_content()
        content_preview = content[:2000] + "..." if len(content) > 2000 else content
        return f"=== Source: {source.get_source_id()} ({source.get_source_type()}) ===\n\n{content_preview}"
    except Exception as e:
        return f"Error retrieving source content: {str(e)}"


@tool
def list_available_sources() -> str:
    """List all available sources in the knowledge graph.

    Returns:
        List of source IDs and their types
    """
    if not _graph_db.sources:
        return "No sources available in the knowledge graph."

    lines = ["=== Available Sources ==="]
    for source_id, source in _graph_db.sources.items():
        lines.append(f"- {source_id} (type: {source.get_source_type()})")

    return "\n".join(lines)


def create_agent(graph_db: GraphDatabase, model_name: str = "gemini-pro-latest"):
    """Create a LangGraph agent with tools for interacting with the knowledge graph.

    Args:
        graph_db: The GraphDatabase instance to use
        model_name: The Gemini model to use (default: gemini-pro-latest)

    Returns:
        A compiled LangGraph agent
    """
    global _graph_db
    _graph_db = graph_db

    llm = _get_llm(model_name)

    tools = [
        search_knowledge_graph,
        get_subgraph_around_node,
        get_source_content,
        list_available_sources,
    ]

    agent = create_react_agent(llm, tools)

    return agent


class SessionManager:
    def __init__(self, model_name: str = "gemini-pro-latest"):
        self._sessions = {}
        self._model_name = model_name
        self._current_session_id = None
        self._current_agent = None

    def create_session(self, session_id: str) -> str:
        db = GraphDatabase()
        self._sessions[session_id] = db
        return session_id

    def get_session(self, session_id: str) -> GraphDatabase:
        return self._sessions.get(session_id)

    def switch_session(self, session_id: str) -> None:
        if session_id not in self._sessions:
            self.create_session(session_id)
        self._current_session_id = session_id
        self._current_agent = create_agent(self._sessions[session_id], self._model_name)

    def get_current_agent(self):
        if self._current_agent is None:
            self.switch_session("default")
        return self._current_agent

    def get_current_db(self) -> GraphDatabase:
        if self._current_session_id is None:
            self.switch_session("default")
        return self._sessions[self._current_session_id]

    def list_sessions(self) -> list:
        return list(self._sessions.keys())


_session_manager = None


def get_session_manager() -> SessionManager:
    global _session_manager
    if _session_manager is None:
        _session_manager = SessionManager()
    return _session_manager


def chat(agent, message: str, history: Optional[List] = None) -> str:
    """Chat with the agent.

    Args:
        agent: The LangGraph agent
        message: The user's message
        history: Optional list of previous messages

    Returns:
        The agent's response
    """
    if history is None:
        history = []

    messages = []
    for msg in history:
        if msg.get("role") == "user":
            messages.append(HumanMessage(content=msg["content"]))
        elif msg.get("role") == "assistant":
            messages.append(AIMessage(content=msg["content"]))

    messages.append(HumanMessage(content=message))

    result = agent.invoke({"messages": messages})

    last_message = result["messages"][-1]

    if isinstance(last_message.content, list):
        for item in last_message.content:
            if isinstance(item, dict) and item.get("type") == "text":
                return item.get("text", "")

    return str(last_message.content)


_graph_db = None
