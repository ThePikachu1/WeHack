import os
import sys
import time

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

import requests
from graph import GraphDatabase
from sources import WebSourceProvider
from orchestration import create_agent, chat
from langchain_core.messages import HumanMessage


WIKIPEDIA_URLS = [
    "https://en.wikipedia.org/wiki/CBRE_Group",
    "https://en.wikipedia.org/wiki/Cushman_%26_Wakefield",
    "https://en.wikipedia.org/wiki/Colliers_International",
    "https://en.wikipedia.org/wiki/JLL_(company)",
]


def fetch_wikipedia_content(url: str) -> str:
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    }
    response = requests.get(url, headers=headers)
    response.raise_for_status()
    return response.text


def load_knowledge_graph(db: GraphDatabase) -> None:
    """Load Wikipedia sources into the knowledge graph."""
    import cProfile
    import pstats
    import io

    print("Loading knowledge graph from Wikipedia sources...")

    profiler = cProfile.Profile()
    profiler.enable()

    for url in WIKIPEDIA_URLS:
        name = url.split("/")[-1].replace("_", " ").replace("-", " ")
        print(f"  Loading {name}...")

        try:
            provider = WebSourceProvider(url)
            db.ingest(provider)
            print(
                f"    Done: {len(db.get_all_nodes())} nodes, {len(db.get_all_edges())} edges"
            )
        except Exception as e:
            print(f"    Error loading {name}: {e}")

    profiler.disable()

    print(
        f"\nKnowledge graph loaded: {len(db.get_all_nodes())} nodes, {len(db.get_all_edges())} edges"
    )

    # Print profiling results
    print("\n" + "=" * 60)
    print("PROFILING RESULTS (top 30 by cumulative time)")
    print("=" * 60)
    s = io.StringIO()
    ps = pstats.Stats(profiler, stream=s).sort_stats("cumulative")
    ps.print_stats(30)
    print(s.getvalue())


def run_repl(agent, db):
    """Run an interactive REPL for chatting with the agent."""
    print("\n" + "=" * 60)
    print("Knowledge Graph Chat REPL")
    print("=" * 60)
    print("Commands:")
    print("  :quit or :exit - Exit the chat")
    print("  :clear - Clear conversation history")
    print("  :sources - List available sources")
    print("  :help - Show this help message")
    print("=" * 60)
    print()

    history = []

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting...")
            break

        if not user_input:
            continue

        if user_input.lower() in [":quit", ":exit", ":q"]:
            print("Goodbye!")
            break

        if user_input.lower() == ":clear":
            history = []
            print("Conversation history cleared.")
            continue

        if user_input.lower() == ":sources":
            sources = list(db.sources.keys()) if hasattr(db, "sources") else []
            if sources:
                print("Available sources:")
                for s in sources:
                    print(f"  - {s}")
            else:
                print("No sources loaded.")
            continue

        if user_input.lower() == ":help":
            print("\nCommands:")
            print("  :quit or :exit - Exit the chat")
            print("  :clear - Clear conversation history")
            print("  :sources - List available sources")
            print("  :help - Show this help message")
            continue

        print("\nAssistant: ", end="", flush=True)
        try:
            response = chat(agent, user_input, history)
            print(response)

            history.append({"role": "user", "content": user_input})
            history.append({"role": "assistant", "content": response})
        except Exception as e:
            print(f"Error: {e}")

        print()


def main():
    import time

    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print("Error: GEMINI_API_KEY not found in environment.")
        print("Please set it in .env file.")
        sys.exit(1)

    print("Initializing knowledge graph...")
    db = GraphDatabase()

    load_knowledge_graph(db)

    print("\nInitializing LangGraph agent...")
    agent_start = time.time()
    try:
        agent = create_agent(db, model_name="gemini-pro-latest")
    except Exception as e:
        print(f"Error initializing agent: {e}")
        print("Falling back to gemini-1.5-pro...")
        try:
            agent = create_agent(db, model_name="gemini-1.5-pro")
        except Exception as e2:
            print(f"Fatal error: {e2}")
            sys.exit(1)

    # Warmup the model to get actual load time
    print("  Warming up model...")
    warmup_start = time.time()
    try:
        result = agent.invoke({"messages": [HumanMessage(content="hi")]})
        warmup_elapsed = time.time() - warmup_start
        agent_elapsed = agent_start + warmup_elapsed
        print(
            f"Agent ready! (init: {time.time() - agent_start:.1f}s, warmup: {warmup_elapsed:.1f}s)"
        )
    except Exception as e:
        print(f"Agent initialized (warmup failed: {e})")

    run_repl(agent, db)


if __name__ == "__main__":
    main()
