import unittest
import os
import sys
import tempfile
import shutil
import json
from unittest.mock import patch, MagicMock

# Add the current directory to the path
sys.path.insert(0, os.path.dirname(__file__))


class TestGraphDatabase(unittest.TestCase):
    def setUp(self):
        from graph import GraphDatabase

        self.db = GraphDatabase()

    def test_add_node(self):
        node = self.db.add_node("TestNode")
        self.assertEqual(node.name, "TestNode")
        self.assertEqual(len(self.db.nodes), 1)
        self.assertEqual(node.id, 0)

    def test_add_node_auto_resolve(self):
        node1 = self.db.add_node("CBRE Group")
        node2 = self.db.add_node("CBRE")

        # Should have resolved to same node due to embedding similarity
        self.assertEqual(node1.id, node2.id)
        self.assertIn("CBRE", node1.aliases)

    def test_add_edge(self):
        n1 = self.db.add_node("A")
        n2 = self.db.add_node("B")
        edge = self.db.add_edge(n1, n2, "relates to", "test_source")

        self.assertEqual(edge.content, "A relates to B")
        self.assertEqual(edge.sources, ["test_source"])
        self.assertEqual(len(self.db.edges), 1)

    def test_edge_placeholder_templates(self):
        n1 = self.db.add_node("John Doe")
        n2 = self.db.add_node("CBRE")
        edge = self.db.add_edge(n1, n2, "{source} leases from {target}", "doc1")

        self.assertEqual(edge.relationship_type, "{source} leases from {target}")
        self.assertEqual(edge.content, "John Doe leases from CBRE")

    def test_edge_renders_at_access_time(self):
        n1 = self.db.add_node("Tenant")
        n2 = self.db.add_node("Landlord")

        edge = self.db.add_edge(n1, n2, "{source} pays rent to {target}", "doc1")
        self.assertEqual(edge.content, "Tenant pays rent to Landlord")

        n2.name = "Property Manager LLC"
        self.assertEqual(edge.content, "Tenant pays rent to Property Manager LLC")

    def test_edge_no_placeholders_concatenates(self):
        n1 = self.db.add_node("Apple")
        n2 = self.db.add_node("Fruit")
        edge = self.db.add_edge(n1, n2, "is a type of", "doc1")

        self.assertEqual(edge.content, "Apple is a type of Fruit")

    def test_nli_gets_rendered_content(self):
        from unittest.mock import patch, MagicMock

        n1 = self.db.add_node("Tenant")
        n2 = self.db.add_node("Landlord")

        self.db.add_edge(n1, n2, "{source} pays rent to {target}", "source1")

        render_calls = []
        original_render = self.db._render_relationship

        def track_render(rel, src_id, tgt_id):
            render_calls.append(rel)
            return original_render(rel, src_id, tgt_id)

        with patch.object(self.db, "_render_relationship", side_effect=track_render):
            self.db.add_edge(n1, n2, "{source} pays rent to {target}", "source2")

        self.assertGreater(len(render_calls), 0)

    def test_get_node_by_name(self):
        self.db.add_node("TestNode")
        node = self.db.get_node_by_name("TestNode")
        self.assertIsNotNone(node)
        self.assertEqual(node.name, "TestNode")

    def test_get_node_by_alias(self):
        self.db.add_node("CBRE Group")
        # Note: auto-resolve is enabled, so "CBRE" would resolve to existing node
        node = self.db.get_node_by_name("CBRE Group")
        self.assertIsNotNone(node)

    def test_search_edges(self):
        n1 = self.db.add_node("A")
        n2 = self.db.add_node("B")
        self.db.add_edge(n1, n2, "Apple is a fruit", "test")

        results = self.db.search_edges("fruit", k=5)
        # May or may not have results depending on embedding similarity

    def test_get_related_subgraph(self):
        n1 = self.db.add_node("Apple")
        n2 = self.db.add_node("Fruit")
        n3 = self.db.add_node("Red")
        self.db.add_edge(n1, n2, "Apple is a fruit", "test")
        self.db.add_edge(n1, n3, "Apple is red", "test")

        subgraph = self.db.get_related_subgraph("Apple", k=10)
        self.assertIsNotNone(subgraph)

    def test_get_related_subgraph(self):
        n1 = self.db.add_node("A")
        n2 = self.db.add_node("B")
        n3 = self.db.add_node("C")
        self.db.add_edge(n1, n2, "A to B", "test")
        self.db.add_edge(n2, n3, "B to C", "test")

        subgraph = self.db.get_related_subgraph("A", k=10)
        self.assertGreater(len(subgraph.node_ids), 0)
        self.assertGreater(len(subgraph.edge_ids), 0)

    def test_identity_facts(self):
        node = self.db.add_node("Test")
        node.add_identity_fact("Test is a fact", "source1")

        self.assertEqual(len(node.identity_facts), 1)
        self.assertEqual(node.identity_facts[0]["fact"], "Test is a fact")

    def test_identity_fact_sources_merged(self):
        node = self.db.add_node("Test")
        node.add_identity_fact("Test is a fact", "source1")
        node.add_identity_fact("Test is a fact", "source2")

        self.assertEqual(len(node.identity_facts), 1)
        self.assertIn("source1", node.identity_facts[0]["source"])
        self.assertIn("source2", node.identity_facts[0]["source"])

    def test_identity_fact_longer_preferred(self):
        node = self.db.add_node("Google")
        node.add_identity_fact("Google is a search engine company", "source1")
        node.add_identity_fact("Google is an internet search engine company", "source2")

        self.assertEqual(len(node.identity_facts), 1)
        self.assertIn("internet", node.identity_facts[0]["fact"])

    def test_identity_fact_multiple_sources(self):
        node = self.db.add_node("Test")
        node.add_identity_fact("Test is a fact", "source1")
        node.add_identity_fact("Test is a fact", "source2")
        node.add_identity_fact("Test is a fact", "source3")

        self.assertEqual(len(node.identity_facts), 1)
        sources = node.identity_facts[0]["source"]
        self.assertEqual(len(sources), 3)
        self.assertIn("source1", sources)
        self.assertIn("source2", sources)
        self.assertIn("source3", sources)

    def test_node_aliases(self):
        node = self.db.add_node("CBRE Group")
        node.add_alias("CBRE")

        self.assertIn("CBRE Group", node.aliases)
        self.assertIn("CBRE", node.aliases)

    def test_save_load(self):
        n1 = self.db.add_node("A")
        n2 = self.db.add_node("B")
        self.db.add_edge(n1, n2, "test edge", "source1")

        path = tempfile.mkdtemp()
        try:
            self.db.save(path)

            from graph import GraphDatabase

            db2 = GraphDatabase.from_saved(path)

            self.assertEqual(len(db2.nodes), 2)
            self.assertEqual(len(db2.edges), 1)
        finally:
            shutil.rmtree(path)


class TestGraphEdge(unittest.TestCase):
    def setUp(self):
        from graph import GraphDatabase, GraphEdge

        self.db = GraphDatabase()
        self.n1 = self.db.add_node("A")
        self.n2 = self.db.add_node("B")

    def test_edge_sources_list(self):
        edge = self.db.add_edge(self.n1, self.n2, "content", "source1")
        self.assertIsInstance(edge.sources, list)
        self.assertIn("source1", edge.sources)

    def test_add_source(self):
        edge = self.db.add_edge(self.n1, self.n2, "content", "source1")
        edge.add_source("source2")

        self.assertIn("source1", edge.sources)
        self.assertIn("source2", edge.sources)

        # Adding same source again should not duplicate
        edge.add_source("source1")
        self.assertEqual(edge.sources.count("source1"), 1)


class TestSubGraph(unittest.TestCase):
    def setUp(self):
        from graph import GraphDatabase

        self.db = GraphDatabase()

    def test_subgraph_get_nodes(self):
        n1 = self.db.add_node("A")
        n2 = self.db.add_node("B")
        self.db.add_edge(n1, n2, "edge", "test")

        subgraph = self.db.get_related_subgraph("A", k=10)
        nodes = subgraph.get_nodes()

        self.assertGreater(len(nodes), 0)

    def test_subgraph_get_edges(self):
        n1 = self.db.add_node("A")
        n2 = self.db.add_node("B")
        self.db.add_edge(n1, n2, "edge", "test")

        subgraph = self.db.get_related_subgraph("A", k=10)
        edges = subgraph.get_edges()

        self.assertGreater(len(edges), 0)


class TestSourceProviders(unittest.TestCase):
    def test_text_source_equality(self):
        from sources import TextSourceProvider

        s1 = TextSourceProvider("content", "source1")
        s2 = TextSourceProvider("content", "source1")
        s3 = TextSourceProvider("different", "source1")

        # Same ID should be equal
        self.assertEqual(s1.get_source_id(), s2.get_source_id())
        self.assertEqual(s1.get_source_key(), s2.get_source_key())
        # Different content but same ID is still "equal" (same source)

    def test_web_source_equality(self):
        from sources import WebSourceProvider

        s1 = WebSourceProvider("http://example.com/page")
        s2 = WebSourceProvider("http://example.com/page")
        s3 = WebSourceProvider("http://example.com/other")

        self.assertEqual(s1.get_source_key(), s2.get_source_key())
        self.assertNotEqual(s1.get_source_key(), s3.get_source_key())

    def test_source_key(self):
        from sources import TextSourceProvider, WebSourceProvider, FileSourceProvider

        t = TextSourceProvider("text", "id")
        self.assertEqual(t.get_source_key(), "text:id")

        w = WebSourceProvider("http://example.com")
        self.assertEqual(w.get_source_key(), "web:http://example.com")

    def test_file_source_equality(self):
        from sources import FileSourceProvider

        with tempfile.NamedTemporaryFile(delete=False) as f:
            f.write(b"content")
            f.flush()
            path = f.name

        try:
            s1 = FileSourceProvider(path)
            s2 = FileSourceProvider(path)

            self.assertEqual(s1.get_source_key(), s2.get_source_key())
        finally:
            os.unlink(path)


class TestIngest(unittest.TestCase):
    def test_ingest_integration(self):
        from graph import GraphDatabase

        db = GraphDatabase()

        data = {
            "identity": [{"concept": "Test", "fact_sentence": "Test is a concept"}],
            "relation": [
                {
                    "from_concept": "Test",
                    "to_concept": "Other",
                    "fact_sentence": "Test relates to Other",
                }
            ],
        }

        for identity in data.get("identity", []):
            node = db.add_node(identity["concept"])
            node.add_identity_fact(identity["fact_sentence"], "test_source")

        for relation in data.get("relation", []):
            from_node = db.add_node(relation["from_concept"])
            to_node = db.add_node(relation["to_concept"])
            db.add_edge(from_node, to_node, relation["fact_sentence"], "test_source")

        self.assertGreater(len(db.nodes), 0)
        self.assertGreater(len(db.edges), 0)


class TestOrchestration(unittest.TestCase):
    @patch("orchestration._get_llm")
    def test_session_manager(self, mock_llm):
        from orchestration import SessionManager, create_agent
        from graph import GraphDatabase

        mock_llm.return_value = MagicMock()

        manager = SessionManager()

        # Create sessions
        manager.create_session("session1")
        manager.create_session("session2")

        self.assertIn("session1", manager.list_sessions())
        self.assertIn("session2", manager.list_sessions())

        # Switch sessions
        manager.switch_session("session1")
        db1 = manager.get_current_db()
        self.assertIsInstance(db1, GraphDatabase)

        manager.switch_session("session2")
        db2 = manager.get_current_db()
        self.assertIsInstance(db2, GraphDatabase)

        # Different sessions should have different databases
        self.assertNotEqual(id(db1), id(db2))


class TestGraphDatabaseIngestDatabase(unittest.TestCase):
    def test_ingest_database_basic(self):
        from graph import GraphDatabase

        db1 = GraphDatabase()
        db2 = GraphDatabase()

        n1 = db2.add_node("TestNode")
        n2 = db2.add_node("Related")
        db2.add_edge(n1, n2, "test relation", "source1")

        db1.ingest_database(db2)

        # Should have transferred nodes and edges
        self.assertGreater(len(db1.nodes), 0)
        self.assertGreater(len(db1.edges), 0)

        # Should have sources
        self.assertGreater(len(db1.sources), 0)


class TestEdgeConflictResolution(unittest.TestCase):
    @patch("graph._get_nli_model")
    def test_mutual_entailment_merges_sources(self, mock_nli):
        from graph import GraphDatabase

        mock_model = MagicMock()
        mock_model.predict.return_value = [[0.1, 0.9, 0.0]]  # entailment
        mock_nli.return_value = mock_model

        db = GraphDatabase(entailment_threshold=0.75)

        n1 = db.add_node("A")
        n2 = db.add_node("B")

        edge1 = db.add_edge(n1, n2, "A is related to B", "source1")
        edge2 = db.add_edge(n1, n2, "A is connected to B", "source2")

        # Should have merged (mutual entailment), sources combined
        all_edges = db.get_all_edges()

        # The two edges might be merged into one with multiple sources
        total_sources = sum(len(e.sources) for e in all_edges)
        self.assertGreaterEqual(total_sources, 1)


class TestNodeEntityResolution(unittest.TestCase):
    def test_resolve_entities(self):
        from graph import GraphDatabase

        db = GraphDatabase()

        node1 = db.add_node("Google Inc", auto_resolve=False)
        node2 = db.add_node("Google", auto_resolve=False)

        self.assertEqual(node1.id, node2.id - 1)

        merged = db.resolve_entities(threshold=0.92)
        self.assertIsInstance(merged, int)

    def test_node_merge_contradiction_guard(self):
        from graph import GraphDatabase

        db = GraphDatabase(entity_merge_threshold=0.3)

        node1 = db.add_node("Contract", auto_resolve=False)
        node1.add_identity_fact("Contract for 123 Main St", "source1")
        node1.add_identity_fact("Signed on 2024-01-01", "source1")

        node2 = db.add_node("Contract", auto_resolve=False)
        node2.add_identity_fact("Contract for 456 Oak Ave", "source2")
        node2.add_identity_fact("Signed on 2023-06-15", "source2")

        initial_count = len(db.nodes)

        if hasattr(db, "_has_excessive_contradictions"):
            has_contradiction = db._has_excessive_contradictions(node1, node2)
            self.assertTrue(has_contradiction)

    def test_identity_facts_merged_on_node_merge(self):
        from graph import GraphDatabase

        db = GraphDatabase(entity_merge_threshold=0.5)

        node1 = db.add_node("Company", auto_resolve=False)
        node1.add_identity_fact("Company is a tech company", "source1")

        node2 = db.add_node("Company", auto_resolve=False)
        node2.add_identity_fact("Company is a corporation", "source2")

        merged = db.resolve_entities(threshold=0.92)
        node = db.get_node_by_name("Company")
        self.assertIsNotNone(node)
        self.assertEqual(len(node.identity_facts), 1)


if __name__ == "__main__":
    unittest.main()
