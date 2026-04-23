import hnswlib
import numpy as np
from sentence_transformers import SentenceTransformer, CrossEncoder
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from sources import SourceProvider

_embedding_model = None
_nli_model = None


def _get_embedding_model() -> SentenceTransformer:
    global _embedding_model
    if _embedding_model is None:
        _embedding_model = SentenceTransformer(
            "nomic-ai/nomic-embed-text-v1.5", trust_remote_code=True
        )
    return _embedding_model


def _get_nli_model() -> CrossEncoder:
    global _nli_model
    if _nli_model is None:
        _nli_model = CrossEncoder("cross-encoder/nli-MiniLM2-L6-H768")
    return _nli_model


EMBEDDING_DIM = 768
ENTITY_SIMILARITY_THRESHOLD = 0.85
ENTITY_SIMILARITY_NLI_THRESHOLD = 0.75
ENTITY_CONTRADICTION_MERGE_THRESHOLD = 0.3


class GraphEdge:
    def __init__(
        self,
        id: int,
        relationship_type: str,
        sources: list[str],
        database: "GraphDatabase",
        source_node_id: int,
        target_node_id: int,
    ):
        assert sources, "Edge must have at least one source"
        assert relationship_type, "Edge must have a relationship type"
        self.id = id
        self.relationship_type = relationship_type
        self.sources = sources
        self.database = database
        self.source_node_id = source_node_id
        self.target_node_id = target_node_id

    @property
    def content(self) -> str:
        source_node = self.database.nodes.get(self.source_node_id)
        target_node = self.database.nodes.get(self.target_node_id)
        if not source_node or not target_node:
            return self.relationship_type
        return self._format_relationship(source_node.name, target_node.name)

    @property
    def display_content(self) -> str:
        return self._format_relationship("{source}", "{target}")

    def check_name_collision(self) -> list[str]:
        warnings = []
        source_node = self.database.nodes.get(self.source_node_id)
        target_node = self.database.nodes.get(self.target_node_id)
        if not source_node or not target_node:
            return warnings

        source_name = source_node.name
        target_name = target_node.name
        rel = self.relationship_type

        if source_name in rel and "{source}" not in rel:
            warnings.append(
                f"{source_name} appears in relationship but not as placeholder"
            )
        if target_name in rel and "{target}" not in rel:
            warnings.append(
                f"{target_name} appears in relationship but not as placeholder"
            )

        if len(source_name) > len(target_name):
            if source_name != target_name:
                longer, shorter = source_name, target_name
                if longer in rel:
                    warnings.append(
                        f"{longer} (longer name) may collide with {shorter}"
                    )

        return warnings

    def _format_relationship(self, source_name: str, target_name: str) -> str:
        rel = self.relationship_type
        has_source_placeholder = "{source}" in rel
        has_target_placeholder = "{target}" in rel

        if has_source_placeholder:
            rel = rel.replace("{source}", source_name)
        if has_target_placeholder:
            rel = rel.replace("{target}", target_name)

        if not has_source_placeholder and not has_target_placeholder:
            return f"{source_name} {rel} {target_name}"

        return rel

    def get_content(self, resolve_names: bool = True) -> str:
        if not resolve_names:
            return self.relationship_type
        return self.content

    def add_source(self, source: str) -> None:
        if source not in self.sources:
            self.sources.append(source)


class GraphNode:
    def __init__(
        self, id: int, name: str, database: "GraphDatabase", aliases: list[str] = None
    ):
        self.id = id
        self.name = name
        self.database = database
        self.edge_ids = []
        self.identity_facts = []
        self.aliases = aliases or [name]

    def add_identity_fact(self, fact: str, source: str) -> None:
        resolved = self.database._check_and_resolve_identity_conflicts(
            fact, source, self.identity_facts
        )
        if resolved is None:
            self.identity_facts.append({"fact": fact, "source": [source]})

    def add_alias(self, alias: str) -> None:
        if alias not in self.aliases:
            self.aliases.append(alias)

    def add_edge(
        self, content: str, target_node: "GraphNode", source: str
    ) -> GraphEdge:
        edge = self.database.add_edge(self, target_node, content, source)
        return edge

    def remove_edge(self, edge_id: int) -> None:
        if edge_id in self.edge_ids:
            self.edge_ids.remove(edge_id)
            edge = self.database.edges.get(edge_id)
            if edge:
                target_node = self.database.nodes.get(edge.target_node_id)
                if target_node and edge_id in target_node.edge_ids:
                    target_node.edge_ids.remove(edge_id)
                del self.database.edges[edge_id]
                self.database._remove_edge_from_index(edge_id)

    def get_edges(self) -> list[GraphEdge]:
        return [
            self.database.edges[eid]
            for eid in self.edge_ids
            if eid in self.database.edges
        ]


class SubGraph:
    def __init__(
        self, node_ids: list[int], edge_ids: list[int], parent_db: "GraphDatabase"
    ):
        self.node_ids = node_ids
        self.edge_ids = edge_ids
        self.parent_db = parent_db

    def get_nodes(self) -> list[GraphNode]:
        return [
            self.parent_db.nodes[nid]
            for nid in self.node_ids
            if nid in self.parent_db.nodes
        ]

    def get_edges(self) -> list[GraphEdge]:
        return [
            self.parent_db.edges[eid]
            for eid in self.edge_ids
            if eid in self.parent_db.edges
        ]

    def get_node(self, node_id: int) -> GraphNode | None:
        if node_id in self.node_ids:
            return self.parent_db.nodes.get(node_id)
        return None

    def get_edge(self, edge_id: int) -> GraphEdge | None:
        if edge_id in self.edge_ids:
            return self.parent_db.edges.get(edge_id)
        return None


class GraphDatabase:
    def __init__(
        self,
        ef_construction: int = 200,
        M: int = 15,
        entailment_threshold: float = 1.0,
        contradiction_threshold: float = 2.0,
        entity_merge_threshold: float = ENTITY_CONTRADICTION_MERGE_THRESHOLD,
    ):
        self.nodes = {}
        self.edges = {}
        self.sources = {}
        self._next_node_id = 0
        self._next_edge_id = 0
        self._hnsw = hnswlib.Index(space="cosine", dim=EMBEDDING_DIM)
        self._hnsw.init_index(max_elements=1000, ef_construction=ef_construction, M=M)
        self._edge_id_to_idx = {}
        self._idx_to_edge_id = {}
        self._nli_model = _get_nli_model()
        self._entailment_threshold = entailment_threshold
        self._contradiction_threshold = contradiction_threshold
        self._entity_merge_threshold = entity_merge_threshold
        self._node_hnsw = hnswlib.Index(space="cosine", dim=EMBEDDING_DIM)
        self._node_hnsw.init_index(
            max_elements=1000, ef_construction=ef_construction, M=M
        )
        self._node_name_to_id = {}

    def _get_next_node_id(self):
        id = self._next_node_id
        self._next_node_id += 1
        return id

    def _get_next_edge_id(self):
        id = self._next_edge_id
        self._next_edge_id += 1
        return id

    def _add_edge_to_index(self, edge: GraphEdge, idx: int = None) -> None:
        model = _get_embedding_model()
        embedding = model.encode(edge.relationship_type, convert_to_numpy=True)
        embedding = embedding / np.linalg.norm(embedding)
        idx = edge.id
        self._hnsw.add_items(embedding.reshape(1, -1), np.array([idx]))
        self._edge_id_to_idx[edge.id] = idx
        self._idx_to_edge_id[idx] = edge.id

    def _update_edge_in_index(self, edge: GraphEdge) -> None:
        if edge.id in self._edge_id_to_idx:
            idx = self._edge_id_to_idx[edge.id]
            self._hnsw.set_ef(max(10, self._hnsw.ef))
            model = _get_embedding_model()
            embedding = model.encode(edge.relationship_type, convert_to_numpy=True)
            embedding = embedding / np.linalg.norm(embedding)
            self._hnsw.knn_query(embedding.reshape(1, -1), k=1)

    def _get_entity_embedding(self, name: str) -> np.ndarray:
        model = _get_embedding_model()
        embedding = model.encode(f"clustering: {name}", convert_to_numpy=True)
        return embedding

    def _find_similar_nodes(
        self, name: str, threshold: float = ENTITY_SIMILARITY_THRESHOLD
    ) -> list[tuple["GraphNode", float]]:
        if not self.nodes:
            return []

        query_embedding = self._get_entity_embedding(name)
        query_embedding = query_embedding / np.linalg.norm(query_embedding)

        current_count = self._node_hnsw.get_current_count()
        if current_count == 0:
            return []

        self._node_hnsw.set_ef(max(10, self._node_hnsw.ef))
        k = min(5, current_count)
        labels, distances = self._node_hnsw.knn_query(query_embedding, k=k)

        if len(distances[0]) == 0:
            return []

        candidates = []
        for idx, dist in zip(labels[0], distances[0]):
            node_id = self._node_name_to_id.get(idx)
            if node_id is not None and node_id in self.nodes:
                similarity = 1.0 - dist
                candidates.append((self.nodes[node_id], similarity))

        for existing_node, similarity in candidates:
            if similarity >= threshold:
                if self._check_mutual_entailment(name, existing_node.name):
                    return [(existing_node, similarity)]

        return []

    def _check_mutual_entailment(self, name_a: str, name_b: str) -> bool:
        model = _get_nli_model()
        pairs_a_to_b = [(name_a, name_b)]
        pairs_b_to_a = [(name_b, name_a)]

        scores_a_to_b = model.predict(pairs_a_to_b)
        scores_b_to_a = model.predict(pairs_b_to_a)

        a_to_b_entailment = scores_a_to_b[0][1]
        b_to_a_entailment = scores_b_to_a[0][1]

        return (
            a_to_b_entailment >= ENTITY_SIMILARITY_NLI_THRESHOLD
            and b_to_a_entailment >= ENTITY_SIMILARITY_NLI_THRESHOLD
        )

    def _add_node_to_index(self, node: GraphNode) -> None:
        embedding = self._get_entity_embedding(node.name)
        embedding = embedding / np.linalg.norm(embedding)
        idx = node.id
        self._node_hnsw.add_items(embedding.reshape(1, -1), np.array([idx]))
        self._node_name_to_id[idx] = node.id

    def _merge_nodes(self, target_node: "GraphNode", source_node: "GraphNode") -> None:
        if self._has_excessive_contradictions(target_node, source_node):
            self._disambiguate_conflicting_nodes(target_node, source_node)
            return

        source_node.add_alias(target_node.name)

        for edge_id in list(source_node.edge_ids):
            edge = self.edges.get(edge_id)
            if edge:
                if edge.source_node_id == source_node.id:
                    edge.source_node_id = target_node.id
                    if edge_id not in target_node.edge_ids:
                        target_node.edge_ids.append(edge_id)
                if edge.target_node_id == source_node.id:
                    edge.target_node_id = target_node.id
                    if edge_id not in target_node.edge_ids:
                        target_node.edge_ids.append(edge_id)

        for edge in self.edges.values():
            if edge.source_node_id == source_node.id:
                edge.source_node_id = target_node.id
                if edge.id not in target_node.edge_ids:
                    target_node.edge_ids.append(edge.id)
            if edge.target_node_id == source_node.id:
                edge.target_node_id = target_node.id
                if edge.id not in target_node.edge_ids:
                    target_node.edge_ids.append(edge.id)

        for fact in source_node.identity_facts:
            target_node.add_identity_fact(fact["fact"], fact["source"])

        source_node_id = source_node.id
        if source_node_id in self.nodes:
            del self.nodes[source_node_id]

        self._check_merged_node_conflicts(target_node)

    def _has_excessive_contradictions(
        self, node_a: GraphNode, node_b: GraphNode
    ) -> bool:
        facts_a = node_a.identity_facts
        facts_b = node_b.identity_facts

        if not facts_a or not facts_b:
            return False

        model = _get_nli_model()
        contradictions = 0
        total_comparisons = 0

        for fact_a in facts_a:
            for fact_b in facts_b:
                pairs_a_to_b = [(fact_a["fact"], fact_b["fact"])]
                pairs_b_to_a = [(fact_b["fact"], fact_a["fact"])]

                try:
                    scores_a_to_b = model.predict(pairs_a_to_b)
                    scores_b_to_a = model.predict(pairs_b_to_a)

                    a_to_b = scores_a_to_b[0][1]
                    b_to_a = scores_b_to_a[0][1]

                    if (
                        a_to_b < -self._entailment_threshold
                        and b_to_a < -self._entailment_threshold
                    ):
                        contradictions += 1
                    total_comparisons += 1
                except Exception:
                    pass

        if total_comparisons == 0:
            return False

        contradiction_ratio = contradictions / total_comparisons
        return contradiction_ratio >= self._entity_merge_threshold

    def _disambiguate_conflicting_nodes(
        self, node_a: GraphNode, node_b: GraphNode
    ) -> None:
        try:
            from google import genai
            from google.genai import types
            import os
        except ImportError:
            return

        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            return

        client = genai.Client(api_key=api_key)

        facts_a_str = "\n".join(
            f"- {f['fact']} (source: {f['source']})" for f in node_a.identity_facts
        )
        facts_b_str = "\n".join(
            f"- {f['fact']} (source: {f['source']})" for f in node_b.identity_facts
        )

        prompt = f"""These two nodes have similar names but their identity facts contradict each other. 
Please suggest more specific names to distinguish them.

Node A ({node_a.name}):
{facts_a_str}

Node B ({node_b.name}):
{facts_b_str}

Respond with JSON in this format:
{{"node_a_name": "more specific name for node A", "node_b_name": "more specific name for node B", "reason": "brief explanation"}}
"""

        try:
            response = client.models.generate_content(
                model="gemini-2.0-flash",
                contents=prompt,
            )
            import json

            result = json.loads(response.text)
            if result.get("node_a_name"):
                node_a.add_alias(node_a.name)
                node_a.name = result["node_a_name"]
            if result.get("node_b_name"):
                node_b.add_alias(node_b.name)
                node_b.name = result["node_b_name"]
        except Exception:
            pass

    def _check_merged_node_conflicts(self, node: GraphNode) -> None:
        edges_by_target = {}
        for edge in node.get_edges():
            if edge.target_node_id not in edges_by_target:
                edges_by_target[edge.target_node_id] = []
            edges_by_target[edge.target_node_id].append(edge)

        for target_id, edges in edges_by_target.items():
            if len(edges) > 1:
                for i in range(1, len(edges)):
                    self._check_and_resolve_conflicts(
                        edges[i].relationship_type,
                        edges[i].sources[0] if edges[i].sources else "",
                        [edges[0]],
                        edges[i].source_node_id,
                        edges[i].target_node_id,
                    )

    def resolve_entities(self, threshold: float = ENTITY_SIMILARITY_THRESHOLD) -> int:
        merged_count = 0
        nodes_to_resolve = list(self.nodes.values())

        for node in nodes_to_resolve:
            if node.id not in self.nodes:
                continue

            similar = self._find_similar_nodes(node.name, threshold)
            for similar_node, score in similar:
                if similar_node.id != node.id and similar_node.id in self.nodes:
                    self._merge_nodes(node, similar_node)
                    merged_count += 1

        return merged_count

    def add_node(self, name: str, auto_resolve: bool = True) -> GraphNode:
        if auto_resolve:
            for existing in self.nodes.values():
                if name in existing.aliases:
                    return existing
            similar = self._find_similar_nodes(name, ENTITY_SIMILARITY_THRESHOLD)
            if similar:
                existing = similar[0][0]
                self._resolve_node_names(existing, name)
                return existing

        node_id = self._get_next_node_id()
        node = GraphNode(node_id, name, self)
        self.nodes[node_id] = node
        self._add_node_to_index(node)
        return node

    def _resolve_node_names(self, existing_node: "GraphNode", new_name: str) -> None:
        if len(new_name) > len(existing_node.name):
            existing_node.add_alias(existing_node.name)
            existing_node.name = new_name
        else:
            existing_node.add_alias(new_name)

    def get_node_by_name(self, name: str) -> GraphNode | None:
        for node in self.nodes.values():
            if node.name == name:
                return node
            if name in node.aliases:
                return node
        return None

    def add_edge(
        self,
        source_node: GraphNode,
        target_node: GraphNode,
        relationship: str,
        source: str,
    ) -> GraphEdge:
        relationship_type = self._to_relationship_type(
            relationship, source_node.name, target_node.name
        )
        existing_edges = [
            e for e in source_node.get_edges() if e.target_node_id == target_node.id
        ]

        if existing_edges:
            merged_edge = self._check_and_resolve_conflicts(
                relationship_type,
                source,
                existing_edges,
                source_node.id,
                target_node.id,
            )
            if merged_edge is not None:
                return merged_edge

        edge_id = self._get_next_edge_id()
        edge = GraphEdge(
            edge_id, relationship_type, [source], self, source_node.id, target_node.id
        )
        self.edges[edge_id] = edge
        source_node.edge_ids.append(edge_id)
        target_node.edge_ids.append(edge_id)
        self._add_edge_to_index(edge)
        return edge

    def _to_relationship_type(
        self, relationship: str, source_name: str, target_name: str
    ) -> str:
        return relationship

    def _parse_relationship(self, rel: str, source_name: str, target_name: str) -> str:
        return rel.replace("{source}", source_name).replace("{target}", target_name)

    def _render_relationship(
        self, rel: str, source_node_id: int, target_node_id: int
    ) -> str:
        source_node = self.nodes.get(source_node_id)
        target_node = self.nodes.get(target_node_id)
        if not source_node or not target_node:
            return rel
        return self._parse_relationship(rel, source_node.name, target_node.name)

    def _check_and_resolve_conflicts(
        self,
        new_relationship_type: str,
        new_source: str,
        existing_edges: list[GraphEdge],
        new_source_node_id: int = None,
        new_target_node_id: int = None,
    ) -> GraphEdge | None:
        if not existing_edges:
            return None

        similar_edges = self._find_similar_edges_for_nli(
            new_relationship_type, existing_edges
        )

        if not similar_edges:
            return None

        model = _get_nli_model()

        new_entailment_scores = {}
        existing_entailment_scores = {}

        if new_source_node_id is not None and new_target_node_id is not None:
            new_content = self._render_relationship(
                new_relationship_type,
                new_source_node_id,
                new_target_node_id,
            )
        else:
            new_content = new_relationship_type

        for existing in similar_edges:
            existing_content = self._render_relationship(
                existing.relationship_type,
                existing.source_node_id,
                existing.target_node_id,
            )
            pairs_new_to_existing = [(new_content, existing_content)]
            pairs_existing_to_new = [(existing_content, new_content)]

            scores_new = model.predict(pairs_new_to_existing)
            scores_existing = model.predict(pairs_existing_to_new)

            new_entailment_scores[existing.id] = scores_new[0][1]
            existing_entailment_scores[existing.id] = scores_existing[0][1]

        for existing in similar_edges:
            a_to_b = new_entailment_scores[existing.id]
            b_to_a = existing_entailment_scores[existing.id]

            if (
                a_to_b > self._entailment_threshold
                and b_to_a > self._entailment_threshold
            ):
                if a_to_b >= b_to_a:
                    existing.add_source(new_source)
                else:
                    existing.relationship_type = new_relationship_type
                    existing.add_source(new_source)
                    self._update_edge_in_index(existing)
                return existing

            elif a_to_b > self._entailment_threshold:
                existing.add_source(new_source)
                return existing
            elif b_to_a > self._entailment_threshold:
                pass

            elif (
                a_to_b < -self._entailment_threshold
                and b_to_a < -self._entailment_threshold
            ):
                self._resolve_contradiction(new_content, new_source, existing)

        return None

    def _check_and_resolve_identity_conflicts(
        self, new_fact: str, new_source: str, existing_facts: list[dict]
    ) -> dict | None:
        if not existing_facts:
            return None

        model = _get_nli_model()

        for existing in existing_facts:
            existing_content = existing["fact"]

            pairs_new_to_existing = [(new_fact, existing_content)]
            pairs_existing_to_new = [(existing_content, new_fact)]

            scores_new = model.predict(pairs_new_to_existing)
            scores_existing = model.predict(pairs_existing_to_new)

            new_to_existing = scores_new[0][1]
            existing_to_new = scores_existing[0][1]

            if (
                new_to_existing > self._entailment_threshold
                and existing_to_new > self._entailment_threshold
            ):
                if len(new_fact) > len(existing_content):
                    existing["fact"] = new_fact
                existing["source"] = self._merge_sources(
                    existing.get("source", []), new_source
                )
                return existing

            elif new_to_existing > self._entailment_threshold:
                existing["source"] = self._merge_sources(
                    existing.get("source", []), new_source
                )
                return existing

            elif (
                new_to_existing < -self._entailment_threshold
                and existing_to_new < -self._entailment_threshold
            ):
                self._resolve_contradiction(new_fact, new_source, existing)

        return None

    def _merge_sources(self, existing_sources: list, new_source: str) -> list:
        if isinstance(existing_sources, str):
            existing_sources = [existing_sources]
        if new_source not in existing_sources:
            return existing_sources + [new_source]
        return existing_sources

    def _find_similar_edges_for_nli(
        self, content: str, existing_edges: list[GraphEdge], threshold: float = 0.7
    ) -> list[GraphEdge]:
        if not existing_edges or not self._hnsw.get_current_count():
            return existing_edges

        model = _get_embedding_model()
        query_embedding = model.encode(content, convert_to_numpy=True)
        query_embedding = query_embedding / np.linalg.norm(query_embedding)

        results = []
        for edge in existing_edges:
            edge_idx = self._edge_id_to_idx.get(edge.id)
            if edge_idx is not None:
                try:
                    labels, distances = self._hnsw.knn_query(query_embedding, k=1)
                    if len(labels[0]) > 0 and labels[0][0] == edge_idx:
                        similarity = 1.0 - distances[0][0]
                        if similarity >= threshold:
                            results.append(edge)
                except Exception:
                    pass

        return results if results else existing_edges[:3]

    def remove_node(self, node_id: int) -> None:
        node = self.nodes.get(node_id)
        if node:
            for edge_id in list(node.edge_ids):
                edge = self.edges.get(edge_id)
                if edge:
                    other_node_id = (
                        edge.target_node_id
                        if edge.source_node_id == node_id
                        else edge.source_node_id
                    )
                    other_node = self.nodes.get(other_node_id)
                    if other_node and edge_id in other_node.edge_ids:
                        other_node.edge_ids.remove(edge_id)
                    del self.edges[edge_id]
                    self._remove_edge_from_index(edge_id)
            del self.nodes[node_id]

    def get_node(self, node_id: int) -> GraphNode | None:
        return self.nodes.get(node_id)

    def get_all_nodes(self) -> list[GraphNode]:
        return list(self.nodes.values())

    def get_all_edges(self) -> list[GraphEdge]:
        return list(self.edges.values())

    def ingest(self, source: "SourceProvider") -> None:
        temp_db = GraphDatabase(
            entailment_threshold=self._entailment_threshold,
            contradiction_threshold=self._contradiction_threshold,
            entity_merge_threshold=self._entity_merge_threshold,
        )
        temp_db.sources[source.get_source_id()] = source

        content = source.get_content()
        source_id = source.get_source_id()

        data = extract_facts(content)

        concept_to_node = {}

        for identity in data.get("identity", []):
            concept = identity["concept"]
            fact_sentence = identity["fact_sentence"]
            if concept not in concept_to_node:
                node = temp_db.add_node(concept, auto_resolve=False)
                concept_to_node[concept] = node
            else:
                node = concept_to_node[concept]
            node.add_identity_fact(fact_sentence, source_id)

        for relation in data.get("relation", []):
            from_concept = relation["from_concept"]
            to_concept = relation["to_concept"]
            fact_sentence = relation["fact_sentence"]

            if from_concept not in concept_to_node:
                node = temp_db.add_node(from_concept, auto_resolve=False)
                concept_to_node[from_concept] = node
            if to_concept not in concept_to_node:
                node = temp_db.add_node(to_concept, auto_resolve=False)
                concept_to_node[to_concept] = node

            temp_db.add_edge(
                concept_to_node[from_concept],
                concept_to_node[to_concept],
                fact_sentence,
                source_id,
            )

        self.ingest_database(temp_db)

    def get_source(self, source_id: str) -> "SourceProvider":
        return self.sources.get(source_id)

    def ingest_database(self, other_db: "GraphDatabase") -> None:
        source_key_to_name = {}

        for src_key, src in other_db.sources.items():
            src_key_self = src.get_source_key()
            if src_key_self in [s.get_source_key() for s in self.sources.values()]:
                continue

            canonical_name = src.get_source_id()
            counter = 1
            while canonical_name in self.sources:
                counter += 1
                canonical_name = f"{src.get_source_id()} ({counter})"

            source_key_to_name[src_key] = canonical_name
            self.sources[canonical_name] = src

        if not source_key_to_name:
            for edge in other_db.edges.values():
                for src in edge.sources:
                    if src not in source_key_to_name:
                        source_key_to_name[src] = src
                        self.sources[src] = src

        nodes_to_merge = []
        node_id_map = {}

        for other_node in other_db.nodes.values():
            similar = self._find_similar_nodes(
                other_node.name, ENTITY_SIMILARITY_THRESHOLD
            )
            if similar:
                existing_node = similar[0][0]
                if self._has_excessive_contradictions(existing_node, other_node):
                    node_id_map[other_node.id] = self._get_next_node_id()
                    new_node = GraphNode(
                        node_id_map[other_node.id],
                        other_node.name,
                        self,
                        aliases=other_node.aliases[:],
                    )
                    new_node.edge_ids = []
                    new_node.identity_facts = other_node.identity_facts[:]
                    self.nodes[node_id_map[other_node.id]] = new_node
                    self._add_node_to_index(new_node)
                else:
                    nodes_to_merge.append((existing_node, other_node))
            else:
                new_id = self._get_next_node_id()
                node_id_map[other_node.id] = new_id

                new_node = GraphNode(
                    new_id, other_node.name, self, aliases=other_node.aliases[:]
                )
                new_node.edge_ids = []
                new_node.identity_facts = other_node.identity_facts[:]
                self.nodes[new_id] = new_node
                self._add_node_to_index(new_node)

        for existing_node, other_node in nodes_to_merge:
            self._merge_nodes(existing_node, other_node)
            node_id_map[other_node.id] = existing_node.id

        for edge in other_db.edges.values():
            src_node_id = node_id_map.get(edge.source_node_id)
            tgt_node_id = node_id_map.get(edge.target_node_id)

            if src_node_id is None or tgt_node_id is None:
                continue

            new_sources = []
            for src_name in edge.sources:
                for other_key, mapped_name in source_key_to_name.items():
                    if src_name == other_key or self.sources.get(src_name):
                        if not any(
                            s.get_source_key()
                            == self.sources.get(mapped_name).get_source_key()
                            for s in new_sources
                            if s
                        ):
                            new_sources.append(self.sources.get(mapped_name))
                if not new_sources:
                    new_sources.append(src_name)

            new_content = edge.relationship_type
            for old_name, new_name in source_key_to_name.items():
                new_content = new_content.replace(old_name, new_name)

            src_node = self.nodes.get(src_node_id)
            tgt_node = self.nodes.get(tgt_node_id)

            if src_node and tgt_node:
                self.add_edge(
                    src_node,
                    tgt_node,
                    new_content,
                    new_sources[0] if new_sources else "",
                )

    def save(self, path: str) -> None:
        import os
        import json
        import pickle

        os.makedirs(path, exist_ok=True)

        nodes_data = []
        for node in self.nodes.values():
            nodes_data.append(
                {
                    "id": node.id,
                    "name": node.name,
                    "aliases": node.aliases,
                    "edge_ids": node.edge_ids,
                    "identity_facts": node.identity_facts,
                }
            )

        edges_data = []
        for edge in self.edges.values():
            edges_data.append(
                {
                    "id": edge.id,
                    "relationship_type": edge.relationship_type,
                    "sources": edge.sources,
                    "source_node_id": edge.source_node_id,
                    "target_node_id": edge.target_node_id,
                }
            )

        with open(os.path.join(path, "graph.json"), "w") as f:
            json.dump(
                {
                    "nodes": nodes_data,
                    "edges": edges_data,
                    "sources": list(self.sources.keys()),
                    "next_node_id": self._next_node_id,
                    "next_edge_id": self._next_edge_id,
                    "entailment_threshold": self._entailment_threshold,
                    "contradiction_threshold": self._contradiction_threshold,
                },
                f,
            )

        with open(os.path.join(path, "hnsw_index.pkl"), "wb") as f:
            pickle.dump(self._hnsw, f)

        with open(os.path.join(path, "node_hnsw_index.pkl"), "wb") as f:
            pickle.dump(self._node_hnsw, f)

        with open(os.path.join(path, "node_name_to_id.pkl"), "wb") as f:
            pickle.dump(self._node_name_to_id, f)

        with open(os.path.join(path, "edge_id_to_idx.pkl"), "wb") as f:
            pickle.dump(self._edge_id_to_idx, f)

        with open(os.path.join(path, "idx_to_edge_id.pkl"), "wb") as f:
            pickle.dump(self._idx_to_edge_id, f)

    @staticmethod
    def from_saved(path: str) -> "GraphDatabase":
        import os
        import json
        import pickle

        with open(os.path.join(path, "graph.json"), "r") as f:
            data = json.load(f)

        db = GraphDatabase(
            entailment_threshold=data.get("entailment_threshold", 1.0),
            contradiction_threshold=data.get("contradiction_threshold", 2.0),
        )

        db._next_node_id = data.get("next_node_id", 0)
        db._next_edge_id = data.get("next_edge_id", 0)

        node_map = {}
        for node_data in data["nodes"]:
            node = GraphNode(
                node_data["id"],
                node_data["name"],
                db,
                aliases=node_data.get("aliases", [node_data["name"]]),
            )
            node.edge_ids = node_data.get("edge_ids", [])
            node.identity_facts = node_data.get("identity_facts", [])
            db.nodes[node.id] = node
            node_map[node.id] = node

        for edge_data in data["edges"]:
            edge = GraphEdge(
                edge_data["id"],
                edge_data.get("relationship_type", edge_data.get("content", "")),
                edge_data["sources"],
                db,
                edge_data["source_node_id"],
                edge_data["target_node_id"],
            )
            db.edges[edge.id] = edge

        with open(os.path.join(path, "hnsw_index.pkl"), "rb") as f:
            db._hnsw = pickle.load(f)

        with open(os.path.join(path, "node_hnsw_index.pkl"), "rb") as f:
            db._node_hnsw = pickle.load(f)

        with open(os.path.join(path, "node_name_to_id.pkl"), "rb") as f:
            db._node_name_to_id = pickle.load(f)

        with open(os.path.join(path, "edge_id_to_idx.pkl"), "rb") as f:
            db._edge_id_to_idx = pickle.load(f)

        with open(os.path.join(path, "idx_to_edge_id.pkl"), "rb") as f:
            db._idx_to_edge_id = pickle.load(f)

        for node in db.nodes.values():
            db._add_node_to_index(node)

        return db

    def search_edges(self, query: str, k: int = 5) -> list[tuple[GraphEdge, float]]:
        if not self.edges:
            return []
        model = _get_embedding_model()
        query_embedding = model.encode(query, convert_to_numpy=True)
        k_actual = min(k, len(self.edges))
        if k_actual == 0:
            return []
        self._hnsw.set_ef(max(k_actual, self._hnsw.ef))
        labels, distances = self._hnsw.knn_query(query_embedding, k=k_actual)
        results = []
        for idx, dist in zip(labels[0], distances[0]):
            edge_id = self._idx_to_edge_id.get(idx)
            if edge_id and edge_id in self.edges:
                edge = self.edges[edge_id]
                similarity = 1.0 - dist
                results.append((edge, similarity))
        return results

    def get_related_subgraph(self, query: str, k: int = 10) -> SubGraph:
        start_node = self.get_node_by_name(query)
        if start_node is None:
            results = self.search_edges(query, k=k)
            if not results:
                return SubGraph([], [], self)
            relevant_edge_ids = set()
            concept_node_ids = set()
            for edge, _ in results:
                relevant_edge_ids.add(edge.id)
                concept_node_ids.add(edge.source_node_id)
                concept_node_ids.add(edge.target_node_id)
        else:
            concept_node_ids = set([start_node.id])
            relevant_edge_ids = set(start_node.edge_ids)
            for edge_id in start_node.edge_ids:
                edge = self.edges.get(edge_id)
                if edge:
                    concept_node_ids.add(edge.target_node_id)

        subgraph_edge_ids = []
        for edge_id in relevant_edge_ids:
            edge = self.edges.get(edge_id)
            if (
                edge
                and edge.source_node_id in concept_node_ids
                and edge.target_node_id in concept_node_ids
            ):
                subgraph_edge_ids.append(edge_id)

        return SubGraph(list(concept_node_ids), subgraph_edge_ids, self)

    def _resolve_contradiction(
        self, new_content: str, new_source: str, existing_edge: GraphEdge
    ) -> None:
        try:
            from google import genai
            from google.genai import types
            import os
            import json
        except ImportError:
            return

        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            return

        client = genai.Client(api_key=api_key)

        source_node = self.nodes.get(existing_edge.source_node_id)
        target_node = self.nodes.get(existing_edge.target_node_id)

        context = f"Source node: {source_node.name if source_node else 'Unknown'}\n"
        context += f"Target node: {target_node.name if target_node else 'Unknown'}\n"
        context += f"Existing relationship: {existing_edge.relationship_type}\n"
        context += f"Existing sources: {', '.join(existing_edge.sources)}\n"
        context += f"New relationship: {new_content}\n"
        context += f"New source: {new_source}\n\n"

        all_sources = list(set(existing_edge.sources + [new_source]))
        context += "Source texts:\n"
        for src_id in all_sources:
            src = self.sources.get(src_id)
            if src:
                try:
                    content = src.get_content()
                    content_preview = (
                        content[:3000] + "..." if len(content) > 3000 else content
                    )
                    context += f"\n--- Source: {src_id} ({src.get_source_type()}) ---\n"
                    context += content_preview + "\n"
                except Exception:
                    pass

        prompt = f"""You are a knowledge graph integrity manager. Two claims about the same relationship are contradictory:

{context}

Determine how to resolve this contradiction by choosing one of these approaches:
1. "revise_both" - Revise both claims to add qualifiers/nuances that make them both potentially true (e.g., "usually", "in most cases", "according to some sources")
2. "prefer_existing" - Keep the existing claim and add the new source to its sources (if the existing claim seems more reliable)
3. "prefer_new" - Replace the existing claim with the new one and add the old source
4. "keep_both" - Keep both as separate claims with different sources (they may be context-dependent)

Respond with ONLY a JSON object in this format:
{{"approach": "revise_both|prefer_existing|prefer_new|keep_both", "revised_existing": "revised claim text or null", "revised_new": "revised claim text or null", "reason": "brief explanation"}}
"""

        try:
            max_retries = 3
            delay = 2.0
            for attempt in range(max_retries):
                try:
                    result = client.models.generate_content(
                        model="gemini-flash-latest",
                        contents=[
                            types.Content(
                                role="user", parts=[types.Part.from_text(text=prompt)]
                            )
                        ],
                        config=types.GenerateContentConfig(
                            response_mime_type="application/json",
                            system_instruction=[
                                types.Part.from_text(
                                    text="You are a knowledge graph integrity assistant. Respond with valid JSON only."
                                )
                            ],
                        ),
                    )
                    break
                except Exception as retry_err:
                    if attempt == max_retries - 1:
                        raise retry_err
                    print(
                        f"Warning: Gemini API error in contradiction resolution, retrying in {delay}s: {retry_err}"
                    )
                    import time

                    time.sleep(delay)
                    delay *= 2

            response = json.loads(result.text)

            if response["approach"] == "revise_both":
                if response.get("revised_existing"):
                    existing_edge.relationship_type = response["revised_existing"]
                    self._update_edge_in_index(existing_edge)
            elif response["approach"] == "prefer_existing":
                existing_edge.add_source(new_source)
            elif response["approach"] == "prefer_new":
                existing_edge.relationship_type = new_content
                existing_edge.add_source(new_source)
                self._update_edge_in_index(existing_edge)
            elif response["approach"] == "keep_both":
                pass

        except Exception as e:
            pass
