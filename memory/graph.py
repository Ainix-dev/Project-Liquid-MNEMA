# memory/graph.py
"""
MNEMA v2 — Relational Memory Graph

Replaces flat memory retrieval with a structured knowledge graph.

Architecture:
    Nodes  → every memory entry (fact, preference, correction, event)
    Edges  → typed relationships between memories:
             temporal    — memory B happened after memory A
             causal      — memory A caused or led to memory B
             refines     — memory B updates or elaborates on memory A
             contradicts — memory B conflicts with memory A
             depends_on  — memory B requires memory A to make sense

On every new memory:
    1. Check for contradictions with existing memories
    2. Find temporally/semantically related memories and link them
    3. If contradiction found — mark older node as superseded, link with
       "contradicts" edge, store belief revision event

Multi-hop retrieval:
    1. Semantic search finds seed nodes by similarity
    2. Graph traversal expands to neighbors (1-2 hops)
    3. Returns enriched context: seed + related + contradiction warnings
"""

import sqlite3
import time
import json
import uuid
from dataclasses import dataclass, field
from typing import Optional
from sentence_transformers import SentenceTransformer
import numpy as np


# ── Edge type constants ──────────────────────────────────────────────────────
EDGE_TEMPORAL    = "temporal"      # B happened after A
EDGE_CAUSAL      = "causal"        # A caused B
EDGE_REFINES     = "refines"       # B updates/elaborates A
EDGE_CONTRADICTS = "contradicts"   # B conflicts with A
EDGE_DEPENDS_ON  = "depends_on"    # B requires A to make sense

# Contradiction similarity threshold — above this = likely contradiction
CONTRADICTION_THRESHOLD = 0.72

# Refinement similarity threshold — above this = likely refinement
REFINEMENT_THRESHOLD = 0.60

# How many hops to traverse during retrieval
MAX_HOPS = 2


@dataclass
class MemoryNode:
    id: str
    content: str
    memory_type: str          # correction | preference | fact | casual | event
    importance: float
    strength: float
    created_at: float
    turn: int
    superseded: bool = False  # True if a contradiction replaced this memory
    metadata: dict = field(default_factory=dict)


@dataclass
class MemoryEdge:
    id: str
    source_id: str
    target_id: str
    edge_type: str
    weight: float             # 0.0 - 1.0, confidence of this relationship
    created_at: float
    metadata: dict = field(default_factory=dict)


class RelationalMemoryGraph:
    """
    Graph-structured memory system for MNEMA v2.
    Backward compatible with v1 MemoryStore — drop-in upgrade.
    """

    def __init__(self, db_path: str = "./data/memory_graph.db",
                 embedder_model: str = "all-MiniLM-L6-v2"):
        self.db_path = db_path
        self.embedder = SentenceTransformer(embedder_model)
        self._init_db()

    # ── Database setup ───────────────────────────────────────────────────────

    def _init_db(self):
        conn = sqlite3.connect(self.db_path)
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS nodes (
                id          TEXT PRIMARY KEY,
                content     TEXT NOT NULL,
                memory_type TEXT NOT NULL,
                importance  REAL DEFAULT 0.5,
                strength    REAL DEFAULT 1.0,
                embedding   TEXT,
                created_at  REAL,
                last_accessed REAL,
                turn        INTEGER,
                superseded  INTEGER DEFAULT 0,
                metadata    TEXT DEFAULT '{}'
            );

            CREATE TABLE IF NOT EXISTS edges (
                id          TEXT PRIMARY KEY,
                source_id   TEXT NOT NULL,
                target_id   TEXT NOT NULL,
                edge_type   TEXT NOT NULL,
                weight      REAL DEFAULT 1.0,
                created_at  REAL,
                metadata    TEXT DEFAULT '{}',
                FOREIGN KEY (source_id) REFERENCES nodes(id),
                FOREIGN KEY (target_id) REFERENCES nodes(id)
            );

            CREATE INDEX IF NOT EXISTS idx_edges_source ON edges(source_id);
            CREATE INDEX IF NOT EXISTS idx_edges_target ON edges(target_id);
            CREATE INDEX IF NOT EXISTS idx_nodes_type   ON nodes(memory_type);
            CREATE INDEX IF NOT EXISTS idx_nodes_strength ON nodes(strength);
        """)
        conn.commit()
        conn.close()

    # ── Core: Add a memory node ──────────────────────────────────────────────

    def add(self, content: str, memory_type: str, importance: float,
            turn: int, metadata: dict = None) -> str:
        """
        Add a new memory node to the graph.
        Automatically:
          - Computes embedding
          - Detects contradictions with existing alive memories only
          - Links temporally to recent memories
          - Links via refinement if very similar to existing
        Returns the new node's ID.
        """
        node_id = str(uuid.uuid4())
        now = time.time()
        embedding = self.embedder.encode(content).tolist()
        metadata = metadata or {}

        conn = sqlite3.connect(self.db_path)

        # Store the node
        conn.execute("""
            INSERT INTO nodes
            (id, content, memory_type, importance, strength, embedding,
             created_at, last_accessed, turn, superseded, metadata)
            VALUES (?, ?, ?, ?, 1.0, ?, ?, ?, ?, 0, ?)
        """, (node_id, content, memory_type, importance,
              json.dumps(embedding), now, now, turn,
              json.dumps(metadata)))
        conn.commit()

        # ── Auto-link: contradictions and refinements ────────────────────────
        # _get_alive_nodes_with_embeddings already filters superseded=0
        # so we only compare against currently-believed memories
        existing = self._get_alive_nodes_with_embeddings(conn)
        existing = [n for n in existing if n["id"] != node_id]

        for existing_node in existing:
            sim = self._cosine_similarity(embedding, existing_node["embedding"])

            if sim >= CONTRADICTION_THRESHOLD:
                # New memory contradicts existing — new supersedes old
                # source = new (node_id), target = old (existing_node)
                self._add_edge(conn, node_id, existing_node["id"],
                               EDGE_CONTRADICTS, weight=sim)
                # Mark older node as superseded
                conn.execute(
                    "UPDATE nodes SET superseded = 1 WHERE id = ?",
                    (existing_node["id"],)
                )
                conn.commit()

            elif sim >= REFINEMENT_THRESHOLD:
                # Moderately similar = refinement/elaboration
                self._add_edge(conn, node_id, existing_node["id"],
                               EDGE_REFINES, weight=sim)

        # ── Auto-link: temporal — connect to most recent alive memory ────────
        recent = conn.execute("""
            SELECT id FROM nodes
            WHERE id != ? AND superseded = 0
            ORDER BY created_at DESC LIMIT 1
        """, (node_id,)).fetchone()

        if recent:
            self._add_edge(conn, recent[0], node_id,
                           EDGE_TEMPORAL, weight=1.0)

        conn.commit()
        conn.close()

        return node_id

    # ── Core: Retrieve with graph traversal ─────────────────────────────────

    def retrieve(self, query: str, top_k: int = 5) -> list[dict]:
        """
        Graph-aware retrieval:
        1. Find seed nodes by semantic similarity
        2. Traverse graph edges up to MAX_HOPS
        3. Return enriched context with relationship labels
        """
        query_embedding = self.embedder.encode(query).tolist()

        conn = sqlite3.connect(self.db_path)
        all_nodes = self._get_alive_nodes_with_embeddings(conn)

        if not all_nodes:
            conn.close()
            return []

        # ── Step 1: Score all alive nodes by similarity ──────────────────────
        scored = []
        for node in all_nodes:
            sim = self._cosine_similarity(query_embedding, node["embedding"])
            scored.append((sim, node))

        scored.sort(key=lambda x: x[0], reverse=True)
        seed_nodes = [node for _, node in scored[:top_k]]
        seed_ids = {n["id"] for n in seed_nodes}

        # ── Step 2: Graph traversal — expand to neighbors ───────────────────
        expanded = {n["id"]: {**n, "hop": 0, "relation": "direct"}
                    for n in seed_nodes}

        for hop in range(1, MAX_HOPS + 1):
            current_ids = [nid for nid, n in expanded.items()
                           if n["hop"] == hop - 1]

            for node_id in current_ids:
                neighbors = self._get_neighbors(conn, node_id)
                for neighbor_id, edge_type, weight in neighbors:
                    if neighbor_id not in expanded:
                        neighbor_data = self._get_node(conn, neighbor_id)
                        # Only expand to alive (non-superseded) nodes
                        if neighbor_data and not neighbor_data["superseded"]:
                            expanded[neighbor_id] = {
                                **neighbor_data,
                                "hop": hop,
                                "relation": edge_type,
                                "relation_weight": weight
                            }

        # ── Step 3: Reinforce accessed seed nodes ────────────────────────────
        now = time.time()
        for node_id in seed_ids:
            conn.execute("""
                UPDATE nodes
                SET strength = MIN(1.0, strength + 0.4 * (1.0 - strength)),
                    last_accessed = ?
                WHERE id = ?
            """, (now, node_id))
        conn.commit()
        conn.close()

        # ── Step 4: Format results ───────────────────────────────────────────
        results = []
        for node_id, node in expanded.items():
            results.append({
                "id": node_id,
                "content": node["content"],
                "type": node["memory_type"],
                "strength": node["strength"],
                "hop": node["hop"],
                "relation": node.get("relation", "direct"),
                "superseded": node.get("superseded", False),
            })

        # Sort: direct hits first, then by strength
        results.sort(key=lambda x: (x["hop"], -x["strength"]))
        return results[:top_k * 2]

    # ── Contradiction detection (public) ────────────────────────────────────

    def get_contradictions(self) -> list[dict]:
        """
        Return all contradiction edges.
        Convention: source = newer node (current belief)
                    target = older node (superseded belief)
        """
        conn = sqlite3.connect(self.db_path)
        rows = conn.execute("""
            SELECT e.id, e.source_id, e.target_id, e.weight,
                   n1.content as source_content,
                   n2.content as target_content
            FROM edges e
            JOIN nodes n1 ON e.source_id = n1.id
            JOIN nodes n2 ON e.target_id = n2.id
            WHERE e.edge_type = ?
        """, (EDGE_CONTRADICTS,)).fetchall()
        conn.close()

        # source = new (current), target = old (superseded)
        return [{
            "edge_id": r[0],
            "newer": r[1], "newer_content": r[4],   # source = new
            "older": r[2], "older_content": r[5],   # target = old
            "confidence": r[3]
        } for r in rows]

    # ── Decay integration ────────────────────────────────────────────────────

    def get_all_for_decay(self) -> list[dict]:
        """Return all alive nodes for the decay engine."""
        conn = sqlite3.connect(self.db_path)
        rows = conn.execute("""
            SELECT id, content, memory_type, importance, strength,
                   created_at, last_accessed, turn
            FROM nodes WHERE superseded = 0
        """).fetchall()
        conn.close()
        return [{
            "id": r[0], "content": r[1], "type": r[2],
            "importance": r[3], "strength": r[4],
            "created_at": r[5], "last_accessed": r[6], "turn": r[7]
        } for r in rows]

    def update_strength(self, node_id: str, new_strength: float):
        """Update decay strength for a node."""
        conn = sqlite3.connect(self.db_path)
        if new_strength < 0.05:
            conn.execute(
                "UPDATE nodes SET superseded = 1 WHERE id = ?", (node_id,))
        else:
            conn.execute(
                "UPDATE nodes SET strength = ? WHERE id = ?",
                (new_strength, node_id))
        conn.commit()
        conn.close()

    # ── Consolidation integration ────────────────────────────────────────────

    def get_consolidation_candidates(self, min_strength: float = 0.6,
                                      limit: int = 50) -> list[dict]:
        """
        Return high-strength alive memories ready for LoRA consolidation.
        Compatible with v1 trainer interface.
        """
        conn = sqlite3.connect(self.db_path)
        rows = conn.execute("""
            SELECT id, content, memory_type, importance, strength, turn
            FROM nodes
            WHERE superseded = 0 AND strength >= ?
            ORDER BY strength DESC, importance DESC
            LIMIT ?
        """, (min_strength, limit)).fetchall()
        conn.close()
        return [{
            "id": r[0], "content": r[1], "type": r[2],
            "importance": r[3], "strength": r[4], "turn": r[5]
        } for r in rows]

    # ── Graph stats ──────────────────────────────────────────────────────────

    def stats(self) -> dict:
        conn = sqlite3.connect(self.db_path)
        node_count = conn.execute(
            "SELECT COUNT(*) FROM nodes WHERE superseded = 0"
        ).fetchone()[0]
        edge_count = conn.execute(
            "SELECT COUNT(*) FROM edges"
        ).fetchone()[0]
        contradiction_count = conn.execute(
            "SELECT COUNT(*) FROM edges WHERE edge_type = ?",
            (EDGE_CONTRADICTS,)
        ).fetchone()[0]
        superseded_count = conn.execute(
            "SELECT COUNT(*) FROM nodes WHERE superseded = 1"
        ).fetchone()[0]
        edge_types = conn.execute("""
            SELECT edge_type, COUNT(*) FROM edges GROUP BY edge_type
        """).fetchall()
        conn.close()
        return {
            "alive_nodes": node_count,
            "superseded_nodes": superseded_count,
            "total_edges": edge_count,
            "contradictions": contradiction_count,
            "edge_breakdown": {r[0]: r[1] for r in edge_types}
        }

    # ── Internal helpers ─────────────────────────────────────────────────────

    def _add_edge(self, conn, source_id: str, target_id: str,
                  edge_type: str, weight: float = 1.0,
                  metadata: dict = None):
        """Insert an edge — skips duplicates of the same type."""
        existing = conn.execute("""
            SELECT id FROM edges
            WHERE source_id = ? AND target_id = ? AND edge_type = ?
        """, (source_id, target_id, edge_type)).fetchone()

        if not existing:
            conn.execute("""
                INSERT INTO edges (id, source_id, target_id, edge_type,
                                   weight, created_at, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (str(uuid.uuid4()), source_id, target_id, edge_type,
                  weight, time.time(), json.dumps(metadata or {})))

    def _get_neighbors(self, conn, node_id: str) -> list[tuple]:
        """Get all neighbors of a node with edge type and weight."""
        rows = conn.execute("""
            SELECT target_id, edge_type, weight FROM edges
            WHERE source_id = ?
            UNION
            SELECT source_id, edge_type, weight FROM edges
            WHERE target_id = ?
        """, (node_id, node_id)).fetchall()
        return [(r[0], r[1], r[2]) for r in rows]

    def _get_node(self, conn, node_id: str) -> Optional[dict]:
        """Fetch a single node by ID."""
        row = conn.execute("""
            SELECT id, content, memory_type, importance, strength,
                   embedding, created_at, turn, superseded
            FROM nodes WHERE id = ?
        """, (node_id,)).fetchone()
        if not row:
            return None
        return {
            "id": row[0], "content": row[1], "memory_type": row[2],
            "importance": row[3], "strength": row[4],
            "embedding": json.loads(row[5]) if row[5] else [],
            "created_at": row[6], "turn": row[7],
            "superseded": bool(row[8])
        }

    def _get_alive_nodes_with_embeddings(self, conn,
                                          memory_type: str = None) -> list[dict]:
        """
        Fetch all alive (non-superseded) nodes that have embeddings.
        These are the only nodes eligible for contradiction checking
        and semantic retrieval.
        """
        if memory_type:
            rows = conn.execute("""
                SELECT id, content, memory_type, importance, strength, embedding
                FROM nodes
                WHERE superseded = 0 AND embedding IS NOT NULL
                AND memory_type = ?
            """, (memory_type,)).fetchall()
        else:
            rows = conn.execute("""
                SELECT id, content, memory_type, importance, strength, embedding
                FROM nodes
                WHERE superseded = 0 AND embedding IS NOT NULL
            """).fetchall()

        result = []
        for r in rows:
            try:
                emb = json.loads(r[5])
                result.append({
                    "id": r[0], "content": r[1], "memory_type": r[2],
                    "importance": r[3], "strength": r[4], "embedding": emb
                })
            except (json.JSONDecodeError, TypeError):
                continue
        return result

    @staticmethod
    def _cosine_similarity(a: list, b: list) -> float:
        """Compute cosine similarity between two embedding vectors."""
        a, b = np.array(a), np.array(b)
        norm_a, norm_b = np.linalg.norm(a), np.linalg.norm(b)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return float(np.dot(a, b) / (norm_a * norm_b))
