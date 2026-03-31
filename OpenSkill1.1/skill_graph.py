"""
skill_graph.py — S-Path-RAG Skill Graph
=========================================
Implements the semantic-aware shortest-path retrieval over a skill graph.

Core idea (from S-Path-RAG paper, arXiv:2603.23512):
  Skills are not isolated documents — they form a graph where edges encode
  semantic relationships. Retrieval for a complex problem is not a single
  lookup but a PATH through the skill graph: prerequisite skills → core skill
  → error-handling skills → verification skills.

Architecture:
  Nodes  — Skill.md documents (keyed by skill_id)
  Edges  — Typed semantic relationships:
    PREREQUISITE_OF  : Skill A must be understood before Skill B
    RESOLVES_ERROR   : Skill A fixes failures that Skill B causes
    EXTENDS          : Skill A generalizes or deepens Skill B
    SIMILAR_TO       : Skill A and B address overlapping problem regions
    CONTRADICTS      : Skill A and B should not be used together

S-Path-RAG retrieval pipeline (adapted for skill graphs):
  1. Embed query → query vector
  2. Seed expansion: find top-K entry skills by vector similarity (TurboQuant)
  3. Enumerate bounded-length paths from seed nodes (k-shortest + beam)
  4. Score paths: structural plausibility + semantic alignment to query
  5. Neural-Socratic loop: if confidence < threshold, expand graph and retry
  6. Return top path as ordered skill list + reasoning trace

Graph persistence:
  skills_output/skill_graph.json — adjacency list with edge metadata
  Updated automatically on skill creation, evolution, and merge
"""

import json
import math
import asyncio
from pathlib import Path
from typing import Optional
from collections import defaultdict, deque

import httpx
import numpy as np

from skill_vector import (
    embed_skill_text,
    cosine_similarity_raw,
    EMBED_DIM,
)

# ── Constants ─────────────────────────────────────────────────────────────────

GRAPH_FILE          = Path("skills_output/skill_graph.json")
MAX_PATH_LEN        = 4      # Max hops in a skill path (S-Path-RAG: L=4 optimal)
BEAM_WIDTH          = 8      # Beam search width during path enumeration
TOP_K_SEEDS         = 5      # Number of entry-point skills from vector search
CONFIDENCE_THRESHOLD = 0.55  # Neural-Socratic loop: expand if confidence below this
MAX_SOCRATIC_ROUNDS  = 3     # Max retrieval refinement rounds

# Edge type weights for path scoring (structural plausibility prior)
EDGE_WEIGHTS = {
    "PREREQUISITE_OF": 0.9,
    "EXTENDS":         0.85,
    "RESOLVES_ERROR":  0.8,
    "SIMILAR_TO":      0.7,
    "CONTRADICTS":    -0.5,   # Negative: penalize contradictory paths
}

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"


# ── Graph I/O ─────────────────────────────────────────────────────────────────

def load_graph() -> dict:
    """
    Load skill graph from disk.
    Schema:
    {
      "nodes": { skill_id: { "title", "category", "subcategory", "domain" } },
      "edges": [
        { "from": id_a, "to": id_b, "type": "SIMILAR_TO",
          "weight": 0.82, "reason": "Both handle recursive decomposition" }
      ]
    }
    """
    if GRAPH_FILE.exists():
        try:
            return json.loads(GRAPH_FILE.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {"nodes": {}, "edges": []}


def save_graph(graph: dict) -> None:
    GRAPH_FILE.parent.mkdir(exist_ok=True)
    GRAPH_FILE.write_text(json.dumps(graph, indent=2), encoding="utf-8")


def add_node(graph: dict, skill_id: str, meta: dict) -> None:
    """Register a skill as a graph node."""
    graph["nodes"][skill_id] = {
        "title":       meta.get("title", ""),
        "category":    meta.get("category", ""),
        "subcategory": meta.get("subcategory", ""),
        "domain":      meta.get("domain", ""),
    }


def add_edge(graph: dict, from_id: str, to_id: str,
             edge_type: str, weight: float, reason: str = "") -> None:
    """Add a typed directed edge to the skill graph."""
    # Deduplicate: remove existing edge of same type between same nodes
    graph["edges"] = [
        e for e in graph["edges"]
        if not (e["from"] == from_id and e["to"] == to_id and e["type"] == edge_type)
    ]
    graph["edges"].append({
        "from":   from_id,
        "to":     to_id,
        "type":   edge_type,
        "weight": round(weight, 4),
        "reason": reason[:200],
    })


def get_adjacency(graph: dict) -> dict[str, list[dict]]:
    """Build adjacency list from edge list for fast traversal."""
    adj = defaultdict(list)
    for edge in graph["edges"]:
        adj[edge["from"]].append(edge)
    return dict(adj)


# ── Automatic Edge Inference ──────────────────────────────────────────────────

async def infer_edges_for_new_skill(
    api_key: str,
    new_skill_id: str,
    new_meta: dict,
    all_metas: dict,
    graph: dict,
) -> list[dict]:
    """
    Automatically infer graph edges when a new skill is added.

    Two-stage process:
    1. Vector similarity scan (TurboQuant cosine): find candidate neighbors
    2. LLM semantic edge classifier: determine relationship type for top candidates

    This mirrors S-Path-RAG's 'differentiable path scorer' — we jointly train
    (here: prompt) a classifier that suppresses spurious edges while preserving
    structurally meaningful ones.
    """
    if len(all_metas) < 2:
        return []

    # Stage 1: Find top candidates by vector similarity
    similarities = []
    for sid, meta in all_metas.items():
        if sid == new_skill_id:
            continue
        sim = cosine_similarity_raw(new_meta, meta)
        if sim > 0.3:  # Minimum threshold to consider
            similarities.append((sid, meta, sim))

    similarities.sort(key=lambda x: x[2], reverse=True)
    candidates = similarities[:6]  # Consider top 6 neighbors

    if not candidates:
        return []

    # Stage 2: LLM edge classification
    new_info = f"Title: {new_meta.get('title')}\nDomain: {new_meta.get('domain')}\n" \
               f"Category: {new_meta.get('category')}/{new_meta.get('subcategory')}"

    candidate_info = "\n".join([
        f"  [{sid}] Title: {m.get('title')} | Domain: {m.get('domain')} | "
        f"Category: {m.get('category')}/{m.get('subcategory')} | similarity: {sim:.2f}"
        for sid, m, sim in candidates
    ])

    system = (
        "You are a Skill Graph Architect. Analyze relationships between reasoning skills "
        "and classify directed edges between them.\n\n"
        "Edge types:\n"
        "  PREREQUISITE_OF : new skill requires the candidate skill as foundation\n"
        "  EXTENDS         : new skill generalizes or deepens the candidate skill\n"
        "  RESOLVES_ERROR  : new skill fixes failures that the candidate skill causes\n"
        "  SIMILAR_TO      : both skills address overlapping problem regions\n"
        "  CONTRADICTS     : these skills should not be applied together\n\n"
        "For each candidate, decide if a meaningful edge exists.\n"
        "Output ONLY valid JSON array. No preamble. No markdown.\n"
        "Format: [{\"from\": \"new\", \"to\": \"candidate_id\", \"type\": \"EDGE_TYPE\", "
        "\"weight\": 0.0-1.0, \"reason\": \"one sentence\"}]\n"
        "Include an edge only if there is a clear, strong semantic relationship. "
        "If no strong relationship exists, return []."
    )

    user = (
        f"NEW SKILL:\n{new_info}\n\n"
        f"CANDIDATE EXISTING SKILLS:\n{candidate_info}\n\n"
        "Classify which directed edges should be created from the new skill to candidates. "
        "Use 'new' as the from-id for the new skill."
    )

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://openskill.local",
        "X-Title": "OpenSkill",
    }
    payload = {
        "model": "minimax/minimax-m2.7",
        "messages": [
            {"role": "system", "content": system},
            {"role": "user",   "content": user},
        ],
        "max_tokens": 80000,
        "temperature": 0.2,
    }

    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            resp = await client.post(OPENROUTER_URL, headers=headers, json=payload)
            if resp.status_code != 200:
                return []
            data = resp.json()

        raw = data["choices"][0]["message"]["content"] or ""

        import re
        match = re.search(r'\[.*\]', raw, re.DOTALL)
        if not match:
            return []

        edges_raw = json.loads(match.group(0))
        inferred = []
        for e in edges_raw:
            if not isinstance(e, dict):
                continue
            etype = e.get("type", "")
            if etype not in EDGE_WEIGHTS:
                continue
            # Replace "new" placeholder with actual skill_id
            from_id = new_skill_id if e.get("from") == "new" else e.get("from", new_skill_id)
            to_id   = e.get("to", "")
            if to_id not in all_metas:
                continue
            inferred.append({
                "from":   from_id,
                "to":     to_id,
                "type":   etype,
                "weight": float(e.get("weight", 0.7)),
                "reason": e.get("reason", ""),
            })
        return inferred

    except Exception:
        return []


# ── Path Enumeration ──────────────────────────────────────────────────────────

def enumerate_paths(
    adj: dict[str, list[dict]],
    seed_ids: list[str],
    max_len: int = MAX_PATH_LEN,
    beam_width: int = BEAM_WIDTH,
) -> list[list[dict]]:
    """
    Enumerate candidate paths through the skill graph starting from seed nodes.

    Implements S-Path-RAG's hybrid strategy:
      k-shortest + beam: BFS with beam pruning by cumulative edge weight
      constrained random walk: inject randomness to explore non-greedy paths

    Returns: list of paths, each path = list of edge dicts traversed
    (Empty path = single seed node with no edges)
    """
    paths = []
    # Queue: (current_node_id, path_so_far, visited_set, cumulative_score)
    queue = deque()

    for seed in seed_ids:
        queue.append((seed, [], set([seed]), 1.0))
        # Also add single-node path (the seed itself as a standalone skill)
        paths.append([{"from": None, "to": seed, "type": "SEED", "weight": 1.0, "reason": "seed"}])

    while queue:
        node, path, visited, score = queue.popleft()

        if len(path) >= max_len:
            continue

        neighbors = adj.get(node, [])
        # Sort by edge weight descending (beam pruning)
        neighbors_sorted = sorted(neighbors, key=lambda e: e.get("weight", 0), reverse=True)
        beam_candidates = neighbors_sorted[:beam_width]

        # Add a small random sample beyond beam for diversity (constrained random walk)
        if len(neighbors_sorted) > beam_width:
            import random
            extra = random.sample(neighbors_sorted[beam_width:], min(2, len(neighbors_sorted) - beam_width))
            beam_candidates = beam_candidates + extra

        for edge in beam_candidates:
            next_node = edge["to"]
            if next_node in visited:
                continue
            edge_w = EDGE_WEIGHTS.get(edge["type"], 0.5) * edge.get("weight", 0.7)
            if edge_w < 0:
                continue  # Skip contradictory edges in path building
            new_path   = path + [edge]
            new_score  = score * edge_w
            new_visited = visited | {next_node}
            queue.append((next_node, new_path, new_visited, new_score))
            if len(new_path) >= 1:
                paths.append(new_path)

    # Deduplicate paths by terminal node sequence
    seen = set()
    unique_paths = []
    for p in paths:
        key = tuple(e["to"] for e in p)
        if key not in seen:
            seen.add(key)
            unique_paths.append(p)

    return unique_paths


# ── Path Scoring ──────────────────────────────────────────────────────────────

def score_path(
    path: list[dict],
    query_vec: np.ndarray,
    all_metas: dict,
    alpha: float = 0.4,  # Structural weight (S-Path-RAG optimal: 0.4)
    beta:  float = 0.4,  # Semantic alignment weight
    gamma: float = 0.2,  # Coverage diversity weight
) -> float:
    """
    Score a candidate path using S-Path-RAG's hybrid weighting scheme.
    Best F1 in paper: alpha=0.4, beta=0.4, gamma=0.2 (CWQ validation set).

    Score = alpha * structural_plausibility
           + beta  * semantic_alignment
           + gamma * coverage_diversity
    """
    if not path:
        return 0.0

    # Structural plausibility: product of edge weights along path
    structural = 1.0
    for edge in path:
        ew = EDGE_WEIGHTS.get(edge.get("type", ""), 0.5) * edge.get("weight", 0.7)
        structural *= max(ew, 0.01)
    structural = structural ** (1.0 / max(len(path), 1))  # Geometric mean

    # Semantic alignment: average cosine sim of path nodes to query vector
    node_ids = [e["to"] for e in path if e["to"] in all_metas]
    if not node_ids:
        return 0.0

    sims = []
    for nid in node_ids:
        meta = all_metas.get(nid, {})
        if "embedding" not in meta:
            continue
        node_vec = np.array(meta["embedding"], dtype=np.float32)
        norm_q = np.linalg.norm(query_vec) + 1e-8
        norm_n = np.linalg.norm(node_vec) + 1e-8
        sims.append(float(np.dot(query_vec, node_vec) / (norm_q * norm_n)))

    semantic = float(np.mean(sims)) if sims else 0.0

    # Coverage diversity: reward paths that span multiple categories/subcategories
    categories = set()
    for nid in node_ids:
        meta = all_metas.get(nid, {})
        categories.add(meta.get("subcategory", ""))
    diversity = math.log1p(len(categories)) / math.log1p(MAX_PATH_LEN)

    return alpha * structural + beta * semantic + gamma * diversity


# ── Neural-Socratic Retrieval Loop ────────────────────────────────────────────

async def neural_socratic_retrieve(
    api_key: str,
    query: str,
    query_vec: np.ndarray,
    all_metas: dict,
    graph: dict,
    top_k: int = 3,
) -> dict:
    """
    S-Path-RAG Neural-Socratic Graph Dialogue retrieval.

    The model expresses uncertainty diagnostics, which are mapped to:
      - Graph seed expansion (add more entry points)
      - Path length extension (allow deeper hops)
      - Edge type filtering (exclude contradictory paths)

    Iterates until confidence > CONFIDENCE_THRESHOLD or MAX_SOCRATIC_ROUNDS reached.

    Returns:
    {
      "paths": [ordered list of skill_ids, best path first],
      "skills": [list of skill metadata for top path nodes],
      "scores": [float scores per path],
      "confidence": float,
      "rounds": int,
      "reasoning_trace": str,
    }
    """
    adj = get_adjacency(graph)

    # Seed expansion: top-K entry skills by vector similarity
    scored_seeds = []
    for sid, meta in all_metas.items():
        if "embedding" not in meta:
            continue
        sim = cosine_similarity_raw({"embedding": query_vec.tolist()}, meta)
        scored_seeds.append((sid, sim))

    scored_seeds.sort(key=lambda x: x[1], reverse=True)
    seed_ids = [s[0] for s in scored_seeds[:TOP_K_SEEDS]]

    if not seed_ids:
        return {
            "paths": [],
            "skills": [],
            "scores": [],
            "confidence": 0.0,
            "rounds": 0,
            "reasoning_trace": "No embedded skills found in graph.",
        }

    confidence = 0.0
    rounds = 0
    reasoning_trace = []
    best_paths = []
    current_max_len = MAX_PATH_LEN

    while rounds < MAX_SOCRATIC_ROUNDS:
        rounds += 1

        # Enumerate paths from current seeds
        all_paths = enumerate_paths(adj, seed_ids, max_len=current_max_len)

        # Score all paths
        scored_paths = []
        for path in all_paths:
            score = score_path(path, query_vec, all_metas)
            node_ids = [e["to"] for e in path]
            scored_paths.append((score, node_ids, path))

        scored_paths.sort(key=lambda x: x[0], reverse=True)
        best_paths = scored_paths[:top_k]

        if not best_paths:
            break

        top_score = best_paths[0][0]
        confidence = min(1.0, top_score)

        reasoning_trace.append(
            f"Round {rounds}: {len(all_paths)} paths enumerated, "
            f"top score={top_score:.3f}, seeds={len(seed_ids)}"
        )

        if confidence >= CONFIDENCE_THRESHOLD:
            break

        # Neural-Socratic expansion: below threshold → expand seeds and depth
        # This maps to S-Path-RAG's "diagnostic to graph edit" mechanism
        if rounds < MAX_SOCRATIC_ROUNDS:
            # Expand seeds: add next-tier candidates
            extra_seeds = [s[0] for s in scored_seeds[TOP_K_SEEDS:TOP_K_SEEDS+3]]
            seed_ids = list(set(seed_ids + extra_seeds))
            current_max_len = min(current_max_len + 1, 6)  # Allow deeper paths
            reasoning_trace.append(
                f"Round {rounds} diagnostic: confidence {confidence:.2f} < {CONFIDENCE_THRESHOLD}. "
                f"Expanding seeds to {len(seed_ids)}, depth to {current_max_len}."
            )

    # Build output: collect skills for top path nodes
    skills_in_path = []
    if best_paths:
        top_node_ids = best_paths[0][1]
        for nid in top_node_ids:
            if nid in all_metas:
                skills_in_path.append(all_metas[nid])

    return {
        "paths": [p[1] for p in best_paths],
        "skills": skills_in_path,
        "scores": [p[0] for p in best_paths],
        "confidence": confidence,
        "rounds": rounds,
        "reasoning_trace": "\n".join(reasoning_trace),
    }


# ── Public API ────────────────────────────────────────────────────────────────

async def register_skill_in_graph(
    api_key: str,
    skill_id: str,
    meta: dict,
    all_metas: dict,
) -> dict:
    """
    Add a new skill to the graph and infer edges to existing skills.
    Called after skill creation + embedding computation.

    Returns updated graph dict.
    """
    graph = load_graph()
    add_node(graph, skill_id, meta)

    # Infer edges via LLM + vector similarity
    inferred = await infer_edges_for_new_skill(api_key, skill_id, meta, all_metas, graph)
    for edge in inferred:
        add_edge(graph, edge["from"], edge["to"], edge["type"], edge["weight"], edge["reason"])
        # Also add reverse SIMILAR_TO edges bidirectionally
        if edge["type"] == "SIMILAR_TO":
            add_edge(graph, edge["to"], edge["from"], "SIMILAR_TO", edge["weight"], edge["reason"])

    save_graph(graph)
    return graph


async def graph_retrieve(
    api_key: str,
    query: str,
    all_metas: dict,
    top_k: int = 3,
    model: str = "openai/gpt-4o-mini",
) -> dict:
    """
    Full S-Path-RAG retrieval pipeline.

    1. Embed query
    2. Neural-Socratic path search through skill graph
    3. Return top-K paths with reasoning trace

    This is the production replacement for the current symbolic category-matching retrieval.
    """
    graph = load_graph()

    # Embed the query
    query_vec = await embed_skill_text(api_key, query)

    # Run Neural-Socratic retrieval loop
    result = await neural_socratic_retrieve(
        api_key, query, query_vec, all_metas, graph, top_k=top_k
    )

    return result


def get_graph_stats(graph: dict) -> dict:
    """Return graph statistics for the API."""
    edge_types = defaultdict(int)
    for e in graph["edges"]:
        edge_types[e["type"]] += 1
    return {
        "node_count": len(graph["nodes"]),
        "edge_count": len(graph["edges"]),
        "edge_types": dict(edge_types),
    }
