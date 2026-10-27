"""Density-based clustering of demand signals into jobs-to-be-done.

Algorithm: DBSCAN over **cosine distance** (eps on 1−cos), which is the practical
"sibling" of HDBSCAN for small/medium batches: deterministic, no tuning of ``k``,
explicit noise label, and it degrades gracefully when there are only a handful of
signals. ``hdbscan`` can be dropped in behind :func:`cluster_vectors` later without
touching callers.

Complexity: O(n²) memory for the similarity matrix. The pipeline caps batch size
(``max_signals``) and the production path swaps in pgvector ANN + HNSW when the
corpus grows past ~50k signals (documented in DESIGN.md).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from llm.embeddings import cosine_pairwise


@dataclass
class Cluster:
    member_indices: list[int]
    centroid: list[float]
    cohesion: float  # mean intra-cluster cosine similarity
    top_terms: list[str] = field(default_factory=list)
    size: int = 0

    def __post_init__(self) -> None:
        self.size = len(self.member_indices)

    def to_dict(self) -> dict[str, Any]:
        return {
            "size": self.size,
            "cohesion": round(self.cohesion, 4),
            "top_terms": self.top_terms,
            "members": self.member_indices,
        }


@dataclass
class ClusterResult:
    labels: list[int]  # -1 == noise
    clusters: list[Cluster]
    noise_indices: list[int]
    params: dict[str, Any]
    similarity: np.ndarray | None = None

    @property
    def n_clusters(self) -> int:
        return len(self.clusters)

    def to_dict(self) -> dict[str, Any]:
        return {
            "n_clusters": self.n_clusters,
            "n_noise": len(self.noise_indices),
            "noise_ratio": round(len(self.noise_indices) / max(len(self.labels), 1), 4),
            "params": self.params,
            "clusters": [cluster.to_dict() for cluster in self.clusters],
        }


class _UnionFind:
    def __init__(self, size: int) -> None:
        self.parent = list(range(size))

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[rb] = ra


def _top_terms(texts: list[str], k: int = 4) -> list[str]:
    """Distinctive unigrams/bigrams for a cluster (TF within cluster)."""
    from market_intel.normalize import WORD_RE

    stop = {
        "the",
        "a",
        "an",
        "and",
        "or",
        "to",
        "of",
        "in",
        "for",
        "with",
        "is",
        "are",
        "how",
        "do",
        "i",
        "my",
        "we",
        "you",
        "it",
        "on",
        "that",
        "this",
        "can",
        "what",
        "when",
        "need",
        "want",
        "best",
        "way",
        "someone",
        "anyone",
        "should",
        "which",
        "from",
    }
    counts: dict[str, int] = {}
    for text in texts:
        tokens = [t for t in WORD_RE.findall(text.lower()) if t not in stop and len(t) > 2]
        for token in tokens:
            counts[token] = counts.get(token, 0) + 1
        for first, second in zip(tokens, tokens[1:], strict=False):
            bigram = f"{first} {second}"
            counts[bigram] = counts.get(bigram, 0) + 2  # favour phrases
    ranked = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
    return [term for term, _ in ranked[:k]]


def cluster_vectors(
    vectors: list[list[float]] | np.ndarray,
    *,
    eps: float = 0.34,
    min_size: int = 3,
    texts: list[str] | None = None,
    max_noise_ratio: float = 0.6,
) -> ClusterResult:
    """Group vectors into clusters of related demand. Never raises on tiny inputs."""
    matrix = np.asarray(vectors, dtype=np.float64)
    n = matrix.shape[0] if matrix.ndim == 2 else 0
    params = {"algorithm": "dbscan-cosine", "eps": eps, "min_size": min_size, "n": n}
    if n == 0:
        return ClusterResult(labels=[], clusters=[], noise_indices=[], params=params)
    if n < min_size:
        return ClusterResult(
            labels=[-1] * n, clusters=[], noise_indices=list(range(n)), params=params
        )

    sim = cosine_pairwise(matrix)
    np.fill_diagonal(sim, 1.0)
    adjacency = sim >= (1.0 - eps)
    degree = adjacency.sum(axis=1)
    core = degree >= min_size

    uf = _UnionFind(n)
    core_indices = np.where(core)[0]
    for i in core_indices:
        for j in np.where(adjacency[i])[0]:
            if j != i and core[j]:
                uf.union(int(i), int(j))

    labels = [-1] * n
    next_label = 0
    root_to_label: dict[int, int] = {}
    for i in core_indices:
        root = uf.find(int(i))
        if root not in root_to_label:
            root_to_label[root] = next_label
            next_label += 1
        labels[int(i)] = root_to_label[root]

    # Border points: non-core points adjacent to a core point inherit its cluster.
    for i in range(n):
        if labels[i] != -1 or core[i]:
            continue
        neighbours = np.where(adjacency[i])[0]
        core_neighbours = [int(j) for j in neighbours if core[j]]
        if core_neighbours:
            best = max(core_neighbours, key=lambda j: sim[i, j])
            labels[i] = labels[best]

    clusters: list[Cluster] = []
    for label in sorted({lab for lab in labels if lab != -1}):
        members = [idx for idx, lab in enumerate(labels) if lab == label]
        if len(members) < 1:
            continue
        sub = sim[np.ix_(members, members)]
        cohesion = float((sub.sum() - len(members)) / max(len(members) * (len(members) - 1), 1))
        centroid = matrix[members].mean(axis=0)
        norm = np.linalg.norm(centroid)
        if norm > 0:
            centroid = centroid / norm
        top_terms = _top_terms([texts[i] for i in members], k=4) if texts else []
        clusters.append(
            Cluster(
                member_indices=members,
                centroid=centroid.tolist(),
                cohesion=cohesion,
                top_terms=top_terms,
            )
        )

    clusters.sort(key=lambda c: c.size, reverse=True)
    noise = [idx for idx, lab in enumerate(labels) if lab == -1]
    params["noise_ratio"] = round(len(noise) / n, 4)
    if (
        max_noise_ratio is not None
        and params["noise_ratio"] > max_noise_ratio
        and len(clusters) <= 1
    ):
        # Everything is "noise" because thresholds are too tight for this batch:
        # report it so the operator can tune eps rather than silently losing signal.
        params["warning"] = "noise_ratio_exceeds_limit"
    return ClusterResult(
        labels=labels, clusters=clusters, noise_indices=noise, params=params, similarity=sim
    )
