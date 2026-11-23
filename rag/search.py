"""Пошук: векторний (Qdrant) · лексичний (BM25) · hybrid (RRF) · hybrid + reranking, з фільтрами за метаданими.

    python -m rag.search "які Go-вакансії вимагають Kubernetes" --mode hybrid_rerank -k 5
    python -m rag.search "медіанна зарплата middle" --doc-type report
    python -m rag.search "досвід з Kafka" --doc-type vacancy --stack Java
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass, field

import numpy as np

from .bm25 import BM25
from .index import CFG, Embedder, collection_name, load_chunks, qdrant_client

MODES = ("vector", "bm25", "hybrid", "hybrid_rerank")


@dataclass
class Hit:
    chunk: dict
    score: float
    ranks: dict = field(default_factory=dict)       # місце в кожному з пошуків (для дебагу і звіту)


def matches(chunk: dict, flt: dict | None) -> bool:
    """Фільтр за метаданими. flt: {"doc_type": "vacancy" | [...], "stack": "Go", "lang": "uk", "doc_id": ...}."""
    if not flt:
        return True
    for key, want in flt.items():
        have = chunk.get("meta", {}).get(key) if key == "stack" else chunk.get(key)
        allowed = want if isinstance(want, (list, tuple, set)) else [want]
        if have not in allowed:
            return False
    return True


_QDRANT = None


def shared_qdrant():
    """Локальний Qdrant блокує папку на один клієнт, тож усі колекції в процесі ділять один."""
    global _QDRANT
    if _QDRANT is None:
        _QDRANT = qdrant_client()
    return _QDRANT


class QdrantBackend:
    def __init__(self, kind: str):
        self.client = shared_qdrant()
        self.name = collection_name(kind)

    def search(self, vec: np.ndarray, k: int, flt: dict | None) -> list[tuple[int, float]]:
        from qdrant_client.models import FieldCondition, Filter, MatchAny, MatchValue
        must = []
        for key, want in (flt or {}).items():
            field_key = "meta.stack" if key == "stack" else key
            cond = MatchAny(any=list(want)) if isinstance(want, (list, tuple, set)) else MatchValue(value=want)
            must.append(FieldCondition(key=field_key, match=cond))
        res = self.client.query_points(self.name, query=vec.tolist(), limit=k,
                                       query_filter=Filter(must=must) if must else None)
        return [(p.id, p.score) for p in res.points]


class NumpyBackend:
    """Той самий інтерфейс без Qdrant: брутфорс по косинусу. Для тестів і як запасний варіант."""

    def __init__(self, vectors: np.ndarray, chunks: list[dict]):
        self.vectors, self.chunks = vectors, chunks

    def search(self, vec: np.ndarray, k: int, flt: dict | None) -> list[tuple[int, float]]:
        sims = self.vectors @ vec
        order = [i for i in np.argsort(-sims) if matches(self.chunks[i], flt)]
        return [(int(i), float(sims[i])) for i in order[:k]]


class Retriever:
    def __init__(self, kind: str = "structure", embedder: Embedder | None = None, backend=None, reranker=None):
        self.chunks = load_chunks(kind)
        self.bm25 = BM25([c["embed_text"] for c in self.chunks], CFG["bm25"]["k1"], CFG["bm25"]["b"])
        self.embedder = embedder or Embedder()
        self.backend = backend or QdrantBackend(kind)
        self._reranker = reranker

    @property
    def reranker(self):
        if self._reranker is None:
            from sentence_transformers import CrossEncoder
            self._reranker = CrossEncoder(CFG["rerank"]["model"])
        return self._reranker

    # --- окремі пошуки -------------------------------------------------------
    def vector(self, query: str, k: int, flt: dict | None = None) -> list[tuple[int, float]]:
        return self.backend.search(self.embedder.query(query), k, flt)

    def lexical(self, query: str, k: int, flt: dict | None = None) -> list[tuple[int, float]]:
        scores = self.bm25.scores(query)
        order = sorted((i for i, s in enumerate(scores) if s > 0 and matches(self.chunks[i], flt)),
                       key=lambda i: -scores[i])
        return [(i, scores[i]) for i in order[:k]]

    @staticmethod
    def rrf(*rankings: list[tuple[int, float]], k: int = 60) -> list[tuple[int, float]]:
        """Reciprocal Rank Fusion: важить лише місце в списку, а не «сирий» бал — тож BM25 і косинус
        не треба зводити до однієї шкали. Документ, високо в обох списках, піднімається найвище."""
        fused: dict[int, float] = {}
        for ranking in rankings:
            for rank, (i, _) in enumerate(ranking, 1):
                fused[i] = fused.get(i, 0.0) + 1.0 / (k + rank)
        return sorted(fused.items(), key=lambda x: -x[1])

    # --- головний метод ------------------------------------------------------
    def search(self, query: str, mode: str = "hybrid_rerank", k: int = 5, flt: dict | None = None) -> list[Hit]:
        assert mode in MODES, f"mode must be one of {MODES}"
        n_cand = CFG["fusion"]["candidates"]
        if mode == "vector":
            res = self.vector(query, k, flt)
            return [Hit(self.chunks[i], s, {"vector": r}) for r, (i, s) in enumerate(res, 1)]
        if mode == "bm25":
            res = self.lexical(query, k, flt)
            return [Hit(self.chunks[i], s, {"bm25": r}) for r, (i, s) in enumerate(res, 1)]

        vec, lex = self.vector(query, n_cand, flt), self.lexical(query, n_cand, flt)
        vrank = {i: r for r, (i, _) in enumerate(vec, 1)}
        lrank = {i: r for r, (i, _) in enumerate(lex, 1)}
        fused = self.rrf(vec, lex, k=CFG["fusion"]["k"])
        hits = [Hit(self.chunks[i], s, {"vector": vrank.get(i), "bm25": lrank.get(i), "fused": r})
                for r, (i, s) in enumerate(fused, 1)]
        if mode == "hybrid":
            return hits[:k]

        top = hits[:CFG["rerank"]["top_n"]]
        scores = self.reranker.predict([(query, h.chunk["embed_text"]) for h in top])
        for h, s in zip(top, scores):
            h.ranks["rerank_score"] = float(s)
            h.score = float(s)
        top.sort(key=lambda h: -h.score)
        return top[:k]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("query")
    ap.add_argument("--mode", choices=MODES, default="hybrid_rerank")
    ap.add_argument("-k", type=int, default=5)
    ap.add_argument("--chunks", choices=["structure", "fixed"], default="structure")
    ap.add_argument("--doc-type", nargs="+", help="vacancy | law | guide | article | report")
    ap.add_argument("--stack", help="для вакансій: Python, Go, Java ...")
    ap.add_argument("--lang", choices=["uk", "en"])
    args = ap.parse_args()
    flt = {}
    if args.doc_type:
        flt["doc_type"] = args.doc_type if len(args.doc_type) > 1 else args.doc_type[0]
    if args.stack:
        flt["stack"] = args.stack
    if args.lang:
        flt["lang"] = args.lang
    for r, h in enumerate(Retriever(args.chunks).search(args.query, args.mode, args.k, flt or None), 1):
        c = h.chunk
        print(f"{r}. [{h.score:.3f}] {c['chunk_id']}  ({c['doc_type']}, ranks={h.ranks})")
        print(f"   {c['title'][:90]}" + (f" › {c['section'][:80]}" if c["section"] else ""))
        print(f"   {c['text'][:220].replace(chr(10), ' ⏎ ')}\n")


if __name__ == "__main__":
    main()
