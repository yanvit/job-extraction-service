"""Індекс: ембеддинги чанків → Qdrant (локальний режим, без сервера) з метаданими у payload.

    python -m rag.index                    # structure-aware чанки → колекція "chunks"
    python -m rag.index --chunks fixed     # фіксовані чанки → колекція "chunks_fixed" (для порівняння)

Ембеддинги кешуються в data/index/emb_cache_<model>.npz за sha256(модель + текст):
повторна побудова індексу не перераховує вже пораховані вектори (stretch: кешування embeddings).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parent.parent
CFG = yaml.safe_load((Path(__file__).parent / "config.yaml").read_text(encoding="utf-8"))
PAYLOAD_FIELDS = ["chunk_id", "doc_id", "title", "section", "source", "format", "doc_type", "lang", "version",
                  "license", "doc_sha256", "chunk_sha256", "position", "pages", "meta", "n_chars"]


def load_chunks(kind: str = "structure") -> list[dict]:
    path = ROOT / CFG["paths"]["chunks" if kind == "structure" else "chunks_fixed"]
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def collection_name(kind: str) -> str:
    return "chunks" if kind == "structure" else "chunks_fixed"


class Embedder:
    """Обгортка над sentence-transformers з префіксами e5 і кешем на диску."""

    def __init__(self, model_name: str | None = None, cache_dir: Path | None = None, model=None):
        self.model_name = model_name or CFG["embeddings"]["model"]
        self._model = model                                    # можна підставити фейкову модель у тестах
        self.cache_path = (cache_dir or ROOT / CFG["paths"]["index_dir"]) / \
            f"emb_cache_{self.model_name.replace('/', '__')}.npz"
        self._cache: dict[str, np.ndarray] = {}
        if self.cache_path.exists():
            data = np.load(self.cache_path)
            self._cache = {k: data[k] for k in data.files}

    @property
    def model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self.model_name)
        return self._model

    def _key(self, text: str) -> str:
        return hashlib.sha256(f"{self.model_name}\n{text}".encode("utf-8")).hexdigest()[:32]

    def _encode(self, texts: list[str]) -> np.ndarray:
        vecs = self.model.encode(texts, batch_size=CFG["embeddings"]["batch_size"],
                                 normalize_embeddings=True, show_progress_bar=len(texts) > 64)
        return np.asarray(vecs, dtype=np.float32)

    def passages(self, texts: list[str]) -> tuple[np.ndarray, int]:
        """→ (вектори, скільки порахували заново)."""
        prefixed = [CFG["embeddings"]["passage_prefix"] + t for t in texts]
        keys = [self._key(t) for t in prefixed]
        missing = [i for i, k in enumerate(keys) if k not in self._cache]
        if missing:
            new = self._encode([prefixed[i] for i in missing])
            for i, v in zip(missing, new):
                self._cache[keys[i]] = v
            self.cache_path.parent.mkdir(parents=True, exist_ok=True)
            np.savez(self.cache_path, **self._cache)
        return np.stack([self._cache[k] for k in keys]), len(missing)

    def query(self, text: str) -> np.ndarray:
        return self._encode([CFG["embeddings"]["query_prefix"] + text])[0]


def qdrant_client():
    from qdrant_client import QdrantClient
    return QdrantClient(path=str(ROOT / CFG["paths"]["index_dir"] / "qdrant"))


def build(kind: str = "structure") -> dict:
    from qdrant_client.models import Distance, PayloadSchemaType, PointStruct, VectorParams

    chunks = load_chunks(kind)
    t0 = time.perf_counter()
    emb = Embedder()
    vecs, computed = emb.passages([c["embed_text"] for c in chunks])
    t_embed = time.perf_counter() - t0

    client = qdrant_client()
    name = collection_name(kind)
    if client.collection_exists(name):
        client.delete_collection(name)
    client.create_collection(name, vectors_config=VectorParams(size=vecs.shape[1], distance=Distance.COSINE))
    points = [PointStruct(id=i, vector=v.tolist(), payload={k: c.get(k) for k in PAYLOAD_FIELDS})
              for i, (c, v) in enumerate(zip(chunks, vecs))]
    for i in range(0, len(points), 256):
        client.upsert(name, points=points[i:i + 256])
    for field in ("doc_type", "format", "lang", "doc_id"):           # індекси для фільтрів за метаданими
        try:
            client.create_payload_index(name, field, PayloadSchemaType.KEYWORD)
        except Exception:  # noqa: BLE001 — локальний режим може не підтримувати payload-індекси; фільтри працюють і без них
            pass
    client.close()
    info = {"collection": name, "chunks": len(chunks), "dim": int(vecs.shape[1]), "model": emb.model_name,
            "embedded_now": computed, "from_cache": len(chunks) - computed, "embed_seconds": round(t_embed, 1)}
    print(info)
    return info


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--chunks", choices=["structure", "fixed"], default="structure")
    build(ap.parse_args().chunks)


if __name__ == "__main__":
    main()
