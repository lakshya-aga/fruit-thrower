"""
vector_store.py — Embed and persist parsed code units using ChromaDB.

Embeddings are handled by ChromaDB's default embedding function (sentence-transformer),
with persistence on disk. Supports upsert (idempotent re-indexing) and semantic search.
"""
import json
import logging
import os
from typing import Optional

logger = logging.getLogger(__name__)

try:
    import chromadb
    from chromadb.config import Settings
    _CHROMA_AVAILABLE = True
except ImportError:
    _CHROMA_AVAILABLE = False

from parser import ParsedUnit

COLLECTION_NAME = "code_units"


class CodeVectorStore:
    """
    Persistent vector store for code units.

    Uses ChromaDB's default embedding function for semantic retrieval over
    repository functions/classes/modules.

    Example:
        >>> store = CodeVectorStore(persist_dir="./.code_index")
        >>> store.upsert(units)
        >>> results = store.search("cusum filter for financial time series")
    """

    def __init__(self, persist_dir: str = "./.code_index", api_key: Optional[str] = None):
        """
        Initialise the vector store.

        :param persist_dir: (str) Directory where ChromaDB persists data.
        :param api_key: unused (kept for backward compatibility).
        """
        if not _CHROMA_AVAILABLE:
            raise ImportError("pip install chromadb")

        self._db = chromadb.PersistentClient(
            path=persist_dir,
            settings=Settings(anonymized_telemetry=False),
        )
        self._collection = self._db.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )

    def _sanitize_metadata_str(self, value: str) -> str:
        """Sanitize string values for safe storage in ChromaDB metadata."""
        if not isinstance(value, str):
            return str(value)
        # Strip null bytes and control chars that could cause issues
        return value.replace("\x00", "")

    def upsert(self, units: list[ParsedUnit], batch_size: int = 50) -> int:
        """
        Embed and upsert code units into the store (idempotent).

        :param units: (list[ParsedUnit]) Units to index.
        :param batch_size: (int) Number of units to embed per API call.
        :return: (int) Number of units upserted.
        """
        # Deduplicate by ID — last occurrence wins (consistent with upsert semantics)
        seen: dict[str, ParsedUnit] = {}
        for u in units:
            seen[u.id] = u
        units = list(seen.values())

        total = 0
        for i in range(0, len(units), batch_size):
            batch = units[i : i + batch_size]
            texts = [u.to_embed_text() for u in batch]
            try:
                self._collection.upsert(
                    ids=[u.id for u in batch],
                    documents=texts,
                    metadatas=[{
                        "kind": self._sanitize_metadata_str(u.kind),
                        "name": self._sanitize_metadata_str(u.name),
                        "module": self._sanitize_metadata_str(u.module),
                        "file_path": self._sanitize_metadata_str(u.file_path),
                        "line_start": u.line_start,
                        "line_end": u.line_end,
                        "signature": self._sanitize_metadata_str(u.signature),
                        "docstring": self._sanitize_metadata_str(u.docstring or ""),
                        "parent": self._sanitize_metadata_str(u.parent or ""),
                        "source": self._sanitize_metadata_str(u.source[:2000]),
                    } for u in batch],
                )
            except Exception:
                logger.error("Failed to upsert batch at offset %d", i, exc_info=True)
                continue
            total += len(batch)
            logger.info("Indexed %d/%d units", total, len(units))
        return total

    def search(
        self,
        query: str,
        n_results: int = 5,
        kind_filter: Optional[str] = None,
        module_filter: Optional[str] = None,
    ) -> list[dict]:
        """
        Semantic search over indexed code units.

        :param query: (str) Natural language or code query.
        :param n_results: (int) Maximum number of results to return.
        :param kind_filter: (str | None) Filter by kind: "function", "class", "method", "module".
        :param module_filter: (str | None) Filter by module prefix, e.g. "filters".
        :return: (list[dict]) Ranked results with metadata and relevance distance.
        """
        where: dict = {}
        if kind_filter:
            where["kind"] = {"$eq": kind_filter}

        results = self._collection.query(
            query_texts=[query],
            n_results=max(n_results * 3, 15) if module_filter else n_results,
            where=where if where else None,
            include=["documents", "metadatas", "distances"],
        )

        hits = []
        for i, meta in enumerate(results["metadatas"][0]):
            if module_filter and module_filter not in meta.get("module", ""):
                continue
            hits.append({
                "score": 1 - results["distances"][0][i],  # cosine sim
                "id": results["ids"][0][i],
                **meta,
            })
            if len(hits) >= n_results:
                break
        return hits

    def count(self) -> int:
        """Return the number of indexed units."""
        return self._collection.count()

    def delete(self, unit_ids: list[str]) -> None:
        """Remove specific units from the index."""
        self._collection.delete(ids=unit_ids)

    def stats(self) -> dict:
        """Return index statistics."""
        all_meta = self._collection.get(include=["metadatas"])["metadatas"]
        kinds: dict[str, int] = {}
        modules: set[str] = set()
        for m in all_meta:
            kinds[m["kind"]] = kinds.get(m["kind"], 0) + 1
            modules.add(m["module"])
        return {
            "total_units": len(all_meta),
            "by_kind": kinds,
            "unique_modules": len(modules),
        }


# ── Fallback: simple TF-IDF store (no API needed) ─────────────────────────

class SimpleTFIDFStore:
    """
    Lightweight keyword-based code store using TF-IDF.

    Use this when the Anthropic API is not available. Persists index as JSON.
    Good for development / offline use.

    Example:
        >>> store = SimpleTFIDFStore("./.code_index_tfidf")
        >>> store.upsert(units)
        >>> results = store.search("cusum filter threshold")
    """

    def __init__(self, persist_dir: str = "./.code_index_tfidf"):
        """
        :param persist_dir: (str) Directory for JSON persistence.
        """
        import sklearn  # noqa: just check it's available
        os.makedirs(persist_dir, exist_ok=True)
        self._persist_dir = persist_dir
        self._index_path = os.path.join(persist_dir, "index.json")
        self._records: list[dict] = []
        self._vectorizer = None
        self._matrix = None
        if os.path.exists(self._index_path):
            self._load()

    def _load(self):
        with open(self._index_path, encoding="utf-8") as f:
            self._records = json.load(f)
        self._fit()

    def _fit(self):
        from sklearn.feature_extraction.text import TfidfVectorizer
        if not self._records:
            return
        self._vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=20000)
        texts = [r["embed_text"] for r in self._records]
        self._matrix = self._vectorizer.fit_transform(texts)

    def upsert(self, units: list[ParsedUnit]) -> int:
        existing_ids = {r["id"] for r in self._records}
        added = 0
        for u in units:
            record = {
                "id": u.id,
                "embed_text": u.to_embed_text(),
                "kind": u.kind,
                "name": u.name,
                "module": u.module,
                "file_path": u.file_path,
                "line_start": u.line_start,
                "line_end": u.line_end,
                "signature": u.signature,
                "docstring": u.docstring or "",
                "parent": u.parent or "",
                "source": u.source,
            }
            if u.id not in existing_ids:
                self._records.append(record)
                existing_ids.add(u.id)
                added += 1
            else:
                # Update in place
                for i, r in enumerate(self._records):
                    if r["id"] == u.id:
                        self._records[i] = record
        with open(self._index_path, "w", encoding="utf-8") as f:
            json.dump(self._records, f, indent=2)
        self._fit()
        logger.info("Indexed %d total units (+%d new)", len(self._records), added)
        return added

    def search(
        self,
        query: str,
        n_results: int = 5,
        kind_filter: Optional[str] = None,
        module_filter: Optional[str] = None,
    ) -> list[dict]:
        """
        Keyword search over indexed code units using TF-IDF cosine similarity.

        :param query: (str) Search query.
        :param n_results: (int) Number of results.
        :param kind_filter: (str | None) Filter by kind.
        :param module_filter: (str | None) Filter by module prefix.
        :return: (list[dict]) Ranked results.
        """
        import numpy as np
        from sklearn.metrics.pairwise import cosine_similarity

        if not self._vectorizer or not self._records:
            return []

        q_vec = self._vectorizer.transform([query])
        sims = cosine_similarity(q_vec, self._matrix).flatten()

        records = self._records
        if kind_filter:
            mask = [i for i, r in enumerate(records) if r["kind"] == kind_filter]
            if mask:
                records = [records[i] for i in mask]
                sims = sims[mask]

        if module_filter:
            mask = [i for i, r in enumerate(records) if module_filter in r["module"]]
            if mask:
                sims = sims[[records.index(records[i]) for i in mask]]
                records = [records[i] for i in mask]

        top_idx = np.argsort(sims)[::-1][:n_results]
        return [{"score": float(sims[i]), **records[i]} for i in top_idx if sims[i] > 0]

    def count(self) -> int:
        return len(self._records)

    def stats(self) -> dict:
        kinds: dict[str, int] = {}
        modules: set[str] = set()
        for r in self._records:
            kinds[r["kind"]] = kinds.get(r["kind"], 0) + 1
            modules.add(r["module"])
        return {"total_units": len(self._records), "by_kind": kinds, "unique_modules": len(modules)}
