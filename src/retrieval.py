"""Retrieval backends for semantic and lexical search over local planning documents."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import re
from typing import Any

from .config import RetrievalConfig
from .ingest import DocumentChunk, IngestionReport, load_documents

try:
    import chromadb
    from sentence_transformers import SentenceTransformer
except ImportError:  # pragma: no cover - optional semantic retrieval stack
    chromadb = None
    SentenceTransformer = None

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


@dataclass
class RetrievedChunk:
    """A search result enriched with retrieval metadata."""

    chunk_id: str
    source: str
    path: str
    doc_type: str
    text: str
    score: float
    metadata: dict[str, object]


class Retriever:
    """Hybrid retriever that prefers Chroma embeddings and falls back to TF-IDF."""

    def __init__(
        self,
        chunks: list[DocumentChunk],
        config: RetrievalConfig | None = None,
        collection_name: str = "planning_documents",
        corpus_fingerprint: str = "",
    ) -> None:
        self.chunks = chunks
        self.config = config or RetrievalConfig()
        self.backend = "tfidf"
        self.vectorizer: TfidfVectorizer | None = None
        self.matrix = None
        self.collection = None
        self.embedding_model = None
        self.collection_name = collection_name
        self.persist_directory = self.config.vector_store_dir
        self.corpus_fingerprint = corpus_fingerprint

        backend = self.config.backend.lower()
        if backend not in {"auto", "chroma", "tfidf"}:
            raise ValueError("backend must be one of: auto, chroma, tfidf")

        if backend in {"auto", "chroma"} and chromadb is not None and SentenceTransformer is not None:
            self.backend = "chroma"
            self._initialize_chroma()
        elif backend == "chroma":
            raise ImportError(
                "Chroma backend requested but sentence-transformers or chromadb is unavailable."
            )
        else:
            self.backend = "tfidf"
            self._initialize_tfidf()

    @classmethod
    def from_directory(
        cls, data_dir: Path, config: RetrievalConfig | None = None
    ) -> tuple["Retriever", IngestionReport]:
        """Load a corpus and build a retriever in one step."""

        config = config or RetrievalConfig()
        report = load_documents(data_dir, config=config)
        if not report.chunks:
            raise ValueError(
                f"No supported documents found in {data_dir}. Add .txt, .md, .pdf, or .docx files."
            )
        collection_name = _collection_name_for_path(data_dir)
        return (
            cls(
                report.chunks,
                config=config,
                collection_name=collection_name,
                corpus_fingerprint=report.fingerprint,
            ),
            report,
        )

    def search(self, query: str, top_k: int | None = None) -> list[RetrievedChunk]:
        """Search the active backend for the top evidence chunks."""

        top_k = top_k or self.config.top_k
        if self.backend == "chroma":
            return self._search_chroma(query, top_k=top_k)
        return self._search_tfidf(query, top_k=top_k)

    def corpus_stats(self) -> dict[str, Any]:
        """Expose corpus details for the CLI, UI, and evaluation layers."""

        return {
            "backend": self.backend,
            "chunk_count": len(self.chunks),
            "document_count": len({chunk.source for chunk in self.chunks}),
            "collection_name": self.collection_name,
            "persist_directory": self.persist_directory,
        }

    def _initialize_chroma(self) -> None:
        """Build or reuse a persistent Chroma collection when the corpus is unchanged."""

        assert chromadb is not None
        assert SentenceTransformer is not None

        persist_path = Path(self.persist_directory)
        persist_path.mkdir(parents=True, exist_ok=True)

        self.embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
        client = chromadb.PersistentClient(path=self.persist_directory)
        manifest_path = persist_path / f"{self.collection_name}_manifest.json"
        should_rebuild = True

        if self.config.reuse_index and manifest_path.exists():
            try:
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                if manifest.get("fingerprint") == self.corpus_fingerprint:
                    self.collection = client.get_collection(name=self.collection_name)
                    should_rebuild = False
            except Exception:
                should_rebuild = True

        if should_rebuild:
            try:
                client.delete_collection(name=self.collection_name)
            except Exception:
                pass

            self.collection = client.create_collection(name=self.collection_name)
            embeddings = self.embedding_model.encode(
                [chunk.text for chunk in self.chunks], normalize_embeddings=True
            ).tolist()
            self.collection.add(
                ids=[chunk.chunk_id for chunk in self.chunks],
                documents=[chunk.text for chunk in self.chunks],
                metadatas=[
                    {
                        "source": chunk.source,
                        "path": chunk.path,
                        "doc_type": chunk.doc_type,
                        **_stringify_metadata(chunk.metadata),
                    }
                    for chunk in self.chunks
                ],
                embeddings=embeddings,
            )
            manifest_path.write_text(
                json.dumps(
                    {
                        "fingerprint": self.corpus_fingerprint,
                        "chunk_count": len(self.chunks),
                        "collection_name": self.collection_name,
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )

    def _initialize_tfidf(self) -> None:
        """Build a lexical fallback index for environments without the embedding stack."""

        self.vectorizer = TfidfVectorizer(stop_words="english")
        self.matrix = self.vectorizer.fit_transform([chunk.text for chunk in self.chunks])

    def _search_chroma(self, query: str, top_k: int = 4) -> list[RetrievedChunk]:
        """Query the semantic vector store and convert results into a consistent shape."""

        assert self.collection is not None
        assert self.embedding_model is not None

        query_embedding = self.embedding_model.encode(
            [query], normalize_embeddings=True
        ).tolist()
        response = self.collection.query(query_embeddings=query_embedding, n_results=top_k)

        ids = response.get("ids", [[]])[0]
        documents = response.get("documents", [[]])[0]
        metadatas = response.get("metadatas", [[]])[0]
        distances = response.get("distances", [[]])[0]

        results: list[RetrievedChunk] = []
        for chunk_id, text, metadata, distance in zip(ids, documents, metadatas, distances):
            score = 1.0 / (1.0 + float(distance))
            results.append(
                RetrievedChunk(
                    chunk_id=chunk_id,
                    source=metadata["source"],
                    path=metadata.get("path", metadata["source"]),
                    doc_type=metadata.get("doc_type", "unknown"),
                    text=text,
                    score=score,
                    metadata=metadata,
                )
            )
        return results

    def _search_tfidf(self, query: str, top_k: int = 4) -> list[RetrievedChunk]:
        """Use lexical similarity when semantic retrieval is unavailable or disabled."""

        assert self.vectorizer is not None
        assert self.matrix is not None

        query_vec = self.vectorizer.transform([query])
        scores = cosine_similarity(query_vec, self.matrix).ravel()
        top_indices = scores.argsort()[::-1][:top_k]

        results: list[RetrievedChunk] = []
        for idx in top_indices:
            if scores[idx] <= 0:
                continue
            chunk = self.chunks[idx]
            results.append(
                RetrievedChunk(
                    chunk_id=chunk.chunk_id,
                    source=chunk.source,
                    path=chunk.path,
                    doc_type=chunk.doc_type,
                    text=chunk.text,
                    score=float(scores[idx]),
                    metadata=chunk.metadata,
                )
            )
        return results


def _collection_name_for_path(data_dir: Path) -> str:
    """Generate a valid Chroma collection name from the corpus location."""

    normalized = re.sub(r"[^a-zA-Z0-9]+", "_", str(data_dir.resolve()))
    return f"planning_{normalized.lower()}"[:60]


def _stringify_metadata(metadata: dict[str, object]) -> dict[str, str]:
    """Chroma metadata is string-oriented, so normalize values before persistence."""

    return {key: str(value) for key, value in metadata.items()}
