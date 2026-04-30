from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .ingest import DocumentChunk, load_text_documents

try:
    import chromadb
    from sentence_transformers import SentenceTransformer
except ImportError:  # pragma: no cover - fallback path for environments without optional deps
    chromadb = None
    SentenceTransformer = None

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


@dataclass
class RetrievedChunk:
    source: str
    text: str
    score: float


class Retriever:
    def __init__(self, chunks: list[DocumentChunk]) -> None:
        self.chunks = chunks
        self.backend = "tfidf"
        self.vectorizer = None
        self.matrix = None
        self.collection = None
        self.embedding_model = None

        if chromadb is not None and SentenceTransformer is not None:
            self.backend = "chroma"
            self.embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
            client = chromadb.Client()
            self.collection = client.create_collection(name="planning_documents")
            embeddings = self.embedding_model.encode(
                [chunk.text for chunk in chunks], normalize_embeddings=True
            ).tolist()
            self.collection.add(
                ids=[str(i) for i in range(len(chunks))],
                documents=[chunk.text for chunk in chunks],
                metadatas=[{"source": chunk.source} for chunk in chunks],
                embeddings=embeddings,
            )
        else:
            self.vectorizer = TfidfVectorizer(stop_words="english")
            self.matrix = self.vectorizer.fit_transform([chunk.text for chunk in chunks])

    @classmethod
    def from_directory(cls, data_dir: Path) -> "Retriever":
        chunks = load_text_documents(data_dir)
        if not chunks:
            raise ValueError(
                f"No .txt documents found in {data_dir}. Add planning documents to continue."
            )
        return cls(chunks)

    def search(self, query: str, top_k: int = 4) -> list[RetrievedChunk]:
        if self.backend == "chroma":
            return self._search_chroma(query, top_k=top_k)
        return self._search_tfidf(query, top_k=top_k)

    def _search_chroma(self, query: str, top_k: int = 4) -> list[RetrievedChunk]:
        assert self.collection is not None
        assert self.embedding_model is not None
        query_embedding = self.embedding_model.encode(
            [query], normalize_embeddings=True
        ).tolist()
        response = self.collection.query(
            query_embeddings=query_embedding,
            n_results=top_k,
        )

        documents = response.get("documents", [[]])[0]
        metadatas = response.get("metadatas", [[]])[0]
        distances = response.get("distances", [[]])[0]

        results: list[RetrievedChunk] = []
        for text, metadata, distance in zip(documents, metadatas, distances):
            score = 1.0 / (1.0 + float(distance))
            results.append(
                RetrievedChunk(
                    source=metadata["source"],
                    text=text,
                    score=score,
                )
            )
        return results

    def _search_tfidf(self, query: str, top_k: int = 4) -> list[RetrievedChunk]:
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
                    source=chunk.source,
                    text=chunk.text,
                    score=float(scores[idx]),
                )
            )
        return results
