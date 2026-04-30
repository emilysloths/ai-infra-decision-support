from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass
class DocumentChunk:
    source: str
    text: str


def load_text_documents(data_dir: Path) -> list[DocumentChunk]:
    documents: list[DocumentChunk] = []
    for path in sorted(data_dir.glob("*.txt")):
        text = path.read_text(encoding="utf-8").strip()
        documents.extend(chunk_text(path.name, text))
    return documents


def chunk_text(source: str, text: str, chunk_size: int = 450) -> list[DocumentChunk]:
    normalized_lines = [line.strip() for line in text.splitlines() if line.strip()]
    normalized = "\n".join(normalized_lines)
    if not normalized:
        return []

    chunks: list[DocumentChunk] = []
    start = 0
    while start < len(normalized):
        end = min(start + chunk_size, len(normalized))
        if end < len(normalized):
            split_at = normalized.rfind("\n", start, end)
            if split_at > start:
                end = split_at
        chunks.append(DocumentChunk(source=source, text=normalized[start:end].strip()))
        start = end + 1
    return chunks
