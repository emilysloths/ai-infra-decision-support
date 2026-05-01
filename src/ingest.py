"""Document ingestion and chunking for local planning corpora."""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
from pathlib import Path
from typing import Any

from .config import RetrievalConfig

try:
    from docx import Document as DocxDocument
except ImportError:  # pragma: no cover - optional dependency
    DocxDocument = None

try:
    from pypdf import PdfReader
except ImportError:  # pragma: no cover - optional dependency
    PdfReader = None


@dataclass
class DocumentRecord:
    """A loaded document before it has been split into retrieval chunks."""

    source: str
    path: str
    doc_type: str
    text: str
    metadata: dict[str, Any]


@dataclass
class DocumentChunk:
    """A retrieval unit enriched with source and chunk metadata."""

    chunk_id: str
    source: str
    path: str
    doc_type: str
    text: str
    metadata: dict[str, Any]


@dataclass
class IngestionIssue:
    """Represents a recoverable warning or a blocking ingestion error."""

    path: str
    severity: str
    message: str


@dataclass
class IngestionReport:
    """Collects chunks together with ingestion diagnostics and source fingerprints."""

    chunks: list[DocumentChunk]
    documents: list[DocumentRecord]
    issues: list[IngestionIssue] = field(default_factory=list)
    fingerprint: str = ""

    @property
    def warnings(self) -> list[IngestionIssue]:
        return [issue for issue in self.issues if issue.severity == "warning"]

    @property
    def errors(self) -> list[IngestionIssue]:
        return [issue for issue in self.issues if issue.severity == "error"]


def load_documents(data_dir: Path, config: RetrievalConfig | None = None) -> IngestionReport:
    """Load all supported documents from a directory and return chunked retrieval units."""

    config = config or RetrievalConfig()
    records: list[DocumentRecord] = []
    issues: list[IngestionIssue] = []
    fingerprints: list[str] = []
    supported_suffixes = {".txt", ".md", ".pdf", ".docx"}

    if not data_dir.exists():
        raise ValueError(f"Data directory does not exist: {data_dir}")

    for path in sorted(data_dir.iterdir()):
        if not path.is_file():
            continue
        if path.suffix.lower() not in supported_suffixes:
            issues.append(
                IngestionIssue(
                    path=str(path),
                    severity="warning",
                    message="Skipped unsupported document type.",
                )
            )
            continue

        try:
            record = load_document(path)
        except Exception as exc:  # pragma: no cover - exercised through runtime errors
            issues.append(
                IngestionIssue(
                    path=str(path),
                    severity="error",
                    message=str(exc),
                )
            )
            continue

        if not record.text.strip():
            issues.append(
                IngestionIssue(
                    path=str(path),
                    severity="warning",
                    message="Document was empty after loading.",
                )
            )
            continue

        records.append(record)
        fingerprints.append(_fingerprint_file(path))

    chunks: list[DocumentChunk] = []
    for record in records:
        chunks.extend(
            chunk_text(
                record,
                chunk_size=config.chunk_size,
                overlap=config.chunk_overlap,
            )
        )

    return IngestionReport(
        chunks=chunks,
        documents=records,
        issues=issues,
        fingerprint=_fingerprint_payload(fingerprints, config),
    )


def load_document(path: Path) -> DocumentRecord:
    """Dispatch document loading based on file extension."""

    suffix = path.suffix.lower()
    loaders = {
        ".txt": _load_text_file,
        ".md": _load_text_file,
        ".pdf": _load_pdf_file,
        ".docx": _load_docx_file,
    }
    if suffix not in loaders:
        raise ValueError(f"Unsupported document type: {suffix}")
    return loaders[suffix](path)


def chunk_text(
    record: DocumentRecord, chunk_size: int = 450, overlap: int = 80
) -> list[DocumentChunk]:
    """Split text into chunks while trying to keep overlap boundaries human-readable."""

    normalized_lines = [line.strip() for line in record.text.splitlines() if line.strip()]
    normalized = "\n".join(normalized_lines)
    if not normalized:
        return []

    chunks: list[DocumentChunk] = []
    start = 0
    chunk_index = 0
    shared_metadata = {**record.metadata, **_extract_structured_metadata(record.text)}
    while start < len(normalized):
        end = min(start + chunk_size, len(normalized))
        if end < len(normalized):
            split_at = normalized.rfind("\n", start, end)
            if split_at > start:
                end = split_at

        chunk_text_value = normalized[start:end].strip()
        if chunk_text_value:
            chunks.append(
                DocumentChunk(
                    chunk_id=f"{record.source}-{chunk_index}",
                    source=record.source,
                    path=record.path,
                    doc_type=record.doc_type,
                    text=chunk_text_value,
                    metadata={**shared_metadata, "chunk_index": chunk_index},
                )
            )
            chunk_index += 1

        if end >= len(normalized):
            break

        # Advance to the next clean token boundary instead of starting mid-word.
        next_start = max(end - overlap, start + 1)
        while next_start < len(normalized) and normalized[next_start] not in {" ", "\n"}:
            next_start += 1
        while next_start < len(normalized) and normalized[next_start] in {" ", "\n"}:
            next_start += 1
        start = next_start
    return chunks


def _load_text_file(path: Path) -> DocumentRecord:
    """Load plain text and Markdown documents with simple title metadata."""

    return DocumentRecord(
        source=path.name,
        path=str(path),
        doc_type=path.suffix.lower().lstrip("."),
        text=path.read_text(encoding="utf-8").strip(),
        metadata={"title": path.stem},
    )


def _load_pdf_file(path: Path) -> DocumentRecord:
    """Extract PDF page text if pypdf is available."""

    if PdfReader is None:
        raise ImportError(
            "PDF ingestion requires pypdf. Install dependencies from requirements.txt."
        )
    reader = PdfReader(str(path))
    page_text = []
    for page_number, page in enumerate(reader.pages, start=1):
        page_text.append(f"[Page {page_number}]\n{page.extract_text() or ''}")
    return DocumentRecord(
        source=path.name,
        path=str(path),
        doc_type="pdf",
        text="\n".join(page_text).strip(),
        metadata={"title": path.stem, "page_count": len(reader.pages)},
    )


def _load_docx_file(path: Path) -> DocumentRecord:
    """Extract DOCX paragraph text if python-docx is available."""

    if DocxDocument is None:
        raise ImportError(
            "DOCX ingestion requires python-docx. Install dependencies from requirements.txt."
        )
    document = DocxDocument(str(path))
    paragraphs = [paragraph.text for paragraph in document.paragraphs if paragraph.text.strip()]
    return DocumentRecord(
        source=path.name,
        path=str(path),
        doc_type="docx",
        text="\n".join(paragraphs).strip(),
        metadata={"title": path.stem, "paragraph_count": len(paragraphs)},
    )


def _extract_structured_metadata(text: str) -> dict[str, Any]:
    """Promote simple `Key: Value` fields into chunk metadata for later scoring."""

    metadata: dict[str, Any] = {}
    for line in text.splitlines():
        line = line.strip()
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        key = key.strip().lower().replace(" ", "_")
        value = value.strip()
        try:
            metadata[key] = float(value)
        except ValueError:
            metadata[key] = value
    return metadata


def _fingerprint_file(path: Path) -> str:
    """Fingerprint source files so vector indexes can be reused when nothing changed."""

    stat = path.stat()
    return f"{path.name}:{stat.st_size}:{stat.st_mtime_ns}"


def _fingerprint_payload(fingerprints: list[str], config: RetrievalConfig) -> str:
    """Combine source file signatures with chunking settings."""

    payload = "|".join(sorted(fingerprints) + [str(config.chunk_size), str(config.chunk_overlap)])
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()
