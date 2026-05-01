# Milestone 2 Checklist

This checklist breaks Milestone 2 into implementation tasks and tracks the current completion state.

## 1. Ingestion and document handling

- [x] Support `.txt` and `.md` ingestion
- [x] Add optional `.pdf` ingestion
- [x] Add optional `.docx` ingestion
- [x] Preserve source metadata for each document
- [x] Preserve chunk-level metadata for retrieval
- [x] Add file-level validation and ingestion error reporting

## 2. Retrieval and indexing

- [x] Refactor retrieval around structured document chunks
- [x] Add a persistent Chroma storage path
- [x] Keep a TF-IDF fallback for lightweight environments
- [x] Add configurable chunk size and overlap
- [x] Add top-k and backend configuration through app settings
- [x] Add index reuse without rebuilding on every startup

## 3. Answer generation and reasoning

- [x] Add structured answer synthesis beyond the current rule-based layer
- [x] Support explicit citations in the final recommendation body
- [x] Add user-visible confidence or evidence quality indicators

## 4. Decision support

- [x] Move decision weights into configuration
- [x] Allow interactive tuning of weights from the UI
- [x] Add side-by-side comparison between candidate options

## 5. Evaluation

- [x] Keep repeatable smoke evaluation cases
- [x] Add retrieval-specific evaluation metrics
- [x] Add benchmark question sets from planning and policy documents
- [x] Export evaluation results to a file

## 6. User interface

- [x] Add document upload support in Streamlit
- [x] Expose backend and retrieval settings in the UI
- [x] Show richer metadata with each evidence item
- [x] Display indexed document counts and status

## 7. Project completion targets

- [x] Ingest planning and policy documents from the local corpus or uploads
- [x] Persist a reusable semantic index
- [x] Produce structured answers with citations
- [x] Support configurable decision criteria
- [x] Expand validation coverage
- [x] Document setup, architecture, and operational behavior
