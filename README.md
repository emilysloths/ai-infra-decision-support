# AI Infrastructure Planning Assistant

An AI-assisted decision support system for infrastructure planning queries. The project combines retrieval-augmented generation patterns, local semantic search, configurable multi-criteria scoring, and benchmark evaluation to answer questions using local planning and policy documents.

## Overview

The assistant is designed to:

1. Load planning and policy documents from a local corpus or uploaded files
2. Split source documents into retrievable chunks with metadata
3. Build or reuse a local retrieval index
4. Accept a user question
5. Retrieve the most relevant evidence
6. Produce a structured recommendation with citations, confidence, and tradeoffs
7. Rank candidate options using configurable decision criteria
8. Evaluate answer and retrieval quality with repeatable benchmarks

## Architecture

The system is organized as a small pipeline:

1. Ingestion
   Documents are loaded from `.txt`, `.md`, `.pdf`, and `.docx` sources. The ingestion layer validates files, captures metadata, and produces chunked retrieval units.

2. Retrieval
   The assistant supports two retrieval modes:
   - Sentence Transformers + Chroma for persistent semantic search
   - TF-IDF + cosine similarity as a lightweight fallback

3. Decision Support
   Retrieved evidence is parsed for planning signals such as resilience, cyber maturity, cost efficiency, and implementation readiness. Those signals are combined with configurable base weights, query-sensitive adjustments, and retrieval relevance bonuses.

4. Answer Synthesis
   The final answer includes an executive summary, a recommendation body with citations, a scorecard, tradeoff notes, and a confidence estimate derived from evidence quality.

5. Interfaces
   The project includes a command-line interface, a Streamlit application, and an evaluation runner for automated validation.

## Project structure

```text
ai_infra_assistant/
  app.py
  requirements.txt
  README.md
  streamlit_app.py
  run_eval.py
  benchmarks/
  data/
  docs/
  src/
  tests/
```

## Components

### [app.py](/C:/Users/amyme/OneDrive/Documents/New%20project/ai_infra_assistant/app.py)

Command-line interface for querying the assistant with configurable retrieval and decision settings.

### [streamlit_app.py](/C:/Users/amyme/OneDrive/Documents/New%20project/ai_infra_assistant/streamlit_app.py)

Interactive web interface with:

- document upload
- backend selection
- chunking controls
- weight tuning
- evidence and comparison views
- benchmark execution

### [run_eval.py](/C:/Users/amyme/OneDrive/Documents/New%20project/ai_infra_assistant/run_eval.py)

Benchmark runner that evaluates retrieval and recommendation behavior and exports results as JSON or CSV.

### [src/config.py](/C:/Users/amyme/OneDrive/Documents/New%20project/ai_infra_assistant/src/config.py)

Defines shared retrieval and decision configuration objects.

### [src/ingest.py](/C:/Users/amyme/OneDrive/Documents/New%20project/ai_infra_assistant/src/ingest.py)

Loads documents, validates source files, captures metadata, chunks content, and produces ingestion reports.

### [src/retrieval.py](/C:/Users/amyme/OneDrive/Documents/New%20project/ai_infra_assistant/src/retrieval.py)

Builds either:

- a persistent Chroma collection backed by Sentence Transformers embeddings, or
- a TF-IDF fallback index

It also exposes corpus statistics and supports index reuse through a corpus fingerprint manifest.

### [src/decision.py](/C:/Users/amyme/OneDrive/Documents/New%20project/ai_infra_assistant/src/decision.py)

Ranks candidate sites by combining:

- configurable planning criteria weights
- query-sensitive weight adjustments
- lexical phrase overlap
- domain-specific evidence bonuses
- retrieval score contribution

### [src/synthesis.py](/C:/Users/amyme/OneDrive/Documents/New%20project/ai_infra_assistant/src/synthesis.py)

Builds the final structured answer, including executive summary, citations, tradeoffs, and confidence indicators.

### [src/evaluate.py](/C:/Users/amyme/OneDrive/Documents/New%20project/ai_infra_assistant/src/evaluate.py)

Loads benchmark cases, computes answer and retrieval metrics, summarizes results, and exports reports.

### [tests](/C:/Users/amyme/OneDrive/Documents/New%20project/ai_infra_assistant/tests)

Contains ingestion, retrieval, evaluation, and end-to-end validation tests.

## Technology stack

- Python
- Sentence Transformers for local embeddings
- Chroma for persistent vector storage and semantic retrieval
- scikit-learn for TF-IDF fallback retrieval
- Streamlit for the interactive interface
- `pypdf` for PDF ingestion
- `python-docx` for DOCX ingestion
- `unittest` for regression coverage

## Why these technologies are used

- Sentence Transformers provides a local semantic embedding workflow without depending on a hosted embedding service.
- Chroma supports simple persistent vector search suitable for local experimentation and lightweight deployments.
- TF-IDF fallback retrieval keeps the assistant usable when the embedding stack is unavailable.
- Streamlit provides a direct path to an interactive interface with minimal web application overhead.
- `pypdf` and `python-docx` extend the assistant beyond plain text sources into common planning document formats.
- `unittest` keeps validation portable across standard Python environments.

## Data model

The local corpus may include site-level records, planning memoranda, and policy guidance. Documents can provide:

- site name
- document type
- region
- policy focus
- resilience score
- cyber maturity score
- cost efficiency score
- implementation readiness score
- strengths, risks, and notes

The assistant keeps this information lightweight on purpose so the same retrieval and decision pipeline can work across both structured site profiles and less structured policy documents.

## Running the project

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the CLI:

```bash
python app.py --question "Which site is best for a resilient regional operations center?"
```

Run the Streamlit app:

```bash
streamlit run streamlit_app.py
```

## Evaluation

Run the benchmark suite:

```bash
python run_eval.py --output artifacts/eval_results.json
```

This produces:

- case-level pass/fail results
- retrieval precision@k
- retrieval recall@k
- summary metrics
- exportable JSON or CSV reports

## Validation

Run the automated tests with:

```bash
python -m unittest discover -s tests -v
```

## Current behavior

Milestone 2 currently supports:

- persistent semantic indexing when the embedding stack is available
- lexical fallback retrieval
- configurable chunking and top-k retrieval
- configurable decision weights
- structured answers with citations and confidence
- document upload in the Streamlit UI
- benchmark evaluation and report export

## Limitations

- The answer synthesis layer is template-based rather than LLM-generated
- The current corpus is still small and local
- Chroma persistence is optimized for local workflows, not distributed deployment
- Evaluation is benchmark-driven and not yet tied to human annotation workflows
