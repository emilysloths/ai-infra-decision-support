# AI Infrastructure Planning Assistant

An AI-assisted decision support system for infrastructure planning queries. The project combines retrieval-augmented generation patterns, local semantic search, and a lightweight multi-criteria scoring layer to answer questions using planning and policy context stored in local documents.

## Overview

The assistant is designed to:

1. Load planning documents from a local `data/` directory
2. Split documents into retrievable chunks
3. Build a local retrieval index
4. Accept a user question
5. Retrieve relevant policy and project context
6. Produce a recommendation, supporting reasoning, and evidence citations
7. Rank candidate options using a basic multi-criteria decision scorecard

## Architecture

The system is organized as a small pipeline:

1. Ingestion
   Raw text documents are loaded from the local data directory and split into chunks.

2. Retrieval
   The assistant indexes chunks for semantic search. When optional dependencies are installed, it uses Sentence Transformers embeddings with Chroma vector search. Otherwise it falls back to TF-IDF with cosine similarity.

3. Decision Support
   Retrieved chunks are parsed for structured planning signals such as resilience, cyber maturity, cost efficiency, and implementation readiness. Those signals are combined with query-aware weighting and retrieval relevance to produce a ranked recommendation.

4. Interfaces
   The project provides both a command-line interface and a Streamlit UI for interactive use.

5. Evaluation
   Smoke evaluation cases and unit tests validate retrieval behavior and recommendation consistency for milestone 1.

## Project structure

```text
ai_infra_assistant/
  app.py
  requirements.txt
  README.md
  streamlit_app.py
  run_eval.py
  data/
  src/
  tests/
```

## Components

### [app.py](/C:/Users/amyme/OneDrive/Documents/New%20project/ai_infra_assistant/app.py)

Command-line entry point. Accepts a planning question and prints:

- recommendation
- decision scorecard
- reasoning summary
- retrieved evidence

### [streamlit_app.py](/C:/Users/amyme/OneDrive/Documents/New%20project/ai_infra_assistant/streamlit_app.py)

Interactive UI for submitting questions and viewing the assistant output in a browser.

### [run_eval.py](/C:/Users/amyme/OneDrive/Documents/New%20project/ai_infra_assistant/run_eval.py)

Runs milestone 1 smoke evaluation cases against the assistant and reports pass/fail results.

### [src/ingest.py](/C:/Users/amyme/OneDrive/Documents/New%20project/ai_infra_assistant/src/ingest.py)

Loads `.txt` documents and chunks them into retrieval units while preserving source references.

### [src/retrieval.py](/C:/Users/amyme/OneDrive/Documents/New%20project/ai_infra_assistant/src/retrieval.py)

Implements retrieval over the local document collection.

- Primary path: Sentence Transformers + Chroma
- Fallback path: TF-IDF + cosine similarity

The fallback path keeps the project runnable in environments where embedding dependencies are unavailable.

### [src/decision.py](/C:/Users/amyme/OneDrive/Documents/New%20project/ai_infra_assistant/src/decision.py)

Builds the final recommendation from retrieved context. This layer:

- extracts site-level attributes from retrieved chunks
- applies weighted scoring across planning criteria
- adjusts weights based on the question
- adds retrieval and phrase-match relevance bonuses
- returns the final recommendation and scorecard

### [src/agent.py](/C:/Users/amyme/OneDrive/Documents/New%20project/ai_infra_assistant/src/agent.py)

Top-level orchestration for the assistant. It connects retrieval and decision support into a single `answer()` flow.

### [src/evaluate.py](/C:/Users/amyme/OneDrive/Documents/New%20project/ai_infra_assistant/src/evaluate.py)

Defines evaluation cases and helper functions for repeatable milestone 1 validation.

### [tests](/C:/Users/amyme/OneDrive/Documents/New%20project/ai_infra_assistant/tests)

Contains `unittest`-based smoke tests for retrieval and evaluation behavior.

## Technology stack

- Python
- Sentence Transformers for local embeddings
- Chroma for vector storage and semantic retrieval
- scikit-learn for TF-IDF fallback retrieval
- Streamlit for the interactive interface
- `unittest` for lightweight validation

## Why these technologies are used

- Sentence Transformers provides a practical local embedding workflow for semantic similarity without requiring a hosted vectorization service.
- Chroma offers a simple vector database interface that is easy to integrate for local development and experimentation.
- TF-IDF fallback retrieval keeps the application usable even when the semantic search stack is not installed.
- Streamlit provides a fast path to a usable interactive interface without introducing backend/frontend boilerplate.
- `unittest` keeps milestone 1 validation simple and portable across Python environments.

## Data model

The sample documents in `data/` represent site and planning context. Each document can contain:

- site name
- document type
- region
- policy focus
- resilience score
- cyber maturity score
- cost efficiency score
- implementation readiness score
- strengths, risks, and notes

These fields are intentionally lightweight so the project can demonstrate retrieval and ranking behavior without requiring a large structured database.

## Running the project

```bash
pip install -r requirements.txt
python app.py --question "Which site is best for a resilient regional operations center?"
streamlit run streamlit_app.py
```

## Validation

Run the milestone 1 checks with:

```bash
python run_eval.py
python -m unittest discover -s tests -v
```

## Current limitations

- The current data source is a small local text corpus
- The recommendation layer is rule-based rather than LLM-generated
- PDF and DOCX ingestion are not implemented yet
- Persistent vector storage is not configured for production use
- The scoring model is intentionally simple and designed for milestone 1

## Next steps

- Add PDF and DOCX ingestion
- Add a richer answer synthesis layer
- Expand the evaluation suite with benchmark question sets
- Support configurable decision weights
- Persist vector indexes between runs
