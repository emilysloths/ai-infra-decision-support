"""Streamlit interface for the planning assistant."""

from __future__ import annotations

from pathlib import Path
import shutil
import tempfile

import pandas as pd
import streamlit as st

from src.agent import InfrastructureAssistant
from src.config import DecisionConfig
from src.evaluate import export_results, load_eval_cases, run_smoke_eval, summarize_results


def _build_uploaded_corpus(uploaded_files: list[object], include_sample_data: bool) -> str:
    """Create a temporary corpus directory from uploaded files plus optional sample data."""

    temp_dir = Path(tempfile.mkdtemp(prefix="infra_assistant_"))
    if include_sample_data:
        sample_dir = Path("data")
        for source_path in sample_dir.glob("*"):
            if source_path.is_file():
                shutil.copy2(source_path, temp_dir / source_path.name)

    for file in uploaded_files:
        target = temp_dir / file.name
        target.write_bytes(file.getvalue())

    return str(temp_dir)


st.set_page_config(page_title="AI Infrastructure Planning Assistant", layout="wide")
st.title("AI Infrastructure Planning Assistant")
st.write(
    "Ask infrastructure planning questions and get retrieval-grounded recommendations "
    "using local policy and project context."
)

st.sidebar.header("Retrieval Settings")
backend = st.sidebar.selectbox("Backend", ["auto", "chroma", "tfidf"], index=0)
top_k = st.sidebar.slider("Top-k evidence chunks", min_value=1, max_value=8, value=4)
chunk_size = st.sidebar.slider("Chunk size", min_value=150, max_value=1200, value=450, step=50)
chunk_overlap = st.sidebar.slider("Chunk overlap", min_value=0, max_value=300, value=80, step=10)
reuse_index = st.sidebar.checkbox("Reuse existing semantic index", value=True)

st.sidebar.header("Decision Weights")
resilience_weight = st.sidebar.slider("Resilience", 0.0, 1.0, 0.30, 0.05)
cyber_weight = st.sidebar.slider("Cyber maturity", 0.0, 1.0, 0.30, 0.05)
cost_weight = st.sidebar.slider("Cost efficiency", 0.0, 1.0, 0.20, 0.05)
readiness_weight = st.sidebar.slider("Implementation readiness", 0.0, 1.0, 0.20, 0.05)

st.sidebar.header("Corpus")
include_sample_data = st.sidebar.checkbox("Include sample corpus", value=True)
uploaded_files = st.sidebar.file_uploader(
    "Upload .txt, .md, .pdf, or .docx files",
    type=["txt", "md", "pdf", "docx"],
    accept_multiple_files=True,
)

data_dir = "data"
if uploaded_files:
    data_dir = _build_uploaded_corpus(uploaded_files, include_sample_data)
elif not include_sample_data:
    st.warning("Enable the sample corpus or upload documents before running the assistant.")

assistant = InfrastructureAssistant(
    data_dir=data_dir,
    top_k=top_k,
    retrieval_backend=backend,
    chunk_size=chunk_size,
    chunk_overlap=chunk_overlap,
    reuse_index=reuse_index,
    decision_config=DecisionConfig(
        resilience=resilience_weight,
        cyber_maturity=cyber_weight,
        cost_efficiency=cost_weight,
        implementation_readiness=readiness_weight,
    ),
)

question = st.text_input(
    "Question",
    value="Which site is best for a resilient regional operations center with strong cyber maturity?",
)

if st.button("Run Assistant"):
    result = assistant.answer(question)
    st.caption(f"Retrieval backend: `{result['retrieval_backend']}`")
    st.caption(f"Corpus stats: `{result['corpus_stats']}`")

    if result["ingestion_warnings"]:
        with st.expander("Ingestion Warnings"):
            st.json(result["ingestion_warnings"])

    if result["ingestion_errors"]:
        with st.expander("Ingestion Errors"):
            st.json(result["ingestion_errors"])

    st.subheader("Executive Summary")
    st.write(result["executive_summary"])

    st.subheader("Recommendation")
    st.write(result["recommendation_body"])

    st.subheader("Confidence")
    st.json(result["confidence"])

    st.subheader("Decision Scorecard")
    st.dataframe(pd.DataFrame(result["scorecard"]), use_container_width=True)

    st.subheader("Comparison Table")
    st.dataframe(pd.DataFrame(result["comparison_table"]), use_container_width=True)

    st.subheader("Tradeoffs")
    for item in result["tradeoffs"]:
        st.write(f"- {item}")

    st.subheader("Evidence")
    for item in result["evidence"]:
        label = f"{item['source']} | {item['doc_type']} | score={item['score']:.4f}"
        with st.expander(label):
            st.caption(f"Path: `{item['path']}`")
            if item["metadata"]:
                st.json(item["metadata"])
            st.write(item["text"])

st.divider()
st.subheader("Benchmark Evaluation")
if st.button("Run Benchmarks"):
    eval_agent = InfrastructureAssistant(
        data_dir=data_dir,
        top_k=top_k,
        retrieval_backend=backend,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        reuse_index=reuse_index,
        decision_config=DecisionConfig(
            resilience=resilience_weight,
            cyber_maturity=cyber_weight,
            cost_efficiency=cost_weight,
            implementation_readiness=readiness_weight,
        ),
    )
    cases = load_eval_cases(Path("benchmarks/benchmark_cases.json"))
    results = run_smoke_eval(eval_agent, cases)
    summary = summarize_results(results)
    st.write(summary)
    st.dataframe(pd.DataFrame(results), use_container_width=True)

    export_path = Path("artifacts/streamlit_eval_results.json")
    export_results(results, summary, export_path)
    st.caption(f"Saved benchmark report to `{export_path}`")
