import streamlit as st

from src.agent import InfrastructureAssistant


st.set_page_config(page_title="AI Infrastructure Planning Assistant", layout="wide")
st.title("AI Infrastructure Planning Assistant")
st.write(
    "Ask infrastructure planning questions and get retrieval-grounded recommendations "
    "using local policy and project context."
)

assistant = InfrastructureAssistant(data_dir="data")

question = st.text_input(
    "Question",
    value="Which site is best for a resilient regional operations center with strong cyber maturity?",
)

if st.button("Run Assistant"):
    result = assistant.answer(question)
    st.subheader("Recommendation")
    st.write(result["recommendation"])

    st.subheader("Decision Scorecard")
    st.dataframe(result["scorecard"], use_container_width=True)

    st.subheader("Reasoning")
    st.write(result["reasoning"])

    st.subheader("Evidence")
    for item in result["evidence"]:
        with st.expander(f"{item['source']} | score={item['score']:.4f}"):
            st.write(item["text"])
