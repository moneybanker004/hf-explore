import streamlit as st
from transformers import pipeline

MODEL_ID = "mrm8488/distilroberta-finetuned-financial-news-sentiment-analysis"


@st.cache_resource
def load_classifier():
    return pipeline("text-classification", model=MODEL_ID)


st.title("Financial Sentiment Checker")
st.write("Paste a financial headline or sentence and the model will label its sentiment.")

text = st.text_input("Headline or sentence")

if st.button("Analyze"):
    if not text.strip():
        st.error("Please enter some text before clicking Analyze.")
    else:
        try:
            with st.spinner("Running the model..."):
                classifier = load_classifier()
                results = classifier(text, top_k=None, truncation=True)

            # Some versions of transformers wrap a single input in an extra list
            if results and isinstance(results[0], list):
                results = results[0]

            best = max(results, key=lambda r: r["score"])
            st.success(f"Overall sentiment: {best['label']} at {best['score']:.1%} confidence")

            st.table(
                [
                    {"Label": r["label"], "Confidence": f"{r['score']:.1%}"}
                    for r in sorted(results, key=lambda r: r["score"], reverse=True)
                ]
            )
        except Exception as error:
            st.error(f"Something went wrong while running the model: {error}")