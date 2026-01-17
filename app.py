import streamlit as st
import pickle
import faiss
from sentence_transformers import SentenceTransformer
import numpy as np
import re

# ----- Load FAISS index and metadata -----
INDEX_FILE = "faiss.index"
META_FILE = "metadata.pkl"

index = faiss.read_index(INDEX_FILE)

with open(META_FILE, "rb") as f:
    meta = pickle.load(f)

texts = meta["texts"]
sources = meta["sources"]

# ----- Streamlit UI -----
st.title("Document-based Q&A Agent")
st.write("Ask a question and get answers grounded in your documents. Only relevant content will be shown.")

query = st.text_input("Your question:")
num_results = st.slider("Number of results", min_value=1, max_value=10, value=5)
threshold = st.slider("Similarity threshold", min_value=0.0, max_value=1.0, value=0.35, step=0.01)

# ----- Initialize model -----
@st.cache_resource
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2", device="cpu")

model = load_model()

# ----- Helper: highlight query in text -----
def highlight_query(text, query):
    pattern = re.compile(re.escape(query), re.IGNORECASE)
    return pattern.sub(lambda m: f"**{m.group(0)}**", text)

# ----- Main Q&A -----
if query:
    query_vec = model.encode([query])
    D, I = index.search(query_vec, len(texts))  # search all vectors

    results = []
    for dist, idx in zip(D[0], I[0]):
        sim = 1 / (1 + dist)  # convert L2 distance to rough similarity
        if sim >= threshold:
            results.append((sim, texts[idx], sources[idx]))

    # Sort by similarity
    results = sorted(results, key=lambda x: x[0], reverse=True)
    results = results[:num_results]

    if results:
        for sim, text, src in results:
            st.markdown(f"**Source:** {src}")
            st.markdown(f"**Similarity score:** {sim:.2f}")
            st.markdown(f"**Answer snippet:** {highlight_query(text, query)}")
            st.markdown("---")
    else:
        st.info("No relevant information found in the documents.")
