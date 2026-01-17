import os
import pickle
from docx import Document
from sentence_transformers import SentenceTransformer
import faiss

import os
if os.path.exists("faiss_index.faiss"):
    os.remove("faiss_index.faiss")

DOCS_DIR = "documents"
INDEX_FILE = "faiss.index"
META_FILE = "metadata.pkl"


def load_documents():
    texts = []
    sources = []

    for file in os.listdir(DOCS_DIR):
        if file.endswith(".docx"):
            path = os.path.join(DOCS_DIR, file)
            doc = Document(path)
            full_text = "\n".join([p.text for p in doc.paragraphs if p.text.strip()])

            if full_text:
                texts.append(full_text)
                sources.append(file)

    return texts, sources


def chunk_text(text, chunk_size=300):
    words = text.split()
    chunks = []

    for i in range(0, len(words), chunk_size):
        chunks.append(" ".join(words[i:i + chunk_size]))

    return chunks


def main():
    texts, sources = load_documents()

    all_chunks = []
    chunk_sources = []

    for text, src in zip(texts, sources):
        chunks = chunk_text(text)
        all_chunks.extend(chunks)
        chunk_sources.extend([src] * len(chunks))

    if not all_chunks:
        raise ValueError("No document content found.")

    model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = model.encode(all_chunks)

    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)

    faiss.write_index(index, INDEX_FILE)

    with open(META_FILE, "wb") as f:
        pickle.dump(
            {"texts": all_chunks, "sources": chunk_sources},
            f
        )

    print("Ingestion complete.")
    print(f"Chunks indexed: {len(all_chunks)}")


if __name__ == "__main__":
    main()
