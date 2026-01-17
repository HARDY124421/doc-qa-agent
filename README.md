# Document-Based Q&A Agent

This project is a document-grounded Q&A system that answers user questions
strictly using the content of provided Word documents.

The system:
- Accepts a natural language question
- Searches relevant documents using vector similarity
- Returns answers grounded only in retrieved content
- Shows source documents and excerpts
- Refuses to answer when relevant information is missing

A Streamlit UI is provided for interactive usage.

---

## How It Works

1. Documents (.docx) are parsed and chunked
2. Text chunks are embedded and stored in a FAISS vector index
3. User questions are embedded and matched against stored chunks
4. Relevant excerpts are retrieved and displayed as grounded answers
5. Source file names and similarity scores are shown to ensure transparency

---

## Tools and Models Used

- Python
- Streamlit (UI)
- FAISS (vector similarity search)
- Sentence Transformers (text embeddings)
- python-docx (document parsing)

---

## How to Run

```bash
### 1. Clone the repository

git clone https://github.com/HARDY124421/doc-qa-agent.git
cd doc-qa-agent

### 2. Create and activate a virtual environment

python3 -m venv venv
source venv/bin/activate

### 3. Install dependencies
pip install -r requirements.txt

### 4. Build or rebuild the document index
python ingest.py

### 5. Run the application
streamlit run app.py

```

---

## Optional / Future Improvements

With more time, the following enhancements could be made:

- Add LLM-based answer synthesis on top of retrieved excerpts
- Improve chunking strategy for better semantic retrieval
- Support PDF and TXT documents in addition to DOCX
