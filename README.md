# 🚀 Advanced RAG System

> A production-ready Retrieval-Augmented Generation (RAG) system built with **LangChain**, **Ollama**, **FAISS**, **BM25**, and modern retrieval techniques for accurate, context-aware question answering.

---

## 📌 Overview

This project demonstrates how modern RAG systems are built in production.

Instead of relying on a single vector search, it combines multiple retrieval strategies such as **Hybrid Search**, **Parent Document Retrieval**, **MultiQuery Retrieval**, **Contextual Compression**, and **Cross-Encoder Reranking** to improve retrieval quality and answer accuracy.

The project is designed as a learning resource as well as a portfolio project for AI/ML and GenAI engineers.

---

# ✨ Features

| Feature | Description |
|----------|-------------|
| 🔍 Dense Retrieval | Semantic search using FAISS |
| 📚 Sparse Retrieval | BM25 keyword search |
| 🔀 Hybrid Search | Combines dense + sparse retrieval |
| 🧠 MultiQuery Retriever | Generates multiple query variations |
| 📄 Parent Document Retriever | Retrieves parent documents for richer context |
| ✂️ Contextual Compression | Removes irrelevant information before sending to the LLM |
| ⭐ Cross Encoder Reranking | Re-ranks retrieved documents for higher accuracy |
| 🏷 Metadata Filtering | Filter documents using metadata |
| 💬 Local LLM | Ollama (Mistral/Llama3/Gemma) |
| 🌐 Interactive UI | Streamlit interface |
| 📂 Multi-format Support | PDF, DOCX, PPTX, TXT, Markdown |

---

# 🏗️ Architecture

```
                User Query
                     │
                     ▼
           Query Processing
                     │
      ┌──────────────┴──────────────┐
      ▼                             ▼
 Dense Retrieval               Sparse Retrieval
   (FAISS)                        (BM25)
      │                             │
      └──────────────┬──────────────┘
                     ▼
             Hybrid Retriever
                     │
                     ▼
          Parent Document Retriever
                     │
                     ▼
           Contextual Compression
                     │
                     ▼
        Cross Encoder Reranker
                     │
                     ▼
               Ollama LLM
                     │
                     ▼
             Final Response
```

---

# 🛠 Tech Stack

| Category | Technology |
|-----------|------------|
| Framework | LangChain |
| LLM | Ollama |
| Models | Mistral / Llama 3 / Gemma |
| Vector Database | FAISS |
| Sparse Retrieval | BM25 |
| Embeddings | Sentence Transformers |
| UI | Streamlit |
| Language | Python 3.13 |
| Document Loaders | PyPDF, Unstructured |
| Reranking | Cross Encoder |

---

# 📂 Project Structure

```
advanced-rag-system/
│
├── config/
│   ├── config.yaml
│   └── .env.example
│
├── data/
│   ├── documents/
│   └── vector_store/
│
├── notebooks/
│
├── src/
│   ├── loaders/
│   ├── embeddings/
│   ├── retrievers/
│   ├── rerankers/
│   ├── llm/
│   ├── pipelines/
│   └── app.py
│
├── utils/
│
├── tests/
│
├── assets/
│
├── README.md
├── requirements.txt
└── .gitignore
```

---

# 🚀 Getting Started

## Prerequisites

- Python 3.13+
- Ollama
- Git

---

## Clone Repository

```bash
git clone https://github.com/Gagan47raj/advanced-rag-system.git

cd advanced-rag-system
```

---

## Create Virtual Environment

### Windows

```bash
python -m venv venv

venv\Scripts\activate
```

### Linux / macOS

```bash
python3 -m venv venv

source venv/bin/activate
```

---

## Install Dependencies

```bash
pip install -r requirements.txt
```

---

## Configure Environment

Copy

```
config/.env.example
```

to

```
config/.env
```

and update the required configuration.

---

## Download LLM

```bash
ollama pull mistral
```

or

```bash
ollama pull llama3
```

---

## Run Application

```bash
streamlit run app.py
```

---

# 📊 Retrieval Pipeline

```
Documents
     │
     ▼
Document Loader
     │
     ▼
Text Splitter
     │
     ▼
Embedding Model
     │
     ▼
FAISS Index
     │
     ▼
Hybrid Retrieval
     │
     ▼
Compression
     │
     ▼
Reranking
     │
     ▼
LLM
     │
     ▼
Answer
```

---

# 📸 Screenshots

> Add screenshots of:

- Streamlit UI
- Document Upload
- Retrieval Results
- Generated Response

---

# 🎯 Learning Objectives

This project demonstrates:

- Retrieval-Augmented Generation (RAG)
- LangChain
- FAISS
- BM25 Retrieval
- Hybrid Search
- Parent Document Retrieval
- MultiQuery Retrieval
- Contextual Compression
- Metadata Filtering
- Cross Encoder Reranking
- Prompt Engineering
- Local LLMs using Ollama

---

# 🛣️ Roadmap

- [x] Basic RAG
- [x] FAISS Retriever
- [x] BM25 Retriever
- [x] Hybrid Search
- [x] Parent Document Retriever
- [x] MultiQuery Retriever
- [x] Contextual Compression
- [x] Metadata Filtering
- [x] Cross Encoder Reranking
- [ ] Agentic RAG
- [ ] Graph RAG
- [ ] Multi-modal RAG
- [ ] Evaluation Pipeline
- [ ] FastAPI Backend
- [ ] Docker Deployment
- [ ] Kubernetes Deployment

---

# 🤝 Contributing

Contributions are welcome.

Feel free to fork the repository, create a feature branch, and submit a pull request.

---

# ⭐ Support

If you found this project helpful, consider giving it a ⭐ on GitHub.

---

# 📄 License

This project is licensed under the MIT License.
