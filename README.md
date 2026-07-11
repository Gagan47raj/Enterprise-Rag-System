# Advanced RAG System 🚀

A comprehensive Retrieval-Augmented Generation (RAG) system built with LangChain, FAISS, and various advanced retrieval techniques.

## 🌟 Features

- **Multi-Strategy Retrieval**: Combines dense (FAISS) and sparse (BM25) retrieval
- **MultiQuery Retriever**: Generates multiple query variations for better retrieval
- **Parent Document Retriever**: Maintains context with parent-child document relationships
- **Contextual Compression**: Compresses retrieved documents for efficiency
- **Reranker**: Re-ranks retrieved documents using cross-encoders
- **Metadata Filtering**: Filter documents based on metadata
- **Interactive UI**: Built with Streamlit for easy interaction

## 🛠️ Tech Stack

- **LangChain**: Orchestration framework
- **FAISS**: Vector similarity search
- **BM25**: Sparse retrieval
- **Ollama + Mistral**: Local LLM
- **Streamlit**: User interface

## 📁 Project Structure
advanced-rag-system/
├── config/ # Configuration files
│ ├── config.yaml # Main configuration
│ └── .env.example # Environment variables template
├── data/ # Data storage
│ ├── documents/ # Input documents
│ └── vector_store/ # FAISS index
├── notebooks/ # Jupyter notebooks for testing
├── src/ # Source code
│ ├── init.py
│ └── main.py # Entry point
├── utils/ # Utility functions
│ ├── helpers.py # Helper functions
│ └── text_processor.py # Text processing
├── tests/ # Test files
├── .gitignore
├── README.md
└── requirements.txt


## 🚀 Quick Start

### Prerequisites

1. Python 3.9+
2. Ollama installed with Mistral model
3. Git

### Installation

```bash
# Clone the repository
git clone https://github.com/Gagan47raj/Enterprise-Rag-System.git
cd advanced-rag-system

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Setup configuration
cp config/.env
# Edit config/.env with your settings

# Pull Ollama model
ollama pull mistral