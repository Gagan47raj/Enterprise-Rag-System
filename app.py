"""
Advanced RAG System - Streamlit Application
Phase 3: Vector Store Integration
"""
import streamlit as st
import sys
from pathlib import Path
import time

# Add project root to Python path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

from utils.helpers import load_config
from src.document_processor import DocumentProcessingPipeline
from src.vector_store import VectorStoreFactory

# Page configuration
st.set_page_config(
    page_title="Advanced RAG System - Phase 3",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if 'vector_store' not in st.session_state:
    st.session_state.vector_store = None
if 'documents_processed' not in st.session_state:
    st.session_state.documents_processed = False
if 'stats' not in st.session_state:
    st.session_state.stats = {}

def process_documents():
    """Process documents and create vector store"""
    with st.spinner("Processing documents..."):
        config = load_config()
        pipeline = DocumentProcessingPipeline(config)
        
        # Process documents
        sample_dir = project_root / 'data' / 'documents'
        chunks, parents = pipeline.process_directory(
            str(sample_dir),
            strategy="recursive",
            use_parent_child=True,
            enhance_metadata=True
        )
        
        # Create vector store
        vector_store = VectorStoreFactory.create_vector_store(
            store_type="hybrid",
            config=config
        )
        
        index_path = project_root / 'data' / 'vector_store' / 'main_index'
        vector_store.index_documents(chunks, str(index_path))
        
        st.session_state.vector_store = vector_store
        st.session_state.documents_processed = True
        st.session_state.stats = vector_store.get_stats()
        st.session_state.chunks = chunks
        
        return True

def main():
    """Main Streamlit application"""
    
    # Sidebar
    with st.sidebar:
        st.title("🎛️ Configuration")
        st.markdown("---")
        
        # System Status
        st.subheader("📊 System Status")
        if st.session_state.documents_processed:
            st.success("Documents Indexed ✅")
            stats = st.session_state.stats
            st.metric("FAISS Vectors", stats.get('faiss_index_size', 0))
            st.metric("BM25 Docs", stats.get('bm25_doc_count', 0))
            st.metric("Embedding Dim", stats.get('embedding_dimension', 0))
        else:
            st.warning("No documents indexed")
            if st.button("🚀 Process Documents", use_container_width=True):
                if process_documents():
                    st.rerun()
        
        st.markdown("---")
        
        # Search Settings
        st.subheader("🔍 Search Settings")
        search_type = st.selectbox(
            "Search Type",
            ["hybrid", "similarity", "mmr", "bm25"],
            help="Select search strategy"
        )
        
        k_docs = st.slider(
            "Number of results",
            min_value=1,
            max_value=10,
            value=3,
            help="Number of documents to retrieve"
        )
        
        if search_type == "hybrid":
            alpha = st.slider(
                "Dense/Sparse Balance (α)",
                min_value=0.0,
                max_value=1.0,
                value=0.5,
                help="0 = BM25 only, 1 = FAISS only"
            )
    
    # Main content area
    st.title("🚀 Advanced RAG System")
    st.markdown("---")
    
    # Query Interface
    st.subheader("💬 Query Interface")
    
    col1, col2 = st.columns([4, 1])
    with col1:
        query = st.text_input(
            "Enter your query:",
            placeholder="Ask something about the documents...",
            key="query_input"
        )
    with col2:
        search_button = st.button("🔍 Search", use_container_width=True)
    
    # Search Results
    if search_button and query:
        if not st.session_state.documents_processed:
            st.error("Please process documents first using the sidebar button!")
        else:
            with st.spinner("Searching..."):
                start_time = time.time()
                
                # Perform search
                results = st.session_state.vector_store.search(
                    query,
                    k=k_docs,
                    search_type=search_type,
                    alpha=alpha if search_type == "hybrid" else 0.5,
                    return_scores=True
                )
                
                end_time = time.time()
                search_time = end_time - start_time
                
                # Display results
                st.markdown(f"### 📝 Results ({len(results)} found in {search_time:.3f}s)")
                
                for i, (doc, score) in enumerate(results, 1):
                    with st.expander(f"Result {i} - Score: {score:.4f}", expanded=(i==1)):
                        st.markdown(doc.page_content)
                        
                        # Metadata
                        if doc.metadata:
                            st.markdown("**Metadata:**")
                            cols = st.columns(3)
                            meta_items = list(doc.metadata.items())[:6]
                            for j, (key, value) in enumerate(meta_items):
                                with cols[j % 3]:
                                    st.caption(f"{key}: {value}")
    
    # Document Overview
    if st.session_state.documents_processed:
        st.markdown("---")
        st.subheader("📚 Document Overview")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Chunks", len(st.session_state.chunks))
        with col2:
            st.metric("Vector Dimension", st.session_state.stats.get('embedding_dimension', 0))
        with col3:
            st.metric("Avg Chunk Size", 
                     f"{sum(len(c.page_content) for c in st.session_state.chunks) // len(st.session_state.chunks)} chars")
        
        # Show sample chunks
        st.markdown("**Sample Chunks:**")
        for i, chunk in enumerate(st.session_state.chunks[:3], 1):
            st.text_area(f"Chunk {i}", chunk.page_content[:200] + "...", height=100)

if __name__ == "__main__":
    main()