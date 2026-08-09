"""
Advanced RAG System - Complete Streamlit Application
"""
import streamlit as st
import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

# Import components
from src.components.sidebar import SidebarComponent
from src.components.search import SearchComponent
from src.components.document_explorer import DocumentExplorer
from src.components.analytics import AnalyticsComponent
from src.services.session_manager import SessionManager

# Page configuration
st.set_page_config(
    page_title="Advanced RAG System",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://github.com/yourusername/advanced-rag-system',
        'Report a bug': 'https://github.com/yourusername/advanced-rag-system/issues',
        'About': """
        # Advanced RAG System
        
        Built with:
        - **LangChain** for orchestration
        - **FAISS** for vector search
        - **BM25** for sparse retrieval
        - **Ollama + Mistral** for LLM
        - **Streamlit** for UI
        
        Features multi-query retrieval, parent document retrieval,
        contextual compression, reranking, and metadata filtering.
        """
    }
)

# Custom CSS
st.markdown("""
<style>
    /* Main container */
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    
    /* Headers */
    h1 {
        color: #1f77b4;
        font-weight: 700;
    }
    
    h2 {
        color: #2c3e50;
        font-weight: 600;
    }
    
    h3 {
        color: #34495e;
    }
    
    /* Metrics */
    [data-testid="stMetricValue"] {
        font-size: 1.5rem;
        font-weight: 600;
        color: #1f77b4;
    }
    
    /* Buttons */
    .stButton > button {
        border-radius: 8px;
        font-weight: 500;
    }
    
    .stButton > button[kind="primary"] {
        background-color: #1f77b4;
    }
    
    /* Expander */
    .streamlit-expanderHeader {
        font-weight: 600;
        background-color: #f8f9fa;
        border-radius: 8px;
    }
    
    /* Search bar */
    [data-testid="stTextInput"] input {
        border-radius: 8px;
        border: 2px solid #e0e0e0;
        padding: 0.5rem 1rem;
        font-size: 1.1rem;
    }
    
    [data-testid="stTextInput"] input:focus {
        border-color: #1f77b4;
        box-shadow: 0 0 0 2px rgba(31, 119, 180, 0.2);
    }
    
    /* Cards */
    .card {
        background-color: white;
        border-radius: 10px;
        padding: 1.5rem;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
        margin-bottom: 1rem;
    }
    
    /* Progress bar */
    .stProgress > div > div {
        background-color: #1f77b4;
    }
    
    /* Sidebar */
    .css-1d391kg {
        background-color: #f8f9fa;
    }
    
    /* Scrollbar */
    ::-webkit-scrollbar {
        width: 8px;
    }
    
    ::-webkit-scrollbar-track {
        background: #f1f1f1;
    }
    
    ::-webkit-scrollbar-thumb {
        background: #888;
        border-radius: 4px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: #555;
    }
</style>
""", unsafe_allow_html=True)

def main():
    """Main application"""
    
    # Initialize session state
    SessionManager.initialize_session()
    
    # Render sidebar
    SidebarComponent.render()
    
    # Main content area with tabs
    st.markdown(
        """
        <div style='text-align: center; padding: 1rem 0;'>
            <h1 style='font-size: 2.5rem; margin-bottom: 0.5rem;'>
                🤖 Advanced RAG System
            </h1>
            <p style='color: #666; font-size: 1.1rem;'>
                Intelligent Document Retrieval with Multi-Strategy Search
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )
    
    # Navigation tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "🔍 Search", 
        "📚 Document Explorer", 
        "📊 Analytics",
        "ℹ️ Help"
    ])
    
    with tab1:
        SearchComponent.render()
    
    with tab2:
        DocumentExplorer.render()
    
    with tab3:
        AnalyticsComponent.render()
    
    with tab4:
        render_help()
    
    # Footer
    st.markdown("---")
    st.markdown(
        """
        <div style='text-align: center; color: #999; padding: 1rem 0;'>
            <p>Advanced RAG System v1.0 | Built with LangChain, FAISS, BM25, Ollama & Streamlit</p>
            <p style='font-size: 0.8rem;'>
                © 2024 Advanced RAG System. All rights reserved.
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )

def render_help():
    """Render help page"""
    st.title("ℹ️ Help & Documentation")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        ### 🚀 Getting Started
        
        1. **Initialize System**
           - Click 'Initialize System' in the sidebar
           - Wait for embedding model to load
        
        2. **Upload Documents**
           - Use the file uploader in the sidebar
           - Supported formats: PDF, TXT, DOCX, CSV
           - Click 'Process Documents' to index
        
        3. **Search**
           - Enter your query in the search box
           - Configure retrieval options in sidebar
           - View results with relevance scores
        
        ### 🔧 Search Strategies
        
        - **Hybrid**: Combines dense (FAISS) and sparse (BM25) retrieval
        - **Similarity**: Pure vector similarity search
        - **MMR**: Maximum Marginal Relevance for diversity
        - **BM25**: Keyword-based sparse retrieval
        """)
    
    with col2:
        st.markdown("""
        ### ⚡ Advanced Features
        
        - **MultiQuery Retrieval**: Generates multiple query variations for better recall
        - **Contextual Compression**: Filters out irrelevant content
        - **Reranker**: Re-ranks results using cross-encoder models
        - **Metadata Filtering**: Filter documents by metadata fields
        
        ### 📊 System Components
        
        | Component | Technology |
        |-----------|-----------|
        | Embeddings | Sentence Transformers |
        | Vector Store | FAISS |
        | Sparse Index | BM25 |
        | LLM | Ollama + Mistral |
        | Reranker | Cross-Encoder |
        | UI | Streamlit |
        
        ### 🛠️ Troubleshooting
        
        - **System won't initialize**: Check Ollama is running
        - **No results**: Try adjusting search strategy
        - **Slow performance**: Reduce documents or use simpler search
        - **Memory issues**: Process fewer documents at once
        """)
    
    st.markdown("---")
    
    # Quick tips
    st.subheader("💡 Quick Tips")
    
    tips = [
        "📄 Start with small documents to test the system",
        "🔍 Use specific queries for better results",
        "⚡ Enable MultiQuery for ambiguous queries",
        "🎯 Use metadata filtering to narrow results",
        "📊 Check Analytics for performance insights",
        "🗑️ Clear documents to start fresh"
    ]
    
    cols = st.columns(3)
    for i, tip in enumerate(tips):
        with cols[i % 3]:
            st.info(tip)

if __name__ == "__main__":
    main()