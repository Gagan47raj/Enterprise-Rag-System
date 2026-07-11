"""
Advanced RAG System - Streamlit Application
"""
import streamlit as st
import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

# Page configuration
st.set_page_config(
    page_title="Advanced RAG System",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

def main():
    """Main Streamlit application"""
    
    # Sidebar
    with st.sidebar:
        st.title("🎛️ Configuration")
        st.markdown("---")
        
        # Model selection
        st.subheader("Model Settings")
        model = st.selectbox(
            "LLM Model",
            ["mistral", "llama2", "codellama"],
            help="Select the Ollama model to use"
        )
        
        # Retrieval settings
        st.subheader("Retrieval Settings")
        k_docs = st.slider(
            "Number of documents to retrieve",
            min_value=1,
            max_value=20,
            value=5,
            help="Number of documents to retrieve for each query"
        )
        
        # Advanced options
        st.subheader("Advanced Options")
        use_multiquery = st.checkbox("Use MultiQuery Retriever", value=True)
        use_reranker = st.checkbox("Use Reranker", value=True)
        use_compression = st.checkbox("Use Contextual Compression", value=True)
        
        st.markdown("---")
        st.markdown("### 📊 System Status")
        st.success("System Ready ✅")
    
    # Main content area
    st.title("🚀 Advanced RAG System")
    st.markdown("---")
    
    # Welcome message
    st.info(
        """
        ### Welcome to the Advanced RAG System!
        
        This system implements state-of-the-art retrieval techniques:
        - **MultiQuery Retriever**: Generates multiple query variations
        - **Parent Document Retriever**: Maintains document context
        - **Contextual Compression**: Optimizes retrieved content
        - **Reranker**: Improves retrieval relevance
        - **Metadata Filtering**: Smart document filtering
        
        📚 **Coming in next phases**: Document upload, query interface, and more!
        """
    )
    
    # Placeholder for future features
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(label="Documents Loaded", value="0", delta="Coming soon")
    
    with col2:
        st.metric(label="Vector Store Size", value="0", delta="Coming soon")
    
    with col3:
        st.metric(label="Avg Response Time", value="0s", delta="Coming soon")

if __name__ == "__main__":
    main()