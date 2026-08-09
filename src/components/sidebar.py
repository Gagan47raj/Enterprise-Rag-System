"""
Sidebar Component for Advanced RAG System
"""
import streamlit as st
from typing import Dict, Any
from pathlib import Path
import sys
import time

project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from src.services.session_manager import SystemInitializer

class SidebarComponent:
    """Streamlit sidebar with configuration options"""
    
    @staticmethod
    def render():
        """Render sidebar"""
        with st.sidebar:
            st.title("🎛️ Advanced RAG System")
            st.markdown("---")
            
            # System Status Section
            SidebarComponent._render_system_status()
            
            st.markdown("---")
            
            # Document Management
            SidebarComponent._render_document_section()
            
            st.markdown("---")
            
            # Retrieval Configuration
            SidebarComponent._render_retrieval_config()
            
            st.markdown("---")
            
            # Search Options
            SidebarComponent._render_search_options()
            
            st.markdown("---")
            
            # Performance Metrics
            SidebarComponent._render_performance_metrics()
            
            st.markdown("---")
            
            # About Section
            SidebarComponent._render_about()
    
    @staticmethod
    def _render_system_status():
        """Render system status section"""
        st.subheader("📊 System Status")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.session_state.system_initialized:
                st.success("✅ Ready")
            else:
                st.warning("⚠️ Not Ready")
        
        with col2:
            if st.session_state.documents_processed:
                st.success("📄 Indexed")
            else:
                st.warning("📄 Empty")
        
        # System initialization button
        if not st.session_state.system_initialized:
            if st.button("🚀 Initialize System", use_container_width=True, type="primary"):
                with st.spinner("Initializing system..."):
                    try:
                        config = st.session_state.config
                        
                        # Initialize embedding manager
                        st.session_state.embedding_manager = SystemInitializer.initialize_embedding_manager(
                            config['models']['embedding_model']
                        )
                        
                        # Initialize vector store
                        st.session_state.vector_store = SystemInitializer.initialize_vector_store(
                            st.session_state.embedding_manager,
                            config
                        )
                        
                        # Initialize retrieval pipeline
                        st.session_state.retrieval_pipeline = SystemInitializer.initialize_retrieval_pipeline(
                            st.session_state.vector_store,
                            st.session_state.embedding_manager,
                            config
                        )
                        
                        st.session_state.system_initialized = True
                        st.success("System initialized!")
                        time.sleep(1)
                        st.rerun()
                        
                    except Exception as e:
                        st.error(f"Initialization failed: {e}")
        
        # System stats when initialized
        if st.session_state.system_initialized:
            st.markdown("---")
            if st.session_state.documents_processed:
                stats = st.session_state.vector_store.get_stats()
                st.metric("Documents", stats.get('bm25_doc_count', 0))
                st.metric("Vectors", stats.get('faiss_index_size', 0))
                st.metric("Dimension", stats.get('embedding_dimension', 0))
    
    @staticmethod
    def _render_document_section():
        """Render document management section"""
        st.subheader("📁 Document Management")
        
        # File uploader
        uploaded_files = st.file_uploader(
            "Upload Documents",
            type=['pdf', 'txt', 'docx', 'csv'],
            accept_multiple_files=True,
            help="Upload documents to add to the knowledge base"
        )
        
        if uploaded_files and st.button("📤 Process Documents", use_container_width=True):
            if not st.session_state.system_initialized:
                st.warning("Please initialize system first!")
            else:
                with st.spinner(f"Processing {len(uploaded_files)} files..."):
                    try:
                        chunks, parents = SystemInitializer.process_documents(
                            uploaded_files,
                            st.session_state.config
                        )
                        
                        if chunks:
                            # Index documents
                            st.session_state.vector_store.index_documents(chunks)
                            st.session_state.chunks.extend(chunks)
                            if parents:
                                st.session_state.parent_docs.extend(parents)
                            st.session_state.documents_processed = True
                            
                            st.success(f"✅ Processed {len(chunks)} chunks from {len(uploaded_files)} files!")
                            time.sleep(1)
                            st.rerun()
                        else:
                            st.warning("No documents processed")
                            
                    except Exception as e:
                        st.error(f"Error: {e}")
        
        # Clear documents option
        if st.session_state.documents_processed:
            st.markdown("---")
            if st.button("🗑️ Clear All Documents", use_container_width=True):
                st.session_state.vector_store = SystemInitializer.initialize_vector_store(
                    st.session_state.embedding_manager,
                    st.session_state.config
                )
                st.session_state.chunks = []
                st.session_state.parent_docs = []
                st.session_state.documents_processed = False
                st.session_state.search_history = []
                st.rerun()
    
    @staticmethod
    def _render_retrieval_config():
        """Render retrieval configuration"""
        st.subheader("🔧 Retrieval Configuration")
        
        # Search type
        st.session_state.search_type = st.selectbox(
            "Search Strategy",
            ["hybrid", "similarity", "mmr", "bm25"],
            help="Select retrieval strategy"
        )
        
        # Number of results
        st.session_state.k_docs = st.slider(
            "Documents to Retrieve",
            min_value=1,
            max_value=20,
            value=5,
            help="Number of documents to retrieve"
        )
        
        # Hybrid search alpha
        if st.session_state.search_type == "hybrid":
            st.session_state.alpha = st.slider(
                "Dense/Sparse Balance",
                min_value=0.0,
                max_value=1.0,
                value=0.5,
                help="0 = BM25 only, 1 = FAISS only"
            )
        
        # Advanced options
        st.markdown("---")
        st.subheader("⚡ Advanced Features")
        
        st.session_state.use_multiquery = st.checkbox(
            "MultiQuery Retrieval",
            value=True,
            help="Generate multiple query variations"
        )
        
        if st.session_state.use_multiquery:
            st.session_state.num_queries = st.slider(
                "Query Variations",
                min_value=1,
                max_value=10,
                value=3
            )
        
        st.session_state.use_compression = st.checkbox(
            "Contextual Compression",
            value=True,
            help="Compress retrieved documents"
        )
        
        if st.session_state.use_compression:
            st.session_state.compression_method = st.selectbox(
                "Compression Method",
                ["cross_encoder", "embeddings", "pipeline"],
                help="Select compression technique"
            )
            
            st.session_state.compression_threshold = st.slider(
                "Compression Threshold",
                min_value=0.0,
                max_value=1.0,
                value=0.3,
                help="Minimum relevance threshold"
            )
        
        st.session_state.use_reranker = st.checkbox(
            "Reranker",
            value=True,
            help="Rerank results for better relevance"
        )
    
    @staticmethod
    def _render_search_options():
        """Render search options"""
        st.subheader("🔍 Search Options")
        
        # Metadata filtering
        st.session_state.use_metadata_filter = st.checkbox(
            "Metadata Filtering",
            value=False,
            help="Filter by document metadata"
        )
        
        if st.session_state.use_metadata_filter:
            # Dynamic metadata filters
            if st.session_state.documents_processed and st.session_state.chunks:
                # Get unique metadata keys
                all_keys = set()
                for chunk in st.session_state.chunks[:100]:
                    all_keys.update(chunk.metadata.keys())
                
                # Filter common metadata fields
                common_fields = ['source', 'file_type', 'category', 'doc_type', 'author']
                available_fields = [k for k in common_fields if k in all_keys]
                
                if available_fields:
                    st.session_state.metadata_field = st.selectbox(
                        "Filter Field",
                        available_fields
                    )
                    
                    # Get unique values for selected field
                    unique_values = set()
                    for chunk in st.session_state.chunks[:100]:
                        if st.session_state.metadata_field in chunk.metadata:
                            unique_values.add(str(chunk.metadata[st.session_state.metadata_field]))
                    
                    if unique_values:
                        st.session_state.metadata_value = st.selectbox(
                            "Filter Value",
                            sorted(unique_values)
                        )
        
        # Display options
        st.markdown("---")
        st.subheader("📊 Display Options")
        
        st.session_state.show_scores = st.checkbox(
            "Show Scores",
            value=True,
            help="Display relevance scores"
        )
        
        st.session_state.show_metadata = st.checkbox(
            "Show Metadata",
            value=True,
            help="Display document metadata"
        )
        
        st.session_state.highlight_query = st.checkbox(
            "Highlight Query Terms",
            value=True,
            help="Highlight query terms in results"
        )
    
    @staticmethod
    def _render_performance_metrics():
        """Render performance metrics"""
        st.subheader("📈 Performance")
        
        metrics = st.session_state.performance_metrics
        
        if metrics['total_queries'] > 0:
            st.metric("Total Queries", metrics['total_queries'])
            st.metric("Avg Time", f"{metrics['avg_retrieval_time']:.3f}s")
        else:
            st.caption("No queries executed yet")
    
    @staticmethod
    def _render_about():
        """Render about section"""
        with st.expander("ℹ️ About"):
            st.markdown("""
            **Advanced RAG System**
            
            Built with:
            - LangChain
            - FAISS
            - BM25
            - Ollama + Mistral
            - Streamlit
            
            Features:
            - MultiQuery Retrieval
            - Parent Document Retrieval
            - Contextual Compression
            - Reranker
            - Metadata Filtering
            - Hybrid Search
            """)