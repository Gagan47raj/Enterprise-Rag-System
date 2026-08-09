"""
Session State Manager for Streamlit Application
"""
import streamlit as st
from typing import Dict, Any, Optional, List
from pathlib import Path
import sys

project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from src.vector_store import HybridVectorStore, EmbeddingManager, VectorStoreFactory
from src.document_processor import DocumentProcessingPipeline
from src.advanced_retrievers import AdvancedRetrievalPipeline
from utils.helpers import load_config
import logging

logger = logging.getLogger(__name__)

class SessionManager:
    """Manage Streamlit session state"""
    
    @staticmethod
    def initialize_session():
        """Initialize all session state variables"""
        
        # System status
        if 'system_initialized' not in st.session_state:
            st.session_state.system_initialized = False
        
        if 'config' not in st.session_state:
            st.session_state.config = load_config()
        
        # Document processing
        if 'documents_processed' not in st.session_state:
            st.session_state.documents_processed = False
        
        if 'chunks' not in st.session_state:
            st.session_state.chunks = []
        
        if 'parent_docs' not in st.session_state:
            st.session_state.parent_docs = []
        
        # Vector store
        if 'vector_store' not in st.session_state:
            st.session_state.vector_store = None
        
        if 'embedding_manager' not in st.session_state:
            st.session_state.embedding_manager = None
        
        # Advanced retrievers
        if 'retrieval_pipeline' not in st.session_state:
            st.session_state.retrieval_pipeline = None
        
        # Search history
        if 'search_history' not in st.session_state:
            st.session_state.search_history = []
        
        if 'current_results' not in st.session_state:
            st.session_state.current_results = None
        
        # UI state
        if 'sidebar_state' not in st.session_state:
            st.session_state.sidebar_state = 'expanded'
        
        if 'current_tab' not in st.session_state:
            st.session_state.current_tab = 'Search'
        
        # Performance metrics
        if 'performance_metrics' not in st.session_state:
            st.session_state.performance_metrics = {
                'total_queries': 0,
                'avg_retrieval_time': 0,
                'total_documents_indexed': 0,
                'cache_hits': 0
            }
    
    @staticmethod
    def reset_session():
        """Reset all session state"""
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        SessionManager.initialize_session()
    
    @staticmethod
    def update_metrics(query_time: float, docs_retrieved: int):
        """Update performance metrics"""
        metrics = st.session_state.performance_metrics
        metrics['total_queries'] += 1
        
        # Update average time
        prev_avg = metrics['avg_retrieval_time']
        n = metrics['total_queries']
        metrics['avg_retrieval_time'] = prev_avg + (query_time - prev_avg) / n
        
        st.session_state.performance_metrics = metrics


class SystemInitializer:
    """Initialize RAG system components"""
    
    @staticmethod
    @st.cache_resource
    def initialize_embedding_manager(model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        """Initialize and cache embedding manager"""
        logger.info(f"Initializing embedding manager: {model_name}")
        return EmbeddingManager(model_name=model_name)
    
    @staticmethod
    def initialize_vector_store(embedding_manager, config=None):
        """Initialize vector store"""
        logger.info("Initializing vector store")
        return VectorStoreFactory.create_vector_store(
            store_type="hybrid",
            config=config,
            embedding_manager=embedding_manager
        )
    
    @staticmethod
    def initialize_retrieval_pipeline(vector_store, embedding_manager, config=None):
        """Initialize retrieval pipeline"""
        logger.info("Initializing retrieval pipeline")
        return AdvancedRetrievalPipeline(
            vector_store=vector_store,
            embedding_manager=embedding_manager,
            config=config
        )
    
    @staticmethod
    def process_documents(uploaded_files, config=None):
        """Process uploaded documents"""
        logger.info(f"Processing {len(uploaded_files)} uploaded files")
        
        pipeline = DocumentProcessingPipeline(config)
        all_chunks = []
        all_parents = []
        
        for uploaded_file in uploaded_files:
            # Save uploaded file temporarily
            temp_dir = Path("data/documents/uploads")
            temp_dir.mkdir(parents=True, exist_ok=True)
            
            temp_path = temp_dir / uploaded_file.name
            with open(temp_path, 'wb') as f:
                f.write(uploaded_file.getbuffer())
            
            try:
                # Process file
                chunks, parents = pipeline.process_file(
                    str(temp_path),
                    strategy="recursive",
                    use_parent_child=True,
                    enhance_metadata=True
                )
                
                all_chunks.extend(chunks)
                if parents:
                    all_parents.extend(parents)
                
                logger.info(f"Processed {uploaded_file.name}: {len(chunks)} chunks")
                
            except Exception as e:
                logger.error(f"Error processing {uploaded_file.name}: {e}")
                st.error(f"Error processing {uploaded_file.name}: {e}")
        
        return all_chunks, all_parents