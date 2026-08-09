"""
Search Component for Advanced RAG System
"""
import streamlit as st
import time
from typing import List, Dict, Any, Optional
from pathlib import Path
import sys
import re

project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from src.services.session_manager import SessionManager

class SearchComponent:
    """Search interface and results display"""
    
    @staticmethod
    def render():
        """Render search interface"""
        st.title("🔍 Advanced RAG Search")
        st.markdown("---")
        
        # Query input
        col1, col2, col3 = st.columns([5, 1, 1])
        
        with col1:
            query = st.text_input(
                "Enter your query:",
                placeholder="Ask anything about your documents...",
                key="search_query",
                label_visibility="collapsed"
            )
        
        with col2:
            search_clicked = st.button(
                "🔍 Search",
                use_container_width=True,
                type="primary"
            )
        
        with col3:
            clear_clicked = st.button(
                "🗑️ Clear",
                use_container_width=True
            )
        
        # Search history suggestions
        if st.session_state.search_history:
            with st.expander("📜 Recent Searches", expanded=False):
                for i, hist_query in enumerate(reversed(st.session_state.search_history[-5:])):
                    if st.button(f"{hist_query[:80]}...", key=f"hist_{i}"):
                        st.session_state.search_query = hist_query
                        st.rerun()
        
        # Execute search
        if (search_clicked or query) and query.strip():
            SearchComponent._execute_search(query.strip())
        
        elif clear_clicked:
            st.session_state.current_results = None
            st.session_state.search_query = ""
            st.rerun()
        
        # Display results
        if st.session_state.current_results:
            SearchComponent._display_results()
    
    @staticmethod
    def _execute_search(query: str):
        """Execute search with current configuration"""
        
        if not st.session_state.system_initialized:
            st.error("⚠️ System not initialized! Please initialize from sidebar.")
            return
        
        if not st.session_state.documents_processed:
            st.warning("⚠️ No documents indexed! Please upload documents first.")
            return
        
        with st.spinner("🔍 Searching..."):
            try:
                start_time = time.time()
                
                # Prepare metadata filter
                filter_dict = None
                if st.session_state.use_metadata_filter:
                    if hasattr(st.session_state, 'metadata_field') and hasattr(st.session_state, 'metadata_value'):
                        filter_dict = {
                            st.session_state.metadata_field: st.session_state.metadata_value
                        }
                
                # Build retrieval stages
                stages = {
                    'multiquery': st.session_state.use_multiquery,
                    'parent_docs': False,
                    'compression': st.session_state.use_compression,
                    'reranker': st.session_state.use_reranker
                }
                
                # Prepare keyword arguments
                multiquery_kwargs = {}
                if st.session_state.use_multiquery:
                    multiquery_kwargs['num_queries'] = st.session_state.num_queries
                
                compression_kwargs = {}
                if st.session_state.use_compression:
                    compression_kwargs = {
                        'method': st.session_state.compression_method,
                        'threshold': st.session_state.compression_threshold
                    }
                
                reranker_kwargs = {
                    'top_k': st.session_state.k_docs,
                    'return_scores': True
                }
                
                # Determine search strategy
                use_advanced = any([
                    st.session_state.use_multiquery,
                    st.session_state.use_compression,
                    st.session_state.use_reranker
                ])
                
                if use_advanced:
                    # Use advanced retrieval pipeline
                    result = st.session_state.retrieval_pipeline.retrieve(
                        query=query,
                        k=st.session_state.k_docs,
                        stages=stages,
                        multiquery_kwargs=multiquery_kwargs,
                        compression_kwargs=compression_kwargs,
                        reranker_kwargs=reranker_kwargs
                    )
                    
                    documents = result['documents']
                    pipeline_info = result['pipeline_info']
                    
                else:
                    # Use direct vector store search
                    alpha = getattr(st.session_state, 'alpha', 0.5)
                    documents = st.session_state.vector_store.search(
                        query=query,
                        k=st.session_state.k_docs,
                        search_type=st.session_state.search_type,
                        alpha=alpha,
                        filter_dict=filter_dict,
                        return_scores=True
                    )
                    
                    pipeline_info = {
                        'query': query,
                        'stages_executed': ['direct_search'],
                        'total_time': time.time() - start_time,
                        'document_counts': {'final': len(documents)}
                    }
                
                end_time = time.time()
                query_time = end_time - start_time
                
                # Update session state
                st.session_state.current_results = {
                    'query': query,
                    'documents': documents,
                    'pipeline_info': pipeline_info,
                    'query_time': query_time,
                    'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
                }
                
                # Update search history
                if query not in st.session_state.search_history:
                    st.session_state.search_history.append(query)
                
                # Update metrics
                SessionManager.update_metrics(query_time, len(documents))
                
            except Exception as e:
                st.error(f"Search error: {e}")
                import traceback
                st.error(traceback.format_exc())
    
    @staticmethod
    def _display_results():
        """Display search results"""
        results = st.session_state.current_results
        
        st.markdown("---")
        
        # Results header
        col1, col2, col3 = st.columns([3, 2, 2])
        
        with col1:
            st.markdown(f"### 📝 Results for: *{results['query']}*")
        
        with col2:
            st.metric("Documents Found", len(results['documents']))
        
        with col3:
            st.metric("Search Time", f"{results['query_time']:.3f}s")
        
        # Pipeline info
        if 'pipeline_info' in results:
            with st.expander("🔧 Pipeline Details", expanded=False):
                info = results['pipeline_info']
                
                cols = st.columns(3)
                with cols[0]:
                    st.markdown("**Stages Executed:**")
                    for stage in info.get('stages_executed', []):
                        st.markdown(f"- {stage}")
                
                with cols[1]:
                    st.markdown("**Timings:**")
                    for stage, timing in info.get('timings', {}).items():
                        st.markdown(f"- {stage}: {timing:.3f}s")
                
                with cols[2]:
                    st.markdown("**Document Counts:**")
                    for stage, count in info.get('document_counts', {}).items():
                        st.markdown(f"- {stage}: {count}")
        
        st.markdown("---")
        
        # Display documents
        if not results['documents']:
            st.info("No relevant documents found. Try adjusting your query or search settings.")
        else:
            for i, item in enumerate(results['documents'], 1):
                # Handle both (doc, score) tuples and Document objects
                if isinstance(item, tuple):
                    doc, score = item
                else:
                    doc = item
                    score = None
                
                with st.expander(
                    f"📄 Result {i}" + (f" - Score: {score:.4f}" if score and st.session_state.show_scores else ""),
                    expanded=(i == 1)
                ):
                    # Display content with highlighting
                    content = doc.page_content
                    
                    if st.session_state.highlight_query:
                        content = SearchComponent._highlight_terms(content, results['query'])
                    
                    st.markdown(content)
                    
                    # Display metadata
                    if st.session_state.show_metadata and doc.metadata:
                        st.markdown("---")
                        st.markdown("**📋 Metadata:**")
                        
                        # Organize metadata into columns
                        meta_items = list(doc.metadata.items())
                        cols = st.columns(3)
                        
                        for j, (key, value) in enumerate(meta_items):
                            with cols[j % 3]:
                                # Truncate long values
                                value_str = str(value)
                                if len(value_str) > 50:
                                    value_str = value_str[:47] + "..."
                                st.caption(f"**{key}:** {value_str}")
        
        # Export options
        st.markdown("---")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("📋 Copy Results", use_container_width=True):
                SearchComponent._copy_results(results)
        
        with col2:
            if st.button("📥 Export as Text", use_container_width=True):
                SearchComponent._export_text(results)
        
        with col3:
            if st.button("🔄 New Search", use_container_width=True):
                st.session_state.current_results = None
                st.session_state.search_query = ""
                st.rerun()
    
    @staticmethod
    def _highlight_terms(content: str, query: str) -> str:
        """Highlight query terms in content"""
        # Split query into terms
        query_terms = set(query.lower().split())
        # Remove common words
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'}
        query_terms = query_terms - stop_words
        
        highlighted = content
        for term in query_terms:
            if len(term) > 2:  # Only highlight terms with 3+ characters
                pattern = re.compile(f'({re.escape(term)})', re.IGNORECASE)
                highlighted = pattern.sub(r'**\1**', highlighted)
        
        return highlighted
    
    @staticmethod
    def _copy_results(results: Dict[str, Any]):
        """Copy results to clipboard"""
        text = f"Query: {results['query']}\n"
        text += f"Time: {results['timestamp']}\n"
        text += f"Documents Found: {len(results['documents'])}\n"
        text += "="*50 + "\n\n"
        
        for i, item in enumerate(results['documents'], 1):
            if isinstance(item, tuple):
                doc, score = item
                text += f"Result {i} (Score: {score:.4f}):\n"
            else:
                doc = item
                text += f"Result {i}:\n"
            
            text += doc.page_content + "\n"
            text += "-"*30 + "\n"
        
        st.code(text, language="text")
        st.success("Results copied! Select and copy the text above.")
    
    @staticmethod
    def _export_text(results: Dict[str, Any]):
        """Export results as downloadable text file"""
        text = f"Advanced RAG System - Search Results\n"
        text += f"Query: {results['query']}\n"
        text += f"Date: {results['timestamp']}\n"
        text += f"Search Time: {results['query_time']:.3f}s\n"
        text += f"Documents Found: {len(results['documents'])}\n"
        text += "="*60 + "\n\n"
        
        for i, item in enumerate(results['documents'], 1):
            if isinstance(item, tuple):
                doc, score = item
                text += f"\nResult {i} (Relevance Score: {score:.4f})\n"
            else:
                doc = item
                text += f"\nResult {i}\n"
            
            text += "-"*40 + "\n"
            text += doc.page_content + "\n"
            
            if doc.metadata:
                text += "\nMetadata:\n"
                for key, value in doc.metadata.items():
                    text += f"  {key}: {value}\n"
            
            text += "-"*40 + "\n"
        
        st.download_button(
            label="📥 Download Results",
            data=text,
            file_name=f"rag_results_{results['timestamp'].replace(' ', '_')}.txt",
            mime="text/plain"
        )