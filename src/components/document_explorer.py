"""
Document Explorer Component
"""
import streamlit as st
from typing import List, Dict, Any
from pathlib import Path
import sys
import pandas as pd

project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

class DocumentExplorer:
    """Document exploration and management interface"""
    
    @staticmethod
    def render():
        """Render document explorer"""
        st.title("📚 Document Explorer")
        st.markdown("---")
        
        if not st.session_state.documents_processed:
            st.info("📄 No documents indexed yet. Upload documents using the sidebar to get started!")
            DocumentExplorer._render_upload_prompt()
            return
        
        # Document statistics
        DocumentExplorer._render_statistics()
        
        st.markdown("---")
        
        # Document browser
        DocumentExplorer._render_document_browser()
        
        st.markdown("---")
        
        # Metadata analysis
        DocumentExplorer._render_metadata_analysis()
    
    @staticmethod
    def _render_upload_prompt():
        """Render upload prompt"""
        st.markdown("""
        ### Getting Started
        
        1. **Initialize System**: Click 'Initialize System' in the sidebar
        2. **Upload Documents**: Use the file uploader in the sidebar
        3. **Start Searching**: Switch to the Search tab
        
        Supported formats: PDF, TXT, DOCX, CSV
        """)
    
    @staticmethod
    def _render_statistics():
        """Render document statistics"""
        if not st.session_state.chunks:
            return
        
        chunks = st.session_state.chunks
        
        # Basic stats
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Chunks", len(chunks))
        
        with col2:
            total_chars = sum(len(c.page_content) for c in chunks)
            st.metric("Total Characters", f"{total_chars:,}")
        
        with col3:
            avg_chars = total_chars // len(chunks) if chunks else 0
            st.metric("Avg Chunk Size", f"{avg_chars:,}")
        
        with col4:
            unique_sources = len(set(c.metadata.get('source', '') for c in chunks))
            st.metric("Unique Sources", unique_sources)
        
        # Source distribution
        st.markdown("---")
        st.subheader("📊 Source Distribution")
        
        source_counts = {}
        for chunk in chunks:
            source = chunk.metadata.get('source', 'Unknown')
            source_counts[source] = source_counts.get(source, 0) + 1
        
        if source_counts:
            source_df = pd.DataFrame(
                list(source_counts.items()),
                columns=['Source', 'Chunks']
            ).sort_values('Chunks', ascending=False)
            
            st.bar_chart(source_df.set_index('Source'))
    
    @staticmethod
    def _render_document_browser():
        """Render document browser"""
        st.subheader("🔍 Browse Documents")
        
        if not st.session_state.chunks:
            st.info("No documents to browse")
            return
        
        # Search within documents
        search_term = st.text_input(
            "Search within documents:",
            placeholder="Filter chunks by content..."
        )
        
        # Filter by metadata
        col1, col2 = st.columns(2)
        
        with col1:
            # Get unique sources
            sources = list(set(
                c.metadata.get('source', 'Unknown') 
                for c in st.session_state.chunks
            ))
            selected_source = st.selectbox(
                "Filter by Source",
                ["All"] + sorted(sources)
            )
        
        with col2:
            # Get unique categories if available
            categories = set()
            for c in st.session_state.chunks:
                if 'category' in c.metadata:
                    categories.add(c.metadata['category'])
            
            if categories:
                selected_category = st.selectbox(
                    "Filter by Category",
                    ["All"] + sorted(categories)
                )
            else:
                selected_category = "All"
        
        # Apply filters
        filtered_chunks = st.session_state.chunks
        
        if search_term:
            filtered_chunks = [
                c for c in filtered_chunks 
                if search_term.lower() in c.page_content.lower()
            ]
        
        if selected_source != "All":
            filtered_chunks = [
                c for c in filtered_chunks 
                if c.metadata.get('source', '') == selected_source
            ]
        
        if selected_category != "All" and categories:
            filtered_chunks = [
                c for c in filtered_chunks 
                if c.metadata.get('category', '') == selected_category
            ]
        
        # Display results
        st.markdown(f"**Showing {len(filtered_chunks)} of {len(st.session_state.chunks)} chunks**")
        
        # Pagination
        chunks_per_page = 5
        total_pages = max(1, (len(filtered_chunks) + chunks_per_page - 1) // chunks_per_page)
        
        if total_pages > 1:
            page = st.number_input(
                "Page",
                min_value=1,
                max_value=total_pages,
                value=1
            )
        else:
            page = 1
        
        start_idx = (page - 1) * chunks_per_page
        end_idx = min(start_idx + chunks_per_page, len(filtered_chunks))
        
        for i, chunk in enumerate(filtered_chunks[start_idx:end_idx], start_idx + 1):
            with st.expander(f"Chunk {i} - {chunk.metadata.get('source', 'Unknown')}"):
                st.markdown(chunk.page_content)
                
                if chunk.metadata:
                    st.markdown("**Metadata:**")
                    cols = st.columns(3)
                    meta_items = list(chunk.metadata.items())[:6]
                    for j, (key, value) in enumerate(meta_items):
                        with cols[j % 3]:
                            st.caption(f"{key}: {value}")
    
    @staticmethod
    def _render_metadata_analysis():
        """Render metadata analysis"""
        if not st.session_state.chunks:
            return
        
        st.subheader("📋 Metadata Analysis")
        
        # Collect all metadata keys
        all_keys = set()
        key_values = {}
        
        for chunk in st.session_state.chunks[:500]:  # Analyze first 500 chunks
            for key, value in chunk.metadata.items():
                all_keys.add(key)
                if key not in key_values:
                    key_values[key] = set()
                key_values[key].add(str(value))
        
        # Display metadata statistics
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Available Metadata Fields:**")
            for key in sorted(all_keys):
                st.markdown(f"- {key} ({len(key_values[key])} unique values)")
        
        with col2:
            # Select a field to analyze
            if all_keys:
                selected_key = st.selectbox(
                    "Analyze Field",
                    sorted(all_keys)
                )
                
                if selected_key in key_values:
                    # Count values
                    value_counts = {}
                    for chunk in st.session_state.chunks[:500]:
                        if selected_key in chunk.metadata:
                            value = str(chunk.metadata[selected_key])
                            value_counts[value] = value_counts.get(value, 0) + 1
                    
                    if value_counts:
                        value_df = pd.DataFrame(
                            list(value_counts.items()),
                            columns=['Value', 'Count']
                        ).sort_values('Count', ascending=False).head(10)
                        
                        st.bar_chart(value_df.set_index('Value'))