"""
Analytics Component
"""
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from typing import List, Dict, Any
from pathlib import Path
import sys
import time

project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

class AnalyticsComponent:
    """Analytics and visualization dashboard"""
    
    @staticmethod
    def render():
        """Render analytics dashboard"""
        st.title("📊 Analytics Dashboard")
        st.markdown("---")
        
        if not st.session_state.documents_processed:
            st.info("Index documents and perform searches to see analytics.")
            return
        
        # Tabs for different analytics
        tab1, tab2, tab3 = st.tabs([
            "📈 Performance", 
            "🔍 Query Analysis", 
            "📄 Document Insights"
        ])
        
        with tab1:
            AnalyticsComponent._render_performance_tab()
        
        with tab2:
            AnalyticsComponent._render_query_analysis()
        
        with tab3:
            AnalyticsComponent._render_document_insights()
    
    @staticmethod
    def _render_performance_tab():
        """Render performance metrics"""
        st.subheader("System Performance")
        
        metrics = st.session_state.performance_metrics
        
        # Performance gauges
        col1, col2, col3 = st.columns(3)
        
        with col1:
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=metrics['total_queries'],
                title={"text": "Total Queries"},
                gauge={'axis': {'range': [0, max(100, metrics['total_queries'] * 2)]}}
            ))
            fig.update_layout(height=200)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=metrics['avg_retrieval_time'] * 1000,
                title={"text": "Avg Time (ms)"},
                gauge={'axis': {'range': [0, 5000]}}
            ))
            fig.update_layout(height=200)
            st.plotly_chart(fig, use_container_width=True)
        
        with col3:
            if st.session_state.vector_store:
                stats = st.session_state.vector_store.get_stats()
                fig = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=stats.get('bm25_doc_count', 0),
                    title={"text": "Documents"},
                    gauge={'axis': {'range': [0, max(1000, stats.get('bm25_doc_count', 0) * 2)]}}
                ))
                fig.update_layout(height=200)
                st.plotly_chart(fig, use_container_width=True)
        
        # Performance over time
        st.markdown("---")
        st.subheader("Performance History")
        
        if st.session_state.search_history:
            history_df = pd.DataFrame({
                'Query #': range(1, metrics['total_queries'] + 1),
                'Queries': metrics['total_queries']
            }, index=[0])
            
            st.line_chart(history_df)
    
    @staticmethod
    def _render_query_analysis():
        """Render query analysis"""
        st.subheader("Query Analysis")
        
        if not st.session_state.search_history:
            st.info("No queries executed yet. Perform searches to see analysis.")
            return
        
        # Query history
        st.markdown("**Recent Queries:**")
        query_df = pd.DataFrame(
            st.session_state.search_history[-10:],
            columns=['Query']
        )
        query_df.index = range(1, len(query_df) + 1)
        st.dataframe(query_df, use_container_width=True)
        
        # Query length distribution
        query_lengths = [len(q.split()) for q in st.session_state.search_history]
        
        fig = px.histogram(
            x=query_lengths,
            nbins=10,
            title="Query Length Distribution",
            labels={'x': 'Number of Words', 'y': 'Frequency'}
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Common terms
        st.markdown("**Common Query Terms:**")
        all_terms = []
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were', 'what', 'how', 'why', 'when', 'where', 'who'}
        
        for query in st.session_state.search_history:
            terms = [t.lower() for t in query.split() if t.lower() not in stop_words]
            all_terms.extend(terms)
        
        from collections import Counter
        term_counts = Counter(all_terms).most_common(15)
        
        if term_counts:
            term_df = pd.DataFrame(term_counts, columns=['Term', 'Count'])
            fig = px.bar(
                term_df, 
                x='Term', 
                y='Count',
                title="Most Common Query Terms"
            )
            st.plotly_chart(fig, use_container_width=True)
    
    @staticmethod
    def _render_document_insights():
        """Render document insights"""
        st.subheader("Document Insights")
        
        if not st.session_state.chunks:
            st.info("No documents indexed")
            return
        
        chunks = st.session_state.chunks
        
        # Chunk size distribution
        chunk_sizes = [len(c.page_content) for c in chunks]
        
        fig = px.histogram(
            x=chunk_sizes,
            nbins=20,
            title="Chunk Size Distribution",
            labels={'x': 'Chunk Size (characters)', 'y': 'Frequency'}
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Source analysis
        source_counts = {}
        for chunk in chunks:
            source = chunk.metadata.get('source', 'Unknown')
            source_counts[source] = source_counts.get(source, 0) + 1
        
        if source_counts:
            source_df = pd.DataFrame(
                list(source_counts.items()),
                columns=['Source', 'Chunks']
            ).sort_values('Chunks', ascending=False)
            
            fig = px.pie(
                source_df.head(10),
                values='Chunks',
                names='Source',
                title="Document Sources"
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Content analysis
        st.markdown("---")
        st.subheader("Content Statistics")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("Avg Chunk Size", f"{sum(chunk_sizes) // len(chunk_sizes):,} chars")
            st.metric("Total Content", f"{sum(chunk_sizes):,} chars")
        
        with col2:
            avg_words = sum(len(c.page_content.split()) for c in chunks) // len(chunks)
            st.metric("Avg Words/Chunk", f"{avg_words:,}")
            st.metric("Total Words", f"{sum(len(c.page_content.split()) for c in chunks):,}")