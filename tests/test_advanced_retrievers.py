"""
Tests for advanced retrievers module
"""
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

import pytest
from langchain.schema import Document
from src.vector_store import HybridVectorStore, EmbeddingManager
from src.advanced_retrievers import (
    MultiQueryRetriever,
    ParentDocumentRetriever,
    ContextualCompressor,
    Reranker,
    AdvancedRetrievalPipeline
)

# Test documents
TEST_DOCS = [
    Document(
        page_content="Artificial Intelligence (AI) is transforming industries through automation. Machine learning algorithms analyze data to identify patterns and make predictions. Deep learning uses neural networks with multiple layers.",
        metadata={"id": 1, "category": "AI"}
    ),
    Document(
        page_content="Natural Language Processing (NLP) enables computers to understand human language. Applications include sentiment analysis, machine translation, and chatbots. NLP uses various techniques like tokenization and parsing.",
        metadata={"id": 2, "category": "NLP"}
    ),
    Document(
        page_content="Retrieval-Augmented Generation (RAG) combines information retrieval with text generation. RAG systems retrieve relevant documents first, then generate responses based on context.",
        metadata={"id": 3, "category": "RAG"}
    ),
    Document(
        page_content="Vector databases store embeddings for similarity search. FAISS is a popular library for vector similarity search and clustering of dense vectors.",
        metadata={"id": 4, "category": "Vector DB"}
    ),
    Document(
        page_content="Large Language Models (LLMs) like GPT and Mistral generate human-like text. These models are trained on massive datasets and can perform various language tasks.",
        metadata={"id": 5, "category": "LLM"}
    )
]

@pytest.fixture
def vector_store():
    """Create test vector store"""
    emb_manager = EmbeddingManager(model_name="sentence-transformers/all-MiniLM-L6-v2")
    store = HybridVectorStore(embedding_manager=emb_manager)
    store.index_documents(TEST_DOCS)
    return store

@pytest.fixture
def embedding_manager():
    """Create embedding manager"""
    return EmbeddingManager(model_name="sentence-transformers/all-MiniLM-L6-v2")

class TestMultiQueryRetriever:
    """Test MultiQuery Retriever"""
    
    def test_initialization(self, vector_store):
        """Test initialization"""
        mq = MultiQueryRetriever(vector_store)
        assert mq.vector_store is not None
        assert mq.default_num_queries > 0
    
    def test_retrieve(self, vector_store):
        """Test retrieval"""
        mq = MultiQueryRetriever(vector_store)
        results = mq.retrieve("What is AI?", k=3)
        assert len(results) > 0
        assert isinstance(results[0], Document)
    
    def test_retrieve_with_scores(self, vector_store):
        """Test retrieval with scores"""
        mq = MultiQueryRetriever(vector_store)
        results = mq.retrieve("machine learning", k=3, return_scores=True)
        assert len(results) > 0
        assert isinstance(results[0], tuple)
        assert isinstance(results[0][1], float)

class TestParentDocumentRetriever:
    """Test Parent Document Retriever"""
    
    def test_create_structure(self, vector_store):
        """Test parent-child structure creation"""
        pd = ParentDocumentRetriever(vector_store)
        parents, children = pd.create_parent_child_structure(TEST_DOCS)
        
        assert len(parents) > 0
        assert len(children) > 0
        assert len(pd.parent_docs) > 0
    
    def test_retrieve_parents(self, vector_store):
        """Test parent document retrieval"""
        pd = ParentDocumentRetriever(vector_store)
        pd.create_parent_child_structure(TEST_DOCS)
        
        # Re-index with children
        emb_manager = EmbeddingManager(model_name="sentence-transformers/all-MiniLM-L6-v2")
        child_store = HybridVectorStore(embedding_manager=emb_manager)
        _, children = pd.create_parent_child_structure(TEST_DOCS)
        child_store.index_documents(children)
        pd.vector_store = child_store
        
        results = pd.retrieve_parents("AI and machine learning", k=2)
        assert len(results) > 0

class TestContextualCompressor:
    """Test Contextual Compressor"""
    
    def test_initialization(self, embedding_manager):
        """Test initialization"""
        compressor = ContextualCompressor(embedding_manager=embedding_manager)
        assert compressor.cross_encoder is not None
    
    def test_compress_embeddings(self, embedding_manager):
        """Test embedding-based compression"""
        compressor = ContextualCompressor(embedding_manager=embedding_manager)
        
        docs = TEST_DOCS[:3]
        compressed = compressor.compress_with_embeddings("AI technology", docs, threshold=0.3)
        assert len(compressed) <= len(docs)
    
    def test_compress_cross_encoder(self, embedding_manager):
        """Test cross-encoder compression"""
        compressor = ContextualCompressor(embedding_manager=embedding_manager)
        
        docs = TEST_DOCS[:3]
        compressed = compressor.compress_with_cross_encoder("NLP applications", docs, threshold=0.1)
        assert len(compressed) <= len(docs)
    
    def test_compress_pipeline(self, embedding_manager):
        """Test pipeline compression"""
        compressor = ContextualCompressor(embedding_manager=embedding_manager)
        
        docs = TEST_DOCS[:3]
        result = compressor.compress("vector search", docs, method="pipeline", threshold=0.2)
        assert len(result) <= len(docs)

class TestReranker:
    """Test Reranker"""
    
    def test_initialization(self):
        """Test initialization"""
        reranker = Reranker()
        assert reranker.model is not None
    
    def test_rerank(self):
        """Test reranking"""
        reranker = Reranker()
        
        query = "What is machine learning?"
        docs = TEST_DOCS[:5]
        
        reranked = reranker.rerank(query, docs, top_k=3, return_scores=True)
        assert len(reranked) == 3
        assert isinstance(reranked[0][1], float)
    
    def test_rerank_no_scores(self):
        """Test reranking without scores"""
        reranker = Reranker()
        
        query = "AI and NLP"
        docs = TEST_DOCS[:5]
        
        reranked = reranker.rerank(query, docs, top_k=3, return_scores=False)
        assert len(reranked) == 3
        assert isinstance(reranked[0], Document)
    
    def test_hybrid_rerank(self):
        """Test hybrid reranking"""
        import numpy as np
        
        reranker = Reranker()
        
        query = "vector databases"
        docs = TEST_DOCS[:5]
        bm25_scores = np.array([0.5, 0.3, 0.8, 0.9, 0.2])
        
        reranked = reranker.hybrid_rerank(query, docs, bm25_scores=bm25_scores, top_k=3)
        assert len(reranked) == 3

class TestAdvancedRetrievalPipeline:
    """Test complete pipeline"""
    
    def test_initialization(self, vector_store, embedding_manager):
        """Test pipeline initialization"""
        pipeline = AdvancedRetrievalPipeline(vector_store, embedding_manager)
        assert pipeline.vector_store is not None
        assert pipeline.multi_query is not None
        assert pipeline.compressor is not None
        assert pipeline.reranker is not None
    
    def test_retrieve_minimal(self, vector_store, embedding_manager):
        """Test minimal pipeline"""
        pipeline = AdvancedRetrievalPipeline(vector_store, embedding_manager)
        
        result = pipeline.retrieve(
            "What is AI?",
            k=3,
            stages={
                'multiquery': False,
                'parent_docs': False,
                'compression': False,
                'reranker': False
            }
        )
        
        assert 'documents' in result
        assert 'pipeline_info' in result
    
    def test_retrieve_full(self, vector_store, embedding_manager):
        """Test full pipeline"""
        pipeline = AdvancedRetrievalPipeline(vector_store, embedding_manager)
        
        result = pipeline.retrieve(
            "Explain RAG systems",
            k=3,
            stages={
                'multiquery': True,
                'parent_docs': False,
                'compression': True,
                'reranker': True
            },
            compression_kwargs={'method': 'cross_encoder', 'threshold': 0.1}
        )
        
        assert 'documents' in result
        assert len(result['pipeline_info']['stages_executed']) > 0
        assert result['pipeline_info']['total_time'] > 0

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])