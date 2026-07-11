"""
Tests for vector store module
"""
import sys
from pathlib import Path
import tempfile
import os

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

import pytest
from langchain.schema import Document
from src.vector_store import (
    EmbeddingManager,
    BM25Index,
    FAISSVectorStore,
    HybridVectorStore,
    VectorStoreFactory
)

# Sample test data
TEST_DOCS = [
    Document(
        page_content="FAISS is a library for efficient similarity search and clustering of dense vectors.",
        metadata={"id": 1, "category": "technology", "source": "doc1"}
    ),
    Document(
        page_content="BM25 is a ranking function used by search engines to estimate the relevance of documents.",
        metadata={"id": 2, "category": "information_retrieval", "source": "doc2"}
    ),
    Document(
        page_content="Machine learning models can learn patterns from data to make predictions.",
        metadata={"id": 3, "category": "ai", "source": "doc3"}
    ),
    Document(
        page_content="Vector embeddings capture the semantic meaning of text for NLP tasks.",
        metadata={"id": 4, "category": "nlp", "source": "doc4"}
    ),
    Document(
        page_content="Hybrid search combines dense and sparse retrieval for better results.",
        metadata={"id": 5, "category": "search", "source": "doc5"}
    )
]

@pytest.fixture
def embedding_manager():
    """Create embedding manager for testing"""
    return EmbeddingManager(model_name="sentence-transformers/all-MiniLM-L6-v2")

@pytest.fixture
def bm25_index():
    """Create BM25 index for testing"""
    index = BM25Index()
    index.build_index(TEST_DOCS)
    return index

@pytest.fixture
def faiss_store(embedding_manager):
    """Create FAISS store for testing"""
    store = FAISSVectorStore(embedding_manager)
    store.create_from_documents(TEST_DOCS)
    return store

@pytest.fixture
def hybrid_store(embedding_manager):
    """Create hybrid store for testing"""
    store = HybridVectorStore(embedding_manager=embedding_manager)
    store.index_documents(TEST_DOCS)
    return store

class TestEmbeddingManager:
    """Test embedding manager"""
    
    def test_initialization(self, embedding_manager):
        """Test embedding manager initialization"""
        assert embedding_manager.embedding_model is not None
        dim = embedding_manager.get_embedding_dimension()
        assert dim > 0
    
    def test_embed_query(self, embedding_manager):
        """Test query embedding"""
        embedding = embedding_manager.embed_query("test query")
        assert len(embedding) > 0
        assert isinstance(embedding, list)
    
    def test_embed_documents(self, embedding_manager):
        """Test document embedding"""
        texts = ["doc1", "doc2", "doc3"]
        embeddings = embedding_manager.embed_documents(texts)
        assert len(embeddings) == 3
        assert len(embeddings[0]) > 0

class TestBM25Index:
    """Test BM25 index"""
    
    def test_index_creation(self, bm25_index):
        """Test BM25 index creation"""
        assert bm25_index.index is not None
        assert len(bm25_index.documents) == len(TEST_DOCS)
    
    def test_search(self, bm25_index):
        """Test BM25 search"""
        results = bm25_index.search("similarity search", k=2)
        assert len(results) == 2
        assert isinstance(results[0][1], float)  # Score should be float
    
    def test_search_with_threshold(self, bm25_index):
        """Test BM25 search with threshold"""
        results = bm25_index.search("machine learning", k=5, threshold=0.5)
        # Should return fewer results with threshold
        assert len(results) <= 5

class TestFAISSVectorStore:
    """Test FAISS vector store"""
    
    def test_creation(self, faiss_store):
        """Test FAISS store creation"""
        assert faiss_store.vector_store is not None
    
    def test_similarity_search(self, faiss_store):
        """Test similarity search"""
        results = faiss_store.search("search engine ranking", k=2)
        assert len(results) == 2
        assert isinstance(results[0], Document)
    
    def test_mmr_search(self, faiss_store):
        """Test MMR search"""
        results = faiss_store.search(
            "information retrieval", 
            k=2, 
            search_type="mmr",
            fetch_k=5
        )
        assert len(results) == 2
    
    def test_metadata_filtering(self, faiss_store):
        """Test metadata filtering"""
        results = faiss_store.search(
            "technology", 
            k=5,
            filter_dict={"category": "technology"}
        )
        assert len(results) > 0
        for doc in results:
            assert doc.metadata.get("category") == "technology"
    
    def test_save_load(self, faiss_store, embedding_manager):
        """Test save and load"""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Save
            faiss_store.save(tmpdir)
            
            # Load
            new_store = FAISSVectorStore(embedding_manager)
            success = new_store.load(tmpdir)
            assert success
            
            # Verify search works
            results = new_store.search("FAISS library", k=1)
            assert len(results) > 0

class TestHybridVectorStore:
    """Test hybrid vector store"""
    
    def test_index_documents(self, hybrid_store):
        """Test document indexing"""
        stats = hybrid_store.get_stats()
        assert stats['faiss_initialized']
        assert stats['bm25_initialized']
    
    def test_hybrid_search(self, hybrid_store):
        """Test hybrid search"""
        results = hybrid_store.search(
            "dense sparse retrieval",
            k=3,
            search_type="hybrid",
            alpha=0.5
        )
        assert len(results) == 3
    
    def test_search_types(self, hybrid_store):
        """Test different search types"""
        # Similarity search
        sim_results = hybrid_store.search("machine learning", k=2, search_type="similarity")
        assert len(sim_results) == 2
        
        # BM25 search
        bm25_results = hybrid_store.search("information retrieval", k=2, search_type="bm25")
        assert len(bm25_results) == 2
        
        # MMR search
        mmr_results = hybrid_store.search("NLP tasks", k=2, search_type="mmr")
        assert len(mmr_results) == 2
        
        # Score return
        score_results = hybrid_store.search(
            "search engine", k=2, 
            search_type="hybrid", return_scores=True
        )
        assert len(score_results) == 2
        assert isinstance(score_results[0][1], float)
    
    def test_save_load(self, hybrid_store, embedding_manager):
        """Test save and load"""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Save
            hybrid_store.save(tmpdir)
            
            # Load
            new_store = HybridVectorStore(embedding_manager=embedding_manager)
            success = new_store.load(tmpdir)
            assert success
            
            # Verify search works
            results = new_store.search("hybrid search", k=2)
            assert len(results) > 0
    
    def test_stats(self, hybrid_store):
        """Test statistics generation"""
        stats = hybrid_store.get_stats()
        assert 'faiss_initialized' in stats
        assert 'embedding_dimension' in stats
        assert stats['embedding_dimension'] > 0

class TestVectorStoreFactory:
    """Test vector store factory"""
    
    def test_create_store(self):
        """Test factory creation"""
        store = VectorStoreFactory.create_vector_store(store_type="hybrid")
        assert isinstance(store, HybridVectorStore)

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])