"""
Vector Store Module for Advanced RAG System
Implements FAISS vector store, BM25 index, and hybrid search
"""
import os
import pickle
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple, Union
import time

import numpy as np
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import OllamaEmbeddings
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi
import nltk
from nltk.tokenize import word_tokenize

from utils.helpers import load_config, timer_decorator

logger = logging.getLogger(__name__)

# Download NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)


class EmbeddingManager:
    """Manage embedding models"""
    
    def __init__(self, 
                 model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
                 device: str = "cpu",
                 use_ollama: bool = False):
        """
        Initialize embedding manager
        
        Args:
            model_name: Name of embedding model
            device: Device to use (cpu/cuda)
            use_ollama: Whether to use Ollama embeddings
        """
        self.model_name = model_name
        self.device = device
        self.use_ollama = use_ollama
        self.embedding_model = None
        
        self._initialize_model()
    
    def _initialize_model(self):
        """Initialize the embedding model"""
        try:
            if self.use_ollama:
                logger.info(f"Initializing Ollama embeddings with {self.model_name}")
                self.embedding_model = OllamaEmbeddings(
                    model=self.model_name,
                    base_url=os.getenv('OLLAMA_HOST', 'http://localhost:11434')
                )
            else:
                logger.info(f"Initializing HuggingFace embeddings: {self.model_name}")
                self.embedding_model = HuggingFaceEmbeddings(
                    model_name=self.model_name,
                    model_kwargs={'device': self.device},
                    encode_kwargs={'normalize_embeddings': True}
                )
            
            # Test embedding
            test_embedding = self.embedding_model.embed_query("test")
            logger.info(f"✅ Embedding model initialized. Dimension: {len(test_embedding)}")
            
        except Exception as e:
            logger.error(f"Failed to initialize embedding model: {e}")
            raise
    
    def embed_documents(self, documents: List[str]) -> List[List[float]]:
        """Embed a list of documents"""
        return self.embedding_model.embed_documents(documents)
    
    def embed_query(self, query: str) -> List[float]:
        """Embed a single query"""
        return self.embedding_model.embed_query(query)
    
    def get_embedding_dimension(self) -> int:
        """Get embedding dimension"""
        test_embedding = self.embed_query("test")
        return len(test_embedding)


class BM25Index:
    """BM25 sparse retrieval index"""
    
    def __init__(self):
        self.index = None
        self.documents = []
        self.tokenized_docs = []
    
    def tokenize(self, text: str) -> List[str]:
        """Tokenize text for BM25"""
        try:
            return word_tokenize(text.lower())
        except:
            return text.lower().split()
    
    @timer_decorator
    def build_index(self, documents: List[Document]):
        """
        Build BM25 index from documents
        
        Args:
            documents: List of documents to index
        """
        logger.info(f"Building BM25 index for {len(documents)} documents...")
        
        self.documents = documents
        self.tokenized_docs = [
            self.tokenize(doc.page_content) 
            for doc in documents
        ]
        
        self.index = BM25Okapi(self.tokenized_docs)
        logger.info("✅ BM25 index built successfully")
    
    def search(self, 
               query: str, 
               k: int = 5,
               threshold: float = 0.0) -> List[Tuple[Document, float]]:
        """
        Search BM25 index
        
        Args:
            query: Search query
            k: Number of results to return
            threshold: Minimum score threshold
        
        Returns:
            List of (document, score) tuples
        """
        if not self.index:
            logger.warning("BM25 index not initialized")
            return []
        
        tokenized_query = self.tokenize(query)
        scores = self.index.get_scores(tokenized_query)
        
        # Get top k indices
        if threshold > 0:
            valid_indices = np.where(scores > threshold)[0]
            top_indices = valid_indices[np.argsort(scores[valid_indices])[-k:][::-1]]
        else:
            top_indices = np.argsort(scores)[-k:][::-1]
        
        results = []
        for idx in top_indices:
            if idx < len(self.documents):
                results.append((self.documents[idx], float(scores[idx])))
        
        return results
    
    def get_batch_scores(self, query: str) -> np.ndarray:
        """Get BM25 scores for all documents"""
        if not self.index:
            return np.array([])
        
        tokenized_query = self.tokenize(query)
        return self.index.get_scores(tokenized_query)
    
    def save(self, path: str):
        """Save BM25 index to disk"""
        save_path = Path(path)
        save_path.mkdir(parents=True, exist_ok=True)
        
        with open(save_path / 'bm25_index.pkl', 'wb') as f:
            pickle.dump({
                'documents': self.documents,
                'tokenized_docs': self.tokenized_docs
            }, f)
        
        logger.info(f"BM25 index saved to {path}")
    
    def load(self, path: str):
        """Load BM25 index from disk"""
        load_path = Path(path) / 'bm25_index.pkl'
        
        if not load_path.exists():
            logger.warning(f"BM25 index not found at {path}")
            return False
        
        with open(load_path, 'rb') as f:
            data = pickle.load(f)
        
        self.documents = data['documents']
        self.tokenized_docs = data['tokenized_docs']
        self.index = BM25Okapi(self.tokenized_docs)
        
        logger.info(f"BM25 index loaded from {path}")
        return True


class FAISSVectorStore:
    """FAISS vector store wrapper"""
    
    def __init__(self, 
                 embedding_manager: EmbeddingManager,
                 dimension: Optional[int] = None):
        """
        Initialize FAISS vector store
        
        Args:
            embedding_manager: Embedding manager instance
            dimension: Vector dimension
        """
        self.embedding_manager = embedding_manager
        self.dimension = dimension or embedding_manager.get_embedding_dimension()
        self.vector_store = None
        self.index_path = None
    
    @timer_decorator
    def create_from_documents(self, 
                             documents: List[Document],
                             index_path: Optional[str] = None) -> FAISS:
        """
        Create FAISS index from documents
        
        Args:
            documents: List of documents
            index_path: Path to save index
        
        Returns:
            FAISS vector store
        """
        logger.info(f"Creating FAISS index from {len(documents)} documents...")
        
        self.vector_store = FAISS.from_documents(
            documents=documents,
            embedding=self.embedding_manager.embedding_model
        )
        
        if index_path:
            self.save(index_path)
        
        logger.info(f"✅ FAISS index created with {len(documents)} vectors")
        return self.vector_store
    
    @timer_decorator
    def add_documents(self, documents: List[Document]):
        """Add documents to existing index"""
        if not self.vector_store:
            logger.warning("Creating new index as none exists")
            return self.create_from_documents(documents)
        
        self.vector_store.add_documents(documents)
        logger.info(f"Added {len(documents)} documents to index")
    
    def search(self, 
               query: str, 
               k: int = 5,
               search_type: str = "similarity",
               filter_dict: Optional[Dict] = None,
               fetch_k: int = 20,
               lambda_mult: float = 0.5) -> List[Document]:
        """
        Search FAISS index
        
        Args:
            query: Search query
            k: Number of results
            search_type: Type of search (similarity, mmr, similarity_score)
            filter_dict: Metadata filter
            fetch_k: Number of documents to fetch for MMR
            lambda_mult: Diversity parameter for MMR
        
        Returns:
            List of documents
        """
        if not self.vector_store:
            logger.warning("FAISS index not initialized")
            return []
        
        if search_type == "similarity":
            return self.vector_store.similarity_search(
                query, k=k, filter=filter_dict
            )
        elif search_type == "similarity_score":
            return self.vector_store.similarity_search_with_score(
                query, k=k, filter=filter_dict
            )
        elif search_type == "mmr":
            return self.vector_store.max_marginal_relevance_search(
                query, k=k, fetch_k=fetch_k, 
                lambda_mult=lambda_mult, filter=filter_dict
            )
        else:
            raise ValueError(f"Unknown search type: {search_type}")
    
    def save(self, path: str):
        """Save FAISS index to disk"""
        if self.vector_store:
            self.vector_store.save_local(path)
            self.index_path = path
            logger.info(f"FAISS index saved to {path}")
    
    def load(self, path: str) -> bool:
        """Load FAISS index from disk"""
        if not os.path.exists(path):
            logger.warning(f"FAISS index not found at {path}")
            return False
        
        self.vector_store = FAISS.load_local(
            path,
            self.embedding_manager.embedding_model,
            allow_dangerous_deserialization=True
        )
        self.index_path = path
        logger.info(f"FAISS index loaded from {path}")
        return True
    
    def merge_from(self, other_store: 'FAISSVectorStore'):
        """Merge another FAISS store into this one"""
        if self.vector_store and other_store.vector_store:
            self.vector_store.merge_from(other_store.vector_store)
            logger.info("Merged vector stores")


class HybridVectorStore:
    """Hybrid vector store combining FAISS and BM25"""
    
    def __init__(self, 
                 embedding_manager: Optional[EmbeddingManager] = None,
                 config: Optional[Dict] = None):
        """
        Initialize hybrid vector store
        
        Args:
            embedding_manager: Embedding manager instance
            config: Configuration dictionary
        """
        self.config = config or load_config()
        
        # Initialize embedding manager
        if embedding_manager:
            self.embedding_manager = embedding_manager
        else:
            model_name = self.config.get('models', {}).get(
                'embedding_model', 
                'sentence-transformers/all-MiniLM-L6-v2'
            )
            self.embedding_manager = EmbeddingManager(model_name=model_name)
        
        # Initialize components
        self.faiss_store = FAISSVectorStore(
            self.embedding_manager,
            dimension=self.config.get('vector_store', {}).get('dimension', 384)
        )
        
        self.bm25_index = BM25Index()
        
        # Retrieval parameters
        retrieval_config = self.config.get('retrieval', {})
        self.default_k = retrieval_config.get('k_retrieval', 5)
        self.fetch_k = retrieval_config.get('fetch_k', 20)
        self.similarity_threshold = retrieval_config.get('similarity_threshold', 0.7)
        
        # BM25 parameters
        bm25_config = self.config.get('bm25', {})
        self.bm25_k1 = bm25_config.get('k1', 1.5)
        self.bm25_b = bm25_config.get('b', 0.75)
        
        logger.info("Hybrid vector store initialized")
    
    @timer_decorator
    def index_documents(self, 
                       documents: List[Document],
                       index_path: Optional[str] = None) -> None:
        """
        Index documents in both FAISS and BM25
        
        Args:
            documents: List of documents
            index_path: Path to save indices
        """
        logger.info(f"Indexing {len(documents)} documents...")
        
        # Create FAISS index
        if index_path:
            faiss_path = Path(index_path) / 'faiss'
            faiss_path.mkdir(parents=True, exist_ok=True)
            self.faiss_store.create_from_documents(documents, str(faiss_path))
        else:
            self.faiss_store.create_from_documents(documents)
        
        # Create BM25 index
        self.bm25_index.build_index(documents)
        
        logger.info("✅ Documents indexed successfully")
    
    def search(self,
               query: str,
               k: Optional[int] = None,
               search_type: str = "hybrid",
               alpha: float = 0.5,
               filter_dict: Optional[Dict] = None,
               return_scores: bool = False) -> Union[List[Document], List[Tuple[Document, float]]]:
        """
        Search using specified strategy
        
        Args:
            query: Search query
            k: Number of results
            search_type: Type of search (similarity, mmr, hybrid, bm25)
            alpha: Weight for dense scores in hybrid search
            filter_dict: Metadata filter
            return_scores: Whether to return scores
        
        Returns:
            List of documents or (document, score) tuples
        """
        k = k or self.default_k
        
        if search_type == "bm25":
            results = self.bm25_index.search(query, k=k)
            if return_scores:
                return results
            return [doc for doc, _ in results]
        
        elif search_type == "similarity":
            return self.faiss_store.search(
                query, k=k, search_type="similarity", 
                filter_dict=filter_dict
            )
        
        elif search_type == "similarity_score":
            return self.faiss_store.search(
                query, k=k, search_type="similarity_score",
                filter_dict=filter_dict
            )
        
        elif search_type == "mmr":
            return self.faiss_store.search(
                query, k=k, search_type="mmr",
                filter_dict=filter_dict, fetch_k=self.fetch_k
            )
        
        elif search_type == "hybrid":
            return self._hybrid_search(query, k, alpha, filter_dict, return_scores)
        
        else:
            raise ValueError(f"Unknown search type: {search_type}")
    
    def _hybrid_search(self,
                      query: str,
                      k: int,
                      alpha: float,
                      filter_dict: Optional[Dict],
                      return_scores: bool) -> Union[List[Document], List[Tuple[Document, float]]]:
        """
        Perform hybrid search combining dense and sparse retrieval
        
        Args:
            query: Search query
            k: Number of results
            alpha: Weight for dense scores
            filter_dict: Metadata filter
            return_scores: Whether to return scores
        
        Returns:
            Combined results
        """
        # Get dense retrieval results
        dense_results = self.faiss_store.search(
            query, 
            k=min(k * 3, self.fetch_k),
            search_type="similarity_score",
            filter_dict=filter_dict
        )
        
        # Get sparse retrieval results
        sparse_results = self.bm25_index.search(query, k=k * 3)
        
        # Combine results using reciprocal rank fusion
        combined_scores = {}
        
        # Process dense results
        for rank, (doc, score) in enumerate(dense_results, 1):
            doc_id = doc.metadata.get('chunk_id', id(doc))
            if doc_id not in combined_scores:
                combined_scores[doc_id] = {
                    'doc': doc,
                    'dense_score': 0,
                    'sparse_score': 0
                }
            # Reciprocal rank score
            combined_scores[doc_id]['dense_score'] = 1.0 / (rank + 60)  # 60 is a constant to prevent division by too small numbers
        
        # Process sparse results
        for rank, (doc, score) in enumerate(sparse_results, 1):
            doc_id = doc.metadata.get('chunk_id', id(doc))
            if doc_id not in combined_scores:
                combined_scores[doc_id] = {
                    'doc': doc,
                    'dense_score': 0,
                    'sparse_score': 0
                }
            combined_scores[doc_id]['sparse_score'] = 1.0 / (rank + 60)
        
        # Calculate final scores
        for doc_id, scores in combined_scores.items():
            scores['final_score'] = (
                alpha * scores['dense_score'] + 
                (1 - alpha) * scores['sparse_score']
            )
        
        # Sort by final score
        sorted_results = sorted(
            combined_scores.values(),
            key=lambda x: x['final_score'],
            reverse=True
        )[:k]
        
        if return_scores:
            return [(item['doc'], item['final_score']) for item in sorted_results]
        return [item['doc'] for item in sorted_results]
    
    def save(self, path: str):
        """Save both indices to disk"""
        save_path = Path(path)
        save_path.mkdir(parents=True, exist_ok=True)
        
        # Save FAISS
        faiss_path = save_path / 'faiss'
        self.faiss_store.save(str(faiss_path))
        
        # Save BM25
        self.bm25_index.save(str(save_path))
        
        # Save configuration
        with open(save_path / 'config.pkl', 'wb') as f:
            pickle.dump({
                'default_k': self.default_k,
                'fetch_k': self.fetch_k,
                'similarity_threshold': self.similarity_threshold
            }, f)
        
        logger.info(f"Hybrid vector store saved to {path}")
    
    def load(self, path: str) -> bool:
        """Load both indices from disk"""
        load_path = Path(path)
        
        if not load_path.exists():
            logger.warning(f"Vector store not found at {path}")
            return False
        
        try:
            # Load FAISS
            faiss_path = load_path / 'faiss'
            if not self.faiss_store.load(str(faiss_path)):
                logger.error("Failed to load FAISS index")
                return False
            
            # Load BM25
            if not self.bm25_index.load(str(load_path)):
                logger.warning("Failed to load BM25 index")
                # Continue even without BM25
            
            # Load configuration
            config_path = load_path / 'config.pkl'
            if config_path.exists():
                with open(config_path, 'rb') as f:
                    config = pickle.load(f)
                    self.default_k = config.get('default_k', self.default_k)
                    self.fetch_k = config.get('fetch_k', self.fetch_k)
                    self.similarity_threshold = config.get('similarity_threshold', self.similarity_threshold)
            
            logger.info(f"Hybrid vector store loaded from {path}")
            return True
            
        except Exception as e:
            logger.error(f"Error loading vector store: {e}")
            return False
    
    def get_stats(self) -> Dict[str, Any]:
        """Get vector store statistics"""
        stats = {
            'faiss_initialized': self.faiss_store.vector_store is not None,
            'bm25_initialized': self.bm25_index.index is not None,
            'embedding_dimension': self.embedding_manager.get_embedding_dimension(),
            'default_k': self.default_k,
            'fetch_k': self.fetch_k
        }
        
        if self.faiss_store.vector_store:
            stats['faiss_index_size'] = self.faiss_store.vector_store.index.ntotal
        
        if self.bm25_index.index:
            stats['bm25_doc_count'] = len(self.bm25_index.documents)
        
        return stats


class VectorStoreFactory:
    """Factory for creating vector stores"""
    
    @staticmethod
    def create_vector_store(store_type: str = "hybrid",
                          config: Optional[Dict] = None,
                          embedding_manager: Optional[EmbeddingManager] = None) -> HybridVectorStore:
        """
        Create a vector store instance
        
        Args:
            store_type: Type of vector store
            config: Configuration dictionary
            embedding_manager: Embedding manager
        
        Returns:
            Vector store instance
        """
        if store_type == "hybrid":
            return HybridVectorStore(
                embedding_manager=embedding_manager,
                config=config
            )
        else:
            raise ValueError(f"Unknown store type: {store_type}")