"""
Advanced Retrievers Module for RAG System
Implements MultiQuery, Parent Document, Contextual Compression, and Reranker
"""
import os
import time
import hashlib
import logging
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path

import numpy as np
from langchain.schema import Document
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_community.llms import Ollama
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.retrievers import ContextualCompressionRetriever
from sentence_transformers import CrossEncoder

from utils.helpers import load_config, timer_decorator

logger = logging.getLogger(__name__)


class MultiQueryRetriever:
    """Generate multiple query variations for better retrieval"""
    
    def __init__(self, 
                 vector_store,
                 llm=None,
                 config: Optional[Dict] = None):
        """
        Initialize MultiQuery Retriever
        
        Args:
            vector_store: Hybrid vector store instance
            llm: Language model for query generation
            config: Configuration dictionary
        """
        self.vector_store = vector_store
        self.config = config or load_config()
        
        # Initialize LLM
        if llm:
            self.llm = llm
        else:
            model_name = self.config.get('models', {}).get('llm', 'mistral')
            self.llm = Ollama(
                model=model_name,
                base_url=os.getenv('OLLAMA_HOST', 'http://localhost:11434'),
                temperature=0.3
            )
        
        # MultiQuery settings
        mq_config = self.config.get('multi_query', {})
        self.default_num_queries = mq_config.get('num_queries', 3)
        self.include_original = mq_config.get('include_original', True)
        
        # Create prompt template
        self.prompt_template = PromptTemplate(
            input_variables=["query", "num_queries"],
            template="""Generate {num_queries} different versions of the following question.
            Each version should rephrase the question from a different perspective while 
            maintaining the original intent. Make the variations diverse in wording and approach.
            
            Original question: {query}
            
            Generate exactly {num_queries} variations, one per line:
            """
        )
        
        self.query_chain = LLMChain(llm=self.llm, prompt=self.prompt_template)
        
        logger.info(f"MultiQuery Retriever initialized with {self.default_num_queries} queries")
    
    @timer_decorator
    def generate_queries(self, 
                        original_query: str, 
                        num_queries: Optional[int] = None) -> List[str]:
        """
        Generate multiple query variations
        
        Args:
            original_query: Original user query
            num_queries: Number of variations to generate
        
        Returns:
            List of query variations including original
        """
        num_queries = num_queries or self.default_num_queries
        
        logger.info(f"Generating {num_queries} query variations for: '{original_query}'")
        
        try:
            response = self.query_chain.run(
                query=original_query, 
                num_queries=num_queries
            )
            
            # Parse variations
            variations = [q.strip() for q in response.split('\n') if q.strip()]
            variations = variations[:num_queries]
            
            # Include original query
            if self.include_original:
                all_queries = [original_query] + variations
            else:
                all_queries = variations
            
            logger.info(f"Generated {len(all_queries)} query variations")
            return all_queries
            
        except Exception as e:
            logger.error(f"Error generating queries: {e}")
            return [original_query]
    
    @timer_decorator
    def retrieve(self, 
                query: str,
                k: int = 5,
                num_queries: Optional[int] = None,
                alpha: float = 0.5,
                return_scores: bool = False) -> Union[List[Document], List[Tuple[Document, float]]]:
        """
        Retrieve documents using multiple query variations
        
        Args:
            query: Original query
            k: Number of documents per query
            num_queries: Number of query variations
            alpha: Hybrid search weight
            return_scores: Whether to return scores
        
        Returns:
            List of unique documents or (document, score) tuples
        """
        logger.info(f"MultiQuery retrieval for: '{query}'")
        
        # Generate query variations
        queries = self.generate_queries(query, num_queries)
        
        # Retrieve documents for each query
        all_docs = {}
        
        for q in queries:
            results = self.vector_store.search(
                q, 
                k=k, 
                search_type="hybrid",
                alpha=alpha,
                return_scores=True
            )
            
            for doc, score in results:
                doc_id = doc.metadata.get('chunk_id', id(doc))
                
                # Keep best score for each document
                if doc_id not in all_docs or score > all_docs[doc_id][1]:
                    all_docs[doc_id] = (doc, score)
        
        # Sort by score
        sorted_docs = sorted(
            all_docs.values(),
            key=lambda x: x[1],
            reverse=True
        )[:k*2]
        
        if return_scores:
            return sorted_docs
        return [doc for doc, _ in sorted_docs]


class ParentDocumentRetriever:
    """Retrieve parent documents for better context"""
    
    def __init__(self,
                 vector_store,
                 config: Optional[Dict] = None):
        """
        Initialize Parent Document Retriever
        
        Args:
            vector_store: Vector store instance
            config: Configuration dictionary
        """
        self.vector_store = vector_store
        self.config = config or load_config()
        
        # Parent document settings
        pd_config = self.config.get('parent_document', {})
        self.parent_chunk_size = pd_config.get('parent_chunk_size', 2000)
        self.child_chunk_size = pd_config.get('child_chunk_size', 500)
        self.chunk_overlap = pd_config.get('child_chunk_overlap', 50)
        
        # Initialize splitters
        self.parent_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.parent_chunk_size,
            chunk_overlap=self.chunk_overlap * 2,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
        
        self.child_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.child_chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
        
        # Storage for parent-child mapping
        self.parent_docs = {}
        self.child_to_parent = {}
        
        logger.info("Parent Document Retriever initialized")
    
    @timer_decorator
    def create_parent_child_structure(self, 
                                     documents: List[Document]) -> Tuple[List[Document], List[Document]]:
        """
        Create parent-child document structure
        
        Args:
            documents: Original documents
        
        Returns:
            Tuple of (parent_docs, child_docs)
        """
        logger.info(f"Creating parent-child structure for {len(documents)} documents")
        
        parent_docs = []
        child_docs = []
        
        for doc_idx, doc in enumerate(documents):
            # Split into parent chunks
            doc_parents = self.parent_splitter.split_documents([doc])
            
            for parent_idx, parent in enumerate(doc_parents):
                parent_id = f"parent_{doc_idx}_{parent_idx}"
                
                # Add parent metadata
                parent.metadata.update({
                    'parent_id': parent_id,
                    'doc_type': 'parent',
                    'content_hash': hashlib.md5(parent.page_content.encode()).hexdigest(),
                    'original_source': doc.metadata.get('source', 'unknown')
                })
                
                # Split parent into child chunks
                doc_children = self.child_splitter.split_documents([parent])
                
                for child_idx, child in enumerate(doc_children):
                    child_id = f"child_{doc_idx}_{parent_idx}_{child_idx}"
                    
                    # Add child metadata
                    child.metadata.update({
                        'child_id': child_id,
                        'parent_id': parent_id,
                        'doc_type': 'child',
                        'parent_content_hash': parent.metadata['content_hash']
                    })
                    
                    # Store mapping
                    self.child_to_parent[child_id] = parent_id
                    child_docs.append(child)
                
                # Store parent
                self.parent_docs[parent_id] = parent
                parent_docs.append(parent)
        
        logger.info(f"Created {len(parent_docs)} parent and {len(child_docs)} child documents")
        return parent_docs, child_docs
    
    @timer_decorator
    def retrieve_parents(self,
                        query: str,
                        k: int = 5,
                        num_children: int = 10) -> List[Document]:
        """
        Retrieve parent documents based on child document matches
        
        Args:
            query: Search query
            k: Number of parent documents to return
            num_children: Number of child documents to retrieve
        
        Returns:
            List of parent documents
        """
        logger.info(f"Parent document retrieval for: '{query}'")
        
        # Retrieve child documents
        child_results = self.vector_store.search(
            query,
            k=num_children,
            search_type="hybrid",
            return_scores=True
        )
        
        # Collect unique parent documents
        parent_ids = set()
        parent_results = []
        
        for child_doc, score in child_results:
            parent_id = child_doc.metadata.get('parent_id')
            
            if parent_id and parent_id not in parent_ids:
                parent_ids.add(parent_id)
                
                if parent_id in self.parent_docs:
                    parent_doc = self.parent_docs[parent_id]
                    
                    # Add retrieval metadata
                    parent_doc.metadata.update({
                        'retrieval_score': score,
                        'retrieved_via_child': child_doc.metadata.get('child_id'),
                        'retrieval_query': query
                    })
                    
                    parent_results.append((parent_doc, score))
        
        # Sort by score and limit
        parent_results.sort(key=lambda x: x[1], reverse=True)
        parent_docs = [doc for doc, _ in parent_results[:k]]
        
        logger.info(f"Retrieved {len(parent_docs)} parent documents from {len(child_results)} children")
        return parent_docs


class ContextualCompressor:
    """Compress retrieved documents to extract relevant information"""
    
    def __init__(self,
                 embedding_manager=None,
                 llm=None,
                 config: Optional[Dict] = None):
        """
        Initialize Contextual Compressor
        
        Args:
            embedding_manager: Embedding manager instance
            llm: Language model
            config: Configuration dictionary
        """
        self.embedding_manager = embedding_manager
        self.config = config or load_config()
        
        # Initialize LLM
        if llm:
            self.llm = llm
        else:
            model_name = self.config.get('models', {}).get('llm', 'mistral')
            self.llm = Ollama(
                model=model_name,
                base_url=os.getenv('OLLAMA_HOST', 'http://localhost:11434'),
                temperature=0.1
            )
        
        # Compression settings
        comp_config = self.config.get('compression', {})
        self.max_tokens = comp_config.get('max_tokens', 1000)
        self.compression_model = comp_config.get(
            'compression_model', 
            'cross-encoder/ms-marco-MiniLM-L-6-v2'
        )
        
        # Initialize cross-encoder
        try:
            self.cross_encoder = CrossEncoder(self.compression_model)
            logger.info(f"Loaded compression model: {self.compression_model}")
        except Exception as e:
            logger.warning(f"Could not load compression model: {e}")
            self.cross_encoder = None
        
        # LLM extraction prompt
        self.extraction_prompt = PromptTemplate(
            input_variables=["question", "context"],
            template="""Extract only the information from the following text that is 
            directly relevant to answering the question. Remove any irrelevant details, 
            examples, or tangential information. Keep the extracted information concise 
            but complete enough to answer the question.
            
            Question: {question}
            
            Text: {context}
            
            Extracted relevant information:"""
        )
        
        self.extraction_chain = LLMChain(llm=self.llm, prompt=self.extraction_prompt)
        
        logger.info("Contextual Compressor initialized")
    
    @timer_decorator
    def compress_with_llm(self,
                         query: str,
                         documents: List[Document]) -> List[Document]:
        """
        Compress documents using LLM extraction
        
        Args:
            query: User query
            documents: Documents to compress
        
        Returns:
            Compressed documents
        """
        logger.info(f"Compressing {len(documents)} documents with LLM")
        
        compressed_docs = []
        
        for i, doc in enumerate(documents):
            try:
                # Extract relevant information
                extracted = self.extraction_chain.run(
                    question=query,
                    context=doc.page_content[:1500]  # Limit input length
                )
                
                # Create compressed document
                compressed_doc = Document(
                    page_content=extracted.strip(),
                    metadata={
                        **doc.metadata,
                        'compressed': True,
                        'compression_method': 'llm_extraction',
                        'original_length': len(doc.page_content),
                        'compressed_length': len(extracted)
                    }
                )
                compressed_docs.append(compressed_doc)
                
            except Exception as e:
                logger.error(f"Error compressing document {i}: {e}")
                compressed_docs.append(doc)
        
        return compressed_docs
    
    @timer_decorator
    def compress_with_embeddings(self,
                                query: str,
                                documents: List[Document],
                                threshold: float = 0.5) -> List[Document]:
        """
        Filter documents using embedding similarity
        
        Args:
            query: User query
            documents: Documents to filter
            threshold: Similarity threshold
        
        Returns:
            Filtered documents
        """
        if not self.embedding_manager:
            logger.warning("Embedding manager not available for compression")
            return documents
        
        logger.info(f"Filtering {len(documents)} documents with embeddings (threshold: {threshold})")
        
        query_embedding = self.embedding_manager.embed_query(query)
        filtered_docs = []
        
        for doc in documents:
            doc_embedding = self.embedding_manager.embed_query(doc.page_content[:500])
            
            # Calculate cosine similarity
            similarity = np.dot(query_embedding, doc_embedding) / (
                np.linalg.norm(query_embedding) * np.linalg.norm(doc_embedding)
            )
            
            if similarity >= threshold:
                doc.metadata['embedding_similarity'] = float(similarity)
                filtered_docs.append(doc)
        
        logger.info(f"Filtered to {len(filtered_docs)} documents")
        return filtered_docs
    
    @timer_decorator
    def compress_with_cross_encoder(self,
                                   query: str,
                                   documents: List[Document],
                                   threshold: float = 0.3) -> List[Document]:
        """
        Filter documents using cross-encoder relevance scores
        
        Args:
            query: User query
            documents: Documents to filter
            threshold: Relevance threshold
        
        Returns:
            Filtered documents
        """
        if not self.cross_encoder:
            logger.warning("Cross-encoder not available")
            return documents
        
        logger.info(f"Filtering {len(documents)} documents with cross-encoder")
        
        # Create query-document pairs
        pairs = [[query, doc.page_content[:500]] for doc in documents]
        
        # Get relevance scores
        scores = self.cross_encoder.predict(pairs)
        
        filtered_docs = []
        for doc, score in zip(documents, scores):
            if score >= threshold:
                doc.metadata['relevance_score'] = float(score)
                filtered_docs.append(doc)
        
        logger.info(f"Cross-encoder filtered to {len(filtered_docs)} documents")
        return filtered_docs
    
    @timer_decorator
    def compress(self,
                query: str,
                documents: List[Document],
                method: str = "cross_encoder",
                threshold: float = 0.3,
                combine_methods: bool = False) -> List[Document]:
        """
        Compress documents using specified method
        
        Args:
            query: User query
            documents: Documents to compress
            method: Compression method (llm, embeddings, cross_encoder, pipeline)
            threshold: Filtering threshold
            combine_methods: Whether to combine multiple methods
        
        Returns:
            Compressed documents
        """
        original_count = len(documents)
        
        if method == "llm":
            compressed = self.compress_with_llm(query, documents)
        elif method == "embeddings":
            compressed = self.compress_with_embeddings(query, documents, threshold)
        elif method == "cross_encoder":
            compressed = self.compress_with_cross_encoder(query, documents, threshold)
        elif method == "pipeline":
            # Apply multiple methods
            compressed = self.compress_with_embeddings(query, documents, threshold)
            if compressed and combine_methods:
                compressed = self.compress_with_cross_encoder(query, compressed, threshold)
        else:
            compressed = documents
        
        reduction = (1 - len(compressed) / original_count) * 100 if original_count > 0 else 0
        logger.info(f"Compression: {original_count} → {len(compressed)} documents ({reduction:.1f}% reduction)")
        
        return compressed


class Reranker:
    """Rerank retrieved documents for improved relevance"""
    
    def __init__(self, 
                 model_name: Optional[str] = None,
                 config: Optional[Dict] = None):
        """
        Initialize Reranker
        
        Args:
            model_name: Cross-encoder model name
            config: Configuration dictionary
        """
        self.config = config or load_config()
        
        # Reranker settings
        reranker_config = self.config.get('reranker', {})
        self.model_name = model_name or reranker_config.get(
            'model', 
            'cross-encoder/ms-marco-MiniLM-L-6-v2'
        )
        self.top_k = reranker_config.get('top_k', 5)
        self.score_threshold = reranker_config.get('score_threshold', 0.5)
        
        # Initialize model
        try:
            self.model = CrossEncoder(self.model_name)
            logger.info(f"Loaded reranker model: {self.model_name}")
        except Exception as e:
            logger.warning(f"Could not load reranker model: {e}")
            self.model = None
    
    @timer_decorator
    def rerank(self,
              query: str,
              documents: List[Document],
              top_k: Optional[int] = None,
              score_threshold: Optional[float] = None,
              return_scores: bool = True) -> Union[List[Document], List[Tuple[Document, float]]]:
        """
        Rerank documents using cross-encoder
        
        Args:
            query: Search query
            documents: Documents to rerank
            top_k: Number of top documents
            score_threshold: Minimum score threshold
            return_scores: Whether to return scores
        
        Returns:
            Reranked documents
        """
        top_k = top_k or self.top_k
        score_threshold = score_threshold or self.score_threshold
        
        logger.info(f"Reranking {len(documents)} documents for: '{query}'")
        
        if not self.model or len(documents) == 0:
            logger.warning("Reranker not available or no documents")
            if return_scores:
                return [(doc, 1.0) for doc in documents[:top_k]]
            return documents[:top_k]
        
        # Create query-document pairs
        pairs = [[query, doc.page_content[:500]] for doc in documents]
        
        # Get relevance scores
        scores = self.model.predict(pairs)
        
        # Create document-score pairs
        doc_scores = list(zip(documents, scores))
        
        # Sort by score (descending)
        doc_scores.sort(key=lambda x: x[1], reverse=True)
        
        # Apply threshold and top_k
        filtered_scores = [
            (doc, float(score))
            for doc, score in doc_scores
            if score >= score_threshold
        ][:top_k]
        
        logger.info(f"Reranked to {len(filtered_scores)} documents")
        
        if return_scores:
            return filtered_scores
        return [doc for doc, _ in filtered_scores]
    
    @timer_decorator
    def hybrid_rerank(self,
                     query: str,
                     documents: List[Document],
                     bm25_scores: Optional[np.ndarray] = None,
                     top_k: Optional[int] = None,
                     alpha: float = 0.6) -> List[Tuple[Document, float]]:
        """
        Hybrid reranking with cross-encoder and BM25 scores
        
        Args:
            query: Search query
            documents: Documents to rerank
            bm25_scores: BM25 scores for documents
            top_k: Number of results
            alpha: Weight for cross-encoder scores
        
        Returns:
            Reranked documents with combined scores
        """
        top_k = top_k or self.top_k
        logger.info(f"Hybrid reranking with alpha={alpha}")
        
        if not self.model:
            return [(doc, 1.0) for doc in documents[:top_k]]
        
        # Get cross-encoder scores
        pairs = [[query, doc.page_content[:500]] for doc in documents]
        ce_scores = self.model.predict(pairs)
        
        # Normalize cross-encoder scores
        if ce_scores.max() > ce_scores.min():
            ce_scores_norm = (ce_scores - ce_scores.min()) / (ce_scores.max() - ce_scores.min())
        else:
            ce_scores_norm = np.ones_like(ce_scores)
        
        # Combine with BM25 scores if available
        if bm25_scores is not None and len(bm25_scores) == len(documents):
            if bm25_scores.max() > bm25_scores.min():
                bm25_norm = (bm25_scores - bm25_scores.min()) / (bm25_scores.max() - bm25_scores.min())
            else:
                bm25_norm = np.ones_like(bm25_scores)
            
            combined_scores = alpha * ce_scores_norm + (1 - alpha) * bm25_norm
        else:
            combined_scores = ce_scores_norm
        
        # Create and sort document-score pairs
        doc_scores = list(zip(documents, combined_scores))
        doc_scores.sort(key=lambda x: x[1], reverse=True)
        
        return [(doc, float(score)) for doc, score in doc_scores[:top_k]]


class AdvancedRetrievalPipeline:
    """Complete advanced retrieval pipeline"""
    
    def __init__(self,
                 vector_store,
                 embedding_manager=None,
                 llm=None,
                 config: Optional[Dict] = None):
        """
        Initialize advanced retrieval pipeline
        
        Args:
            vector_store: Hybrid vector store
            embedding_manager: Embedding manager
            llm: Language model
            config: Configuration
        """
        self.vector_store = vector_store
        self.embedding_manager = embedding_manager
        self.config = config or load_config()
        
        # Initialize components
        self.multi_query = MultiQueryRetriever(vector_store, llm, config)
        self.parent_doc = ParentDocumentRetriever(vector_store, config)
        self.compressor = ContextualCompressor(embedding_manager, llm, config)
        self.reranker = Reranker(config=config)
        
        # Pipeline settings
        self.stages_enabled = {
            'multiquery': True,
            'parent_docs': False,  # Only if parent structure created
            'compression': True,
            'reranker': True
        }
        
        self.parent_structure_created = False
        
        logger.info("Advanced Retrieval Pipeline initialized")
    
    def create_parent_structure(self, documents: List[Document]):
        """Create parent-child document structure"""
        self.parent_doc.create_parent_child_structure(documents)
        self.parent_structure_created = True
        logger.info("Parent-child structure created")
    
    @timer_decorator
    def retrieve(self,
                query: str,
                k: int = 5,
                stages: Optional[Dict[str, bool]] = None,
                multiquery_kwargs: Optional[Dict] = None,
                compression_kwargs: Optional[Dict] = None,
                reranker_kwargs: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Execute complete retrieval pipeline
        
        Args:
            query: User query
            k: Number of final results
            stages: Which stages to enable
            multiquery_kwargs: MultiQuery parameters
            compression_kwargs: Compression parameters
            reranker_kwargs: Reranker parameters
        
        Returns:
            Dictionary with results and metadata
        """
        stages = stages or self.stages_enabled
        pipeline_info = {
            'query': query,
            'stages_executed': [],
            'timings': {},
            'document_counts': {}
        }
        
        logger.info(f"Starting advanced retrieval pipeline for: '{query}'")
        
        # Stage 1: MultiQuery Retrieval
        if stages.get('multiquery', True):
            logger.info("Stage 1: MultiQuery Retrieval")
            start_time = time.time()
            
            mq_kwargs = multiquery_kwargs or {}
            retrieved_docs = self.multi_query.retrieve(
                query, 
                k=k*2,
                **mq_kwargs
            )
            
            stage_time = time.time() - start_time
            pipeline_info['stages_executed'].append('multiquery')
            pipeline_info['timings']['multiquery'] = stage_time
            pipeline_info['document_counts']['after_multiquery'] = len(retrieved_docs)
            
            logger.info(f"MultiQuery: {len(retrieved_docs)} docs in {stage_time:.3f}s")
        else:
            retrieved_docs = self.vector_store.search(query, k=k*2, search_type="hybrid")
        
        # Stage 2: Parent Document Retrieval
        if stages.get('parent_docs', False) and self.parent_structure_created:
            logger.info("Stage 2: Parent Document Retrieval")
            start_time = time.time()
            
            parent_docs = self.parent_doc.retrieve_parents(query, k=k*2)
            if parent_docs:
                retrieved_docs = parent_docs
            
            stage_time = time.time() - start_time
            pipeline_info['stages_executed'].append('parent_docs')
            pipeline_info['timings']['parent_docs'] = stage_time
            pipeline_info['document_counts']['after_parent_docs'] = len(retrieved_docs)
        
        # Stage 3: Contextual Compression
        if stages.get('compression', True):
            logger.info("Stage 3: Contextual Compression")
            start_time = time.time()
            
            comp_kwargs = compression_kwargs or {'method': 'cross_encoder', 'threshold': 0.3}
            compressed_docs = self.compressor.compress(query, retrieved_docs, **comp_kwargs)
            
            if compressed_docs:
                retrieved_docs = compressed_docs
            
            stage_time = time.time() - start_time
            pipeline_info['stages_executed'].append('compression')
            pipeline_info['timings']['compression'] = stage_time
            pipeline_info['document_counts']['after_compression'] = len(retrieved_docs)
        
        # Stage 4: Reranker
        if stages.get('reranker', True):
            logger.info("Stage 4: Reranking")
            start_time = time.time()
            
            reranker_kw = reranker_kwargs or {'top_k': k, 'return_scores': True}
            final_docs = self.reranker.rerank(query, retrieved_docs, **reranker_kw)
            
            stage_time = time.time() - start_time
            pipeline_info['stages_executed'].append('reranker')
            pipeline_info['timings']['reranker'] = stage_time
            pipeline_info['document_counts']['final'] = len(final_docs)
        else:
            final_docs = [(doc, 1.0) for doc in retrieved_docs[:k]]
        
        # Calculate total time
        pipeline_info['total_time'] = sum(pipeline_info['timings'].values())
        
        logger.info(f"Pipeline complete in {pipeline_info['total_time']:.3f}s")
        logger.info(f"Final documents: {len(final_docs)}")
        
        return {
            'documents': final_docs,
            'pipeline_info': pipeline_info
        }