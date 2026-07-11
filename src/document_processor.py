"""
Document Processing Module for Advanced RAG System
Handles document loading, chunking, and metadata enhancement
"""
import os
import re
import hashlib
from typing import List, Dict, Any, Tuple, Optional
from pathlib import Path
from datetime import datetime
import logging

from langchain_core.documents import Document
from langchain_text_splitters import (
    RecursiveCharacterTextSplitter,
    CharacterTextSplitter,
    TokenTextSplitter,
    MarkdownTextSplitter,
)
from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    Docx2txtLoader,
    UnstructuredFileLoader,
    DirectoryLoader,
    CSVLoader
)

from utils.helpers import load_config, timer_decorator
from utils.text_processor import clean_text, split_into_sentences, extract_metadata_from_text

logger = logging.getLogger(__name__)

class DocumentLoader:
    """Handle loading of various document types"""
    
    SUPPORTED_FORMATS = {
        '.pdf': PyPDFLoader,
        '.txt': TextLoader,
        '.docx': Docx2txtLoader,
        '.csv': CSVLoader,
    }
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or load_config()
    
    def load_file(self, file_path: str) -> List[Document]:
        """
        Load a single document file
        
        Args:
            file_path: Path to document file
        
        Returns:
            List of Document objects
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        file_extension = file_path.suffix.lower()
        
        if file_extension not in self.SUPPORTED_FORMATS:
            # Use unstructured loader as fallback
            logger.warning(f"Using unstructured loader for {file_extension}")
            loader = UnstructuredFileLoader(str(file_path))
        else:
            loader_class = self.SUPPORTED_FORMATS[file_extension]
            loader = loader_class(str(file_path))
        
        try:
            documents = loader.load()
            logger.info(f"Loaded {len(documents)} documents from {file_path.name}")
            
            # Add source metadata
            for doc in documents:
                doc.metadata['source'] = str(file_path)
                doc.metadata['filename'] = file_path.name
                doc.metadata['file_type'] = file_extension
                doc.metadata['load_date'] = datetime.now().isoformat()
            
            return documents
        
        except Exception as e:
            logger.error(f"Error loading {file_path}: {e}")
            raise
    
    def load_directory(self, 
                      directory_path: str, 
                      glob_pattern: str = "**/*",
                      recursive: bool = True) -> List[Document]:
        """
        Load all supported documents from a directory
        
        Args:
            directory_path: Path to directory
            glob_pattern: Pattern to match files
            recursive: Search subdirectories
        
        Returns:
            List of Document objects
        """
        directory_path = Path(directory_path)
        
        if not directory_path.exists():
            raise FileNotFoundError(f"Directory not found: {directory_path}")
        
        all_documents = []
        
        # Find supported files
        supported_patterns = list(self.SUPPORTED_FORMATS.keys())
        files_found = []
        
        for pattern in supported_patterns:
            if recursive:
                files = directory_path.rglob(f"*{pattern}")
            else:
                files = directory_path.glob(f"*{pattern}")
            files_found.extend(files)
        
        # Load each file
        for file_path in files_found:
            try:
                documents = self.load_file(str(file_path))
                all_documents.extend(documents)
                logger.info(f"Loaded {file_path.name}: {len(documents)} pages/sections")
            except Exception as e:
                logger.error(f"Skipping {file_path.name}: {e}")
        
        logger.info(f"Total documents loaded: {len(all_documents)}")
        return all_documents


class DocumentChunker:
    """Advanced document chunking with multiple strategies"""
    
    def __init__(self, 
                 chunk_size: int = 500, 
                 chunk_overlap: int = 50,
                 strategy: str = "recursive"):
        """
        Initialize document chunker
        
        Args:
            chunk_size: Size of each chunk
            chunk_overlap: Overlap between chunks
            strategy: Chunking strategy (recursive, character, token, semantic)
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.strategy = strategy
        
        # Initialize splitters
        self.splitters = {
            'recursive': RecursiveCharacterTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                separators=["\n\n", "\n", ". ", "! ", "? ", " ", ""],
                length_function=len
            ),
            'character': CharacterTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                separator="\n"
            ),
            'token': TokenTextSplitter(
                chunk_size=chunk_size // 4,  # Convert char size to approximate tokens
                chunk_overlap=chunk_overlap // 4
            ),
            'markdown': MarkdownTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap
            )
        }
    
    @timer_decorator
    def chunk_documents(self, 
                       documents: List[Document], 
                       strategy: Optional[str] = None) -> List[Document]:
        """
        Chunk documents using specified strategy
        
        Args:
            documents: List of documents to chunk
            strategy: Chunking strategy (defaults to initialized strategy)
        
        Returns:
            List of chunked documents
        """
        strategy = strategy or self.strategy
        
        if strategy not in self.splitters:
            raise ValueError(f"Unknown strategy: {strategy}. Available: {list(self.splitters.keys())}")
        
        splitter = self.splitters[strategy]
        chunks = splitter.split_documents(documents)
        
        # Add chunk metadata
        for i, chunk in enumerate(chunks):
            chunk.metadata['chunk_id'] = f"chunk_{i}"
            chunk.metadata['chunk_index'] = i
            chunk.metadata['chunk_strategy'] = strategy
            chunk.metadata['chunk_size'] = len(chunk.page_content)
        
        logger.info(f"Chunked {len(documents)} documents into {len(chunks)} chunks using {strategy} strategy")
        return chunks
    
    def get_chunk_statistics(self, chunks: List[Document]) -> Dict[str, Any]:
        """
        Get statistics about chunks
        
        Args:
            chunks: List of chunked documents
        
        Returns:
            Dictionary with chunk statistics
        """
        lengths = [len(chunk.page_content) for chunk in chunks]
        
        stats = {
            "total_chunks": len(chunks),
            "avg_length": sum(lengths) / len(lengths) if lengths else 0,
            "min_length": min(lengths) if lengths else 0,
            "max_length": max(lengths) if lengths else 0,
            "median_length": sorted(lengths)[len(lengths)//2] if lengths else 0,
            "total_characters": sum(lengths),
            "chunks_below_100": sum(1 for l in lengths if l < 100),
            "chunks_above_1000": sum(1 for l in lengths if l > 1000)
        }
        
        return stats


class ParentChildProcessor:
    """Handle parent-child document relationships"""
    
    def __init__(self, 
                 parent_chunk_size: int = 2000,
                 child_chunk_size: int = 500,
                 chunk_overlap: int = 50):
        """
        Initialize parent-child processor
        
        Args:
            parent_chunk_size: Size of parent chunks
            child_chunk_size: Size of child chunks
            chunk_overlap: Overlap between chunks
        """
        self.parent_splitter = RecursiveCharacterTextSplitter(
            chunk_size=parent_chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
        
        self.child_splitter = RecursiveCharacterTextSplitter(
            chunk_size=child_chunk_size,
            chunk_overlap=chunk_overlap // 2,  # Less overlap for children
            separators=["\n\n", "\n", ". ", " ", ""]
        )
        
        self.parent_chunk_size = parent_chunk_size
        self.child_chunk_size = child_chunk_size
    
    @timer_decorator
    def create_parent_child_documents(self, 
                                     documents: List[Document]) -> Tuple[List[Document], List[Document]]:
        """
        Create parent and child document pairs
        
        Args:
            documents: Original documents
        
        Returns:
            Tuple of (parent_docs, child_docs)
        """
        # Create parent documents (larger chunks for context)
        parent_docs = self.parent_splitter.split_documents(documents)
        
        # Create child documents (smaller chunks for retrieval)
        child_docs = self.child_splitter.split_documents(documents)
        
        # Add metadata to distinguish parent and child
        for i, parent_doc in enumerate(parent_docs):
            parent_doc.metadata.update({
                'parent_id': f"parent_{i}",
                'doc_type': 'parent',
                'chunk_size_category': 'large',
                'content_hash': hashlib.md5(parent_doc.page_content.encode()).hexdigest()
            })
        
        for i, child_doc in enumerate(child_docs):
            child_doc.metadata.update({
                'child_id': f"child_{i}",
                'doc_type': 'child',
                'chunk_size_category': 'small',
                'content_hash': hashlib.md5(child_doc.page_content.encode()).hexdigest()
            })
        
        # Link children to parents based on content overlap
        self._link_children_to_parents(parent_docs, child_docs)
        
        logger.info(f"Created {len(parent_docs)} parent and {len(child_docs)} child documents")
        return parent_docs, child_docs
    
    def _link_children_to_parents(self, 
                                 parent_docs: List[Document], 
                                 child_docs: List[Document]):
        """Link child documents to their parent documents"""
        for child_doc in child_docs:
            # Find parent that contains the child content
            for parent_doc in parent_docs:
                if child_doc.page_content[:100] in parent_doc.page_content:
                    child_doc.metadata['parent_id'] = parent_doc.metadata['parent_id']
                    if 'child_ids' not in parent_doc.metadata:
                        parent_doc.metadata['child_ids'] = []
                    parent_doc.metadata['child_ids'].append(child_doc.metadata['child_id'])
                    break


class MetadataEnhancer:
    """Enhance document metadata with extracted information"""
    
    @staticmethod
    def extract_dates(text: str) -> List[str]:
        """Extract dates from text"""
        date_patterns = [
            r'\d{4}-\d{2}-\d{2}',  # YYYY-MM-DD
            r'\d{2}/\d{2}/\d{4}',  # MM/DD/YYYY
            r'\b(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|'
            r'Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)'
            r'\s+\d{1,2},?\s+\d{4}\b'
        ]
        
        dates = []
        for pattern in date_patterns:
            dates.extend(re.findall(pattern, text, re.IGNORECASE))
        return list(set(dates))
    
    @staticmethod
    def extract_keywords(text: str, max_keywords: int = 10) -> List[str]:
        """Extract potential keywords from text (simplified)"""
        # Remove common words and get unique significant words
        common_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 
                       'to', 'for', 'of', 'with', 'by', 'from', 'is', 'are', 'was', 'were'}
        
        words = re.findall(r'\b[a-zA-Z]{4,}\b', text.lower())
        significant_words = [w for w in words if w not in common_words]
        
        # Count frequency and get top keywords
        from collections import Counter
        word_freq = Counter(significant_words)
        keywords = [word for word, _ in word_freq.most_common(max_keywords)]
        
        return keywords
    
    @staticmethod
    def generate_summary(text: str, max_sentences: int = 3) -> str:
        """Generate simple extractive summary"""
        sentences = split_into_sentences(text)
        if not sentences:
            return text[:200]
        
        # Take first few sentences as summary
        summary = ' '.join(sentences[:max_sentences])
        return summary[:500]  # Limit summary length
    
    def enhance_document_metadata(self, documents: List[Document]) -> List[Document]:
        """
        Enhance metadata for all documents
        
        Args:
            documents: List of documents to enhance
        
        Returns:
            Documents with enhanced metadata
        """
        for doc in documents:
            # Basic text statistics
            doc.metadata.update({
                'char_count': len(doc.page_content),
                'word_count': len(doc.page_content.split()),
                'sentence_count': len(split_into_sentences(doc.page_content)),
                'enhanced_date': datetime.now().isoformat()
            })
            
            # Extract dates
            dates = self.extract_dates(doc.page_content)
            if dates:
                doc.metadata['extracted_dates'] = dates
            
            # Extract keywords
            keywords = self.extract_keywords(doc.page_content)
            if keywords:
                doc.metadata['keywords'] = keywords
            
            # Generate summary
            doc.metadata['summary'] = self.generate_summary(doc.page_content)
            
            # Content hash for deduplication
            doc.metadata['content_hash'] = hashlib.md5(
                doc.page_content.encode()
            ).hexdigest()
        
        logger.info(f"Enhanced metadata for {len(documents)} documents")
        return documents


class DocumentProcessingPipeline:
    """Complete document processing pipeline"""
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize processing pipeline
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or load_config()
        
        # Initialize components
        self.loader = DocumentLoader(self.config)
        self.metadata_enhancer = MetadataEnhancer()
        
        # Get chunking parameters from config
        parent_config = self.config.get('parent_document', {})
        self.chunk_size = parent_config.get('child_chunk_size', 500)
        self.chunk_overlap = parent_config.get('child_chunk_overlap', 50)
        self.parent_chunk_size = parent_config.get('parent_chunk_size', 2000)
        
        self.chunker = DocumentChunker(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap
        )
        
        self.parent_child_processor = ParentChildProcessor(
            parent_chunk_size=self.parent_chunk_size,
            child_chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap
        )
    
    @timer_decorator
    def process_file(self, 
                    file_path: str, 
                    strategy: str = "recursive",
                    use_parent_child: bool = True,
                    enhance_metadata: bool = True) -> Tuple[List[Document], Optional[List[Document]]]:
        """
        Process a single file through the pipeline
        
        Args:
            file_path: Path to the file
            strategy: Chunking strategy
            use_parent_child: Whether to create parent-child documents
            enhance_metadata: Whether to enhance metadata
        
        Returns:
            Tuple of (chunks, parent_docs or None)
        """
        logger.info(f"Processing file: {file_path}")
        
        # Step 1: Load document
        documents = self.loader.load_file(file_path)
        
        # Step 2: Clean text
        for doc in documents:
            doc.page_content = clean_text(doc.page_content)
        
        # Step 3: Process based on strategy
        if use_parent_child:
            parent_docs, child_docs = self.parent_child_processor.create_parent_child_documents(documents)
            
            # Enhance metadata
            if enhance_metadata:
                child_docs = self.metadata_enhancer.enhance_document_metadata(child_docs)
                parent_docs = self.metadata_enhancer.enhance_document_metadata(parent_docs)
            
            return child_docs, parent_docs
        else:
            # Regular chunking
            chunks = self.chunker.chunk_documents(documents, strategy)
            
            # Enhance metadata
            if enhance_metadata:
                chunks = self.metadata_enhancer.enhance_document_metadata(chunks)
            
            return chunks, None
    
    @timer_decorator
    def process_directory(self, 
                         directory_path: str,
                         strategy: str = "recursive",
                         use_parent_child: bool = True,
                         enhance_metadata: bool = True) -> Tuple[List[Document], Optional[List[Document]]]:
        """
        Process all documents in a directory
        
        Args:
            directory_path: Path to directory
            strategy: Chunking strategy
            use_parent_child: Whether to create parent-child documents
            enhance_metadata: Whether to enhance metadata
        
        Returns:
            Tuple of (all_chunks, all_parents or None)
        """
        logger.info(f"Processing directory: {directory_path}")
        
        # Load all documents
        documents = self.loader.load_directory(directory_path)
        
        if not documents:
            logger.warning("No documents found in directory")
            return [], None
        
        # Step 2: Clean text
        for doc in documents:
            doc.page_content = clean_text(doc.page_content)
        
        # Step 3: Process based on strategy
        if use_parent_child:
            parent_docs, child_docs = self.parent_child_processor.create_parent_child_documents(documents)
            
            # Enhance metadata
            if enhance_metadata:
                child_docs = self.metadata_enhancer.enhance_document_metadata(child_docs)
                parent_docs = self.metadata_enhancer.enhance_document_metadata(parent_docs)
            
            return child_docs, parent_docs
        else:
            # Regular chunking
            chunks = self.chunker.chunk_documents(documents, strategy)
            
            # Enhance metadata
            if enhance_metadata:
                chunks = self.metadata_enhancer.enhance_document_metadata(chunks)
            
            return chunks, None
    
    def get_processing_stats(self, 
                            chunks: List[Document], 
                            parent_docs: Optional[List[Document]] = None) -> Dict[str, Any]:
        """
        Get processing statistics
        
        Args:
            chunks: List of document chunks
            parent_docs: Optional list of parent documents
        
        Returns:
            Statistics dictionary
        """
        stats = {
            'total_chunks': len(chunks),
            'chunk_stats': self.chunker.get_chunk_statistics(chunks),
            'unique_sources': len(set(doc.metadata.get('source', '') for doc in chunks)),
            'total_characters': sum(len(doc.page_content) for doc in chunks)
        }
        
        if parent_docs:
            stats['total_parents'] = len(parent_docs)
            stats['parent_stats'] = self.chunker.get_chunk_statistics(parent_docs)
        
        return stats


# Utility function for quick processing
def quick_process_file(file_path: str) -> List[Document]:
    """
    Quick processing function for testing
    
    Args:
        file_path: Path to file
    
    Returns:
        Processed chunks
    """
    pipeline = DocumentProcessingPipeline()
    chunks, _ = pipeline.process_file(
        file_path, 
        strategy="recursive",
        use_parent_child=False,
        enhance_metadata=True
    )
    return chunks