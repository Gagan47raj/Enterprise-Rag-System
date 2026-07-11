"""
Tests for document processing module
"""
import sys
from pathlib import Path
import tempfile
import os

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

import pytest
from src.document_processor import (
    DocumentLoader,
    DocumentChunker,
    ParentChildProcessor,
    MetadataEnhancer,
    DocumentProcessingPipeline
)
from langchain.schema import Document

# Sample test data
SAMPLE_TEXT = """
This is a test document for the Advanced RAG System.
It contains multiple paragraphs to test various processing functions.

The RAG system combines retrieval and generation capabilities.
It uses FAISS for vector storage and BM25 for sparse retrieval.

Machine learning has transformed natural language processing.
Deep learning models can understand context and generate responses.
"""

@pytest.fixture
def sample_document():
    """Create a sample document for testing"""
    return Document(
        page_content=SAMPLE_TEXT,
        metadata={"source": "test.txt", "page": 1}
    )

@pytest.fixture
def document_loader():
    """Create document loader instance"""
    return DocumentLoader()

@pytest.fixture
def document_chunker():
    """Create document chunker instance"""
    return DocumentChunker(chunk_size=200, chunk_overlap=30)

@pytest.fixture
def metadata_enhancer():
    """Create metadata enhancer instance"""
    return MetadataEnhancer()

@pytest.fixture
def processing_pipeline():
    """Create processing pipeline instance"""
    return DocumentProcessingPipeline()

class TestDocumentLoader:
    """Test document loading functionality"""
    
    def test_load_text_file(self, document_loader):
        """Test loading a text file"""
        # Create temporary file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write("Test content")
            temp_path = f.name
        
        try:
            docs = document_loader.load_file(temp_path)
            assert len(docs) > 0
            assert docs[0].page_content.strip() == "Test content"
        finally:
            os.unlink(temp_path)
    
    def test_file_not_found(self, document_loader):
        """Test loading non-existent file"""
        with pytest.raises(FileNotFoundError):
            document_loader.load_file("nonexistent.txt")

class TestDocumentChunker:
    """Test document chunking functionality"""
    
    def test_recursive_chunking(self, document_chunker, sample_document):
        """Test recursive chunking strategy"""
        chunks = document_chunker.chunk_documents(
            [sample_document], 
            strategy="recursive"
        )
        assert len(chunks) > 0
        for chunk in chunks:
            assert len(chunk.page_content) <= 200 + 30  # chunk_size + overlap
            assert 'chunk_id' in chunk.metadata
    
    def test_chunk_statistics(self, document_chunker, sample_document):
        """Test chunk statistics generation"""
        chunks = document_chunker.chunk_documents([sample_document])
        stats = document_chunker.get_chunk_statistics(chunks)
        
        assert 'total_chunks' in stats
        assert 'avg_length' in stats
        assert stats['total_chunks'] == len(chunks)

class TestMetadataEnhancer:
    """Test metadata enhancement"""
    
    def test_date_extraction(self, metadata_enhancer):
        """Test date extraction from text"""
        text_with_dates = "The meeting was on 2024-01-15 and follow-up on 01/20/2024."
        dates = metadata_enhancer.extract_dates(text_with_dates)
        assert len(dates) >= 2
    
    def test_keyword_extraction(self, metadata_enhancer):
        """Test keyword extraction"""
        text = "Machine learning and artificial intelligence are transforming technology. " * 5
        keywords = metadata_enhancer.extract_keywords(text)
        assert len(keywords) > 0
        assert 'machine' in keywords or 'learning' in keywords
    
    def test_enhance_metadata(self, metadata_enhancer, sample_document):
        """Test metadata enhancement"""
        enhanced = metadata_enhancer.enhance_document_metadata([sample_document])
        assert len(enhanced) == 1
        assert 'char_count' in enhanced[0].metadata
        assert 'word_count' in enhanced[0].metadata
        assert 'content_hash' in enhanced[0].metadata

class TestParentChildProcessor:
    """Test parent-child document processing"""
    
    def test_create_parent_child(self, sample_document):
        """Test parent-child document creation"""
        processor = ParentChildProcessor(
            parent_chunk_size=400,
            child_chunk_size=200,
            chunk_overlap=30
        )
        
        parents, children = processor.create_parent_child_documents([sample_document])
        
        assert len(parents) > 0
        assert len(children) > 0
        assert 'parent_id' in parents[0].metadata
        assert 'child_id' in children[0].metadata
        assert parents[0].metadata['doc_type'] == 'parent'
        assert children[0].metadata['doc_type'] == 'child'

class TestProcessingPipeline:
    """Test complete processing pipeline"""
    
    def test_process_text_file(self, processing_pipeline):
        """Test processing a text file"""
        # Create temporary file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write(SAMPLE_TEXT)
            temp_path = f.name
        
        try:
            chunks, parents = processing_pipeline.process_file(
                temp_path,
                strategy="recursive",
                use_parent_child=True
            )
            
            assert len(chunks) > 0
            assert parents is not None
            assert len(parents) > 0
        finally:
            os.unlink(temp_path)
    
    def test_processing_stats(self, processing_pipeline):
        """Test processing statistics"""
        # Create temporary file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write(SAMPLE_TEXT)
            temp_path = f.name
        
        try:
            chunks, parents = processing_pipeline.process_file(
                temp_path,
                use_parent_child=False
            )
            
            stats = processing_pipeline.get_processing_stats(chunks)
            assert 'total_chunks' in stats
            assert stats['total_chunks'] > 0
        finally:
            os.unlink(temp_path)

if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v", "--tb=short"])