"""
Main entry point for Advanced RAG System - Phase 3
"""
import os
import sys
from pathlib import Path
import time

import sys

if sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from dotenv import load_dotenv
from utils.helpers import setup_logging, load_config, validate_environment, timer_decorator
from src.document_processor import DocumentProcessingPipeline
from src.vector_store import HybridVectorStore, VectorStoreFactory

def process_and_index_documents(config):
    """Process documents and create vector indices"""
    logger = setup_logging()
    
    # Step 1: Process documents
    logger.info("="*60)
    logger.info("Phase 3: Vector Store Implementation")
    logger.info("="*60)
    
    logger.info("\n📄 Step 1: Processing Documents...")
    pipeline = DocumentProcessingPipeline(config)
    
    # Process sample documents
    sample_dir = project_root / 'data' / 'documents'
    chunks, parents = pipeline.process_directory(
        str(sample_dir),
        strategy="recursive",
        use_parent_child=True,
        enhance_metadata=True
    )
    
    logger.info(f"✅ Processed {len(chunks)} chunks and {len(parents) if parents else 0} parent documents")
    
    # Step 2: Create vector store
    logger.info("\n🔢 Step 2: Creating Vector Store...")
    
    vector_store = VectorStoreFactory.create_vector_store(
        store_type="hybrid",
        config=config
    )
    
    # Index documents
    index_path = project_root / 'data' / 'vector_store' / 'main_index'
    vector_store.index_documents(chunks, str(index_path))
    
    # Get statistics
    stats = vector_store.get_stats()
    logger.info("✅ Vector store created successfully")
    
    # Step 3: Test search
    logger.info("\n🔍 Step 3: Testing Search...")
    test_queries = [
        "What is FAISS used for?",
        "Explain machine learning",
        "How does BM25 work?"
    ]
    
    for query in test_queries:
        logger.info(f"\nQuery: '{query}'")
        results = vector_store.search(query, k=2, search_type="hybrid")
        
        for i, doc in enumerate(results, 1):
            logger.info(f"  Result {i}: {doc.page_content[:100]}...")
    
    return vector_store, stats

@timer_decorator
def main():
    """Main function"""
    # Load environment variables
    load_dotenv('config/.env')
    
    # Setup logging
    logger = setup_logging()
    
    # Load configuration
    config = load_config()
    
    # Validate environment
    if not validate_environment():
        logger.error("Environment validation failed")
        sys.exit(1)
    
    try:
        # Process and index documents
        vector_store, stats = process_and_index_documents(config)
        
        # Display results
        print("\n" + "="*60)
        print("✅ Phase 3 Complete: Vector Store Implementation")
        print("="*60)
        print(f"📊 Vector Store Statistics:")
        print(f"   • FAISS initialized: {stats['faiss_initialized']}")
        print(f"   • BM25 initialized: {stats['bm25_initialized']}")
        print(f"   • Embedding dimension: {stats['embedding_dimension']}")
        if 'faiss_index_size' in stats:
            print(f"   • FAISS vectors: {stats['faiss_index_size']}")
        if 'bm25_doc_count' in stats:
            print(f"   • BM25 documents: {stats['bm25_doc_count']}")
        print("="*60)
        print("\n✅ Ready for Phase 4: Advanced Retrievers!")
        
    except Exception as e:
        logger.error(f"Error in main: {e}")
        raise

if __name__ == "__main__":
    main()