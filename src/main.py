"""
Main entry point for Advanced RAG System
"""
import os
import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from dotenv import load_dotenv
from utils.helpers import setup_logging, load_config, validate_environment, timer_decorator
from src.document_processor import DocumentProcessingPipeline

def process_sample_documents(config):
    """Process sample documents for testing"""
    logger = setup_logging()
    
    # Initialize pipeline
    pipeline = DocumentProcessingPipeline(config)
    
    # Check if sample documents exist
    sample_dir = project_root / 'data' / 'documents'
    if not any(sample_dir.iterdir()):
        logger.warning("No documents found in data/documents/")
        logger.info("Creating sample document for testing...")
        
        # Create sample document
        sample_file = sample_dir / 'sample.txt'
        with open(sample_file, 'w') as f:
            f.write("""
            Advanced RAG System Documentation
            
            The Retrieval-Augmented Generation (RAG) system combines the power of 
            large language models with external knowledge retrieval. This system 
            implements several advanced techniques to improve retrieval accuracy 
            and response quality.
            
            Key Features:
            1. MultiQuery Retriever: Generates multiple query variations
            2. Parent Document Retriever: Maintains document context
            3. Contextual Compression: Optimizes retrieved content
            4. Reranker: Improves retrieval relevance
            5. Metadata Filtering: Smart document filtering
            
            Technical Implementation:
            The system uses FAISS for dense vector retrieval and BM25 for sparse 
            retrieval. This hybrid approach ensures both semantic understanding 
            and keyword matching capabilities.
            
            Performance Considerations:
            The system is optimized for local deployment using Ollama and Mistral,
            providing fast inference without external API dependencies.
            """)
    
    # Process documents
    logger.info("Processing documents...")
    chunks, parents = pipeline.process_directory(
        str(sample_dir),
        strategy="recursive",
        use_parent_child=True,
        enhance_metadata=True
    )
    
    # Display statistics
    stats = pipeline.get_processing_stats(chunks, parents)
    logger.info("Processing complete!")
    
    return chunks, parents, stats

@timer_decorator
def main():
    """Main function to run the RAG system"""
    # Load environment variables
    load_dotenv('config/.env')
    
    # Setup logging
    logger = setup_logging()
    logger.info("="*60)
    logger.info("Starting Advanced RAG System - Phase 2")
    logger.info("="*60)
    
    # Load configuration
    config = load_config()
    logger.info("Configuration loaded successfully")
    
    # Validate environment
    if not validate_environment():
        logger.error("Environment validation failed")
        sys.exit(1)
    
    logger.info("Environment validated successfully")
    
    # Process documents
    try:
        chunks, parents, stats = process_sample_documents(config)
        
        # Display results
        print("\n" + "="*50)
        print("✅ Document Processing Complete!")
        print("="*50)
        print(f"📄 Total chunks created: {stats['total_chunks']}")
        print(f"📊 Average chunk size: {stats['chunk_stats']['avg_length']:.0f} chars")
        print(f"📈 Unique sources: {stats['unique_sources']}")
        
        if parents:
            print(f"👨‍👧 Parent documents: {stats['total_parents']}")
        
        print("="*50)
        print("\nReady for Phase 3: Vector Store Implementation")
        
    except Exception as e:
        logger.error(f"Error during processing: {e}")
        raise

if __name__ == "__main__":
    main()