"""
Main entry point for Advanced RAG System - Phase 4
"""
import os
import sys
from pathlib import Path
import time

project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from dotenv import load_dotenv
from utils.helpers import setup_logging, load_config, validate_environment, timer_decorator
from src.document_processor import DocumentProcessingPipeline
from src.vector_store import HybridVectorStore, VectorStoreFactory, EmbeddingManager
from src.advanced_retrievers import AdvancedRetrievalPipeline

@timer_decorator
def main():
    """Main function"""
    load_dotenv('config/.env')
    logger = setup_logging()
    
    logger.info("="*60)
    logger.info("Phase 4: Advanced Retrievers Implementation")
    logger.info("="*60)
    
    config = load_config()
    
    if not validate_environment():
        logger.error("Environment validation failed")
        sys.exit(1)
    
    try:
        # Setup components
        logger.info("\n📦 Setting up components...")
        
        # Process documents
        pipeline = DocumentProcessingPipeline(config)
        sample_dir = project_root / 'data' / 'documents'
        chunks, parents = pipeline.process_directory(str(sample_dir))
        
        # Create vector store
        embedding_manager = EmbeddingManager(
            model_name=config['models']['embedding_model']
        )
        vector_store = VectorStoreFactory.create_vector_store(
            store_type="hybrid",
            config=config,
            embedding_manager=embedding_manager
        )
        vector_store.index_documents(chunks)
        
        # Create advanced retrieval pipeline
        retriever = AdvancedRetrievalPipeline(
            vector_store,
            embedding_manager,
            config=config
        )
        
        logger.info("✅ Setup complete")
        
        # Test different retrieval strategies
        test_queries = [
            "What is the RAG system?",
            "How does document retrieval work?",
            "Explain advanced retrieval techniques"
        ]
        
        logger.info("\n🔍 Testing Retrieval Strategies...")
        
        for query in test_queries:
            logger.info(f"\n{'='*60}")
            logger.info(f"Query: '{query}'")
            
            # Test 1: Basic hybrid search
            start = time.time()
            basic_results = vector_store.search(query, k=3, search_type="hybrid")
            basic_time = time.time() - start
            logger.info(f"Basic Search: {len(basic_results)} docs in {basic_time:.3f}s")
            
            # Test 2: Full advanced pipeline
            result = retriever.retrieve(
                query,
                k=3,
                stages={
                    'multiquery': True,
                    'parent_docs': False,
                    'compression': True,
                    'reranker': True
                }
            )
            
            info = result['pipeline_info']
            logger.info(f"Advanced Pipeline: {info['document_counts']}")
            logger.info(f"Total time: {info['total_time']:.3f}s")
            
            # Show top result
            if result['documents']:
                top_doc, top_score = result['documents'][0]
                logger.info(f"Top result: {top_doc.page_content[:100]}...")
        
        # Summary
        print("\n" + "="*60)
        print("✅ Phase 4 Complete: Advanced Retrievers")
        print("="*60)
        print("Implemented Features:")
        print("  • MultiQuery Retriever")
        print("  • Parent Document Retriever")
        print("  • Contextual Compression (LLM, Embeddings, Cross-Encoder)")
        print("  • Reranker (Cross-Encoder + Hybrid)")
        print("  • Complete Retrieval Pipeline")
        print("="*60)
        print("\n✅ Ready for Phase 5: Streamlit UI Integration!")
        
    except Exception as e:
        logger.error(f"Error: {e}")
        raise

if __name__ == "__main__":
    main()