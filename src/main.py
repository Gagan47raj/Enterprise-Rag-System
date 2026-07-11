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
from utils.helpers import setup_logging, load_config, validate_environment

def main():
    """Main function to run the RAG system"""
    # Load environment variables
    load_dotenv('config/.env')
    
    # Setup logging
    logger = setup_logging()
    logger.info("Starting Advanced RAG System...")
    
    # Load configuration
    config = load_config()
    logger.info("Configuration loaded successfully")
    
    # Validate environment
    if not validate_environment():
        logger.error("Environment validation failed")
        sys.exit(1)
    
    logger.info("Environment validated successfully")
    
    # System ready
    logger.info("Advanced RAG System is ready!")
    print("\n" + "="*50)
    print("🚀 Advanced RAG System Initialized Successfully!")
    print("="*50)
    print(f"• LLM Model: {config['models']['llm']}")
    print(f"• Embedding Model: {config['models']['embedding_model']}")
    print(f"• Vector Store: {config['vector_store']['type']}")
    print("="*50)

if __name__ == "__main__":
    main()