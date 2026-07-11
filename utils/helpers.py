"""
Utility helper functions for Advanced RAG System
"""
import os
import yaml
from pathlib import Path
from typing import Dict, Any, List
import logging
from datetime import datetime
import json
import requests

def setup_logging(log_level: str = "INFO") -> logging.Logger:
    """
    Setup logging configuration
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR)
    
    Returns:
        Logger instance
    """
    # Create logs directory if not exists
    Path("logs").mkdir(exist_ok=True)
    
    # Configure logging
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(f"logs/rag_system_{datetime.now().strftime('%Y%m%d')}.log"),
            logging.StreamHandler()
        ]
    )
    
    return logging.getLogger(__name__)

def load_config(config_path: str = "config/config.yaml") -> Dict[str, Any]:
    """
    Load configuration from YAML file
    
    Args:
        config_path: Path to configuration file
    
    Returns:
        Configuration dictionary
    """
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        return config
    except Exception as e:
        logging.error(f"Error loading config: {e}")
        raise

def validate_environment() -> bool:
    """
    Validate that all required environment variables are set
    
    Returns:
        True if environment is valid, False otherwise
    """
    required_vars = ['OLLAMA_HOST', 'OLLAMA_MODEL']
    missing_vars = [var for var in required_vars if not os.getenv(var)]
    
    if missing_vars:
        logging.error(f"Missing environment variables: {missing_vars}")
        return False
    return True

def check_ollama_connection() -> bool:
    """
    Check if Ollama server is accessible
    
    Returns:
        True if connection successful, False otherwise
    """
    
    try:
        response = requests.get(f"{os.getenv('OLLAMA_HOST', 'http://localhost:11434')}/api/tags")
        return response.status_code == 200
    except:
        return False

def format_document_metadata(doc_metadata: Dict) -> str:
    """
    Format document metadata for display
    
    Args:
        doc_metadata: Document metadata dictionary
    
    Returns:
        Formatted string
    """
    important_fields = ['source', 'page', 'author', 'date', 'doc_type']
    formatted = []
    
    for field in important_fields:
        if field in doc_metadata:
            formatted.append(f"{field}: {doc_metadata[field]}")
    
    return " | ".join(formatted) if formatted else "No metadata available"

def timer_decorator(func):
    """
    Decorator to measure function execution time
    """
    import time
    from functools import wraps
    
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        logging.debug(f"{func.__name__} took {end_time - start_time:.2f} seconds")
        return result
    return wrapper