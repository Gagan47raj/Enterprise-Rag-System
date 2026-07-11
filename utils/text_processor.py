"""
Text processing utilities for document handling
"""
import re
from typing import List, Dict
import nltk
from nltk.tokenize import sent_tokenize

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

def clean_text(text: str) -> str:
    """
    Clean and normalize text
    
    Args:
        text: Input text
    
    Returns:
        Cleaned text
    """
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text)
    
    # Remove special characters but keep punctuation
    text = re.sub(r'[^\w\s.,!?;:()-]', '', text)
    
    # Normalize line breaks
    text = text.replace('\n', ' ').replace('\r', ' ')
    
    return text.strip()

def split_into_sentences(text: str) -> List[str]:
    """
    Split text into sentences
    
    Args:
        text: Input text
    
    Returns:
        List of sentences
    """
    return sent_tokenize(text)

def extract_metadata_from_text(text: str) -> Dict:
    """
    Extract basic metadata from text
    
    Args:
        text: Input text
    
    Returns:
        Dictionary with extracted metadata
    """
    metadata = {
        'char_count': len(text),
        'word_count': len(text.split()),
        'sentence_count': len(sent_tokenize(text)),
    }
    return metadata

def truncate_text(text: str, max_length: int = 1000) -> str:
    """
    Truncate text to max length while keeping complete sentences
    
    Args:
        text: Input text
        max_length: Maximum character length
    
    Returns:
        Truncated text
    """
    if len(text) <= max_length:
        return text
    
    sentences = sent_tokenize(text)
    truncated = ""
    
    for sentence in sentences:
        if len(truncated) + len(sentence) <= max_length:
            truncated += sentence + " "
        else:
            break
    
    return truncated.strip()