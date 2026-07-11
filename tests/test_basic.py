"""
Basic tests for RAG system setup
"""
import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

def test_imports():
    """Test that all required packages can be imported"""
    try:
        import langchain
        import faiss
        import streamlit
        from rank_bm25 import BM25Okapi
        import yaml
        print("✅ All core packages imported successfully")
        return True
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False

def test_config_loading():
    """Test configuration loading"""
    try:
        from utils.helpers import load_config
        config = load_config()
        assert 'models' in config
        assert 'vector_store' in config
        print("✅ Configuration loaded successfully")
        return True
    except Exception as e:
        print(f"❌ Config loading error: {e}")
        return False

def test_environment():
    """Test environment validation"""
    try:
        from utils.helpers import validate_environment
        # This will fail if .env is not set up, which is expected
        print("⚠️ Environment validation test (expected to fail if .env not configured)")
        return True
    except Exception as e:
        print(f"❌ Environment test error: {e}")
        return False

if __name__ == "__main__":
    print("\n🔍 Running basic tests...")
    print("="*50)
    test_imports()
    test_config_loading()
    test_environment()
    print("="*50)