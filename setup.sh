#!/bin/bash

echo "🚀 Setting up Advanced RAG System..."
echo "===================================="

# Check Python version
python_version=$(python3 --version 2>&1 | awk '{print $2}')
echo "Python version: $python_version"

# Create virtual environment
echo ""
echo "📦 Creating virtual environment..."
python3 -m venv venv

# Activate virtual environment
echo "🔌 Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo "⬆️  Upgrading pip..."
pip install --upgrade pip

# Install requirements
echo ""
echo "📥 Installing dependencies..."
pip install -r requirements.txt

# Create directories
echo ""
echo "📁 Creating project directories..."
mkdir -p data/documents
mkdir -p data/vector_store
mkdir -p logs
mkdir -p config

# Setup environment
echo ""
echo "⚙️  Setting up environment..."
if [ ! -f config/.env ]; then
    cp config/.env.example config/.env
    echo "Created .env file from example"
fi

# Pull Ollama model
echo ""
echo "🤖 Pulling Mistral model..."
if command -v ollama &> /dev/null; then
    ollama pull mistral
    echo "Mistral model pulled successfully"
else
    echo "⚠️  Ollama not found. Please install Ollama and pull Mistral manually."
fi

# Download NLTK data
echo ""
echo "📚 Downloading NLTK data..."
python3 -c "import nltk; nltk.download('punkt'); nltk.download('averaged_perceptron_tagger')"

echo ""
echo "✅ Setup complete!"
echo "===================================="
echo ""
echo "To run the application:"
echo "1. Activate virtual environment: source venv/bin/activate"
echo "2. Run Streamlit: streamlit run app.py"
echo ""
echo "For more information, see README.md"