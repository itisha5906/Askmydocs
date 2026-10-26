#!/usr/bin/env bash
# setup.sh — Linux / macOS setup script for AskMyDocs
set -euo pipefail

echo "============================================"
echo " AskMyDocs — Linux/macOS Setup"
echo "============================================"

# Create virtual environment
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

# Activate venv
# shellcheck disable=SC1091
source venv/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install dependencies
echo "Installing requirements..."
pip install -r requirements.txt

# Copy .env if it does not exist
if [ ! -f ".env" ]; then
    echo "Copying .env.example to .env ..."
    cp .env.example .env
    echo "IMPORTANT: Edit .env and add your GROQ_API_KEY before running the app."
fi

# Create data directory
mkdir -p data/faiss_index

echo ""
echo "Setup complete!"
echo ""
echo "Next steps:"
echo "  1. Edit .env and set your GROQ_API_KEY"
echo "  2. Run: source venv/bin/activate"
echo "  3. Run: streamlit run app.py"
