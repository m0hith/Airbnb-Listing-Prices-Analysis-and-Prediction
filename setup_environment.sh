#!/bin/bash

# Airbnb Price Analysis - Environment Setup Script
# This script sets up a Python virtual environment for the project

echo " ====== Setting up Python environment for Airbnb Price Analysis ======"

# Check if Python 3 is installed
if ! command -v python3 &> /dev/null; then
    echo "Python 3 is not installed. Please install Python 3.8 or higher."
    exit 1
fi

echo "Python 3 found: $(python3 --version)"

# Create virtual environment
echo "Creating virtual environment..."
python3 -m venv venv

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip

# Install requirements
echo "Installing required packages..."
pip install -r requirements.txt

# Install NLTK data (for textblob)
echo "Downloading NLTK data..."
python3 -c "import nltk; nltk.download('punkt', quiet=True); nltk.download('brown', quiet=True); nltk.download('wordnet', quiet=True)"

echo ""
echo "Environment setup complete!"
echo ""
echo "To activate the environment in the future, run:"
echo "  source venv/bin/activate"
echo ""
echo "To start Jupyter Notebook, run:"
echo " jupyter notebook"
echo ""

