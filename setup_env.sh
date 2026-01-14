#!/bin/bash
# Environment setup script for zastupitelstvo-transcriber
# Creates a Python virtual environment and installs dependencies

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo "======================================"
echo "Meeting Transcriber Setup"
echo "======================================"
echo ""

# Check Python version
echo "Checking Python version..."
PYTHON_CMD=""

# Try python3.12 first (recommended)
if command -v python3.12 &> /dev/null; then
    PYTHON_CMD="python3.12"
    echo -e "${GREEN}✓ Found Python 3.12 (recommended)${NC}"
# Try python3.11
elif command -v python3.11 &> /dev/null; then
    PYTHON_CMD="python3.11"
    echo -e "${YELLOW}✓ Found Python 3.11 (should work)${NC}"
# Try python3.10
elif command -v python3.10 &> /dev/null; then
    PYTHON_CMD="python3.10"
    echo -e "${YELLOW}✓ Found Python 3.10 (should work)${NC}"
# Fall back to python3
elif command -v python3 &> /dev/null; then
    PYTHON_CMD="python3"
    PYTHON_VERSION=$($PYTHON_CMD --version 2>&1 | awk '{print $2}')
    echo -e "${YELLOW}⚠ Found python3 version $PYTHON_VERSION${NC}"
    echo -e "${YELLOW}⚠ Python 3.12 is recommended for best compatibility${NC}"
else
    echo -e "${RED}✗ Python 3 not found!${NC}"
    echo "Please install Python 3.10, 3.11, or 3.12"
    exit 1
fi

echo "Using: $PYTHON_CMD"
$PYTHON_CMD --version
echo ""

# Check if venv already exists
if [ -d "venv" ]; then
    echo -e "${YELLOW}⚠ Virtual environment already exists${NC}"
    read -p "Remove and recreate? (y/N) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "Removing existing venv..."
        rm -rf venv
    else
        echo "Keeping existing venv. Exiting."
        exit 0
    fi
fi

# Create virtual environment
echo "Creating virtual environment..."
$PYTHON_CMD -m venv venv
echo -e "${GREEN}✓ Virtual environment created${NC}"
echo ""

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip
echo ""

# Ask which components to install
echo "Which components do you need?"
echo ""
echo "1) Full pipeline (all components) - ~2GB download"
echo "2) Transcriber only - ~2GB download"
echo "3) Analyzer only - ~500MB download"
echo "4) Article generator only - <10MB download"
echo "5) Custom selection"
echo ""
read -p "Enter choice [1-5]: " CHOICE

case $CHOICE in
    1)
        echo "Installing full pipeline..."
        pip install -r requirements.txt
        ;;
    2)
        echo "Installing transcriber..."
        pip install -r requirements-transcriber.txt
        ;;
    3)
        echo "Installing analyzer..."
        pip install -r requirements-analyzer.txt
        ;;
    4)
        echo "Installing article generator..."
        pip install -r requirements-article-generator.txt
        ;;
    5)
        echo "Select components to install:"
        INSTALL_FILES=""

        read -p "Install transcriber? (y/N) " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            INSTALL_FILES="$INSTALL_FILES -r requirements-transcriber.txt"
        fi

        read -p "Install analyzer? (y/N) " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            INSTALL_FILES="$INSTALL_FILES -r requirements-analyzer.txt"
        fi

        read -p "Install article generator? (y/N) " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            INSTALL_FILES="$INSTALL_FILES -r requirements-article-generator.txt"
        fi

        if [ -z "$INSTALL_FILES" ]; then
            echo -e "${RED}No components selected. Exiting.${NC}"
            exit 1
        fi

        echo "Installing selected components..."
        pip install $INSTALL_FILES
        ;;
    *)
        echo -e "${RED}Invalid choice. Exiting.${NC}"
        exit 1
        ;;
esac

echo ""
echo -e "${GREEN}======================================"
echo "✓ Setup Complete!"
echo "======================================${NC}"
echo ""
echo "To activate the environment:"
echo "  source venv/bin/activate"
echo ""
echo "To run the pipeline:"
echo "  python process_meeting.py --audio input/meeting.opus --date 2025-01-15 --number 23"
echo ""
echo "For help:"
echo "  python process_meeting.py --help"
echo ""

# Check for HF_TOKEN
if [ -z "$HF_TOKEN" ]; then
    echo -e "${YELLOW}⚠ Note: HF_TOKEN environment variable not set${NC}"
    echo "The transcriber requires a HuggingFace token for speaker diarization."
    echo "Get your token at: https://huggingface.co/settings/tokens"
    echo ""
    echo "Set it with:"
    echo "  export HF_TOKEN='your_token_here'"
    echo ""
fi
